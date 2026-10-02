// cuSPARSE on complex values, checked against dense complex arithmetic done
// here on the host: what torch.sparse reaches with complex64 and complex128
// tensors, and what a complex iterative solver does.
//
//   SpMV, SpMM     A, A^T and A^H; B, B^T and B^H; a real A with complex
//                  vectors; row- and column-major dense operands
//   SDDMM          A and A^T, B and B^T. NVIDIA documents no conjugate
//                  transpose here; its 13.0 takes one and computes neither
//                  A^H B nor anything else a caller could mean (for op(A) =
//                  A^H, one term of each inner product), so it is not exercised
//   SpSV, SpSM     lower and upper, A and A^H, unit and non-unit diagonal
//   SpGEMM         sparse times sparse
//   conversion     sparse to dense; dense to CSR and to CSC
//   csrgeam2       C = alpha A + beta B in cuComplex and cuDoubleComplex
//   refusals       what an RTX 3060's cuSPARSE 13.0 refuses, with its status:
//                  a conjugate transpose of a real operand (INVALID_VALUE), a
//                  complex type combination it does not take (NOT_SUPPORTED),
//                  SpSM with op(B) = B^H (NOT_SUPPORTED)
//
// Single precision is compared at 2e-5 of the largest reference value, double
// at 1e-12: NVIDIA's accumulates in the compute type, the shim in double, so
// the two agree to rounding rather than bit for bit.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}
static void expect(int got, int want, const char* what) {
  std::printf("%-4s %s (status %d, expected %d)\n", got == want ? "ok" : "FAIL", what, got, want);
  if (got != want) ++failures;
}

#define CK(x)                                                                    \
  do {                                                                           \
    const int r_ = (int)(x);                                                     \
    if (r_ != 0) {                                                               \
      std::printf("FAIL %s returned %d\n", #x, r_);                             \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

template <class T> T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

template <class R> struct Prec;
template <> struct Prec<float> {
  static constexpr cudaDataType real = CUDA_R_32F, cplx = CUDA_C_32F;
  static constexpr double tol = 2e-5;
  static constexpr const char* name = "C_32F";
};
template <> struct Prec<double> {
  static constexpr cudaDataType real = CUDA_R_64F, cplx = CUDA_C_64F;
  static constexpr double tol = 1e-12;
  static constexpr const char* name = "C_64F";
};

static double hash01(int i, int j, int seed) {
  return std::fmod(std::fabs(std::sin(12.9898 * (i + 1) + 78.233 * (j + 1) + seed)) * 43758.5453, 1.0);
}
static cdouble cval(int i, int j, int seed) {
  return {std::cos(0.7 * i - 0.3 * j + seed) * 2.0, std::sin(0.4 * i + 0.9 * j - seed)};
}

// A sparse complex matrix: dense (row-major) on the host and its CSR arrays.
struct CSparse {
  int m, n;
  std::vector<cdouble> dense;
  std::vector<int> off, col;
  std::vector<cdouble> val;
};
static CSparse make_sparse(int m, int n, int seed, double density = 0.45) {
  CSparse s{m, n, std::vector<cdouble>((size_t)m * n), {0}, {}, {}};
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      if (hash01(i, j, seed) < density) {
        const cdouble v = cval(i, j, seed);
        s.dense[(size_t)i * n + j] = v;
        s.col.push_back(j);
        s.val.push_back(v);
      }
    s.off.push_back((int)s.col.size());
  }
  return s;
}

template <class R> std::vector<std::complex<R>> narrow(const std::vector<cdouble>& v) {
  std::vector<std::complex<R>> out;
  for (const cdouble& x : v) out.push_back({(R)x.real(), (R)x.imag()});
  return out;
}
template <class R> std::vector<cdouble> widen(const std::vector<std::complex<R>>& v) {
  return std::vector<cdouble>(v.begin(), v.end());
}
// The values rounded as the device holds them, so the reference sees what the library sees.
template <class R> cdouble rnd(cdouble v) { return {(double)(R)v.real(), (double)(R)v.imag()}; }

static double max_rel(const std::vector<cdouble>& ref, const std::vector<cdouble>& got) {
  double err = 0, scale = 1e-30;
  for (size_t i = 0; i < ref.size(); ++i) {
    err = std::fmax(err, std::abs(ref[i] - got[i]));
    scale = std::fmax(scale, std::abs(ref[i]));
  }
  return err / scale;
}

static cusparseHandle_t h;
static const cusparseOperation_t kOps[3] = {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                            CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE};
static const char* kOpName[3] = {"N", "T", "C"};
// op(M)(i, j) for a row-major rows x cols matrix.
static cdouble op_at(const std::vector<cdouble>& d, int cols, int o, int i, int j) {
  if (o == 0) return d[(size_t)i * cols + j];
  const cdouble v = d[(size_t)j * cols + i];
  return o == 2 ? std::conj(v) : v;
}

template <class R> static void spmv() {
  using C = std::complex<R>;
  const cudaDataType ct = Prec<R>::cplx;
  const int m = 7, n = 5;
  CSparse a = make_sparse(m, n, 3);
  for (auto& v : a.val) v = rnd<R>(v);
  for (auto& v : a.dense) v = rnd<R>(v);
  int* doff = upload(a.off);
  int* dcol = upload(a.col);
  C* dval = upload(narrow<R>(a.val));
  const C alpha((R)1.5, (R)-0.5), beta((R)0.25, (R)0.75);
  for (int o = 0; o < 3; ++o) {
    const int rows = o ? n : m, cols = o ? m : n;
    std::vector<cdouble> x, y0;
    for (int i = 0; i < cols; ++i) x.push_back(rnd<R>(cval(i, 1, 5)));
    for (int i = 0; i < rows; ++i) y0.push_back(rnd<R>(cval(i, 2, 7)));
    C* dx = upload(narrow<R>(x));
    C* dy = upload(narrow<R>(y0));
    cusparseSpMatDescr_t A;
    cusparseDnVecDescr_t X, Y;
    CK(cusparseCreateCsr(&A, m, n, (int64_t)a.val.size(), doff, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, ct));
    CK(cusparseCreateDnVec(&X, cols, dx, ct));
    CK(cusparseCreateDnVec(&Y, rows, dy, ct));
    size_t bytes = 0;
    CK(cusparseSpMV_bufferSize(h, kOps[o], &alpha, A, X, &beta, Y, ct, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max<size_t>(bytes, 16));
    CK(cusparseSpMV(h, kOps[o], &alpha, A, X, &beta, Y, ct, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    std::vector<cdouble> ref;
    for (int i = 0; i < rows; ++i) {
      cdouble s = 0;
      for (int j = 0; j < cols; ++j) s += op_at(a.dense, n, o, i, j) * x[j];
      ref.push_back(cdouble(alpha) * s + cdouble(beta) * y0[i]);
    }
    const double e = max_rel(ref, widen(download(dy, rows)));
    char what[160];
    std::snprintf(what, sizeof what, "SpMV %s, op(A) = A%s", Prec<R>::name, o == 0 ? "" : o == 1 ? "^T" : "^H");
    check(e < Prec<R>::tol, what, e);
    cudaFree(buf);
    cusparseDestroySpMat(A);
    cusparseDestroyDnVec(X);
    cusparseDestroyDnVec(Y);
    cudaFree(dx);
    cudaFree(dy);
  }
  {  // A real matrix times complex vectors: A of the compute type's precision.
    std::vector<R> rv;
    for (const cdouble& v : a.val) rv.push_back((R)v.real());
    R* drv = upload(rv);
    std::vector<cdouble> x;
    for (int i = 0; i < m; ++i) x.push_back(rnd<R>(cval(i, 3, 2)));
    C* dx = upload(narrow<R>(x));
    C* dy = upload(std::vector<C>(n));
    cusparseSpMatDescr_t A;
    cusparseDnVecDescr_t X, Y;
    CK(cusparseCreateCsr(&A, m, n, (int64_t)rv.size(), doff, dcol, drv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                         CUSPARSE_INDEX_BASE_ZERO, Prec<R>::real));
    CK(cusparseCreateDnVec(&X, m, dx, ct));
    CK(cusparseCreateDnVec(&Y, n, dy, ct));
    const C zero(0, 0);
    size_t bytes = 0;
    CK(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_TRANSPOSE, &alpha, A, X, &zero, Y, ct, CUSPARSE_SPMV_ALG_DEFAULT,
                               &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max<size_t>(bytes, 16));
    CK(cusparseSpMV(h, CUSPARSE_OPERATION_TRANSPOSE, &alpha, A, X, &zero, Y, ct, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    std::vector<cdouble> ref;
    for (int j = 0; j < n; ++j) {
      cdouble s = 0;
      for (int i = 0; i < m; ++i) s += a.dense[(size_t)i * n + j].real() * x[i];
      ref.push_back(cdouble(alpha) * s);
    }
    double e = max_rel(ref, widen(download(dy, n)));
    check(e < Prec<R>::tol, "SpMV, a real A transposed with complex vectors", e);
    // A conjugate transpose of that real A: refused.
    expect(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &alpha, A, X, &zero, Y, ct,
                                   CUSPARSE_SPMV_ALG_DEFAULT, &bytes),
           CUSPARSE_STATUS_INVALID_VALUE, "SpMV refuses A^H of a real A");
    expect(cusparseSpMV(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &alpha, A, X, &zero, Y, ct,
                        CUSPARSE_SPMV_ALG_DEFAULT, buf),
           CUSPARSE_STATUS_INVALID_VALUE, "and so does SpMV itself");
    cudaFree(buf);
    cusparseDestroySpMat(A);
    cusparseDestroyDnVec(X);
    cusparseDestroyDnVec(Y);
    cudaFree(drv);
    cudaFree(dx);
    cudaFree(dy);
  }
  cudaFree(doff);
  cudaFree(dcol);
  cudaFree(dval);
}

template <class R> static void spmm() {
  using C = std::complex<R>;
  const cudaDataType ct = Prec<R>::cplx;
  const int m = 6, k = 5, n = 3;
  CSparse a = make_sparse(m, k, 11);
  for (auto& v : a.val) v = rnd<R>(v);
  for (auto& v : a.dense) v = rnd<R>(v);
  int* doff = upload(a.off);
  int* dcol = upload(a.col);
  C* dval = upload(narrow<R>(a.val));
  const C alpha((R)0.5, (R)1.25), beta((R)-1.0, (R)0.5);
  double worst = 0;
  bool all = true;
  for (int oa = 0; oa < 3; ++oa)
    for (int ob = 0; ob < 3; ++ob)
      for (cusparseOrder_t order : {CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW}) {
        const int rows = oa ? k : m, inner = oa ? m : k;
        const int br = ob ? n : inner, bc = ob ? inner : n;  // B as stored
        std::vector<cdouble> bd((size_t)br * bc), c0((size_t)rows * n);  // row-major host copies
        for (int i = 0; i < br; ++i)
          for (int j = 0; j < bc; ++j) bd[(size_t)i * bc + j] = rnd<R>(cval(i, j, 17));
        for (int i = 0; i < rows; ++i)
          for (int j = 0; j < n; ++j) c0[(size_t)i * n + j] = rnd<R>(cval(i, j, 23));
        auto store = [&](const std::vector<cdouble>& rm, int r, int c) {
          std::vector<cdouble> out((size_t)r * c);
          for (int i = 0; i < r; ++i)
            for (int j = 0; j < c; ++j)
              out[order == CUSPARSE_ORDER_COL ? (size_t)j * r + i : (size_t)i * c + j] = rm[(size_t)i * c + j];
          return out;
        };
        C* dB = upload(narrow<R>(store(bd, br, bc)));
        C* dC = upload(narrow<R>(store(c0, rows, n)));
        cusparseSpMatDescr_t A;
        cusparseDnMatDescr_t B, Cm;
        CK(cusparseCreateCsr(&A, m, k, (int64_t)a.val.size(), doff, dcol, dval, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, ct));
        CK(cusparseCreateDnMat(&B, br, bc, order == CUSPARSE_ORDER_COL ? br : bc, dB, ct, order));
        CK(cusparseCreateDnMat(&Cm, rows, n, order == CUSPARSE_ORDER_COL ? rows : n, dC, ct, order));
        size_t bytes = 0;
        CK(cusparseSpMM_bufferSize(h, kOps[oa], kOps[ob], &alpha, A, B, &beta, Cm, ct, CUSPARSE_SPMM_ALG_DEFAULT,
                                   &bytes));
        void* buf = nullptr;
        cudaMalloc(&buf, std::max<size_t>(bytes, 16));
        CK(cusparseSpMM(h, kOps[oa], kOps[ob], &alpha, A, B, &beta, Cm, ct, CUSPARSE_SPMM_ALG_DEFAULT, buf));
        std::vector<cdouble> ref;
        for (int i = 0; i < rows; ++i)
          for (int j = 0; j < n; ++j) {
            cdouble s = 0;
            for (int x = 0; x < inner; ++x) s += op_at(a.dense, k, oa, i, x) * op_at(bd, bc, ob, x, j);
            ref.push_back(cdouble(alpha) * s + cdouble(beta) * c0[(size_t)i * n + j]);
          }
        ref = store(ref, rows, n);
        const double e = max_rel(ref, widen(download(dC, (size_t)rows * n)));
        if (!(e < Prec<R>::tol)) {
          std::printf("     SpMM op(A)=%s op(B)=%s %s-major: %.2e\n", kOpName[oa], kOpName[ob],
                      order == CUSPARSE_ORDER_COL ? "column" : "row", e);
          all = false;
        }
        worst = std::fmax(worst, e);
        cudaFree(buf);
        cusparseDestroySpMat(A);
        cusparseDestroyDnMat(B);
        cusparseDestroyDnMat(Cm);
        cudaFree(dB);
        cudaFree(dC);
      }
  check(all, "SpMM, every op(A) x op(B) in both dense orders", worst);
  cudaFree(doff);
  cudaFree(dcol);
  cudaFree(dval);
}

template <class R> static void sddmm() {
  using C = std::complex<R>;
  const cudaDataType ct = Prec<R>::cplx;
  const int m = 5, n = 6, k = 4;
  CSparse c = make_sparse(m, n, 29);
  for (auto& v : c.val) v = rnd<R>(v);
  int* doff = upload(c.off);
  int* dcol = upload(c.col);
  const C alpha((R)0.75, (R)-0.25), beta((R)0.5, (R)0.5);
  double worst = 0;
  bool all = true;
  for (int oa = 0; oa < 2; ++oa)
    for (int ob = 0; ob < 2; ++ob) {
      const int ar = oa ? k : m, ac = oa ? m : k, br = ob ? n : k, bc = ob ? k : n;
      std::vector<cdouble> ad((size_t)ar * ac), bd((size_t)br * bc);  // row-major
      for (int i = 0; i < ar; ++i)
        for (int j = 0; j < ac; ++j) ad[(size_t)i * ac + j] = rnd<R>(cval(i, j, 31));
      for (int i = 0; i < br; ++i)
        for (int j = 0; j < bc; ++j) bd[(size_t)i * bc + j] = rnd<R>(cval(i, j, 37));
      C* dA = upload(narrow<R>(ad));
      C* dB = upload(narrow<R>(bd));
      C* dv = upload(narrow<R>(c.val));
      cusparseDnMatDescr_t A, B;
      cusparseSpMatDescr_t Cs;
      CK(cusparseCreateDnMat(&A, ar, ac, ac, dA, ct, CUSPARSE_ORDER_ROW));
      CK(cusparseCreateDnMat(&B, br, bc, bc, dB, ct, CUSPARSE_ORDER_ROW));
      CK(cusparseCreateCsr(&Cs, m, n, (int64_t)c.val.size(), doff, dcol, dv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, ct));
      size_t bytes = 0;
      CK(cusparseSDDMM_bufferSize(h, kOps[oa], kOps[ob], &alpha, A, B, &beta, Cs, ct, CUSPARSE_SDDMM_ALG_DEFAULT,
                                  &bytes));
      void* buf = nullptr;
      cudaMalloc(&buf, std::max<size_t>(bytes, 16));  // NVIDIA's refuses NULL even when it asked for none
      CK(cusparseSDDMM_preprocess(h, kOps[oa], kOps[ob], &alpha, A, B, &beta, Cs, ct, CUSPARSE_SDDMM_ALG_DEFAULT,
                                  buf));
      CK(cusparseSDDMM(h, kOps[oa], kOps[ob], &alpha, A, B, &beta, Cs, ct, CUSPARSE_SDDMM_ALG_DEFAULT, buf));
      std::vector<cdouble> ref;
      for (int i = 0; i < m; ++i)
        for (int e = c.off[i]; e < c.off[i + 1]; ++e) {
          cdouble s = 0;
          for (int x = 0; x < k; ++x) s += op_at(ad, ac, oa, i, x) * op_at(bd, bc, ob, x, c.col[e]);
          ref.push_back(cdouble(alpha) * s + cdouble(beta) * c.val[e]);
        }
      const double e = max_rel(ref, widen(download(dv, c.val.size())));
      if (!(e < Prec<R>::tol)) {
        std::printf("     SDDMM op(A)=%s op(B)=%s: %.2e\n", kOpName[oa], kOpName[ob], e);
        all = false;
      }
      worst = std::fmax(worst, e);
      cudaFree(buf);
      cusparseDestroyDnMat(A);
      cusparseDestroyDnMat(B);
      cusparseDestroySpMat(Cs);
      cudaFree(dA);
      cudaFree(dB);
      cudaFree(dv);
    }
  check(all, "SDDMM, A and A^T against B and B^T", worst);
  cudaFree(doff);
  cudaFree(dcol);
}

// A triangular system with a well-conditioned diagonal, entries on both
// sides of it so the fill mode has something to ignore.
static CSparse make_triangular(int n, int seed) {
  CSparse s = make_sparse(n, n, seed, 0.5);
  s.off = {0};
  s.col.clear();
  s.val.clear();
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      cdouble& v = s.dense[(size_t)i * n + j];
      if (i == j) v = cdouble(3.0 + i, 0.5 * i - 1.0);
      if (v == cdouble(0)) continue;
      s.col.push_back(j);
      s.val.push_back(v);
    }
    s.off.push_back((int)s.col.size());
  }
  return s;
}
// op(T) x = b by dense substitution, T the triangle of `d` that fill names.
static std::vector<cdouble> tri_solve(const std::vector<cdouble>& d, int n, bool lower, bool unit, int o,
                                      std::vector<cdouble> b) {
  auto t = [&](int i, int j) -> cdouble {  // op(T)(i, j)
    const int r = o ? j : i, c = o ? i : j;
    if (lower ? c > r : c < r) return 0;
    if (r == c && unit) return 1;
    return op_at(d, n, o, i, j);
  };
  const bool low = o ? !lower : lower;
  for (int s = 0; s < n; ++s) {
    const int i = low ? s : n - 1 - s;
    cdouble v = b[i];
    for (int j = 0; j < n; ++j)
      if (j != i && (low ? j < i : j > i)) v -= t(i, j) * b[j];
    b[i] = v / t(i, i);
  }
  return b;
}

template <class R> static void triangular() {
  using C = std::complex<R>;
  const cudaDataType ct = Prec<R>::cplx;
  const int n = 6;
  CSparse a = make_triangular(n, 41);
  for (auto& v : a.val) v = rnd<R>(v);
  for (auto& v : a.dense) v = rnd<R>(v);
  int* doff = upload(a.off);
  int* dcol = upload(a.col);
  C* dval = upload(narrow<R>(a.val));
  const C alpha((R)2.0, (R)-1.0);
  std::vector<cdouble> b;
  for (int i = 0; i < n; ++i) b.push_back(rnd<R>(cval(i, 4, 9)));
  C* db = upload(narrow<R>(b));
  double worst = 0;
  bool all = true;
  for (int lower = 0; lower < 2; ++lower)
    for (int unit = 0; unit < 2; ++unit)
      for (int o = 0; o < 3; ++o) {
        C* dy = upload(std::vector<C>(n));
        cusparseSpMatDescr_t A;
        cusparseDnVecDescr_t X, Y;
        CK(cusparseCreateCsr(&A, n, n, (int64_t)a.val.size(), doff, dcol, dval, CUSPARSE_INDEX_32I,
                             CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, ct));
        cusparseFillMode_t fill = lower ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER;
        cusparseDiagType_t diag = unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT;
        CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof fill));
        CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_DIAG_TYPE, &diag, sizeof diag));
        CK(cusparseCreateDnVec(&X, n, db, ct));
        CK(cusparseCreateDnVec(&Y, n, dy, ct));
        cusparseSpSVDescr_t d;
        CK(cusparseSpSV_createDescr(&d));
        size_t bytes = 0;
        CK(cusparseSpSV_bufferSize(h, kOps[o], &alpha, A, X, Y, ct, CUSPARSE_SPSV_ALG_DEFAULT, d, &bytes));
        void* buf = nullptr;
        cudaMalloc(&buf, std::max<size_t>(bytes, 16));
        CK(cusparseSpSV_analysis(h, kOps[o], &alpha, A, X, Y, ct, CUSPARSE_SPSV_ALG_DEFAULT, d, buf));
        CK(cusparseSpSV_solve(h, kOps[o], &alpha, A, X, Y, ct, CUSPARSE_SPSV_ALG_DEFAULT, d));
        std::vector<cdouble> rhs = b;
        for (auto& v : rhs) v *= cdouble(alpha);
        const auto ref = tri_solve(a.dense, n, lower, unit, o, rhs);
        const double e = max_rel(ref, widen(download(dy, n)));
        if (!(e < Prec<R>::tol * 10)) {
          std::printf("     SpSV %s %s op=%s: %.2e\n", lower ? "lower" : "upper", unit ? "unit" : "non-unit",
                      kOpName[o], e);
          all = false;
        }
        worst = std::fmax(worst, e);
        cusparseSpSV_destroyDescr(d);
        cudaFree(buf);
        cusparseDestroySpMat(A);
        cusparseDestroyDnVec(X);
        cusparseDestroyDnVec(Y);
        cudaFree(dy);
      }
  check(all, "SpSV, lower and upper, unit and not, A, A^T and A^H", worst);

  // SpSM: op(A) C = alpha op(B), B 6 x 3 column-major (or 3 x 6 for B^T).
  const int nrhs = 3;
  worst = 0;
  all = true;
  for (int oa : {0, 2})
    for (int ob : {0, 1}) {
      const int br = ob ? nrhs : n, bc = ob ? n : nrhs;
      std::vector<cdouble> bm((size_t)br * bc);  // column-major
      for (int j = 0; j < bc; ++j)
        for (int i = 0; i < br; ++i) bm[(size_t)j * br + i] = rnd<R>(cval(i, j, 43));
      C* dB = upload(narrow<R>(bm));
      C* dC = upload(std::vector<C>((size_t)n * nrhs));
      cusparseSpMatDescr_t A;
      cusparseDnMatDescr_t B, Cm;
      CK(cusparseCreateCsr(&A, n, n, (int64_t)a.val.size(), doff, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, ct));
      cusparseFillMode_t fill = CUSPARSE_FILL_MODE_UPPER;
      CK(cusparseSpMatSetAttribute(A, CUSPARSE_SPMAT_FILL_MODE, &fill, sizeof fill));
      CK(cusparseCreateDnMat(&B, br, bc, br, dB, ct, CUSPARSE_ORDER_COL));
      CK(cusparseCreateDnMat(&Cm, n, nrhs, n, dC, ct, CUSPARSE_ORDER_COL));
      cusparseSpSMDescr_t d;
      CK(cusparseSpSM_createDescr(&d));
      size_t bytes = 0;
      CK(cusparseSpSM_bufferSize(h, kOps[oa], kOps[ob], &alpha, A, B, Cm, ct, CUSPARSE_SPSM_ALG_DEFAULT, d, &bytes));
      void* buf = nullptr;
      cudaMalloc(&buf, std::max<size_t>(bytes, 16));
      CK(cusparseSpSM_analysis(h, kOps[oa], kOps[ob], &alpha, A, B, Cm, ct, CUSPARSE_SPSM_ALG_DEFAULT, d, buf));
      CK(cusparseSpSM_solve(h, kOps[oa], kOps[ob], &alpha, A, B, Cm, ct, CUSPARSE_SPSM_ALG_DEFAULT, d));
      const auto got = widen(download(dC, (size_t)n * nrhs));
      std::vector<cdouble> ref, gotv;
      for (int j = 0; j < nrhs; ++j) {
        std::vector<cdouble> rhs(n);
        for (int i = 0; i < n; ++i) rhs[i] = cdouble(alpha) * (ob ? bm[(size_t)i * br + j] : bm[(size_t)j * br + i]);
        const auto x = tri_solve(a.dense, n, false, false, oa, rhs);
        ref.insert(ref.end(), x.begin(), x.end());
      }
      const double e = max_rel(ref, got);
      if (!(e < Prec<R>::tol * 10)) {
        std::printf("     SpSM op(A)=%s op(B)=%s: %.2e\n", kOpName[oa], kOpName[ob], e);
        all = false;
      }
      worst = std::fmax(worst, e);
      if (oa == 0 && ob == 1)  // the shape B^H needs, so only the operation is wrong
        expect(cusparseSpSM_bufferSize(h, kOps[0], kOps[2], &alpha, A, B, Cm, ct, CUSPARSE_SPSM_ALG_DEFAULT, d,
                                       &bytes),
               CUSPARSE_STATUS_NOT_SUPPORTED, "SpSM refuses op(B) = B^H");
      cusparseSpSM_destroyDescr(d);
      cudaFree(buf);
      cusparseDestroySpMat(A);
      cusparseDestroyDnMat(B);
      cusparseDestroyDnMat(Cm);
      cudaFree(dB);
      cudaFree(dC);
    }
  check(all, "SpSM, upper, A and A^H against B and B^T", worst);
  cudaFree(doff);
  cudaFree(dcol);
  cudaFree(dval);
  cudaFree(db);
}

template <class R> static void spgemm_and_convert() {
  using C = std::complex<R>;
  const cudaDataType ct = Prec<R>::cplx;
  const int m = 5, k = 6, n = 4;
  CSparse a = make_sparse(m, k, 51), b = make_sparse(k, n, 53);
  for (auto* s : {&a, &b}) {
    for (auto& v : s->val) v = rnd<R>(v);
    for (auto& v : s->dense) v = rnd<R>(v);
  }
  int* daoff = upload(a.off);
  int* dacol = upload(a.col);
  C* daval = upload(narrow<R>(a.val));
  int* dboff = upload(b.off);
  int* dbcol = upload(b.col);
  C* dbval = upload(narrow<R>(b.val));
  int* dcoff = upload(std::vector<int>(m + 1));
  cusparseSpMatDescr_t A, B, Cs;
  CK(cusparseCreateCsr(&A, m, k, (int64_t)a.val.size(), daoff, dacol, daval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, ct));
  CK(cusparseCreateCsr(&B, k, n, (int64_t)b.val.size(), dboff, dbcol, dbval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, ct));
  CK(cusparseCreateCsr(&Cs, m, n, 0, dcoff, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, ct));
  const C alpha((R)1.0, (R)2.0), beta((R)0, (R)0);
  const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpGEMMDescr_t g;
  CK(cusparseSpGEMM_createDescr(&g));
  size_t b1 = 0, b2 = 0;
  CK(cusparseSpGEMM_workEstimation(h, N, N, &alpha, A, B, &beta, Cs, ct, CUSPARSE_SPGEMM_DEFAULT, g, &b1, nullptr));
  void *w1 = nullptr, *w2 = nullptr;
  cudaMalloc(&w1, std::max<size_t>(b1, 16));
  CK(cusparseSpGEMM_workEstimation(h, N, N, &alpha, A, B, &beta, Cs, ct, CUSPARSE_SPGEMM_DEFAULT, g, &b1, w1));
  CK(cusparseSpGEMM_compute(h, N, N, &alpha, A, B, &beta, Cs, ct, CUSPARSE_SPGEMM_DEFAULT, g, &b2, nullptr));
  cudaMalloc(&w2, std::max<size_t>(b2, 16));
  CK(cusparseSpGEMM_compute(h, N, N, &alpha, A, B, &beta, Cs, ct, CUSPARSE_SPGEMM_DEFAULT, g, &b2, w2));
  int64_t cr, cc, cnnz;
  CK(cusparseSpMatGetSize(Cs, &cr, &cc, &cnnz));
  int* dccol = upload(std::vector<int>(std::max<int64_t>(cnnz, 1)));
  C* dcval = upload(std::vector<C>(std::max<int64_t>(cnnz, 1)));
  CK(cusparseCsrSetPointers(Cs, dcoff, dccol, dcval));
  CK(cusparseSpGEMM_copy(h, N, N, &alpha, A, B, &beta, Cs, ct, CUSPARSE_SPGEMM_DEFAULT, g));
  // To dense, then compare with the dense product.
  C* dd = upload(std::vector<C>((size_t)m * n));
  cusparseDnMatDescr_t D;
  CK(cusparseCreateDnMat(&D, m, n, m, dd, ct, CUSPARSE_ORDER_COL));
  size_t bytes = 0;
  CK(cusparseSparseToDense_bufferSize(h, Cs, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytes));
  void* buf = nullptr;
  cudaMalloc(&buf, std::max<size_t>(bytes, 16));
  CK(cusparseSparseToDense(h, Cs, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, buf));
  std::vector<cdouble> ref((size_t)m * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      cdouble s = 0;
      for (int x = 0; x < k; ++x) s += a.dense[(size_t)i * k + x] * b.dense[(size_t)x * n + j];
      ref[(size_t)j * m + i] = cdouble(alpha) * s;
    }
  double e = max_rel(ref, widen(download(dd, (size_t)m * n)));
  check(e < Prec<R>::tol, "SpGEMM, then SparseToDense: the dense product", e);

  // Dense to CSR and to CSC: the nonzeros of A's dense form, in order.
  std::vector<cdouble> acm((size_t)m * k);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < k; ++j) acm[(size_t)j * m + i] = a.dense[(size_t)i * k + j];
  C* dad = upload(narrow<R>(acm));
  cusparseDnMatDescr_t AD;
  CK(cusparseCreateDnMat(&AD, m, k, m, dad, ct, CUSPARSE_ORDER_COL));
  for (int csc = 0; csc < 2; ++csc) {
    const int nnz = (int)a.val.size(), lead = csc ? k : m;
    int* o = upload(std::vector<int>(lead + 1));
    int* ix = upload(std::vector<int>(nnz));
    C* v = upload(std::vector<C>(nnz));
    cusparseSpMatDescr_t S;
    if (csc)
      CK(cusparseCreateCsc(&S, m, k, 0, o, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, ct));
    else
      CK(cusparseCreateCsr(&S, m, k, 0, o, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                           CUSPARSE_INDEX_BASE_ZERO, ct));
    CK(cusparseDenseToSparse_bufferSize(h, AD, S, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bytes));
    void* w = nullptr;
    cudaMalloc(&w, std::max<size_t>(bytes, 16));
    CK(cusparseDenseToSparse_analysis(h, AD, S, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, w));
    int64_t rows_, cols_, got_nnz = 0;
    CK(cusparseSpMatGetSize(S, &rows_, &cols_, &got_nnz));
    if (csc)
      CK(cusparseCscSetPointers(S, o, ix, v));
    else
      CK(cusparseCsrSetPointers(S, o, ix, v));
    CK(cusparseDenseToSparse_convert(h, AD, S, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, w));
    std::vector<int> ro{0}, ri;
    std::vector<cdouble> rv;
    for (int x = 0; x < lead; ++x) {
      for (int y = 0; y < (csc ? m : k); ++y) {
        const cdouble val = csc ? a.dense[(size_t)y * k + x] : a.dense[(size_t)x * k + y];
        if (val == cdouble(0)) continue;
        ri.push_back(y);
        rv.push_back(val);
      }
      ro.push_back((int)ri.size());
    }
    const bool same = got_nnz == nnz && download(o, lead + 1) == ro && download(ix, nnz) == ri &&
                      max_rel(rv, widen(download(v, nnz))) == 0.0;
    check(same, csc ? "DenseToSparse into CSC: every nonzero, bit for bit"
                    : "DenseToSparse into CSR: every nonzero, bit for bit",
          0);
    cudaFree(w);
    cusparseDestroySpMat(S);
    cudaFree(o);
    cudaFree(ix);
    cudaFree(v);
  }
  cusparseDestroyDnMat(AD);
  cudaFree(dad);
  cusparseSpGEMM_destroyDescr(g);
  cusparseDestroyDnMat(D);
  cusparseDestroySpMat(A);
  cusparseDestroySpMat(B);
  cusparseDestroySpMat(Cs);
  for (void* x : {(void*)daoff, (void*)dacol, (void*)daval, (void*)dboff, (void*)dbcol, (void*)dbval, (void*)dcoff,
                  (void*)dccol, (void*)dcval, (void*)dd, buf, w1, w2})
    cudaFree(x);
}

// csrgeam2 in cuComplex and cuDoubleComplex: C = alpha A + beta B.
template <class R, class T, class Geam, class Size>
static void geam(Geam geam2, Size size, const char* what) {
  const int m = 5, n = 6;
  CSparse a = make_sparse(m, n, 61), b = make_sparse(m, n, 67);
  for (auto* s : {&a, &b}) {
    for (auto& v : s->val) v = rnd<R>(v);
    for (auto& v : s->dense) v = rnd<R>(v);
  }
  cusparseMatDescr_t d;
  CK(cusparseCreateMatDescr(&d));
  int* dao = upload(a.off);
  int* dac = upload(a.col);
  int* dbo = upload(b.off);
  int* dbc = upload(b.col);
  T* dav = reinterpret_cast<T*>(upload(narrow<R>(a.val)));
  T* dbv = reinterpret_cast<T*>(upload(narrow<R>(b.val)));
  int* dco = upload(std::vector<int>(m + 1));
  const std::complex<R> al((R)1.5, (R)0.5), be((R)-0.5, (R)2.0);
  const T* alpha = reinterpret_cast<const T*>(&al);
  const T* beta = reinterpret_cast<const T*>(&be);
  size_t bytes = 0;
  CK(size(h, m, n, alpha, d, (int)a.val.size(), dav, dao, dac, beta, d, (int)b.val.size(), dbv, dbo, dbc, d, nullptr,
          dco, nullptr, &bytes));
  void* buf = nullptr;
  cudaMalloc(&buf, std::max<size_t>(bytes, 16));
  int nnz = 0;
  CK(cusparseXcsrgeam2Nnz(h, m, n, d, (int)a.val.size(), dao, dac, d, (int)b.val.size(), dbo, dbc, d, dco, &nnz,
                          buf));
  int* dcc = upload(std::vector<int>(std::max(nnz, 1)));
  T* dcv = reinterpret_cast<T*>(upload(std::vector<std::complex<R>>(std::max(nnz, 1))));
  CK(geam2(h, m, n, alpha, d, (int)a.val.size(), dav, dao, dac, beta, d, (int)b.val.size(), dbv, dbo, dbc, d, dcv, dco,
           dcc, buf));
  const auto off = download(dco, m + 1);
  const auto col = download(dcc, nnz);
  const auto val = widen(download(reinterpret_cast<std::complex<R>*>(dcv), nnz));
  std::vector<cdouble> ref, got;
  bool pattern = off[m] == nnz;
  for (int i = 0; i < m && pattern; ++i)
    for (int e = off[i]; e < off[i + 1]; ++e) {
      const int j = col[e];
      ref.push_back(cdouble(al) * a.dense[(size_t)i * n + j] + cdouble(be) * b.dense[(size_t)i * n + j]);
      got.push_back(val[e]);
      if (a.dense[(size_t)i * n + j] == cdouble(0) && b.dense[(size_t)i * n + j] == cdouble(0)) pattern = false;
    }
  const double e = pattern ? max_rel(ref, got) : 1.0;
  check(pattern && e < Prec<R>::tol, what, e);
  cusparseDestroyMatDescr(d);
  for (void* x : {(void*)dao, (void*)dac, (void*)dbo, (void*)dbc, (void*)dav, (void*)dbv, (void*)dco, (void*)dcc,
                  (void*)dcv, buf})
    cudaFree(x);
}

// Complex type combinations NVIDIA's takes, and one it does not.
static void combinations() {
  std::vector<std::complex<float>> v(4, {1, 1});
  std::vector<int> off = {0, 1, 2}, col = {0, 1};
  int* doff = upload(off);
  int* dcol = upload(col);
  auto* dv = upload(v);
  auto* dx = upload(std::vector<std::complex<double>>(2, {1, 0}));
  auto* dy = upload(std::vector<std::complex<double>>(2));
  cusparseSpMatDescr_t A;
  cusparseDnVecDescr_t X, Y;
  CK(cusparseCreateCsr(&A, 2, 2, 2, doff, dcol, dv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                       CUDA_C_32F));
  CK(cusparseCreateDnVec(&X, 2, dx, CUDA_C_64F));
  CK(cusparseCreateDnVec(&Y, 2, dy, CUDA_C_64F));
  const std::complex<double> one(1, 0), zero(0, 0);
  size_t bytes = 0;
  expect(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, &zero, Y, CUDA_C_64F,
                                 CUSPARSE_SPMV_ALG_DEFAULT, &bytes),
         CUSPARSE_STATUS_NOT_SUPPORTED, "SpMV refuses a C_32F matrix with C_64F vectors");
  cusparseDestroySpMat(A);
  cusparseDestroyDnVec(X);
  cusparseDestroyDnVec(Y);
  for (void* x : {(void*)doff, (void*)dcol, (void*)dv, (void*)dx, (void*)dy}) cudaFree(x);
}

int main() {
  if (cusparseCreate(&h)) {
    std::printf("FAIL: cusparseCreate\n");
    return 1;
  }
  spmv<float>();
  spmv<double>();
  spmm<float>();
  spmm<double>();
  sddmm<float>();
  sddmm<double>();
  triangular<float>();
  triangular<double>();
  spgemm_and_convert<float>();
  spgemm_and_convert<double>();
  geam<float, cuComplex>(cusparseCcsrgeam2, cusparseCcsrgeam2_bufferSizeExt, "csrgeam2 in cuComplex");
  geam<double, cuDoubleComplex>(cusparseZcsrgeam2, cusparseZcsrgeam2_bufferSizeExt, "csrgeam2 in cuDoubleComplex");
  combinations();
  cusparseDestroy(h);
  std::printf(failures ? "FAIL: %d complex sparse checks\n" : "PASS: every complex sparse check\n", failures);
  return failures ? 1 : 0;
}
