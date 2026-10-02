// cuSPARSE's blocked (BSR) matrices, checked against dense arithmetic done
// here on the host.
//
//   generic API    cusparseCreateBsr with SpMV (either block order) and SpMM
//                  and SDDMM (row-major blocks), and what NVIDIA's refuses:
//                  op(A) other than A in SpMV and SpMM, column-major blocks in
//                  SpMM, BSR in SparseToDense, non-square blocks, a batch
//                  stride shorter than a member
//   legacy API     bsrmv, bsrxmv, bsrmm, bsrsv2 and bsrsm2 (with their zero
//                  pivots), bsric02 and bsrilu02 (with ILU's numeric boost),
//                  csric02 and csrilu02, csr2bsr and bsr2csr, csr2gebsr and
//                  gebsr2csr
//
// The statuses and conventions asserted here are what an RTX 3060 answered
// with cuSPARSE 13.0. CUDA 12.0's header, which CI compiles this with, has no
// generic BSR (it arrived in 12.1), so those entry points are looked up by
// name in the cuSPARSE the program runs against.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <type_traits>
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
template <class T> static double max_rel(const std::vector<T>& ref, const std::vector<T>& got) {
  double err = 0, scale = 1e-30;
  for (size_t i = 0; i < ref.size(); ++i) {
    err = std::fmax(err, std::abs(ref[i] - got[i]));
    scale = std::fmax(scale, std::abs(ref[i]));
  }
  return err / scale;
}

static double hash01(int i, int j, int seed) {
  return std::fmod(std::fabs(std::sin(12.9898 * (i + 1) + 78.233 * (j + 1) + seed)) * 43758.5453, 1.0);
}

// A BSR matrix of mb x nb blocks of rbd x cbd, each stored row- or
// column-major, with a dense row-major copy of the element matrix.
template <class T> struct Bsr {
  int mb, nb, rbd, cbd;
  bool row_major;
  std::vector<int> off, col;
  std::vector<T> val;
  int rows() const { return mb * rbd; }
  int cols() const { return nb * cbd; }
  size_t at(int k, int r, int c) const {
    return (size_t)k * rbd * cbd + (row_major ? (size_t)r * cbd + c : (size_t)c * rbd + r);
  }
  std::vector<T> dense() const {
    std::vector<T> d((size_t)rows() * cols());
    for (int i = 0; i < mb; ++i)
      for (int k = off[i]; k < off[i + 1]; ++k)
        for (int r = 0; r < rbd; ++r)
          for (int c = 0; c < cbd; ++c) d[(size_t)(i * rbd + r) * cols() + col[k] * cbd + c] = val[at(k, r, c)];
    return d;
  }
  std::vector<char> mask() const {
    std::vector<char> m((size_t)rows() * cols(), 0);
    for (int i = 0; i < mb; ++i)
      for (int k = off[i]; k < off[i + 1]; ++k)
        for (int r = 0; r < rbd; ++r)
          for (int c = 0; c < cbd; ++c) m[(size_t)(i * rbd + r) * cols() + col[k] * cbd + c] = 1;
    return m;
  }
};
template <class T> T value(int i, int j, int seed);
template <> double value<double>(int i, int j, int seed) { return std::cos(0.7 * i - 0.3 * j + seed) * 2.0; }
template <> cdouble value<cdouble>(int i, int j, int seed) {
  return {std::cos(0.7 * i - 0.3 * j + seed) * 2.0, std::sin(0.4 * i + 0.9 * j - seed)};
}
// Blocks where `keep` says, the diagonal block always when `diag`; values
// from `value`, the diagonal elements made dominant when `dominant`.
template <class T>
static Bsr<T> make_bsr(int mb, int nb, int rbd, int cbd, bool row_major, int seed, bool diag, double dominant = 0) {
  Bsr<T> b{mb, nb, rbd, cbd, row_major, {0}, {}, {}};
  for (int i = 0; i < mb; ++i) {
    for (int j = 0; j < nb; ++j)
      if ((diag && i == j) || hash01(i, j, seed) < 0.45) b.col.push_back(j);
    b.off.push_back((int)b.col.size());
  }
  b.val.resize(b.col.size() * rbd * cbd);
  for (int i = 0; i < mb; ++i)
    for (int k = b.off[i]; k < b.off[i + 1]; ++k)
      for (int r = 0; r < rbd; ++r)
        for (int c = 0; c < cbd; ++c) {
          const int gi = i * rbd + r, gj = b.col[k] * cbd + c;
          T v = value<T>(gi, gj, seed);
          if (gi == gj && dominant != 0) v = T(dominant + gi);
          b.val[b.at(k, r, c)] = v;
        }
  return b;
}

static cusparseHandle_t h;

/* ---- the generic API ---- */

using CreateBsrFn = cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, int64_t, void*,
                                         void*, void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t,
                                         cudaDataType, cusparseOrder_t);
using BsrBatchFn = cusparseStatus_t (*)(cusparseSpMatDescr_t, int, int64_t, int64_t, int64_t);
static const cusparseFormat_t kBsr = static_cast<cusparseFormat_t>(6);  // CUSPARSE_FORMAT_BSR

static void generic() {
  auto create = reinterpret_cast<CreateBsrFn>(dlsym(RTLD_DEFAULT, "cusparseCreateBsr"));
  auto batch = reinterpret_cast<BsrBatchFn>(dlsym(RTLD_DEFAULT, "cusparseBsrSetStridedBatch"));
  if (!create || !batch) {
    std::printf("FAIL the cuSPARSE this runs against has no cusparseCreateBsr\n");
    ++failures;
    return;
  }
  const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE, T = CUSPARSE_OPERATION_TRANSPOSE;
  for (int row_major = 0; row_major < 2; ++row_major)
    for (int base = 0; base < 2; ++base) {
      const auto b = make_bsr<double>(3, 4, 3, 3, row_major, 5 + base, false);
      std::vector<int> off = b.off, col = b.col;
      for (int& v : off) v += base;
      for (int& v : col) v += base;
      int* doff = upload(off);
      int* dcol = upload(col);
      double* dval = upload(b.val);
      cusparseSpMatDescr_t A;
      CK(create(&A, b.mb, b.nb, (int64_t)b.col.size(), 3, 3, doff, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F,
                row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL));
      int64_t rows = 0, cols = 0, nnz = 0;
      cusparseFormat_t fmt;
      CK(cusparseSpMatGetSize(A, &rows, &cols, &nnz));
      CK(cusparseSpMatGetFormat(A, &fmt));
      if (row_major == 0 && base == 0)
        check(rows == 9 && cols == 12 && nnz == (int64_t)b.val.size() && fmt == kBsr,
              "a BSR descriptor reports its size in elements, nnz counting every block element", 0);
      const auto d = b.dense();
      std::vector<double> x(12), y0(9);
      for (int i = 0; i < 12; ++i) x[i] = 0.5 + 0.25 * i;
      for (int i = 0; i < 9; ++i) y0[i] = std::sin(i + 1.0);
      double* dx = upload(x);
      double* dy = upload(y0);
      cusparseDnVecDescr_t X, Y;
      CK(cusparseCreateDnVec(&X, 12, dx, CUDA_R_64F));
      CK(cusparseCreateDnVec(&Y, 9, dy, CUDA_R_64F));
      const double alpha = 1.5, beta = -0.5;
      size_t bytes = 0;
      CK(cusparseSpMV_bufferSize(h, N, &alpha, A, X, &beta, Y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
      void* buf = nullptr;
      cudaMalloc(&buf, std::max<size_t>(bytes, 16));
      CK(cusparseSpMV(h, N, &alpha, A, X, &beta, Y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
      std::vector<double> ref(9);
      for (int i = 0; i < 9; ++i) {
        double s = 0;
        for (int j = 0; j < 12; ++j) s += d[(size_t)i * 12 + j] * x[j];
        ref[i] = alpha * s + beta * y0[i];
      }
      double e = max_rel(ref, download(dy, 9));
      char what[160];
      std::snprintf(what, sizeof what, "SpMV on BSR, %s-major blocks, %s-based", row_major ? "row" : "column",
                    base ? "one" : "zero");
      check(e < 1e-14, what, e);
      cusparseDnVecDescr_t Xt, Yt;
      CK(cusparseCreateDnVec(&Xt, 9, dy, CUDA_R_64F));
      CK(cusparseCreateDnVec(&Yt, 12, dx, CUDA_R_64F));
      if (row_major == 0 && base == 0)
        expect(cusparseSpMV_bufferSize(h, T, &alpha, A, Xt, &beta, Yt, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes),
               CUSPARSE_STATUS_NOT_SUPPORTED, "SpMV on BSR refuses op(A) = A^T");
      cusparseDestroyDnVec(Xt);
      cusparseDestroyDnVec(Yt);

      // SpMM: row-major blocks only.
      const int n = 4;
      for (int ob = 0; ob < 2; ++ob)
        for (cusparseOrder_t order : {CUSPARSE_ORDER_COL, CUSPARSE_ORDER_ROW}) {
          const int br = ob ? n : 12, bc = ob ? 12 : n;
          std::vector<double> bm((size_t)br * bc), c0((size_t)9 * n, 0.0);
          for (size_t i = 0; i < bm.size(); ++i) bm[i] = std::cos(0.37 * (double)i);
          auto belem = [&](int i, int j) {  // B(i, j) as stored
            return bm[order == CUSPARSE_ORDER_COL ? (size_t)j * br + i : (size_t)i * bc + j];
          };
          double* dB = upload(bm);
          double* dC = upload(c0);
          cusparseDnMatDescr_t B, C;
          CK(cusparseCreateDnMat(&B, br, bc, order == CUSPARSE_ORDER_COL ? br : bc, dB, CUDA_R_64F, order));
          CK(cusparseCreateDnMat(&C, 9, n, order == CUSPARSE_ORDER_COL ? 9 : n, dC, CUDA_R_64F, order));
          const cusparseStatus_t st = cusparseSpMM_bufferSize(h, N, ob ? T : N, &alpha, A, B, &beta, C, CUDA_R_64F,
                                                              CUSPARSE_SPMM_ALG_DEFAULT, &bytes);
          if (!row_major) {
            if (ob == 0 && order == CUSPARSE_ORDER_COL && base == 0)
              expect(st, CUSPARSE_STATUS_NOT_SUPPORTED, "SpMM on BSR refuses column-major blocks");
          } else {
            CK(st);
            void* w = nullptr;
            cudaMalloc(&w, std::max<size_t>(bytes, 16));
            CK(cusparseSpMM(h, N, ob ? T : N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, w));
            cudaFree(w);
            const auto got = download(dC, (size_t)9 * n);
            std::vector<double> r, g;
            for (int i = 0; i < 9; ++i)
              for (int j = 0; j < n; ++j) {
                double s = 0;
                for (int x2 = 0; x2 < 12; ++x2) s += d[(size_t)i * 12 + x2] * (ob ? belem(j, x2) : belem(x2, j));
                r.push_back(alpha * s);
                g.push_back(got[order == CUSPARSE_ORDER_COL ? (size_t)j * 9 + i : (size_t)i * n + j]);
              }
            e = max_rel(r, g);
            std::snprintf(what, sizeof what, "SpMM on BSR, op(B) = B%s, %s-major dense, %s-based", ob ? "^T" : "",
                          order == CUSPARSE_ORDER_COL ? "column" : "row", base ? "one" : "zero");
            check(e < 1e-14, what, e);
          }
          cusparseDestroyDnMat(B);
          cusparseDestroyDnMat(C);
          cudaFree(dB);
          cudaFree(dC);
        }
      if (row_major && base == 0) {
        cusparseDnMatDescr_t Bt, Ct;
        CK(cusparseCreateDnMat(&Bt, 9, 4, 9, dy, CUDA_R_64F, CUSPARSE_ORDER_COL));
        CK(cusparseCreateDnMat(&Ct, 12, 4, 12, dx, CUDA_R_64F, CUSPARSE_ORDER_COL));
        expect(cusparseSpMM_bufferSize(h, T, N, &alpha, A, Bt, &beta, Ct, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT,
                                       &bytes),
               CUSPARSE_STATUS_NOT_SUPPORTED, "SpMM on BSR refuses op(A) = A^T");
        cusparseDestroyDnMat(Bt);
        cusparseDestroyDnMat(Ct);
        cusparseDnMatDescr_t D;
        CK(cusparseCreateDnMat(&D, 9, 12, 9, dx, CUDA_R_64F, CUSPARSE_ORDER_COL));
        expect(cusparseSparseToDense_bufferSize(h, A, D, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, &bytes),
               CUSPARSE_STATUS_NOT_SUPPORTED, "SparseToDense refuses BSR");
        cusparseDestroyDnMat(D);
        expect(batch(A, 2, b.mb + 1, (int64_t)b.col.size(), (int64_t)b.val.size() - 1), CUSPARSE_STATUS_INVALID_VALUE,
               "BsrSetStridedBatch refuses a value stride shorter than one member");
      }
      cudaFree(buf);
      cusparseDestroySpMat(A);
      cusparseDestroyDnVec(X);
      cusparseDestroyDnVec(Y);
      for (void* p : {(void*)doff, (void*)dcol, (void*)dval, (void*)dx, (void*)dy}) cudaFree(p);
    }

  {  // Complex values, SpMV with complex alpha and beta.
    const auto b = make_bsr<cdouble>(3, 3, 2, 2, true, 13, false);
    std::vector<std::complex<float>> v;
    for (const cdouble& x : b.val) v.push_back({(float)x.real(), (float)x.imag()});
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    auto* dv = upload(v);
    std::vector<std::complex<float>> x(6);
    for (int i = 0; i < 6; ++i) x[i] = {(float)std::cos(i * 0.5), (float)std::sin(i * 0.25)};
    auto* dx = upload(x);
    auto* dy = upload(std::vector<std::complex<float>>(6));
    cusparseSpMatDescr_t A;
    cusparseDnVecDescr_t X, Y;
    CK(create(&A, 3, 3, (int64_t)b.col.size(), 2, 2, doff, dcol, dv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
              CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F, CUSPARSE_ORDER_ROW));
    CK(cusparseCreateDnVec(&X, 6, dx, CUDA_C_32F));
    CK(cusparseCreateDnVec(&Y, 6, dy, CUDA_C_32F));
    const std::complex<float> alpha(0.5f, 1.0f), beta(0, 0);
    size_t bytes = 0;
    CK(cusparseSpMV_bufferSize(h, N, &alpha, A, X, &beta, Y, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max<size_t>(bytes, 16));
    CK(cusparseSpMV(h, N, &alpha, A, X, &beta, Y, CUDA_C_32F, CUSPARSE_SPMV_ALG_DEFAULT, buf));
    const auto d = b.dense();
    std::vector<cdouble> ref(6), got(6);
    const auto out = download(dy, 6);
    for (int i = 0; i < 6; ++i) {
      cdouble s = 0;
      for (int j = 0; j < 6; ++j) s += d[(size_t)i * 6 + j] * cdouble(x[j]);
      ref[i] = cdouble(alpha) * s;
      got[i] = out[i];
    }
    const double e = max_rel(ref, got);
    check(e < 2e-6, "SpMV on a complex BSR matrix", e);
    cudaFree(buf);
    cusparseDestroySpMat(A);
    cusparseDestroyDnVec(X);
    cusparseDestroyDnVec(Y);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, (void*)dx, (void*)dy}) cudaFree(p);
  }

  {  // SDDMM into a BSR pattern: k a whole number of blocks.
    const auto c = make_bsr<double>(2, 3, 2, 2, true, 19, false);
    const int m = 4, n = 6, k = 4;
    std::vector<double> a((size_t)m * k), bm((size_t)k * n);  // column-major
    for (size_t i = 0; i < a.size(); ++i) a[i] = std::sin(0.3 * (double)i + 1);
    for (size_t i = 0; i < bm.size(); ++i) bm[i] = std::cos(0.2 * (double)i);
    int* doff = upload(c.off);
    int* dcol = upload(c.col);
    double* dv = upload(c.val);
    double* dA = upload(a);
    double* dB = upload(bm);
    cusparseSpMatDescr_t C;
    cusparseDnMatDescr_t A, B;
    CK(create(&C, 2, 3, (int64_t)c.col.size(), 2, 2, doff, dcol, dv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
              CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CK(cusparseCreateDnMat(&A, m, k, m, dA, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CK(cusparseCreateDnMat(&B, k, n, k, dB, CUDA_R_64F, CUSPARSE_ORDER_COL));
    const double alpha = 2.0, beta = 0.5;
    size_t bytes = 0;
    CK(cusparseSDDMM_bufferSize(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max<size_t>(bytes, 16));
    CK(cusparseSDDMM(h, N, N, &alpha, A, B, &beta, C, CUDA_R_64F, CUSPARSE_SDDMM_ALG_DEFAULT, buf));
    std::vector<double> ref(c.val.size());
    for (int i = 0; i < c.mb; ++i)
      for (int e = c.off[i]; e < c.off[i + 1]; ++e)
        for (int r = 0; r < 2; ++r)
          for (int cc = 0; cc < 2; ++cc) {
            const int gi = i * 2 + r, gj = c.col[e] * 2 + cc;
            double s = 0;
            for (int x = 0; x < k; ++x) s += a[(size_t)x * m + gi] * bm[(size_t)gj * k + x];
            ref[c.at(e, r, cc)] = alpha * s + beta * c.val[c.at(e, r, cc)];
          }
    const double e = max_rel(ref, download(dv, c.val.size()));
    check(e < 1e-14, "SDDMM into a BSR pattern", e);
    cudaFree(buf);
    cusparseDestroySpMat(C);
    cusparseDestroyDnMat(A);
    cusparseDestroyDnMat(B);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, (void*)dA, (void*)dB}) cudaFree(p);
  }

  {
    cusparseSpMatDescr_t A;
    int* p = upload(std::vector<int>{0, 1});
    expect(create(&A, 1, 1, 1, 2, 3, p, p, p, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                  CUDA_R_64F, CUSPARSE_ORDER_ROW),
           CUSPARSE_STATUS_INVALID_VALUE, "cusparseCreateBsr refuses non-square blocks");
    cudaFree(p);
  }
}

/* ---- the legacy API ---- */

static cusparseMatDescr_t descr(cusparseIndexBase_t base = CUSPARSE_INDEX_BASE_ZERO,
                                cusparseFillMode_t fill = CUSPARSE_FILL_MODE_LOWER,
                                cusparseDiagType_t diag = CUSPARSE_DIAG_TYPE_NON_UNIT) {
  cusparseMatDescr_t d;
  cusparseCreateMatDescr(&d);
  cusparseSetMatIndexBase(d, base);
  cusparseSetMatFillMode(d, fill);
  cusparseSetMatDiagType(d, diag);
  return d;
}
static cusparseDirection_t dir_of(bool row_major) {
  return row_major ? CUSPARSE_DIRECTION_ROW : CUSPARSE_DIRECTION_COLUMN;
}

static void legacy_products() {
  const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE, T = CUSPARSE_OPERATION_TRANSPOSE;
  for (int row_major = 0; row_major < 2; ++row_major) {
    const auto b = make_bsr<double>(4, 3, 3, 3, row_major, 23, false);
    const auto d = b.dense();
    const int m = b.rows(), n = b.cols();
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    double* dv = upload(b.val);
    std::vector<double> x(n), y0(m);
    for (int i = 0; i < n; ++i) x[i] = std::cos(0.4 * i);
    for (int i = 0; i < m; ++i) y0[i] = std::sin(0.3 * i + 2);
    double* dx = upload(x);
    double* dy = upload(y0);
    cusparseMatDescr_t D = descr();
    const double alpha = 0.75, beta = 1.25;
    CK(cusparseDbsrmv(h, dir_of(row_major), N, b.mb, b.nb, (int)b.col.size(), &alpha, D, dv, doff, dcol, 3, dx, &beta,
                      dy));
    std::vector<double> ref(m);
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int j = 0; j < n; ++j) s += d[(size_t)i * n + j] * x[j];
      ref[i] = alpha * s + beta * y0[i];
    }
    double e = max_rel(ref, download(dy, m));
    check(e < 1e-14, row_major ? "bsrmv, row-major blocks" : "bsrmv, column-major blocks", e);

    if (row_major) {
      expect(cusparseDbsrmv(h, dir_of(true), T, b.mb, b.nb, (int)b.col.size(), &alpha, D, dv, doff, dcol, 3, dx,
                            &beta, dy),
             CUSPARSE_STATUS_INVALID_VALUE, "bsrmv refuses op(A) = A^T");
      expect(cusparseDbsrmv(h, dir_of(true), N, b.mb, b.nb, (int)b.col.size(), &alpha, D, dv, doff, dcol, 1, dx,
                            &beta, dy),
             CUSPARSE_STATUS_INVALID_VALUE, "bsrmv refuses blocks of 1");
      cusparseSetMatType(D, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
      expect(cusparseDbsrmv(h, dir_of(true), N, b.mb, b.nb, (int)b.col.size(), &alpha, D, dv, doff, dcol, 3, dx,
                            &beta, dy),
             CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED, "bsrmv refuses a symmetric matrix type");
      cusparseSetMatType(D, CUSPARSE_MATRIX_TYPE_GENERAL);

      // bsrxmv: block rows 1 and 3 only, row 3 without its first block.
      cudaMemcpy(dy, y0.data(), m * sizeof(double), cudaMemcpyHostToDevice);
      std::vector<int> mask = {1, 3}, end(b.off.begin() + 1, b.off.end()), start(b.off.begin(), b.off.end() - 1);
      start[3] = std::min(start[3] + 1, end[3]);
      int* dmask = upload(mask);
      int* dstart = upload(start);
      int* dend = upload(end);
      CK(cusparseDbsrxmv(h, dir_of(true), N, 2, b.mb, b.nb, (int)b.col.size(), &alpha, D, dv, dmask, dstart, dend,
                         dcol, 3, dx, &beta, dy));
      std::vector<double> refx = y0;
      for (int bi : mask)
        for (int r = 0; r < 3; ++r) {
          double s = 0;
          for (int k = start[bi]; k < end[bi]; ++k)
            for (int c = 0; c < 3; ++c) s += b.val[b.at(k, r, c)] * x[b.col[k] * 3 + c];
          refx[bi * 3 + r] = alpha * s + beta * y0[bi * 3 + r];
        }
      e = max_rel(refx, download(dy, m));
      check(e < 1e-14, "bsrxmv updates the masked block rows over [start, end) and leaves the rest", e);
      for (void* p : {(void*)dmask, (void*)dstart, (void*)dend}) cudaFree(p);
    }

    // bsrmm: C = alpha A op(B) + beta C, C m x 5 column-major.
    const int nc = 5;
    for (int tb = 0; tb < 2; ++tb) {
      const int ldb = tb ? nc + 1 : n + 2, ldc = m + 1;
      std::vector<double> bm((size_t)ldb * (tb ? n : nc)), c0((size_t)ldc * nc);
      for (size_t i = 0; i < bm.size(); ++i) bm[i] = std::sin(0.11 * (double)i);
      for (size_t i = 0; i < c0.size(); ++i) c0[i] = std::cos(0.13 * (double)i);
      double* dB = upload(bm);
      double* dC = upload(c0);
      CK(cusparseDbsrmm(h, dir_of(row_major), N, tb ? T : N, b.mb, nc, b.nb, (int)b.col.size(), &alpha, D, dv, doff,
                        dcol, 3, dB, ldb, &beta, dC, ldc));
      std::vector<double> r = c0;
      for (int j = 0; j < nc; ++j)
        for (int i = 0; i < m; ++i) {
          double s = 0;
          for (int k = 0; k < n; ++k) s += d[(size_t)i * n + k] * (tb ? bm[(size_t)k * ldb + j] : bm[(size_t)j * ldb + k]);
          r[(size_t)j * ldc + i] = alpha * s + beta * c0[(size_t)j * ldc + i];
        }
      e = max_rel(r, download(dC, c0.size()));
      check(e < 1e-14, tb ? "bsrmm, op(B) = B^T, padded leading dimensions" : "bsrmm, padded leading dimensions", e);
      cudaFree(dB);
      cudaFree(dC);
    }
    cusparseDestroyMatDescr(D);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, (void*)dx, (void*)dy}) cudaFree(p);
  }

  {  // Complex: Zbsrmv.
    const auto b = make_bsr<cdouble>(3, 3, 2, 2, true, 29, false);
    const auto d = b.dense();
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    auto* dv = upload(b.val);
    std::vector<cdouble> x(6);
    for (int i = 0; i < 6; ++i) x[i] = {std::cos(i * 0.3), std::sin(i * 0.7)};
    auto* dx = upload(x);
    auto* dy = upload(std::vector<cdouble>(6));
    cusparseMatDescr_t D = descr();
    const cdouble alpha(1, -1), beta(0, 0);
    CK(cusparseZbsrmv(h, CUSPARSE_DIRECTION_ROW, N, 3, 3, (int)b.col.size(),
                      reinterpret_cast<const cuDoubleComplex*>(&alpha), D,
                      reinterpret_cast<const cuDoubleComplex*>(dv), doff, dcol, 2,
                      reinterpret_cast<const cuDoubleComplex*>(dx), reinterpret_cast<const cuDoubleComplex*>(&beta),
                      reinterpret_cast<cuDoubleComplex*>(dy)));
    std::vector<cdouble> ref(6);
    for (int i = 0; i < 6; ++i) {
      cdouble s = 0;
      for (int j = 0; j < 6; ++j) s += d[(size_t)i * 6 + j] * x[j];
      ref[i] = alpha * s;
    }
    const double e = max_rel(ref, download(dy, 6));
    check(e < 1e-14, "Zbsrmv", e);
    cusparseDestroyMatDescr(D);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, (void*)dx, (void*)dy}) cudaFree(p);
  }
}

// op(T) x = b with T the triangle of the element matrix the fill mode names.
template <class S>
static std::vector<S> tri_ref(const std::vector<S>& d, int n, bool lower, bool unit, int op, std::vector<S> b) {
  auto t = [&](int i, int j) -> S {  // op(T)(i, j)
    const int r = op ? j : i, c = op ? i : j;
    if (lower ? c > r : c < r) return S(0);
    if (r == c && unit) return S(1);
    S v = d[(size_t)r * n + c];
    if constexpr (!std::is_same_v<S, double>)
      if (op == 2) v = std::conj(v);
    return v;
  };
  const bool low = op ? !lower : lower;
  for (int s = 0; s < n; ++s) {
    const int i = low ? s : n - 1 - s;
    S v = b[i];
    for (int j = 0; j < n; ++j)
      if (j != i && (low ? j < i : j > i)) v -= t(i, j) * b[j];
    b[i] = v / t(i, i);
  }
  return b;
}

static void legacy_solves() {
  const int bd = 3, mb = 4, n = mb * bd;
  for (int row_major = 0; row_major < 2; ++row_major) {
    const auto b = make_bsr<double>(mb, mb, bd, bd, row_major, 31, true, 6.0);
    const auto d = b.dense();
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    double* dv = upload(b.val);
    std::vector<double> f(n);
    for (int i = 0; i < n; ++i) f[i] = std::cos(0.5 * i) + 0.25;
    double* df = upload(f);
    double* dx = upload(std::vector<double>(n));
    double worst = 0;
    bool all = true;
    for (int lower = 0; lower < 2; ++lower)
      for (int unit = 0; unit < 2; ++unit)
        for (int op = 0; op < 2; ++op) {
          cusparseMatDescr_t D = descr(CUSPARSE_INDEX_BASE_ZERO, lower ? CUSPARSE_FILL_MODE_LOWER : CUSPARSE_FILL_MODE_UPPER,
                                       unit ? CUSPARSE_DIAG_TYPE_UNIT : CUSPARSE_DIAG_TYPE_NON_UNIT);
          const auto trans = op ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
          bsrsv2Info_t info;
          CK(cusparseCreateBsrsv2Info(&info));
          int bytes = 0;
          CK(cusparseDbsrsv2_bufferSize(h, dir_of(row_major), trans, mb, (int)b.col.size(), D, dv, doff, dcol, bd, info,
                                        &bytes));
          void* buf = nullptr;
          cudaMalloc(&buf, std::max(bytes, 16));
          CK(cusparseDbsrsv2_analysis(h, dir_of(row_major), trans, mb, (int)b.col.size(), D, dv, doff, dcol, bd, info,
                                      CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
          const double alpha = 1.5;
          CK(cusparseDbsrsv2_solve(h, dir_of(row_major), trans, mb, (int)b.col.size(), &alpha, D, dv, doff, dcol, bd,
                                   info, df, dx, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
          int pos = -7;
          const int zp = cusparseXbsrsv2_zeroPivot(h, info, &pos);
          std::vector<double> rhs = f;
          for (auto& v : rhs) v *= alpha;
          const double e = max_rel(tri_ref(d, n, lower, unit, op, rhs), download(dx, n));
          if (!(e < 1e-13) || zp != 0 || pos != -1) {
            std::printf("     bsrsv2 %s %s op=%d: %.2e, zero pivot %d at %d\n", lower ? "lower" : "upper",
                        unit ? "unit" : "non-unit", op, e, zp, pos);
            all = false;
          }
          worst = std::fmax(worst, e);
          cudaFree(buf);
          cusparseDestroyBsrsv2Info(info);
          cusparseDestroyMatDescr(D);
        }
    check(all, row_major ? "bsrsv2, lower and upper, unit and not, A and A^T, row-major blocks"
                         : "bsrsv2, lower and upper, unit and not, A and A^T, column-major blocks",
          worst);

    // bsrsm2 with 3 right-hand sides, B and X stored plainly and transposed.
    for (int tx = 0; tx < 2; ++tx) {
      const int nrhs = 3, ld = tx ? nrhs + 1 : n + 1;
      std::vector<double> bm((size_t)ld * (tx ? n : nrhs));
      for (size_t i = 0; i < bm.size(); ++i) bm[i] = std::sin(0.21 * (double)i + 0.5);
      double* dB = upload(bm);
      double* dX = upload(std::vector<double>(bm.size()));
      cusparseMatDescr_t D = descr(CUSPARSE_INDEX_BASE_ZERO, CUSPARSE_FILL_MODE_UPPER);
      const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE, T = CUSPARSE_OPERATION_TRANSPOSE;
      bsrsm2Info_t info;
      CK(cusparseCreateBsrsm2Info(&info));
      int bytes = 0;
      CK(cusparseDbsrsm2_bufferSize(h, dir_of(row_major), T, tx ? T : N, mb, nrhs, (int)b.col.size(), D, dv, doff,
                                    dcol, bd, info, &bytes));
      void* buf = nullptr;
      cudaMalloc(&buf, std::max(bytes, 16));
      CK(cusparseDbsrsm2_analysis(h, dir_of(row_major), T, tx ? T : N, mb, nrhs, (int)b.col.size(), D, dv, doff, dcol,
                                  bd, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
      const double alpha = -2.0;
      CK(cusparseDbsrsm2_solve(h, dir_of(row_major), T, tx ? T : N, mb, nrhs, (int)b.col.size(), &alpha, D, dv, doff,
                               dcol, bd, info, dB, ld, dX, ld, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
      const auto got = download(dX, bm.size());
      std::vector<double> r, g;
      for (int j = 0; j < nrhs; ++j) {
        std::vector<double> rhs(n);
        for (int i = 0; i < n; ++i) rhs[i] = alpha * (tx ? bm[(size_t)i * ld + j] : bm[(size_t)j * ld + i]);
        const auto x = tri_ref(d, n, false, false, 1, rhs);
        for (int i = 0; i < n; ++i) {
          r.push_back(x[i]);
          g.push_back(tx ? got[(size_t)i * ld + j] : got[(size_t)j * ld + i]);
        }
      }
      const double e = max_rel(r, g);
      check(e < 1e-13, tx ? "bsrsm2, upper A^T, B and X stored transposed" : "bsrsm2, upper A^T", e);
      cudaFree(buf);
      cusparseDestroyBsrsm2Info(info);
      cusparseDestroyMatDescr(D);
      cudaFree(dB);
      cudaFree(dX);
    }
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, (void*)df, (void*)dx}) cudaFree(p);
  }

  {  // Complex A^H: Cbsrsv2 lower, conjugate transpose.
    auto b = make_bsr<cdouble>(3, 3, 2, 2, true, 37, true, 4.0);
    std::vector<std::complex<float>> v;
    for (auto& x : b.val) {
      v.push_back({(float)x.real(), (float)x.imag()});
      x = cdouble(v.back());
    }
    const auto d = b.dense();
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    auto* dv = upload(v);
    std::vector<std::complex<float>> f(6);
    for (int i = 0; i < 6; ++i) f[i] = {(float)std::cos(i), (float)std::sin(2.0 * i)};
    auto* df = upload(f);
    auto* dx = upload(std::vector<std::complex<float>>(6));
    cusparseMatDescr_t D = descr();
    bsrsv2Info_t info;
    CK(cusparseCreateBsrsv2Info(&info));
    int bytes = 0;
    const auto C = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    CK(cusparseCbsrsv2_bufferSize(h, CUSPARSE_DIRECTION_ROW, C, 3, (int)b.col.size(), D,
                                  reinterpret_cast<cuComplex*>(dv), doff, dcol, 2, info, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max(bytes, 16));
    CK(cusparseCbsrsv2_analysis(h, CUSPARSE_DIRECTION_ROW, C, 3, (int)b.col.size(), D,
                                reinterpret_cast<cuComplex*>(dv), doff, dcol, 2, info, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                                buf));
    const std::complex<float> alpha(1, 0);
    CK(cusparseCbsrsv2_solve(h, CUSPARSE_DIRECTION_ROW, C, 3, (int)b.col.size(),
                             reinterpret_cast<const cuComplex*>(&alpha), D, reinterpret_cast<cuComplex*>(dv), doff,
                             dcol, 2, info, reinterpret_cast<cuComplex*>(df), reinterpret_cast<cuComplex*>(dx),
                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    std::vector<cdouble> rhs(f.begin(), f.end());
    const auto out = download(dx, 6);
    const double e = max_rel(tri_ref(d, 6, true, false, 2, rhs), std::vector<cdouble>(out.begin(), out.end()));
    check(e < 2e-6, "Cbsrsv2, lower, A^H", e);
    cudaFree(buf);
    cusparseDestroyBsrsv2Info(info);
    cusparseDestroyMatDescr(D);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, (void*)df, (void*)dx}) cudaFree(p);
  }
}

// Zero pivots: by block row, at analysis for a missing diagonal block, at
// solve time for a zero on the diagonal, in the matrix's own base.
static void zero_pivots() {
  const auto N = CUSPARSE_OPERATION_NON_TRANSPOSE;
  const std::vector<double> v = {4, 0, 2, 5, 6, 7, 0, 0};  // blocks (0,0) and (1,?), 2x2 row-major
  double* dv = upload(v);
  double* df = upload(std::vector<double>{1, 2, 3, 4});
  double* dx = upload(std::vector<double>(4));
  void* buf = nullptr;
  cudaMalloc(&buf, 4096);
  const double one = 1;
  int pos = -7;
  {  // Block row 1 holds only block (1,0): its diagonal block is missing.
    int* doff = upload(std::vector<int>{0, 1, 2});
    int* dcol = upload(std::vector<int>{0, 0});
    cusparseMatDescr_t D = descr();
    bsrsv2Info_t info;
    CK(cusparseCreateBsrsv2Info(&info));
    CK(cusparseDbsrsv2_analysis(h, CUSPARSE_DIRECTION_ROW, N, 2, 2, D, dv, doff, dcol, 2, info,
                                CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    const int zp = cusparseXbsrsv2_zeroPivot(h, info, &pos);
    check(zp == CUSPARSE_STATUS_ZERO_PIVOT && pos == 1, "bsrsv2_analysis: a missing diagonal block is block row 1's",
          pos);
    cusparseSetMatDiagType(D, CUSPARSE_DIAG_TYPE_UNIT);
    bsrsv2Info_t info2;
    CK(cusparseCreateBsrsv2Info(&info2));
    CK(cusparseDbsrsv2_analysis(h, CUSPARSE_DIRECTION_ROW, N, 2, 2, D, dv, doff, dcol, 2, info2,
                                CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    pos = -7;
    expect(cusparseXbsrsv2_zeroPivot(h, info2, &pos), CUSPARSE_STATUS_SUCCESS,
           "with a unit diagonal nothing is missing");
    cusparseDestroyBsrsv2Info(info);
    cusparseDestroyBsrsv2Info(info2);
    cusparseDestroyMatDescr(D);
    cudaFree(doff);
    cudaFree(dcol);
  }
  for (int base = 0; base < 2; ++base) {  // Block (1,1) = [6 7; 0 0]: a zero at element (3,3).
    int* doff = upload(std::vector<int>{base, 1 + base, 2 + base});
    int* dcol = upload(std::vector<int>{base, 1 + base});
    cusparseMatDescr_t D = descr(base ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO);
    bsrsv2Info_t info;
    CK(cusparseCreateBsrsv2Info(&info));
    CK(cusparseDbsrsv2_analysis(h, CUSPARSE_DIRECTION_ROW, N, 2, 2, D, dv, doff, dcol, 2, info,
                                CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    pos = -7;
    const int before = cusparseXbsrsv2_zeroPivot(h, info, &pos);
    CK(cusparseDbsrsv2_solve(h, CUSPARSE_DIRECTION_ROW, N, 2, 2, &one, D, dv, doff, dcol, 2, info, df, dx,
                             CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    int pos2 = -7;
    const int after = cusparseXbsrsv2_zeroPivot(h, info, &pos2);
    check(before == CUSPARSE_STATUS_SUCCESS && pos == -1 && after == CUSPARSE_STATUS_ZERO_PIVOT && pos2 == 1 + base,
          base ? "a numeric zero shows at solve time, at block row 2 one-based"
               : "a numeric zero shows at solve time, not at analysis, at block row 1",
          pos2);
    cusparseDestroyBsrsv2Info(info);
    cusparseDestroyMatDescr(D);
    cudaFree(doff);
    cudaFree(dcol);
  }
  for (void* p : {(void*)dv, (void*)df, (void*)dx, buf}) cudaFree(p);
}

// IC(0) and ILU(0) of a dense matrix restricted to `mask`, as references.
static std::vector<double> ilu0(std::vector<double> a, const std::vector<char>& mask, int n) {
  for (int i = 1; i < n; ++i)
    for (int k = 0; k < i; ++k) {
      if (!mask[(size_t)i * n + k]) continue;
      a[(size_t)i * n + k] /= a[(size_t)k * n + k];
      for (int j = k + 1; j < n; ++j)
        if (mask[(size_t)i * n + j] && mask[(size_t)k * n + j]) a[(size_t)i * n + j] -= a[(size_t)i * n + k] * a[(size_t)k * n + j];
    }
  return a;
}
static std::vector<double> ic0(std::vector<double> a, const std::vector<char>& mask, int n) {
  for (int j = 0; j < n; ++j) {
    double dsq = a[(size_t)j * n + j];
    for (int k = 0; k < j; ++k)
      if (mask[(size_t)j * n + k]) dsq -= a[(size_t)j * n + k] * a[(size_t)j * n + k];
    const double ljj = std::sqrt(dsq);
    a[(size_t)j * n + j] = ljj;
    for (int i = j + 1; i < n; ++i) {
      if (!mask[(size_t)i * n + j]) continue;
      double s = a[(size_t)i * n + j];
      for (int k = 0; k < j; ++k)
        if (mask[(size_t)i * n + k] && mask[(size_t)j * n + k]) s -= a[(size_t)i * n + k] * a[(size_t)j * n + k];
      a[(size_t)i * n + j] = s / ljj;
    }
  }
  return a;
}

static void legacy_factorizations() {
  const int bd = 2, mb = 5, n = bd * mb;
  for (int row_major = 0; row_major < 2; ++row_major) {
    // A symmetric pattern of blocks, symmetric values, a dominant diagonal.
    auto b = make_bsr<double>(mb, mb, bd, bd, row_major, 43, true, 8.0);
    {
      auto d = b.dense();
      std::vector<std::vector<int>> rows(mb);
      for (int i = 0; i < mb; ++i)
        for (int k = b.off[i]; k < b.off[i + 1]; ++k) {
          rows[i].push_back(b.col[k]);
          rows[b.col[k]].push_back(i);
        }
      Bsr<double> s{mb, mb, bd, bd, (bool)row_major, {0}, {}, {}};
      for (auto& r : rows) {
        std::sort(r.begin(), r.end());
        r.erase(std::unique(r.begin(), r.end()), r.end());
        s.col.insert(s.col.end(), r.begin(), r.end());
        s.off.push_back((int)s.col.size());
      }
      s.val.resize(s.col.size() * bd * bd);
      for (int i = 0; i < mb; ++i)
        for (int k = s.off[i]; k < s.off[i + 1]; ++k)
          for (int r = 0; r < bd; ++r)
            for (int c = 0; c < bd; ++c) {
              const int gi = i * bd + r, gj = s.col[k] * bd + c;
              s.val[s.at(k, r, c)] = gi == gj ? 8.0 + gi : 0.5 * std::cos(0.3 * (gi + gj)) + 0.1 * std::sin(gi * gj);
            }
      b = s;
    }
    const auto d = b.dense();
    const auto mask = b.mask();
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    cusparseMatDescr_t D = descr();
    const int nnzb = (int)b.col.size();

    // IC(0): the lower triangle of the element matrix factored in place; the
    // rest, the upper half of each diagonal block included, left as it was.
    double* dv = upload(b.val);
    bsric02Info_t ic;
    CK(cusparseCreateBsric02Info(&ic));
    int bytes = 0;
    CK(cusparseDbsric02_bufferSize(h, dir_of(row_major), mb, nnzb, D, dv, doff, dcol, bd, ic, &bytes));
    void* buf = nullptr;
    cudaMalloc(&buf, std::max(bytes, 16));
    CK(cusparseDbsric02_analysis(h, dir_of(row_major), mb, nnzb, D, dv, doff, dcol, bd, ic,
                                 CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    CK(cusparseDbsric02(h, dir_of(row_major), mb, nnzb, D, dv, doff, dcol, bd, ic, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        buf));
    int pos = -7;
    const int zp = cusparseXbsric02_zeroPivot(h, ic, &pos);
    auto got = b;
    got.val = download(dv, b.val.size());
    const auto L = ic0(d, mask, n);
    std::vector<double> r, g;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j)
        if (mask[(size_t)i * n + j]) {
          r.push_back(j <= i ? L[(size_t)i * n + j] : d[(size_t)i * n + j]);
          g.push_back(got.dense()[(size_t)i * n + j]);
        }
    double e = max_rel(r, g);
    check(e < 1e-14 && zp == 0 && pos == -1,
          row_major ? "bsric02, row-major blocks: IC(0) below the diagonal, the rest untouched"
                    : "bsric02, column-major blocks: IC(0) below the diagonal, the rest untouched",
          e);
    cusparseDestroyBsric02Info(ic);
    cudaFree(dv);

    // ILU(0) over the whole element pattern.
    dv = upload(b.val);
    bsrilu02Info_t ilu;
    CK(cusparseCreateBsrilu02Info(&ilu));
    CK(cusparseDbsrilu02_bufferSize(h, dir_of(row_major), mb, nnzb, D, dv, doff, dcol, bd, ilu, &bytes));
    CK(cusparseDbsrilu02_analysis(h, dir_of(row_major), mb, nnzb, D, dv, doff, dcol, bd, ilu,
                                  CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    CK(cusparseDbsrilu02(h, dir_of(row_major), mb, nnzb, D, dv, doff, dcol, bd, ilu, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                         buf));
    got.val = download(dv, b.val.size());
    const auto LU = ilu0(d, mask, n);
    r.clear();
    g.clear();
    const auto gd = got.dense();
    for (size_t i = 0; i < mask.size(); ++i)
      if (mask[i]) {
        r.push_back(LU[i]);
        g.push_back(gd[i]);
      }
    e = max_rel(r, g);
    check(e < 1e-14, row_major ? "bsrilu02, row-major blocks" : "bsrilu02, column-major blocks", e);
    cusparseDestroyBsrilu02Info(ilu);
    cudaFree(dv);

    if (row_major) {
      // The same matrices as CSR (blocks of 1): csric02 and csrilu02.
      std::vector<int> off{0}, col;
      std::vector<double> val;
      for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j)
          if (mask[(size_t)i * n + j]) {
            col.push_back(j);
            val.push_back(d[(size_t)i * n + j]);
          }
        off.push_back((int)col.size());
      }
      int* co = upload(off);
      int* cc = upload(col);
      double* cv = upload(val);
      csric02Info_t ci;
      CK(cusparseCreateCsric02Info(&ci));
      CK(cusparseDcsric02_bufferSize(h, n, (int)val.size(), D, cv, co, cc, ci, &bytes));
      CK(cusparseDcsric02_analysis(h, n, (int)val.size(), D, cv, co, cc, ci, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf));
      CK(cusparseDcsric02(h, n, (int)val.size(), D, cv, co, cc, ci, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf));
      auto out = download(cv, val.size());
      r.clear();
      g.clear();
      for (int i = 0; i < n; ++i)
        for (int k = off[i]; k < off[i + 1]; ++k) {
          r.push_back(col[k] <= i ? L[(size_t)i * n + col[k]] : d[(size_t)i * n + col[k]]);
          g.push_back(out[k]);
        }
      e = max_rel(r, g);
      check(e < 1e-14, "csric02", e);
      cudaMemcpy(cv, val.data(), val.size() * sizeof(double), cudaMemcpyHostToDevice);
      csrilu02Info_t cl;
      CK(cusparseCreateCsrilu02Info(&cl));
      CK(cusparseDcsrilu02_bufferSize(h, n, (int)val.size(), D, cv, co, cc, cl, &bytes));
      CK(cusparseDcsrilu02_analysis(h, n, (int)val.size(), D, cv, co, cc, cl, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf));
      CK(cusparseDcsrilu02(h, n, (int)val.size(), D, cv, co, cc, cl, CUSPARSE_SOLVE_POLICY_NO_LEVEL, buf));
      out = download(cv, val.size());
      r.clear();
      g.clear();
      for (int i = 0; i < n; ++i)
        for (int k = off[i]; k < off[i + 1]; ++k) {
          r.push_back(LU[(size_t)i * n + col[k]]);
          g.push_back(out[k]);
        }
      e = max_rel(r, g);
      check(e < 1e-14, "csrilu02", e);
      cusparseDestroyCsric02Info(ci);
      cusparseDestroyCsrilu02Info(cl);
      for (void* p : {(void*)co, (void*)cc, (void*)cv}) cudaFree(p);
    }
    cudaFree(buf);
    cusparseDestroyMatDescr(D);
    cudaFree(doff);
    cudaFree(dcol);
  }

  {  // Pivots: IC(0) of a matrix that is not positive definite, ILU(0)'s zero, and its boost.
    const std::vector<int> off = {0, 2, 4}, col = {0, 1, 0, 1};
    int* doff = upload(off);
    int* dcol = upload(col);
    cusparseMatDescr_t D = descr();
    void* buf = nullptr;
    cudaMalloc(&buf, 4096);
    // Blocks (0,0) (0,1) (1,0) (1,1), row-major; 9 and 99 lie above the diagonal.
    const std::vector<double> notspd = {1, 9, 3, 1, 1, 2, 3, 4, 1, 3, 2, 4, 6, 99, 1, 7};
    double* dv = upload(notspd);
    bsric02Info_t ic;
    CK(cusparseCreateBsric02Info(&ic));
    CK(cusparseDbsric02_analysis(h, CUSPARSE_DIRECTION_ROW, 2, 4, D, dv, doff, dcol, 2, ic,
                                 CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    CK(cusparseDbsric02(h, CUSPARSE_DIRECTION_ROW, 2, 4, D, dv, doff, dcol, 2, ic, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                        buf));
    int pos = -7;
    const int zp = cusparseXbsric02_zeroPivot(h, ic, &pos);
    const auto out = download(dv, 16);
    // 1 - 3^2 = -8 at element (1,1): reported, and the square root of its magnitude carries on.
    const std::vector<double> want = {1, 9, 3, std::sqrt(8.0), 1, 2, 3, 4, 1, 0, 2, -1 / std::sqrt(2.0),
                                      std::sqrt(5.0), 99, -1 / std::sqrt(5.0), std::sqrt(2.3)};
    const double e = max_rel(want, out);
    check(zp == CUSPARSE_STATUS_ZERO_PIVOT && pos == 0 && e < 1e-15,
          "bsric02 reports a pivot that is not positive at its block row and goes on with sqrt(|d|)", e);
    cusparseDestroyBsric02Info(ic);
    cudaFree(dv);

    const std::vector<double> singular = {1, 1, 1, 1, 1, 2, 3, 4, 1, 3, 2, 4, 6, 99, 1, 7};
    for (int boost = 0; boost < 2; ++boost) {
      dv = upload(singular);
      bsrilu02Info_t ilu;
      CK(cusparseCreateBsrilu02Info(&ilu));
      double tol = 1e-3, val = 0.5;
      if (boost) CK(cusparseDbsrilu02_numericBoost(h, ilu, 1, &tol, &val));
      CK(cusparseDbsrilu02_analysis(h, CUSPARSE_DIRECTION_ROW, 2, 4, D, dv, doff, dcol, 2, ilu,
                                    CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
      CK(cusparseDbsrilu02(h, CUSPARSE_DIRECTION_ROW, 2, 4, D, dv, doff, dcol, 2, ilu, CUSPARSE_SOLVE_POLICY_USE_LEVEL,
                           buf));
      pos = -7;
      const int z = cusparseXbsrilu02_zeroPivot(h, ilu, &pos);
      const auto o = download(dv, 16);
      if (boost)
        check(z == 0 && pos == -1 && o[3] == 0.5 && o[9] == 4 && o[15] == -272,
              "bsrilu02's numeric boost replaces the zero pivot and reports none", o[3]);
      else
        check(z == CUSPARSE_STATUS_ZERO_PIVOT && pos == 0 && o[3] == 0 && std::isinf(o[9]),
              "bsrilu02 reports a zero pivot at its block row and divides by it", o[3]);
      cusparseDestroyBsrilu02Info(ilu);
      cudaFree(dv);
    }
    cudaFree(buf);
    cusparseDestroyMatDescr(D);
    cudaFree(doff);
    cudaFree(dcol);
  }

  {  // Complex ILU(0): Zbsrilu02 against the definition, checked by L U = A on the pattern.
    const auto b = make_bsr<cdouble>(3, 3, 2, 2, true, 47, true, 5.0);
    const int nn = 6;
    int* doff = upload(b.off);
    int* dcol = upload(b.col);
    auto* dv = upload(b.val);
    cusparseMatDescr_t D = descr();
    bsrilu02Info_t ilu;
    CK(cusparseCreateBsrilu02Info(&ilu));
    void* buf = nullptr;
    cudaMalloc(&buf, 4096);
    CK(cusparseZbsrilu02_analysis(h, CUSPARSE_DIRECTION_ROW, 3, (int)b.col.size(), D,
                                  reinterpret_cast<cuDoubleComplex*>(dv), doff, dcol, 2, ilu,
                                  CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    CK(cusparseZbsrilu02(h, CUSPARSE_DIRECTION_ROW, 3, (int)b.col.size(), D, reinterpret_cast<cuDoubleComplex*>(dv),
                         doff, dcol, 2, ilu, CUSPARSE_SOLVE_POLICY_USE_LEVEL, buf));
    auto got = b;
    got.val = download(dv, b.val.size());
    const auto f = got.dense(), a = b.dense();
    const auto mask = b.mask();
    // ILU(0)'s defining property: (L U)(i, j) = A(i, j) wherever A has an entry.
    double err = 0, scale = 0;
    for (int i = 0; i < nn; ++i)
      for (int j = 0; j < nn; ++j) {
        if (!mask[(size_t)i * nn + j]) continue;
        cdouble s = 0;
        for (int k = 0; k <= std::min(i, j); ++k) {
          const cdouble l = k == i ? 1.0 : f[(size_t)i * nn + k];
          s += l * f[(size_t)k * nn + j];
        }
        err = std::fmax(err, std::abs(s - a[(size_t)i * nn + j]));
        scale = std::fmax(scale, std::abs(a[(size_t)i * nn + j]));
      }
    check(err / scale < 1e-14, "Zbsrilu02: L U equals A on A's pattern", err / scale);
    cusparseDestroyBsrilu02Info(ilu);
    cusparseDestroyMatDescr(D);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dv, buf}) cudaFree(p);
  }
}

// CSR -> BSR -> CSR, with padding where the size is not a multiple of the
// block, column-major blocks and a one-based result; then general blocks.
static void conversions() {
  const int m = 7, n = 8;
  std::vector<int> off{0}, col;
  std::vector<double> val;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      if (hash01(i, j, 59) < 0.3) {
        col.push_back(j);
        val.push_back(1.0 + i * 10 + j);
      }
    off.push_back((int)col.size());
  }
  std::vector<double> dense((size_t)m * n, 0.0);
  for (int i = 0; i < m; ++i)
    for (int k = off[i]; k < off[i + 1]; ++k) dense[(size_t)i * n + col[k]] = val[k];
  int* doff = upload(off);
  int* dcol = upload(col);
  double* dv = upload(val);
  cusparseMatDescr_t A = descr(), C = descr(CUSPARSE_INDEX_BASE_ONE);
  for (int rbd : {3, 2})
    for (int cbd : {3, 4}) {
      const bool square = rbd == cbd;
      if (!square && rbd == 3) continue;
      const int mb = (m + rbd - 1) / rbd, nb = (n + cbd - 1) / cbd;
      int* boff = upload(std::vector<int>(mb + 1));
      int nnzb = -1;
      void* buf = nullptr;
      if (square) {
        CK(cusparseXcsr2bsrNnz(h, CUSPARSE_DIRECTION_COLUMN, m, n, A, doff, dcol, rbd, C, boff, &nnzb));
      } else {
        int bytes = 0;
        CK(cusparseDcsr2gebsr_bufferSize(h, CUSPARSE_DIRECTION_COLUMN, m, n, A, dv, doff, dcol, rbd, cbd, &bytes));
        cudaMalloc(&buf, std::max(bytes, 16));
        CK(cusparseXcsr2gebsrNnz(h, CUSPARSE_DIRECTION_COLUMN, m, n, A, doff, dcol, C, boff, rbd, cbd, &nnzb, buf));
      }
      int* bcol = upload(std::vector<int>(std::max(nnzb, 1)));
      double* bval = upload(std::vector<double>((size_t)std::max(nnzb, 1) * rbd * cbd));
      if (square)
        CK(cusparseDcsr2bsr(h, CUSPARSE_DIRECTION_COLUMN, m, n, A, dv, doff, dcol, rbd, C, bval, boff, bcol));
      else
        CK(cusparseDcsr2gebsr(h, CUSPARSE_DIRECTION_COLUMN, m, n, A, dv, doff, dcol, C, bval, boff, bcol, rbd, cbd,
                              buf));
      Bsr<double> b{mb, nb, rbd, cbd, false, download(boff, mb + 1), download(bcol, nnzb),
                    download(bval, (size_t)nnzb * rbd * cbd)};
      bool ok = b.off[0] == 1 && b.off[mb] == nnzb + 1;
      for (int& x : b.off) x -= 1;
      for (int& x : b.col) x -= 1;
      for (int i = 0; i < mb && ok; ++i)
        for (int k = b.off[i] + 1; k < b.off[i + 1]; ++k) ok = ok && b.col[k - 1] < b.col[k];
      int want = 0;  // nonzero blocks
      for (int bi = 0; bi < mb; ++bi)
        for (int bj = 0; bj < nb; ++bj) {
          bool any = false;
          for (int i = bi * rbd; i < std::min(m, (bi + 1) * rbd); ++i)
            for (int k = off[i]; k < off[i + 1]; ++k) any = any || col[k] / cbd == bj;
          want += any;
        }
      ok = ok && nnzb == want;
      if (ok) {
        const auto bd = b.dense();
        for (int i = 0; i < b.rows() && ok; ++i)
          for (int j = 0; j < b.cols() && ok; ++j)
            ok = bd[(size_t)i * b.cols() + j] == (i < m && j < n ? dense[(size_t)i * n + j] : 0.0);
      }
      char what[160];
      std::snprintf(what, sizeof what, "csr2%sbsr, %dx%d blocks, column-major, one-based, zero padded",
                    square ? "" : "ge", rbd, cbd);
      check(ok, what, 0);

      // And back: every element of every block becomes an entry.
      const int mm = mb * rbd, total = nnzb * rbd * cbd;
      int* coff = upload(std::vector<int>(mm + 1));
      int* ccol = upload(std::vector<int>(total));
      double* cval = upload(std::vector<double>(total));
      if (square)
        CK(cusparseDbsr2csr(h, CUSPARSE_DIRECTION_COLUMN, mb, nb, C, bval, boff, bcol, rbd, A, cval, coff, ccol));
      else
        CK(cusparseDgebsr2csr(h, CUSPARSE_DIRECTION_COLUMN, mb, nb, C, bval, boff, bcol, rbd, cbd, A, cval, coff,
                              ccol));
      const auto o2 = download(coff, mm + 1), c2 = download(ccol, total);
      const auto v2 = download(cval, total);
      ok = o2[0] == 0 && o2[mm] == total;
      for (int i = 0; i < mm && ok; ++i) {
        ok = o2[i + 1] - o2[i] == (b.off[i / rbd + 1] - b.off[i / rbd]) * cbd;
        for (int k = o2[i]; k < o2[i + 1] && ok; ++k)
          ok = (k == o2[i] || c2[k - 1] < c2[k]) && v2[k] == (i < m && c2[k] < n ? dense[(size_t)i * n + c2[k]] : 0.0);
      }
      std::snprintf(what, sizeof what, "%sbsr2csr back, %dx%d blocks: every block element an entry",
                    square ? "" : "ge", rbd, cbd);
      check(ok, what, 0);
      for (void* p : {(void*)boff, (void*)bcol, (void*)bval, (void*)coff, (void*)ccol, (void*)cval, buf}) cudaFree(p);
    }
  cusparseDestroyMatDescr(A);
  cusparseDestroyMatDescr(C);
  for (void* p : {(void*)doff, (void*)dcol, (void*)dv}) cudaFree(p);
}

int main() {
  if (cusparseCreate(&h)) {
    std::printf("FAIL: cusparseCreate\n");
    return 1;
  }
  generic();
  legacy_products();
  legacy_solves();
  zero_pivots();
  legacy_factorizations();
  conversions();
  cusparseDestroy(h);
  std::printf(failures ? "FAIL: %d BSR checks\n" : "PASS: every BSR check\n", failures);
  return failures ? 1 : 0;
}
