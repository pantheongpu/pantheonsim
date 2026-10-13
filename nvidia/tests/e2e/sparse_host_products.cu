// cuSPARSE's SpMV and SpMM on compressed-row matrices -- the products that run row by row on the host's threads -- against the
// loops they replaced: the matrix expanded to (row, column, value) triples and every triple added to its output in storage
// order, in double (std::complex<double> for complex values), the result rounded to the type of the output.
// Every result is compared bit for bit (two NaNs count as equal). Covered: single and double, real and complex values; 32- and
// 64-bit indices; zero and one based; empty rows and repeated columns; alpha and beta (beta zero, one, other); a vector and a
// matrix on the right in both orders with padded leading dimensions and every operation; infinities and NaNs among the
// values; and a matrix that is not sound (a column out of range, offsets that stop short), which keeps its old answer.
// Not run against the card: this pins the library to its own arithmetic.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

using cd = std::complex<double>;
static int failures = 0, checks = 0;
static void report(bool ok, const char* what) {
  ++checks;
  if (!ok) {
    ++failures;
    std::printf("FAIL %s\n", what);
  }
}

// An element type: its CUDA data type, storage, and conversions through cd.
template <class T> struct Ty;
template <> struct Ty<float> {
  static constexpr cudaDataType dt = CUDA_R_32F;
  static constexpr bool cplx = false;
  static cd get(float v) { return {v, 0.0}; }
  static float put(cd v) { return (float)v.real(); }
};
template <> struct Ty<double> {
  static constexpr cudaDataType dt = CUDA_R_64F;
  static constexpr bool cplx = false;
  static cd get(double v) { return {v, 0.0}; }
  static double put(cd v) { return v.real(); }
};
template <> struct Ty<cuComplex> {
  static constexpr cudaDataType dt = CUDA_C_32F;
  static constexpr bool cplx = true;
  static cd get(cuComplex v) { return {v.x, v.y}; }
  static cuComplex put(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
};
template <> struct Ty<cuDoubleComplex> {
  static constexpr cudaDataType dt = CUDA_C_64F;
  static constexpr bool cplx = true;
  static cd get(cuDoubleComplex v) { return {v.x, v.y}; }
  static cuDoubleComplex put(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
};

template <class S> static bool part_same(S a, S b) { return std::memcmp(&a, &b, sizeof(S)) == 0 || (std::isnan(a) && std::isnan(b)); }
static bool same(float a, float b) { return part_same(a, b); }
static bool same(double a, double b) { return part_same(a, b); }
static bool same(cuComplex a, cuComplex b) { return part_same(a.x, b.x) && part_same(a.y, b.y); }
static bool same(cuDoubleComplex a, cuDoubleComplex b) { return part_same(a.x, b.x) && part_same(a.y, b.y); }
template <class T> static bool same_all(const std::vector<T>& a, const std::vector<T>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!same(a[i], b[i])) return false;
  return true;
}

template <class T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 16);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

static std::mt19937_64 rng(31337);
enum class Mix { Plain, Zeros, Special };
template <class T> static std::vector<T> gen(size_t n, Mix mix) {
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  const double sp[] = {0.0, -0.0, 1e30, -1e30, 1e-310, INFINITY, -INFINITY, NAN};
  std::vector<T> v(n);
  auto one = [&]() {
    switch (mix) {
      case Mix::Plain: return u(rng);
      case Mix::Zeros: return rng() % 3 == 0 ? (rng() % 2 ? 0.0 : -0.0) : u(rng);
      default: return rng() % 12 == 0 ? sp[rng() % 8] : u(rng);
    }
  };
  for (auto& x : v) x = Ty<T>::put(cd(one(), Ty<T>::cplx ? one() : 0.0));
  return v;
}

// A random CSR matrix with empty rows and repeated columns.
struct Csr {
  int64_t rows, cols;
  std::vector<int64_t> offs, idx;  // zero based
};
static Csr make_csr(int64_t rows, int64_t cols) {
  Csr m{rows, cols, {}, {}};
  m.offs.push_back(0);
  for (int64_t r = 0; r < rows; ++r) {
    const int per = (int)(rng() % 10);
    for (int k = 0; k < per; ++k) m.idx.push_back((int64_t)(rng() % (uint64_t)cols));
    if (rng() % 7 == 0 && per > 1) m.idx.push_back(m.idx.back());   // a repeated column
    m.offs.push_back((int64_t)m.idx.size());
  }
  return m;
}

template <class T, class I> static void upload(const Csr& m, bool base1, I** doffs, I** didx) {
  std::vector<I> o(m.offs.size()), c(m.idx.size());
  for (size_t i = 0; i < o.size(); ++i) o[i] = (I)(m.offs[i] + (base1 ? 1 : 0));
  for (size_t i = 0; i < c.size(); ++i) c[i] = (I)(m.idx[i] + (base1 ? 1 : 0));
  *doffs = dev(o);
  *didx = dev(c);
}

template <class T> static cd conj_if(cd v, bool conj) { return conj ? std::conj(v) : v; }

// The reference SpMV, the triple loop: y[r] += a * op(A)(entry) * x[c], entries in storage order.
template <class T>
static std::vector<T> ref_spmv(const Csr& m, const std::vector<T>& val, cusparseOperation_t op, cd alpha, cd beta, const std::vector<T>& x,
                               const std::vector<T>& y0, bool* invalid) {
  const bool tr = op != CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int64_t out = tr ? m.cols : m.rows;
  std::vector<cd> y(out, cd(0));
  if (beta != cd(0)) for (int64_t i = 0; i < out; ++i) y[i] = Ty<T>::get(y0[i]) * beta;
  *invalid = false;
  for (int64_t r = 0; r < m.rows; ++r)
    for (int64_t k = m.offs[r]; k < m.offs[r + 1]; ++k) {
      const int64_t row = tr ? m.idx[k] : r, col = tr ? r : m.idx[k];
      if (row < 0 || row >= out || col < 0 || col >= (tr ? m.rows : m.cols)) { *invalid = true; return y0; }
      y[row] += alpha * conj_if<T>(Ty<T>::get(val[k]), op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) * Ty<T>::get(x[col]);
    }
  std::vector<T> r(out);
  for (int64_t i = 0; i < out; ++i) r[i] = Ty<T>::put(y[i]);
  return r;
}

template <class T, class I>
static void spmv_case(const char* name, bool base1, Mix mix, cusparseOperation_t op, int64_t rows, int64_t cols) {
  if (Ty<T>::cplx == false && op == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) return;
  const Csr m = make_csr(rows, cols);
  const auto val = gen<T>(m.idx.size(), mix);
  const bool tr = op != CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int64_t nx = tr ? rows : cols, ny = tr ? cols : rows;
  const auto x = gen<T>(nx, mix), y0 = gen<T>(ny, Mix::Plain);
  const cd alphas[] = {{1.0, 0.0}, {0.75, -0.5}, {2.0, 0.0}}, betas[] = {{0.0, 0.0}, {1.0, 0.0}, {-0.5, 0.25}};
  for (int q = 0; q < 3; ++q) {
    cd alpha = alphas[q], beta = betas[q];
    if (!Ty<T>::cplx) alpha = alpha.real(), beta = beta.real();
    I *doffs, *didx;
    upload<T, I>(m, base1, &doffs, &didx);
    T *dval = dev(val), *dx = dev(x), *dy = dev(y0);
    cusparseHandle_t h;
    cusparseCreate(&h);
    cusparseSpMatDescr_t A;
    cusparseCreateCsr(&A, rows, cols, (int64_t)m.idx.size(), doffs, didx, dval, sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I,
                      sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I, base1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO, Ty<T>::dt);
    cusparseDnVecDescr_t X, Y;
    cusparseCreateDnVec(&X, nx, dx, Ty<T>::dt);
    cusparseCreateDnVec(&Y, ny, dy, Ty<T>::dt);
    const T al = Ty<T>::put(alpha), be = Ty<T>::put(beta);
    size_t bytes = 0;
    cusparseSpMV_bufferSize(h, op, &al, A, X, &be, Y, Ty<T>::dt, CUSPARSE_SPMV_ALG_DEFAULT, &bytes);
    void* buf = nullptr;
    cudaMalloc(&buf, bytes + 16);
    const cusparseStatus_t st = cusparseSpMV(h, op, &al, A, X, &be, Y, Ty<T>::dt, CUSPARSE_SPMV_ALG_DEFAULT, buf);
    bool invalid = false;
    const auto want = ref_spmv<T>(m, val, op, Ty<T>::get(al), Ty<T>::get(be), x, y0, &invalid);
    char what[200];
    std::snprintf(what, sizeof what, "SpMV %s op %d %lldx%lld base %d alpha %d beta %d mix %d", name, (int)op, (long long)rows, (long long)cols, base1, q, q, (int)mix);
    report(st == CUSPARSE_STATUS_SUCCESS && same_all(host(dy, ny), want), what);
    cudaFree(doffs); cudaFree(didx); cudaFree(dval); cudaFree(dx); cudaFree(dy); cudaFree(buf);
    cusparseDestroySpMat(A); cusparseDestroyDnVec(X); cusparseDestroyDnVec(Y); cusparseDestroy(h);
  }
}

// The reference SpMM.
template <class T>
static std::vector<T> ref_spmm(const Csr& m, const std::vector<T>& val, cusparseOperation_t opA, cusparseOperation_t opB, cd alpha, cd beta,
                               const std::vector<T>& B, int64_t brows, int64_t bcols, int64_t ldb, bool brow, const std::vector<T>& C0,
                               int64_t crows, int64_t ccols, int64_t ldc, bool crow) {
  const bool tA = opA != CUSPARSE_OPERATION_NON_TRANSPOSE, tB = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
  (void)brows;
  auto bat = [&](int64_t r, int64_t c) { return brow ? (size_t)r * ldb + c : (size_t)c * ldb + r; };
  auto cat = [&](int64_t r, int64_t c) { return crow ? (size_t)r * ldc + c : (size_t)c * ldc + r; };
  const size_t nc = crow ? (size_t)crows * ldc : (size_t)ccols * ldc;
  std::vector<cd> c(nc, cd(0));
  if (beta != cd(0)) for (size_t i = 0; i < nc; ++i) c[i] = Ty<T>::get(C0[i]) * beta;
  const int64_t n = tB ? brows : bcols;
  for (int64_t r = 0; r < m.rows; ++r)
    for (int64_t k = m.offs[r]; k < m.offs[r + 1]; ++k) {
      const int64_t row = tA ? m.idx[k] : r, col = tA ? r : m.idx[k];
      const cd av = alpha * conj_if<T>(Ty<T>::get(val[k]), opA == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
      (void)crows;
      for (int64_t j = 0; j < n; ++j) {
        const size_t bi = tB ? bat(j, col) : bat(col, j);
        c[cat(row, j)] += av * conj_if<T>(Ty<T>::get(B[bi]), opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE);
      }
    }
  std::vector<T> out(nc);
  for (size_t i = 0; i < nc; ++i) out[i] = Ty<T>::put(c[i]);
  // (a padded slot between columns is read back as it was: beta * old, or zero)
  return out;
}

template <class T, class I>
static void spmm_case(const char* name, bool base1, Mix mix, cusparseOperation_t opB, bool brow, bool crow, int64_t rows, int64_t cols, int64_t n, int pad) {
  if (!Ty<T>::cplx && opB == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE) return;
  const Csr m = make_csr(rows, cols);
  const auto val = gen<T>(m.idx.size(), mix);
  const bool tB = opB != CUSPARSE_OPERATION_NON_TRANSPOSE;
  const int64_t brows = tB ? n : cols, bcols = tB ? cols : n;
  const int64_t ldb = (brow ? bcols : brows) + pad, ldc = (crow ? n : rows) + pad;
  const size_t nb = brow ? (size_t)brows * ldb : (size_t)bcols * ldb, nc = crow ? (size_t)rows * ldc : (size_t)n * ldc;
  const auto B = gen<T>(nb, mix), C0 = gen<T>(nc, Mix::Plain);
  const cd alphas[] = {{1.0, 0.0}, {0.75, -0.5}}, betas[] = {{0.0, 0.0}, {-0.5, 0.25}};
  for (int q = 0; q < 2; ++q) {
    cd alpha = alphas[q], beta = betas[q];
    if (!Ty<T>::cplx) alpha = alpha.real(), beta = beta.real();
    I *doffs, *didx;
    upload<T, I>(m, base1, &doffs, &didx);
    T *dval = dev(val), *db = dev(B), *dc = dev(C0);
    cusparseHandle_t h;
    cusparseCreate(&h);
    cusparseSpMatDescr_t A;
    cusparseCreateCsr(&A, rows, cols, (int64_t)m.idx.size(), doffs, didx, dval, sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I,
                      sizeof(I) == 4 ? CUSPARSE_INDEX_32I : CUSPARSE_INDEX_64I, base1 ? CUSPARSE_INDEX_BASE_ONE : CUSPARSE_INDEX_BASE_ZERO, Ty<T>::dt);
    cusparseDnMatDescr_t Bd, Cd;
    cusparseCreateDnMat(&Bd, brows, bcols, ldb, db, Ty<T>::dt, brow ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&Cd, rows, n, ldc, dc, Ty<T>::dt, crow ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL);
    const T al = Ty<T>::put(alpha), be = Ty<T>::put(beta);
    size_t bytes = 0;
    cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, opB, &al, A, Bd, &be, Cd, Ty<T>::dt, CUSPARSE_SPMM_ALG_DEFAULT, &bytes);
    void* buf = nullptr;
    cudaMalloc(&buf, bytes + 16);
    const cusparseStatus_t st = cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, opB, &al, A, Bd, &be, Cd, Ty<T>::dt, CUSPARSE_SPMM_ALG_DEFAULT, buf);
    const auto want = ref_spmm<T>(m, val, CUSPARSE_OPERATION_NON_TRANSPOSE, opB, Ty<T>::get(al), Ty<T>::get(be), B, brows, bcols, ldb, brow, C0, rows, n, ldc, crow);
    char what[220];
    std::snprintf(what, sizeof what, "SpMM %s opB %d B %s C %s %lldx%lld n %lld pad %d base %d beta %d mix %d", name, (int)opB, brow ? "row" : "col", crow ? "row" : "col",
                  (long long)rows, (long long)cols, (long long)n, pad, base1, q, (int)mix);
    report(st == CUSPARSE_STATUS_SUCCESS && same_all(host(dc, nc), want), what);
    cudaFree(doffs); cudaFree(didx); cudaFree(dval); cudaFree(db); cudaFree(dc); cudaFree(buf);
    cusparseDestroySpMat(A); cusparseDestroyDnMat(Bd); cusparseDestroyDnMat(Cd); cusparseDestroy(h);
  }
}

// A matrix that is not sound keeps the answer the triple loops gave: an out-of-range column refuses (INVALID_VALUE, y untouched).
static void check_unsound() {
  const int64_t rows = 40, cols = 30;
  Csr m = make_csr(rows, cols);
  if (m.idx.size() < 3) m.idx.assign(3, 0), m.offs.assign(rows + 1, 3);
  m.idx[m.idx.size() / 2] = cols + 4;   // out of range
  const auto val = gen<double>(m.idx.size(), Mix::Plain), x = gen<double>(cols, Mix::Plain), y0 = gen<double>(rows, Mix::Plain);
  int *doffs, *didx;
  upload<double, int>(m, false, &doffs, &didx);
  double *dval = dev(val), *dx = dev(x), *dy = dev(y0);
  cusparseHandle_t h;
  cusparseCreate(&h);
  cusparseSpMatDescr_t A;
  cusparseCreateCsr(&A, rows, cols, (int64_t)m.idx.size(), doffs, didx, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);
  cusparseDnVecDescr_t X, Y;
  cusparseCreateDnVec(&X, cols, dx, CUDA_R_64F);
  cusparseCreateDnVec(&Y, rows, dy, CUDA_R_64F);
  const double one = 1.0, zero = 0.0;
  const cusparseStatus_t st = cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, A, X, &zero, Y, CUDA_R_64F, CUSPARSE_SPMV_ALG_DEFAULT, nullptr);
  report(st == CUSPARSE_STATUS_INVALID_VALUE && same_all(host(dy, rows), y0), "SpMV with a column out of range is refused and leaves y alone");
  cudaFree(doffs); cudaFree(didx); cudaFree(dval); cudaFree(dx); cudaFree(dy);
}

template <class T, class I> static void all_cases(const char* name) {
  for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Special})
    for (const bool base1 : {false, true}) {
      for (const cusparseOperation_t op : {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE}) {
        spmv_case<T, I>(name, base1, mix, op, 120, 90);
        if (op == CUSPARSE_OPERATION_NON_TRANSPOSE) spmv_case<T, I>(name, base1, mix, op, 3000, 2500);   // enough rows to be cut into pieces
      }
      for (const cusparseOperation_t opB : {CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE})
        for (const bool brow : {false, true})
          for (const bool crow : {false, true}) {
            spmm_case<T, I>(name, base1, mix, opB, brow, crow, 50, 40, 11, (int)(rng() % 3));
            if (opB == CUSPARSE_OPERATION_NON_TRANSPOSE) spmm_case<T, I>(name, base1, mix, opB, brow, crow, 700, 500, 19, 1);
          }
    }
}

int main() {
  all_cases<float, int>("float/int32");
  all_cases<double, int64_t>("double/int64");
  all_cases<cuComplex, int>("cuComplex/int32");
  all_cases<cuDoubleComplex, int64_t>("cuDoubleComplex/int64");
  check_unsound();
  std::printf("%d checks, %d failed\n", checks, failures);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
