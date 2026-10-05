// The rest of cuSOLVER's dense API, checked by each result's defining property
// so the same program checks NVIDIA's libraries and VirtualGPU's, plus what an
// RTX 3060 (CUDA 13.0) answers where the API leaves room:
//
//   sytrd/hetrd, orgtr, ormtr   Q^H A Q is the tridiagonal (d, e); ormtr is
//                               Q C; a transpose that is not the type's own
//                               is INVALID_VALUE with info -3
//   gebrd, orgbr                Q^H A P is the bidiagonal; m < n NOT_SUPPORTED
//   potri, lauum                A A^-1 = I; U U^H
//   syevdx/heevdx, Xsyevdx      the chosen eigenpairs; W past meig holds the
//                               tridiagonal's diagonal, as NVIDIA's leaves it
//   sygvd, sygvdx, sygvj        the generalized equation for each itype; B
//                               keeps its Cholesky factor; info n + i (sygvj: i)
//                               for a B that is not positive definite
//   Xgetrf/Xgetrs, Xtrtri       64-bit pivots, no pivots, every op; Xtrtri's
//                               unit diagonal written as ones
//   Xgesvd, Xgesvdp, Xgesvdr    A = U S V^H; Xgesvd's m < n INVALID_VALUE;
//                               Xgesvdr exact on a rank-k matrix
//   Xlarft                      I - V T V^H is the reflectors' product
//   Xgeev                       left eigenvectors refused with INTERNAL_ERROR
//   <t1><t2>gesv/gels, IRSX*    refinement converges; an inconsistent least
//                               squares system runs out (-50) and falls back
//   modes, Jacobi getters       defaults and refusals
//
// CUDA 12.0's header has no Xlarft, Xgeev or handle modes: those are looked
// up by name. Tolerances: 1e-4 relative in single precision, 1e-10 in double.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

using cd = std::complex<double>;
static int failures = 0;
static cusolverDnHandle_t h;
static cusolverDnParams_t params;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}
#define CK(x)                                                \
  do {                                                       \
    const int r_ = (int)(x);                                 \
    if (r_ != 0) {                                           \
      std::printf("FAIL %s returned %d\n", #x, r_);         \
      ++failures;                                            \
      return;                                                \
    }                                                        \
  } while (0)

template <class T> T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T) + 64);
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}
static int dev_int(const int* d) { return download(d, 1)[0]; }

template <class T> constexpr bool cx = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;
template <class T> constexpr bool single = std::is_same_v<T, float> || std::is_same_v<T, cuComplex>;
template <class T> using Re = std::conditional_t<single<T>, float, double>;
template <class T> const char* nm() { return std::is_same_v<T, float> ? "S" : std::is_same_v<T, double> ? "D" : std::is_same_v<T, cuComplex> ? "C" : "Z"; }
template <class T> double tol() { return single<T> ? 1e-4 : 1e-10; }
template <class T> cudaDataType dt() {
  return std::is_same_v<T, float> ? CUDA_R_32F : std::is_same_v<T, double> ? CUDA_R_64F : std::is_same_v<T, cuComplex> ? CUDA_C_32F : CUDA_C_64F;
}
static cd get(float v) { return v; }
static cd get(double v) { return v; }
static cd get(cuComplex v) { return {v.x, v.y}; }
static cd get(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> T put(cd v) {
  if constexpr (std::is_same_v<T, float>) return (float)v.real();
  else if constexpr (std::is_same_v<T, double>) return v.real();
  else if constexpr (std::is_same_v<T, cuComplex>) return make_cuComplex((float)v.real(), (float)v.imag());
  else return make_cuDoubleComplex(v.real(), v.imag());
}
// Picks the entry point for T among the four precisions.
template <class T, class S, class D, class C, class Z> auto pick(S s, D d, C c, Z z) {
  if constexpr (std::is_same_v<T, float>) return s;
  else if constexpr (std::is_same_v<T, double>) return d;
  else if constexpr (std::is_same_v<T, cuComplex>) return c;
  else return z;
}

// Host matrices in complex double, column-major, packed (ld = rows).
struct HM {
  int r = 0, c = 0;
  std::vector<cd> v;
  HM(int r_, int c_) : r(r_), c(c_), v((size_t)r_ * c_, 0.0) {}
  cd& operator()(int i, int j) { return v[(size_t)j * r + i]; }
  cd operator()(int i, int j) const { return v[(size_t)j * r + i]; }
};
static HM mul(const HM& a, const HM& b) {
  HM o(a.r, b.c);
  for (int j = 0; j < b.c; ++j)
    for (int k = 0; k < a.c; ++k)
      for (int i = 0; i < a.r; ++i) o(i, j) += a(i, k) * b(k, j);
  return o;
}
static HM herm(const HM& a) {
  HM o(a.c, a.r);
  for (int j = 0; j < a.c; ++j)
    for (int i = 0; i < a.r; ++i) o(j, i) = std::conj(a(i, j));
  return o;
}
static double maxabs(const HM& a) {
  double m = 0;
  for (const cd& x : a.v) m = std::max(m, std::abs(x));
  return m;
}
static double diff(const HM& a, const HM& b) {
  double m = 0;
  for (size_t i = 0; i < a.v.size(); ++i) m = std::max(m, std::abs(a.v[i] - b.v[i]));
  return m;
}
static HM eye(int n) {
  HM o(n, n);
  for (int i = 0; i < n; ++i) o(i, i) = 1;
  return o;
}
// From / to device layouts with a leading dimension.
template <class T> HM from_dev(const std::vector<T>& v, int rows, int cols, int ld) {
  HM o(rows, cols);
  for (int j = 0; j < cols; ++j)
    for (int i = 0; i < rows; ++i) o(i, j) = get(v[(size_t)j * ld + i]);
  return o;
}
template <class T> std::vector<T> to_dev(const HM& a, int ld, cd pad = 77.0) {
  std::vector<T> v((size_t)ld * a.c, put<T>(pad));
  for (int j = 0; j < a.c; ++j)
    for (int i = 0; i < a.r; ++i) v[(size_t)j * ld + i] = put<T>(a(i, j));
  return v;
}
// Test matrices; their entries rounded to T so host and device see the same values.
template <class T> HM general(int m, int n, int seed) {
  HM a(m, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      a(i, j) = get(put<T>(cd(std::sin(1.3 * i + 0.7 * j + seed) + 0.25 * std::cos(i * j + seed),
                               cx<T> ? 0.5 * std::cos(0.9 * i - 1.1 * j + seed) : 0.0)));
  return a;
}
template <class T> HM hermitian(int n, int seed, double shift) {
  HM g = general<T>(n, n, seed), a(n, n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) a(i, j) = 0.5 * (g(i, j) + std::conj(g(j, i))) + (i == j ? shift : 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) a(i, j) = get(put<T>(a(i, j)));
  for (int i = 0; i < n; ++i) a(i, i) = a(i, i).real();
  return a;
}
// The full Hermitian matrix whose `upper` (or lower) triangle a holds.
static HM mirror(const HM& a, bool upper) {
  HM f(a.r, a.c);
  for (int j = 0; j < a.c; ++j)
    for (int i = 0; i < a.r; ++i) f(i, j) = (upper ? i <= j : i >= j) ? a(i, j) : std::conj(a(j, i));
  return f;
}

/* ---- sytrd / orgtr / ormtr ---- */

template <class T> void tridiagonal(bool upper) {
  const int n = 6, lda = 8;
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const HM a0 = hermitian<T>(n, 3, 0.0);
  T* dA = upload(to_dev<T>(a0, lda));
  Re<T>* dd = upload(std::vector<Re<T>>(n));
  Re<T>* de = upload(std::vector<Re<T>>(n));
  T* dtau = upload(std::vector<T>(n));
  int* di = upload(std::vector<int>{-1});
  int lw = 0;
  CK(pick<T>(cusolverDnSsytrd_bufferSize, cusolverDnDsytrd_bufferSize, cusolverDnChetrd_bufferSize,
             cusolverDnZhetrd_bufferSize)(h, uplo, n, dA, lda, dd, de, dtau, &lw));
  T* w = upload(std::vector<T>(std::max(lw, 1)));
  CK(pick<T>(cusolverDnSsytrd, cusolverDnDsytrd, cusolverDnChetrd, cusolverDnZhetrd)(h, uplo, n, dA, lda, dd, de, dtau,
                                                                                       w, lw, di));
  const auto d = download(dd, n), e = download(de, n - 1);
  const auto reduced = download(dA, (size_t)lda * n);
  T* dQ = upload(reduced);
  CK(pick<T>(cusolverDnSorgtr_bufferSize, cusolverDnDorgtr_bufferSize, cusolverDnCungtr_bufferSize,
             cusolverDnZungtr_bufferSize)(h, uplo, n, dQ, lda, dtau, &lw));
  T* w2 = upload(std::vector<T>(std::max(lw, 1)));
  CK(pick<T>(cusolverDnSorgtr, cusolverDnDorgtr, cusolverDnCungtr, cusolverDnZungtr)(h, uplo, n, dQ, lda, dtau, w2, lw,
                                                                                       di));
  const auto qraw = download(dQ, (size_t)lda * n);
  const HM q = from_dev(qraw, n, n, lda);
  HM t(n, n);
  for (int i = 0; i < n; ++i) t(i, i) = d[i];
  for (int i = 0; i + 1 < n; ++i) t(i + 1, i) = t(i, i + 1) = (double)e[i];
  const HM full = mirror(a0, upper);
  const double s = maxabs(full);
  const double err = diff(mul(mul(herm(q), full), q), t) / s;
  const double orth = diff(mul(herm(q), q), eye(n));
  bool pad_zero = true;  // NVIDIA's orgtr writes zeros below Q, through lda
  for (int j = 0; j < n; ++j)
    for (int i = n; i < lda; ++i) pad_zero = pad_zero && get(qraw[(size_t)j * lda + i]) == 0.0;
  char what[160];
  std::snprintf(what, sizeof what, "%s%s %s: Q^H A Q is the tridiagonal (d, e), Q unitary, zeros below it", nm<T>(),
                cx<T> ? "hetrd/ungtr" : "sytrd/orgtr", upper ? "upper" : "lower");
  check(err < 10 * tol<T>() && orth < 10 * tol<T>() && pad_zero, what, std::max(err, orth));
  // ormtr: Q C and C Q^H against the Q orgtr formed.
  const auto op_h = cx<T> ? CUBLAS_OP_C : CUBLAS_OP_T;
  const auto mtr = pick<T>(cusolverDnSormtr, cusolverDnDormtr, cusolverDnCunmtr, cusolverDnZunmtr);
  const auto mtr_bs = pick<T>(cusolverDnSormtr_bufferSize, cusolverDnDormtr_bufferSize, cusolverDnCunmtr_bufferSize,
                              cusolverDnZunmtr_bufferSize);
  {
    const HM c = general<T>(n, 3, 9);
    T* dC = upload(to_dev<T>(c, n + 1));
    CK(mtr_bs(h, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, n, 3, dA, lda, dtau, dC, n + 1, &lw));
    T* w3 = upload(std::vector<T>(std::max(lw, 1)));
    CK(mtr(h, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, n, 3, dA, lda, dtau, dC, n + 1, w3, lw, di));
    const double e1 = diff(from_dev(download(dC, (size_t)(n + 1) * 3), n, 3, n + 1), mul(q, c)) / maxabs(c);
    const HM c2 = general<T>(3, n, 11);
    T* dC2 = upload(to_dev<T>(c2, 3));
    CK(mtr(h, CUBLAS_SIDE_RIGHT, uplo, op_h, 3, n, dA, lda, dtau, dC2, 3, w3, lw, di));
    const double e2 = diff(from_dev(download(dC2, (size_t)3 * n), 3, n, 3), mul(c2, herm(q))) / maxabs(c2);
    std::snprintf(what, sizeof what, "%s%s %s: Q C and C Q^H", nm<T>(), cx<T> ? "unmtr" : "ormtr", upper ? "upper" : "lower");
    check(e1 < 10 * tol<T>() && e2 < 10 * tol<T>(), what, std::max(e1, e2));
    // The transpose a type does not take: INVALID_VALUE, info -3.
    const int bad = mtr(h, CUBLAS_SIDE_LEFT, uplo, cx<T> ? CUBLAS_OP_T : CUBLAS_OP_C, n, 3, dA, lda, dtau, dC, n + 1, w3,
                        lw, di);
    std::snprintf(what, sizeof what, "%s%s refuses op %s: INVALID_VALUE, info -3", nm<T>(), cx<T> ? "unmtr" : "ormtr",
                  cx<T> ? "T" : "C");
    check(bad == CUSOLVER_STATUS_INVALID_VALUE && dev_int(di) == -3, what, bad);
    for (void* p : {(void*)dC, (void*)dC2, (void*)w3}) cudaFree(p);
  }
  for (void* p : {(void*)dA, (void*)dd, (void*)de, (void*)dtau, (void*)di, (void*)w, (void*)dQ, (void*)w2}) cudaFree(p);
}

/* ---- gebrd / orgbr ---- */

template <class T> void bidiagonal() {
  const int m = 7, n = 5, lda = 8;
  const HM a0 = general<T>(m, n, 5);
  T* dA = upload(to_dev<T>(a0, lda));
  Re<T>* dD = upload(std::vector<Re<T>>(n));
  Re<T>* dE = upload(std::vector<Re<T>>(n));
  T* tq = upload(std::vector<T>(n));
  T* tp = upload(std::vector<T>(n));
  int* di = upload(std::vector<int>{-1});
  int lw = 0;
  CK(pick<T>(cusolverDnSgebrd_bufferSize, cusolverDnDgebrd_bufferSize, cusolverDnCgebrd_bufferSize,
             cusolverDnZgebrd_bufferSize)(h, m, n, &lw));
  T* w = upload(std::vector<T>(std::max(lw, 1)));
  const auto brd = pick<T>(cusolverDnSgebrd, cusolverDnDgebrd, cusolverDnCgebrd, cusolverDnZgebrd);
  CK(brd(h, m, n, dA, lda, dD, dE, tq, tp, w, lw, di));
  const auto red = download(dA, (size_t)lda * n);
  const auto d = download(dD, n), e = download(dE, n - 1);
  const auto gbr = pick<T>(cusolverDnSorgbr, cusolverDnDorgbr, cusolverDnCungbr, cusolverDnZungbr);
  const auto gbr_bs = pick<T>(cusolverDnSorgbr_bufferSize, cusolverDnDorgbr_bufferSize, cusolverDnCungbr_bufferSize,
                              cusolverDnZungbr_bufferSize);
  T* dQ = upload(red);
  CK(gbr_bs(h, CUBLAS_SIDE_LEFT, m, n, n, dQ, lda, tq, &lw));
  T* w2 = upload(std::vector<T>(std::max(lw, 1)));
  CK(gbr(h, CUBLAS_SIDE_LEFT, m, n, n, dQ, lda, tq, w2, lw, di));
  T* dP = upload(red);
  CK(gbr(h, CUBLAS_SIDE_RIGHT, n, n, m, dP, lda, tp, w2, lw, di));
  const HM q = from_dev(download(dQ, (size_t)lda * n), m, n, lda);
  const HM ph = from_dev(download(dP, (size_t)lda * n), n, n, lda);
  HM b(n, n);
  for (int i = 0; i < n; ++i) b(i, i) = d[i];
  for (int i = 0; i + 1 < n; ++i) b(i, i + 1) = e[i];
  const double err = diff(mul(mul(herm(q), a0), herm(ph)), b) / maxabs(a0);
  char what[160];
  std::snprintf(what, sizeof what, "%sgebrd + %s: Q^H A P is the upper bidiagonal (d, e)", nm<T>(), cx<T> ? "ungbr" : "orgbr");
  check(err < 10 * tol<T>(), what, err);
  const int st = brd(h, n, m, dA, lda, dD, dE, tq, tp, w, lw, di);
  std::snprintf(what, sizeof what, "%sgebrd with m < n: NOT_SUPPORTED, as NVIDIA's", nm<T>());
  check(st == CUSOLVER_STATUS_NOT_SUPPORTED, what, st);
  for (void* p : {(void*)dA, (void*)dD, (void*)dE, (void*)tq, (void*)tp, (void*)di, (void*)w, (void*)dQ, (void*)w2, (void*)dP})
    cudaFree(p);
}

/* ---- potri, lauum ---- */

template <class T> void inverse(bool upper) {
  const int n = 6, lda = 7;
  const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  const HM a0 = hermitian<T>(n, 7, n + 2.0);
  T* dA = upload(to_dev<T>(a0, lda));
  int* di = upload(std::vector<int>{-1});
  int lw = 0;
  CK(pick<T>(cusolverDnSpotrf_bufferSize, cusolverDnDpotrf_bufferSize, cusolverDnCpotrf_bufferSize,
             cusolverDnZpotrf_bufferSize)(h, uplo, n, dA, lda, &lw));
  T* w = upload(std::vector<T>(lw + 64));
  CK(pick<T>(cusolverDnSpotrf, cusolverDnDpotrf, cusolverDnCpotrf, cusolverDnZpotrf)(h, uplo, n, dA, lda, w, lw, di));
  const auto factor = download(dA, (size_t)lda * n);
  T* dB = upload(factor);
  CK(pick<T>(cusolverDnSpotri_bufferSize, cusolverDnDpotri_bufferSize, cusolverDnCpotri_bufferSize,
             cusolverDnZpotri_bufferSize)(h, uplo, n, dA, lda, &lw));
  T* w2 = upload(std::vector<T>(std::max(lw, 1)));
  CK(pick<T>(cusolverDnSpotri, cusolverDnDpotri, cusolverDnCpotri, cusolverDnZpotri)(h, uplo, n, dA, lda, w2, lw, di));
  const HM inv = mirror(from_dev(download(dA, (size_t)lda * n), n, n, lda), upper);
  const double err = diff(mul(a0, inv), eye(n));
  CK(pick<T>(cusolverDnSlauum, cusolverDnDlauum, cusolverDnClauum, cusolverDnZlauum)(h, uplo, n, dB, lda, w2, lw, di));
  HM f(n, n);  // the factor's triangle
  const HM fr = from_dev(factor, n, n, lda);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i)
      if (upper ? i <= j : i >= j) f(i, j) = fr(i, j);
  const HM want = upper ? mul(f, herm(f)) : mul(herm(f), f);
  const HM got = mirror(from_dev(download(dB, (size_t)lda * n), n, n, lda), upper);
  const double err2 = diff(got, want) / maxabs(want);
  char what[160];
  std::snprintf(what, sizeof what, "%spotri %s: A A^-1 = I; lauum: %s", nm<T>(), upper ? "upper" : "lower",
                upper ? "U U^H" : "L^H L");
  check(err < 100 * tol<T>() && err2 < 10 * tol<T>(), what, std::max(err, err2));
  for (void* p : {(void*)dA, (void*)di, (void*)w, (void*)dB, (void*)w2}) cudaFree(p);
}

/* ---- syevdx, sygvd and friends ---- */

// Max over the chosen columns of |A x - w x| / ||A||.
static double eig_err(const HM& a, const HM& x, const std::vector<double>& w, int count, const HM* b = nullptr, int itype = 1) {
  double e = 0;
  for (int j = 0; j < count; ++j) {
    HM v(a.r, 1);
    for (int i = 0; i < a.r; ++i) v(i, 0) = x(i, j);
    HM lhs = itype == 1 ? mul(a, v) : itype == 2 ? mul(a, mul(*b, v)) : mul(*b, mul(a, v));
    HM rhs = itype == 1 && b ? mul(*b, v) : v;
    for (int i = 0; i < a.r; ++i) e = std::max(e, std::abs(lhs(i, 0) - w[j] * rhs(i, 0)));
  }
  return e / std::max(maxabs(a), 1e-30);
}

template <class T> void eig_range() {
  const int n = 7, lda = 8;
  const HM a0 = hermitian<T>(n, 11, 0.0);
  const auto ev = pick<T>(cusolverDnSsyevdx, cusolverDnDsyevdx, cusolverDnCheevdx, cusolverDnZheevdx);
  const auto ev_bs = pick<T>(cusolverDnSsyevdx_bufferSize, cusolverDnDsyevdx_bufferSize, cusolverDnCheevdx_bufferSize,
                             cusolverDnZheevdx_bufferSize);
  // The tridiagonal's diagonal: what NVIDIA's leaves in W past meig.
  std::vector<double> tri_d(n);
  {
    T* dA = upload(to_dev<T>(a0, lda));
    Re<T>* dd = upload(std::vector<Re<T>>(n));
    Re<T>* de = upload(std::vector<Re<T>>(n));
    T* dt = upload(std::vector<T>(n));
    int* di = upload(std::vector<int>{0});
    T* w = upload(std::vector<T>(4096));
    CK(pick<T>(cusolverDnSsytrd, cusolverDnDsytrd, cusolverDnChetrd, cusolverDnZhetrd)(h, CUBLAS_FILL_MODE_LOWER, n, dA,
                                                                                         lda, dd, de, dt, w, 4096, di));
    const auto d = download(dd, n);
    for (int i = 0; i < n; ++i) tri_d[i] = d[i];
    for (void* p : {(void*)dA, (void*)dd, (void*)de, (void*)dt, (void*)di, (void*)w}) cudaFree(p);
  }
  for (int which = 0; which < 3; ++which) {
    const bool upper = which == 1;
    const auto range = which == 0 ? CUSOLVER_EIG_RANGE_I : which == 1 ? CUSOLVER_EIG_RANGE_I : CUSOLVER_EIG_RANGE_V;
    const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    T* dA = upload(to_dev<T>(a0, lda));
    Re<T>* dW = upload(std::vector<Re<T>>(n, (Re<T>)-99));
    int* di = upload(std::vector<int>{-1});
    const Re<T> vl = -0.5, vu = 1.5;
    int meig = -1, lw = 0;
    CK(ev_bs(h, CUSOLVER_EIG_MODE_VECTOR, range, uplo, n, dA, lda, vl, vu, 2, 5, &meig, dW, &lw));
    T* w = upload(std::vector<T>(std::max(lw, 1)));
    CK(ev(h, CUSOLVER_EIG_MODE_VECTOR, range, uplo, n, dA, lda, vl, vu, 2, 5, &meig, dW, w, lw, di));
    const auto wv = download(dW, n);
    std::vector<double> wd(wv.begin(), wv.end());
    const HM x = from_dev(download(dA, (size_t)lda * n), n, n, lda);
    double err = eig_err(a0, x, wd, std::max(meig, 0));
    bool sel = range == CUSOLVER_EIG_RANGE_I ? meig == 4 : meig >= 1;
    for (int j = 0; j < meig; ++j) {
      if (range == CUSOLVER_EIG_RANGE_V) sel = sel && wd[j] > vl - 1e-6 && wd[j] <= vu + 1e-6;
      if (j) sel = sel && wd[j] >= wd[j - 1];
    }
    double past = 0;
    for (int j = std::max(meig, 0); j < n; ++j) past = std::max(past, std::abs(wd[j] - tri_d[j]));
    char what[200];
    std::snprintf(what, sizeof what, "%s%s %s, %s: the chosen eigenpairs (meig %d); W past meig is the tridiagonal's diagonal",
                  nm<T>(), cx<T> ? "heevdx" : "syevdx", range == CUSOLVER_EIG_RANGE_I ? "il 2..iu 5" : "(vl, vu]",
                  upper ? "upper" : "lower", meig);
    check(sel && err < 10 * tol<T>() && past < 10 * tol<T>(), what, std::max(err, past));
    for (void* p : {(void*)dA, (void*)dW, (void*)di, (void*)w}) cudaFree(p);
  }
}

template <class T> void generalized() {
  const int n = 6, lda = 7;
  const HM a0 = hermitian<T>(n, 13, 0.0), b0 = hermitian<T>(n, 17, n + 1.5);
  const auto gvd = pick<T>(cusolverDnSsygvd, cusolverDnDsygvd, cusolverDnChegvd, cusolverDnZhegvd);
  const auto gvd_bs = pick<T>(cusolverDnSsygvd_bufferSize, cusolverDnDsygvd_bufferSize, cusolverDnChegvd_bufferSize,
                              cusolverDnZhegvd_bufferSize);
  for (int itype = 1; itype <= 3; ++itype)
    for (bool upper : {false, true}) {
      const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
      T* dA = upload(to_dev<T>(a0, lda));
      T* dB = upload(to_dev<T>(b0, lda));
      Re<T>* dW = upload(std::vector<Re<T>>(n));
      int* di = upload(std::vector<int>{-1});
      int lw = 0;
      CK(gvd_bs(h, (cusolverEigType_t)itype, CUSOLVER_EIG_MODE_VECTOR, uplo, n, dA, lda, dB, lda, dW, &lw));
      T* w = upload(std::vector<T>(std::max(lw, 1)));
      CK(gvd(h, (cusolverEigType_t)itype, CUSOLVER_EIG_MODE_VECTOR, uplo, n, dA, lda, dB, lda, dW, w, lw, di));
      const auto wv = download(dW, n);
      std::vector<double> wd(wv.begin(), wv.end());
      const HM x = from_dev(download(dA, (size_t)lda * n), n, n, lda);
      const double err = eig_err(a0, x, wd, n, &b0, itype);
      // B's stored triangle is its Cholesky factor.
      const HM bf = from_dev(download(dB, (size_t)lda * n), n, n, lda);
      HM f(n, n);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          if (upper ? i <= j : i >= j) f(i, j) = bf(i, j);
      const double ferr = diff(upper ? mul(herm(f), f) : mul(f, herm(f)), b0) / maxabs(b0);
      char what[160];
      std::snprintf(what, sizeof what, "%s%s itype %d %s: the generalized eigenpairs, B its Cholesky factor", nm<T>(),
                    cx<T> ? "hegvd" : "sygvd", itype, upper ? "upper" : "lower");
      check(dev_int(di) == 0 && err < 100 * tol<T>() && ferr < 10 * tol<T>(), what, std::max(err, ferr));
      for (void* p : {(void*)dA, (void*)dB, (void*)dW, (void*)di, (void*)w}) cudaFree(p);
    }
  // sygvdx, a range; sygvj with its getters; B not positive definite.
  {
    T* dA = upload(to_dev<T>(a0, lda));
    T* dB = upload(to_dev<T>(b0, lda));
    Re<T>* dW = upload(std::vector<Re<T>>(n));
    int* di = upload(std::vector<int>{-1});
    T* w = upload(std::vector<T>(8192));
    int meig = -1;
    CK(pick<T>(cusolverDnSsygvdx, cusolverDnDsygvdx, cusolverDnChegvdx, cusolverDnZhegvdx)(
        h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n, dA, lda, dB, lda,
        (Re<T>)0, (Re<T>)0, 2, 4, &meig, dW, w, 8192, di));
    const auto wv = download(dW, n);
    std::vector<double> wd(wv.begin(), wv.end());
    const double err = eig_err(a0, from_dev(download(dA, (size_t)lda * n), n, n, lda), wd, meig, &b0, 1);
    char what[160];
    std::snprintf(what, sizeof what, "%s%s il 2..iu 4: three generalized eigenpairs", nm<T>(), cx<T> ? "hegvdx" : "sygvdx");
    check(meig == 3 && err < 100 * tol<T>(), what, err);
    cudaMemcpy(dA, to_dev<T>(a0, lda).data(), sizeof(T) * lda * n, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, to_dev<T>(b0, lda).data(), sizeof(T) * lda * n, cudaMemcpyHostToDevice);
    syevjInfo_t sj;
    CK(cusolverDnCreateSyevjInfo(&sj));
    CK(pick<T>(cusolverDnSsygvj, cusolverDnDsygvj, cusolverDnChegvj, cusolverDnZhegvj)(
        h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, dA, lda, dB, lda, dW, w, 8192, di, sj));
    const auto wj = download(dW, n);
    std::vector<double> wjd(wj.begin(), wj.end());
    const double errj = eig_err(a0, from_dev(download(dA, (size_t)lda * n), n, n, lda), wjd, n, &b0, 1);
    double res = -1;
    int sweeps = -1;
    const int g1 = cusolverDnXsyevjGetResidual(h, sj, &res), g2 = cusolverDnXsyevjGetSweeps(h, sj, &sweeps);
    std::snprintf(what, sizeof what, "%s%s: the generalized eigenpairs; its residual and sweeps reported", nm<T>(),
                  cx<T> ? "hegvj" : "sygvj");
    check(errj < 100 * tol<T>() && g1 == 0 && g2 == 0 && sweeps > 0 && res >= 0 && res < 1e-3, what, errj);
    // B with a negative second pivot: info n + 2 from sygvd, 2 from sygvj.
    HM bad = b0;
    bad(1, 1) = -50.0;
    cudaMemcpy(dB, to_dev<T>(bad, lda).data(), sizeof(T) * lda * n, cudaMemcpyHostToDevice);
    CK(gvd(h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, lda, dB, lda, dW, w, 8192, di));
    const int info_d = dev_int(di);
    cudaMemcpy(dB, to_dev<T>(bad, lda).data(), sizeof(T) * lda * n, cudaMemcpyHostToDevice);
    CK(pick<T>(cusolverDnSsygvj, cusolverDnDsygvj, cusolverDnChegvj, cusolverDnZhegvj)(
        h, CUSOLVER_EIG_TYPE_1, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, lda, dB, lda, dW, w, 8192, di, sj));
    const int info_j = dev_int(di);
    std::snprintf(what, sizeof what, "%s: B not positive definite at 2 -> sygvd info %d (n + 2), sygvj info %d (2)",
                  nm<T>(), info_d, info_j);
    check(info_d == n + 2 && info_j == 2, what, 0);
    cusolverDnDestroySyevjInfo(sj);
    for (void* p : {(void*)dA, (void*)dB, (void*)dW, (void*)di, (void*)w}) cudaFree(p);
  }
}

/* ---- the 64-bit API ---- */

template <class T> void x_lu() {
  const int n = 6, lda = 8, nrhs = 2;
  const HM a0 = general<T>(n, n, 19);
  for (bool piv : {true, false}) {
    HM a = a0;
    if (!piv)
      for (int i = 0; i < n; ++i) a(i, i) += 4.0;  // safe without pivoting
    T* dA = upload(to_dev<T>(a, lda));
    int64_t* dp = upload(std::vector<int64_t>(n, -1));
    int* di = upload(std::vector<int>{-1});
    size_t dw = 0, hw = 0;
    CK(cusolverDnXgetrf_bufferSize(h, params, n, n, dt<T>(), dA, lda, dt<T>(), &dw, &hw));
    void* w = nullptr;
    cudaMalloc(&w, dw + 16);
    std::vector<char> hwk(hw + 16);
    CK(cusolverDnXgetrf(h, params, n, n, dt<T>(), dA, lda, piv ? dp : nullptr, dt<T>(), w, dw, hwk.data(), hw, di));
    double err = 0;
    for (auto op : {CUBLAS_OP_N, CUBLAS_OP_T, CUBLAS_OP_C}) {
      const HM b = general<T>(n, nrhs, 23);
      T* dB = upload(to_dev<T>(b, n + 1));
      CK(cusolverDnXgetrs(h, params, op, n, nrhs, dt<T>(), dA, lda, piv ? dp : nullptr, dt<T>(), dB, n + 1, di));
      const HM x = from_dev(download(dB, (size_t)(n + 1) * nrhs), n, nrhs, n + 1);
      HM opa = op == CUBLAS_OP_N ? a : herm(a);
      if (op == CUBLAS_OP_T)
        for (cd& v : opa.v) v = std::conj(v);
      err = std::max(err, diff(mul(opa, x), b) / maxabs(b));
      cudaFree(dB);
    }
    bool pivots_ok = true;
    if (piv)
      for (int64_t p : download(dp, n)) pivots_ok = pivots_ok && p >= 1 && p <= n;
    char what[160];
    std::snprintf(what, sizeof what, "%sXgetrf/Xgetrs %s: op(A) X = B for N, T and C", nm<T>(),
                  piv ? "with 64-bit pivots" : "without pivoting");
    check(dev_int(di) == 0 && pivots_ok && err < 100 * tol<T>(), what, err);
    for (void* p : {(void*)dA, (void*)dp, (void*)di, w}) cudaFree(p);
  }
  // A short leading dimension: info -4 (LAPACK's numbering), with SUCCESS when
  // pivoting and INVALID_VALUE without, as NVIDIA's answers.
  T* dA = upload(to_dev<T>(a0, lda));
  int* di = upload(std::vector<int>{-1});
  int64_t* dp = upload(std::vector<int64_t>(n));
  void* w = nullptr;
  cudaMalloc(&w, 1 << 20);
  std::vector<char> hwk(1 << 16);
  const int st = cusolverDnXgetrf(h, params, n, n, dt<T>(), dA, n - 1, dp, dt<T>(), w, 1 << 20, hwk.data(), 1 << 16, di);
  const int i1 = dev_int(di);
  const int st2 = cusolverDnXgetrf(h, params, n, n, dt<T>(), dA, n - 1, nullptr, dt<T>(), w, 1 << 20, hwk.data(), 1 << 16, di);
  char what[160];
  std::snprintf(what, sizeof what, "%sXgetrf lda < m: info -4; SUCCESS with pivots, INVALID_VALUE without", nm<T>());
  check(st == 0 && i1 == -4 && st2 == CUSOLVER_STATUS_INVALID_VALUE && dev_int(di) == -4, what, st);
  for (void* p : {(void*)dA, (void*)di, (void*)dp, w}) cudaFree(p);
}

template <class T> void x_trtri() {
  const int n = 6, lda = 7;
  for (int variant = 0; variant < 2; ++variant) {
    const bool upper = variant == 1, unit = variant == 1;
    HM a = general<T>(n, n, 29);
    for (int i = 0; i < n; ++i) a(i, i) += 3.0;
    T* dA = upload(to_dev<T>(a, lda));
    int* di = upload(std::vector<int>{-1});
    size_t dw = 0, hw = 0;
    const auto uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    const auto diag = unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
    CK(cusolverDnXtrtri_bufferSize(h, uplo, diag, n, dt<T>(), dA, lda, &dw, &hw));
    void* w = nullptr;
    cudaMalloc(&w, dw + 16);
    std::vector<char> hwk(hw + 16);
    CK(cusolverDnXtrtri(h, uplo, diag, n, dt<T>(), dA, lda, w, dw, hwk.data(), hw, di));
    HM t(n, n), ti(n, n);
    const HM got = from_dev(download(dA, (size_t)lda * n), n, n, lda);
    bool ones = true;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        if (upper ? i <= j : i >= j) {
          t(i, j) = i == j && unit ? 1.0 : a(i, j);
          ti(i, j) = got(i, j);
          if (i == j && unit) ones = ones && got(i, j) == 1.0;
        }
    const double err = diff(mul(t, ti), eye(n));
    char what[160];
    std::snprintf(what, sizeof what, "%sXtrtri %s%s: T T^-1 = I%s", nm<T>(), upper ? "upper" : "lower",
                  unit ? " unit" : "", unit ? ", the unit diagonal written as ones" : "");
    check(dev_int(di) == 0 && ones && err < 100 * tol<T>(), what, err);
    for (void* p : {(void*)dA, (void*)di, w}) cudaFree(p);
  }
}

template <class T> void x_syevdx() {
  const int n = 7, lda = 7;
  const HM a0 = hermitian<T>(n, 31, 0.0);
  T* dA = upload(to_dev<T>(a0, lda));
  Re<T>* dW = upload(std::vector<Re<T>>(n));
  int* di = upload(std::vector<int>{-1});
  Re<T> vl = 0, vu = 0;
  int64_t meig = -1;
  const cudaDataType tw = single<T> ? CUDA_R_32F : CUDA_R_64F;
  size_t dw = 0, hw = 0;
  CK(cusolverDnXsyevdx_bufferSize(h, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n,
                                  dt<T>(), dA, lda, &vl, &vu, 3, 5, &meig, tw, dW, dt<T>(), &dw, &hw));
  void* w = nullptr;
  cudaMalloc(&w, dw + 16);
  std::vector<char> hwk(hw + 16);
  CK(cusolverDnXsyevdx(h, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n, dt<T>(), dA,
                       lda, &vl, &vu, 3, 5, &meig, tw, dW, dt<T>(), w, dw, hwk.data(), hw, di));
  const auto wv = download(dW, n);
  std::vector<double> wd(wv.begin(), wv.end());
  const double err = eig_err(a0, from_dev(download(dA, (size_t)lda * n), n, n, lda), wd, (int)meig);
  char what[120];
  std::snprintf(what, sizeof what, "%sXsyevdx il 3..iu 5: three eigenpairs", nm<T>());
  check(meig == 3 && err < 10 * tol<T>(), what, err);
  // A bad il/iu: INVALID_VALUE, info -12 (iu), numbering params as 1.
  const int st = cusolverDnXsyevdx(h, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_RANGE_I, CUBLAS_FILL_MODE_LOWER, n,
                                   dt<T>(), dA, lda, &vl, &vu, 3, 2, &meig, tw, dW, dt<T>(), w, dw, hwk.data(), hw, di);
  std::snprintf(what, sizeof what, "%sXsyevdx iu < il: INVALID_VALUE, info -12", nm<T>());
  check(st == CUSOLVER_STATUS_INVALID_VALUE && dev_int(di) == -12, what, st);
  for (void* p : {(void*)dA, (void*)dW, (void*)di, w}) cudaFree(p);
}

// A = U S V^H to tolerance, the vectors orthonormal.
static double svd_err(const HM& a, const std::vector<double>& s, const HM& u, const HM& v, int k) {
  HM us(a.r, k), vk(a.c, k);
  for (int j = 0; j < k; ++j) {
    for (int i = 0; i < a.r; ++i) us(i, j) = u(i, j) * s[j];
    for (int i = 0; i < a.c; ++i) vk(i, j) = v(i, j);
  }
  HM uk(a.r, k);
  for (int j = 0; j < k; ++j)
    for (int i = 0; i < a.r; ++i) uk(i, j) = u(i, j);
  return std::max(diff(mul(us, herm(vk)), a) / maxabs(a),
                  std::max(diff(mul(herm(uk), uk), eye(k)), diff(mul(herm(vk), vk), eye(k))));
}

template <class T> void x_svd() {
  const int m = 7, n = 5, k = n;
  const HM a0 = general<T>(m, n, 37);
  const cudaDataType tr = single<T> ? CUDA_R_32F : CUDA_R_64F;
  for (signed char job : {'A', 'S', 'O'}) {
    const int lda = m + 1, ldu = m + 1, ldvt = n + 1;
    T* dA = upload(to_dev<T>(a0, lda));
    Re<T>* dS = upload(std::vector<Re<T>>(k));
    T* dU = upload(std::vector<T>((size_t)ldu * m));
    T* dVT = upload(std::vector<T>((size_t)ldvt * n));
    int* di = upload(std::vector<int>{-1});
    const signed char ju = job, jv = job == 'O' ? 'S' : job;
    size_t dw = 0, hw = 0;
    CK(cusolverDnXgesvd_bufferSize(h, params, ju, jv, m, n, dt<T>(), dA, lda, tr, dS, dt<T>(), dU, ldu, dt<T>(), dVT,
                                   ldvt, dt<T>(), &dw, &hw));
    void* w = nullptr;
    cudaMalloc(&w, dw + 16);
    std::vector<char> hwk(hw + 16);
    CK(cusolverDnXgesvd(h, params, ju, jv, m, n, dt<T>(), dA, lda, tr, dS, dt<T>(), dU, ldu, dt<T>(), dVT, ldvt, dt<T>(),
                        w, dw, hwk.data(), hw, di));
    const auto sv = download(dS, k);
    std::vector<double> s(sv.begin(), sv.end());
    const HM u = job == 'O' ? from_dev(download(dA, (size_t)lda * n), m, k, lda)
                            : from_dev(download(dU, (size_t)ldu * m), m, job == 'A' ? m : k, ldu);
    const HM vt = from_dev(download(dVT, (size_t)ldvt * n), job == 'A' ? n : k, n, ldvt);
    const double err = svd_err(a0, s, u, herm(vt), k);
    bool desc = true;
    for (int i = 1; i < k; ++i) desc = desc && s[i] <= s[i - 1];
    char what[120];
    std::snprintf(what, sizeof what, "%sXgesvd jobu %c: A = U S V^H, S descending", nm<T>(), (char)ju);
    check(dev_int(di) == 0 && desc && err < 100 * tol<T>(), what, err);
    if (job == 'A') {
      const int st = cusolverDnXgesvd(h, params, 'N', 'N', n, m, dt<T>(), dA, lda, tr, dS, dt<T>(), dU, ldu, dt<T>(),
                                      dVT, ldvt, dt<T>(), w, dw, hwk.data(), hw, di);
      std::snprintf(what, sizeof what, "%sXgesvd m < n: INVALID_VALUE, as NVIDIA's", nm<T>());
      check(st == CUSOLVER_STATUS_INVALID_VALUE, what, st);
    }
    for (void* p : {(void*)dA, (void*)dS, (void*)dU, (void*)dVT, (void*)di, w}) cudaFree(p);
  }
  for (int econ : {0, 1}) {  // Xgesvdp: V itself, not V^H
    const int lda = m, ldu = m, ldv = n;
    T* dA = upload(to_dev<T>(a0, lda));
    Re<T>* dS = upload(std::vector<Re<T>>(k));
    T* dU = upload(std::vector<T>((size_t)ldu * m));
    T* dV = upload(std::vector<T>((size_t)ldv * n));
    int* di = upload(std::vector<int>{-1});
    size_t dw = 0, hw = 0;
    CK(cusolverDnXgesvdp_bufferSize(h, params, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, dt<T>(), dA, lda, tr, dS, dt<T>(),
                                    dU, ldu, dt<T>(), dV, ldv, dt<T>(), &dw, &hw));
    void* w = nullptr;
    cudaMalloc(&w, dw + 16);
    std::vector<char> hwk(hw + 16);
    double err_sigma = -1;
    CK(cusolverDnXgesvdp(h, params, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, dt<T>(), dA, lda, tr, dS, dt<T>(), dU, ldu,
                         dt<T>(), dV, ldv, dt<T>(), w, dw, hwk.data(), hw, di, &err_sigma));
    const auto sv = download(dS, k);
    std::vector<double> s(sv.begin(), sv.end());
    const double err = svd_err(a0, s, from_dev(download(dU, (size_t)ldu * m), m, econ ? k : m, ldu),
                               from_dev(download(dV, (size_t)ldv * n), n, n, ldv), k);
    char what[120];
    std::snprintf(what, sizeof what, "%sXgesvdp econ %d: A = U S V^H", nm<T>(), econ);
    check(dev_int(di) == 0 && err < 100 * tol<T>() && err_sigma >= 0, what, err);
    for (void* p : {(void*)dA, (void*)dS, (void*)dU, (void*)dV, (void*)di, w}) cudaFree(p);
  }
  {  // Xgesvdr on a matrix of exact rank 3: its three singular triplets.
    const int mm = 24, nn = 16, kk = 3, p = 5, lda = mm;
    HM a(mm, nn);
    const double sig[3] = {9.0, 4.0, 1.5};
    for (int r = 0; r < kk; ++r) {
      HM x(mm, 1), y(nn, 1);
      for (int i = 0; i < mm; ++i) x(i, 0) = cd(std::sin(1.7 * i * (r + 1) + r), cx<T> ? 0.4 * std::cos(i - r) : 0.0);
      for (int i = 0; i < nn; ++i) y(i, 0) = std::cos(0.9 * i * (r + 1) + 2 * r);
      // Gram-Schmidt against the earlier ones, so sig are the singular values.
      static std::vector<HM> xs, ys;
      if (r == 0) xs.clear(), ys.clear();
      for (const HM& q : xs) { cd d = 0; for (int i = 0; i < mm; ++i) d += std::conj(q(i, 0)) * x(i, 0); for (int i = 0; i < mm; ++i) x(i, 0) -= d * q(i, 0); }
      for (const HM& q : ys) { cd d = 0; for (int i = 0; i < nn; ++i) d += std::conj(q(i, 0)) * y(i, 0); for (int i = 0; i < nn; ++i) y(i, 0) -= d * q(i, 0); }
      double nx = 0, ny = 0;
      for (const cd& v : x.v) nx += std::norm(v);
      for (const cd& v : y.v) ny += std::norm(v);
      for (cd& v : x.v) v /= std::sqrt(nx);
      for (cd& v : y.v) v /= std::sqrt(ny);
      xs.push_back(x);
      ys.push_back(y);
      for (int j = 0; j < nn; ++j)
        for (int i = 0; i < mm; ++i) a(i, j) += sig[r] * x(i, 0) * std::conj(y(j, 0));
    }
    for (cd& v : a.v) v = get(put<T>(v));
    T* dA = upload(to_dev<T>(a, lda));
    Re<T>* dS = upload(std::vector<Re<T>>(nn, (Re<T>)-1));
    T* dU = upload(std::vector<T>((size_t)mm * kk));
    T* dV = upload(std::vector<T>((size_t)nn * kk));
    int* di = upload(std::vector<int>{-1});
    size_t dw = 0, hw = 0;
    CK(cusolverDnXgesvdr_bufferSize(h, params, 'S', 'S', mm, nn, kk, p, 2, dt<T>(), dA, lda, tr, dS, dt<T>(), dU, mm,
                                    dt<T>(), dV, nn, dt<T>(), &dw, &hw));
    void* w = nullptr;
    cudaMalloc(&w, dw + 16);
    std::vector<char> hwk(hw + 16);
    CK(cusolverDnXgesvdr(h, params, 'S', 'S', mm, nn, kk, p, 2, dt<T>(), dA, lda, tr, dS, dt<T>(), dU, mm, dt<T>(), dV,
                         nn, dt<T>(), w, dw, hwk.data(), hw, di));
    const auto sv = download(dS, kk);
    std::vector<double> s(sv.begin(), sv.end());
    double serr = 0;
    for (int i = 0; i < kk; ++i) serr = std::max(serr, std::abs(s[i] - sig[i]) / sig[0]);
    const double err = svd_err(a, s, from_dev(download(dU, (size_t)mm * kk), mm, kk, mm),
                               from_dev(download(dV, (size_t)nn * kk), nn, kk, nn), kk);
    char what[120];
    std::snprintf(what, sizeof what, "%sXgesvdr k = 3 on a rank-3 matrix: its singular triplets", nm<T>());
    check(dev_int(di) == 0 && serr < 100 * tol<T>() && err < 100 * tol<T>(), what, std::max(serr, err));
    for (void* p2 : {(void*)dA, (void*)dS, (void*)dU, (void*)dV, (void*)di, w}) cudaFree(p2);
  }
}

using XlarftFn = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int, int, int64_t, int64_t, cudaDataType,
                                      const void*, int64_t, cudaDataType, const void*, cudaDataType, void*, int64_t,
                                      cudaDataType, void*, size_t, void*, size_t);

template <class T> void x_larft() {
  auto larft = reinterpret_cast<XlarftFn>(dlsym(RTLD_DEFAULT, "cusolverDnXlarft"));
  if (!larft) {
    std::printf("FAIL the cuSOLVER this runs against has no cusolverDnXlarft\n");
    ++failures;
    return;
  }
  // Forward: the reflectors geqrf leaves; H(1)...H(k) is the Q orgqr forms.
  const int n = 7, k = 4, ldv = 8, ldt = 5;
  const HM a0 = general<T>(n, k, 43);
  T* dA = upload(to_dev<T>(a0, ldv));
  T* dtau = upload(std::vector<T>(k));
  int* di = upload(std::vector<int>{-1});
  T* w = upload(std::vector<T>(8192));
  CK(pick<T>(cusolverDnSgeqrf, cusolverDnDgeqrf, cusolverDnCgeqrf, cusolverDnZgeqrf)(h, n, k, dA, ldv, dtau, w, 8192, di));
  T* dT = upload(std::vector<T>((size_t)ldt * k, put<T>(9.0)));
  auto larft_bs = reinterpret_cast<cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, int, int, int64_t,
                                                         int64_t, cudaDataType, const void*, int64_t, cudaDataType,
                                                         const void*, cudaDataType, void*, int64_t, cudaDataType,
                                                         size_t*, size_t*)>(dlsym(RTLD_DEFAULT, "cusolverDnXlarft_bufferSize"));
  size_t dw = 0, hw = 0;
  if (larft_bs) CK(larft_bs(h, params, 0, 0, n, k, dt<T>(), dA, ldv, dt<T>(), dtau, dt<T>(), dT, ldt, dt<T>(), &dw, &hw));
  void* dwork = nullptr;
  cudaMalloc(&dwork, dw + 256);
  std::vector<char> hwork(hw + 256);
  CK(larft(h, params, 0 /* forward */, 0 /* column-wise */, n, k, dt<T>(), dA, ldv, dt<T>(), dtau, dt<T>(), dT, ldt,
           dt<T>(), dwork, dw + 256, hwork.data(), hw + 256));
  const HM fac = from_dev(download(dA, (size_t)ldv * k), n, k, ldv);
  HM v(n, k);
  for (int j = 0; j < k; ++j)
    for (int i = j; i < n; ++i) v(i, j) = i == j ? cd(1.0) : fac(i, j);
  const HM t = from_dev(download(dT, (size_t)ldt * k), k, k, ldt);
  bool tri = true;  // the strict lower part written as zeros, as NVIDIA's writes it
  for (int j = 0; j < k; ++j)
    for (int i = j + 1; i < k; ++i) tri = tri && t(i, j) == 0.0;
  HM q = eye(n);
  const HM vt = mul(mul(v, t), herm(v));
  for (size_t i = 0; i < q.v.size(); ++i) q.v[i] -= vt.v[i];
  // Q from orgqr, all n columns.
  std::vector<T> full((size_t)ldv * n, put<T>(0.0));
  const auto facv = download(dA, (size_t)ldv * k);
  std::copy(facv.begin(), facv.end(), full.begin());
  T* dQ = upload(full);
  CK(pick<T>(cusolverDnSorgqr, cusolverDnDorgqr, cusolverDnCungqr, cusolverDnZungqr)(h, n, n, k, dQ, ldv, dtau, w, 8192, di));
  const double err = diff(q, from_dev(download(dQ, (size_t)ldv * n), n, n, ldv));
  char what[120];
  std::snprintf(what, sizeof what, "%sXlarft forward: I - V T V^H is the reflectors' product, T upper", nm<T>());
  check(tri && err < 100 * tol<T>(), what, err);
  const int st = larft(h, params, 0, 1 /* row-wise */, n, k, dt<T>(), dA, ldv, dt<T>(), dtau, dt<T>(), dT, ldt, dt<T>(),
                       dwork, dw + 256, hwork.data(), hw + 256);
  std::snprintf(what, sizeof what, "%sXlarft row-wise: INVALID_VALUE (column-wise only)", nm<T>());
  check(st == CUSOLVER_STATUS_INVALID_VALUE, what, st);
  for (void* p : {(void*)dA, (void*)dtau, (void*)di, (void*)w, (void*)dT, (void*)dQ, dwork}) cudaFree(p);
}

using XgeevBsFn = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t,
                                       int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*,
                                       cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t,
                                       cudaDataType, size_t*, size_t*);

static void geev_left() {
  auto bs = reinterpret_cast<XgeevBsFn>(dlsym(RTLD_DEFAULT, "cusolverDnXgeev_bufferSize"));
  if (!bs) {
    std::printf("FAIL the cuSOLVER this runs against has no cusolverDnXgeev\n");
    ++failures;
    return;
  }
  double* dA = upload(std::vector<double>(64, 1.0));
  size_t dw = 0, hw = 0;
  const int left = bs(h, params, CUSOLVER_EIG_MODE_VECTOR, CUSOLVER_EIG_MODE_VECTOR, 4, CUDA_R_64F, dA, 4, CUDA_R_64F,
                      dA, CUDA_R_64F, dA, 4, CUDA_R_64F, dA, 4, CUDA_R_64F, &dw, &hw);
  check(left == CUSOLVER_STATUS_INTERNAL_ERROR, "Xgeev with left eigenvectors: INTERNAL_ERROR, as NVIDIA's (right only)",
        left);
  const int cvr = bs(h, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, 4, CUDA_R_64F, dA, 4, CUDA_C_64F,
                     dA, CUDA_R_64F, dA, 4, CUDA_C_64F, dA, 4, CUDA_R_64F, &dw, &hw);
  check(cvr == CUSOLVER_STATUS_INVALID_VALUE, "Xgeev of a real A with a complex VR: INVALID_VALUE", cvr);
  cudaFree(dA);
}

/* ---- iterative refinement ---- */

static void refinement() {
  const int n = 20, lda = 21, nrhs = 2;
  HM a(n, n), b(n, nrhs);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) a(i, j) = std::sin(1.3 * i + 0.7 * j + 1) + 0.25 * std::cos(i * j + 1) + (i == j ? 6 : 0);
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i) b(i, j) = std::cos(0.3 * i + j);
  using GesvFn = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, int*, double*, int, double*, int,
                                      void*, size_t, int*, int*);
  using GesvBsFn = cusolverStatus_t (*)(cusolverDnHandle_t, int, int, double*, int, int*, double*, int, double*, int,
                                        void*, size_t*);
  struct { const char* name; GesvBsFn bs; GesvFn f; } solvers[] = {
      {"DDgesv", cusolverDnDDgesv_bufferSize, cusolverDnDDgesv}, {"DSgesv", cusolverDnDSgesv_bufferSize, cusolverDnDSgesv},
      {"DHgesv", cusolverDnDHgesv_bufferSize, cusolverDnDHgesv}};
  for (const auto& s : solvers) {
    const auto av = to_dev<double>(a, lda);
    double* dA = upload(av);
    double* dB = upload(to_dev<double>(b, n));
    double* dX = upload(std::vector<double>((size_t)n * nrhs));
    int* dp = upload(std::vector<int>(n));
    int* di = upload(std::vector<int>{-1});
    size_t lw = 0;
    CK(s.bs(h, n, nrhs, dA, lda, dp, dB, n, dX, n, nullptr, &lw));
    void* w = nullptr;
    cudaMalloc(&w, lw + 16);
    int iter = -999;
    CK(s.f(h, n, nrhs, dA, lda, dp, dB, n, dX, n, w, lw, &iter, di));
    const double err = diff(mul(a, from_dev(download(dX, (size_t)n * nrhs), n, nrhs, n)), b) / maxabs(b);
    const bool same = download(dA, av.size()) == av;
    char what[160];
    std::snprintf(what, sizeof what, "%s: converged (niters %d >= 0) to double accuracy, A left as it was", s.name, iter);
    check(iter >= 0 && dev_int(di) == 0 && same && err < 1e-12, what, err);
    for (void* p : {(void*)dA, (void*)dB, (void*)dX, (void*)dp, (void*)di, w}) cudaFree(p);
  }
  {  // An inconsistent least-squares system never meets the residual test: -50, then the fallback.
    const int m = 30, nn = 20;
    HM g(m, nn), gb(m, 1);
    for (int j = 0; j < nn; ++j)
      for (int i = 0; i < m; ++i) g(i, j) = std::sin(1.3 * i + 0.7 * j + 1) + 0.25 * std::cos(i * j + 1) + (i == j ? 3 : 0);
    for (int i = 0; i < m; ++i) gb(i, 0) = std::cos(0.3 * i);
    double* dA = upload(to_dev<double>(g, m));
    double* dB = upload(to_dev<double>(gb, m));
    double* dX = upload(std::vector<double>(nn));
    int* di = upload(std::vector<int>{-1});
    size_t lw = 0;
    CK(cusolverDnDSgels_bufferSize(h, m, nn, 1, dA, m, dB, m, dX, nn, nullptr, &lw));
    void* w = nullptr;
    cudaMalloc(&w, lw + 16);
    int iter = 0;
    CK(cusolverDnDSgels(h, m, nn, 1, dA, m, dB, m, dX, nn, w, lw, &iter, di));
    const HM x = from_dev(download(dX, nn), nn, 1, nn);
    HM r = mul(g, x);
    for (int i = 0; i < m; ++i) r(i, 0) = gb(i, 0) - r(i, 0);
    const double ne = maxabs(mul(herm(g), r));  // the normal equations
    check(iter == -50 && ne < 1e-10, "DSgels on an inconsistent system: niters -50, the fallback's least-squares x", ne);
    for (void* p : {(void*)dA, (void*)dB, (void*)dX, (void*)di, w}) cudaFree(p);
  }
  {  // The expert interface: unset precisions, a GMRES refinement with two right-hand sides.
    cusolverDnIRSParams_t p;
    cusolverDnIRSInfos_t inf;
    CK(cusolverDnIRSParamsCreate(&p));
    CK(cusolverDnIRSInfosCreate(&inf));
    double* dA = upload(to_dev<double>(a, lda));
    double* dB = upload(to_dev<double>(b, n));
    double* dX = upload(std::vector<double>((size_t)n * nrhs));
    int* di = upload(std::vector<int>{-1});
    void* w = nullptr;
    cudaMalloc(&w, 1 << 22);
    int iter = 0;
    const int unset = cusolverDnIRSXgesv(h, p, inf, n, 1, dA, lda, dB, n, dX, n, w, 1 << 22, &iter, di);
    check(unset == CUSOLVER_STATUS_IRS_PARAMS_INVALID_PREC, "IRSXgesv with no precisions set: IRS_PARAMS_INVALID_PREC",
          unset);
    CK(cusolverDnIRSParamsSetSolverPrecisions(p, CUSOLVER_R_64F, CUSOLVER_R_32F));
    CK(cusolverDnIRSParamsSetRefinementSolver(p, CUSOLVER_IRS_REFINE_GMRES));
    const int two = cusolverDnIRSXgesv(h, p, inf, n, 2, dA, lda, dB, n, dX, n, w, 1 << 22, &iter, di);
    check(two == CUSOLVER_STATUS_IRS_NOT_SUPPORTED, "IRSXgesv GMRES with two right-hand sides: IRS_NOT_SUPPORTED", two);
    // Fresh params and infos: NVIDIA's can answer ALLOC_FAILED on infos reused after a refused call.
    cusolverDnIRSInfosDestroy(inf);
    cusolverDnIRSParamsDestroy(p);
    CK(cusolverDnIRSParamsCreate(&p));
    CK(cusolverDnIRSInfosCreate(&inf));
    CK(cusolverDnIRSParamsSetSolverPrecisions(p, CUSOLVER_R_64F, CUSOLVER_R_32F));
    CK(cusolverDnIRSParamsSetRefinementSolver(p, CUSOLVER_IRS_REFINE_CLASSICAL));
    CK(cusolverDnIRSInfosRequestResidual(inf));
    CK(cusolverDnIRSXgesv(h, p, inf, n, 1, dA, lda, dB, n, dX, n, w, 1 << 22, &iter, di));
    int ni = -1, outer = -1;
    void* hist = nullptr;
    CK(cusolverDnIRSInfosGetNiters(inf, &ni));
    CK(cusolverDnIRSInfosGetOuterNiters(inf, &outer));
    CK(cusolverDnIRSInfosGetResidualHistory(inf, &hist));
    HM b1(n, 1);
    for (int i = 0; i < n; ++i) b1(i, 0) = b(i, 0);
    const double err = diff(mul(a, from_dev(download(dX, n), n, 1, n)), b1) / maxabs(b1);
    const double* hh = static_cast<const double*>(hist);
    const bool hist_ok = hh && ni >= 0 && hh[51 + ni] < hh[51];  // (maxiters + 1) rows: the residual falls
    check(ni == iter && outer == ni && hist_ok && err < 1e-12,
          "IRSXgesv classical, double over single: converged, niters and its residual history reported", err);
    cusolverDnIRSInfosDestroy(inf);
    cusolverDnIRSParamsDestroy(p);
    for (void* q : {(void*)dA, (void*)dB, (void*)dX, (void*)di, w}) cudaFree(q);
  }
}

/* ---- the handle's modes ---- */

static void modes() {
  using GetFn = cusolverStatus_t (*)(cusolverDnHandle_t, int*);
  using SetFn = cusolverStatus_t (*)(cusolverDnHandle_t, int);
  auto gd = reinterpret_cast<GetFn>(dlsym(RTLD_DEFAULT, "cusolverDnGetDeterministicMode"));
  auto sd = reinterpret_cast<SetFn>(dlsym(RTLD_DEFAULT, "cusolverDnSetDeterministicMode"));
  auto gm = reinterpret_cast<GetFn>(dlsym(RTLD_DEFAULT, "cusolverDnGetMathMode"));
  auto sm = reinterpret_cast<SetFn>(dlsym(RTLD_DEFAULT, "cusolverDnSetMathMode"));
  if (!gd || !sd || !gm || !sm) {
    std::printf("FAIL the cuSOLVER this runs against has no handle modes\n");
    ++failures;
    return;
  }
  int det = -1, math = -1;
  CK(gd(h, &det));
  CK(gm(h, &math));
  const int bad_det = sd(h, 0), bad_math = sm(h, 9);
  CK(sd(h, 2));
  int det2 = -1;
  CK(gd(h, &det2));
  CK(sd(h, 1));
  check(det == 1 && math == 1 && det2 == 2 && bad_det == CUSOLVER_STATUS_INTERNAL_ERROR &&
            bad_math == CUSOLVER_STATUS_INVALID_VALUE,
        "modes: deterministic and default math by default; a bad mode refused as NVIDIA's refuses it", 0);
  // The Jacobi getters refuse after a batched call, as NVIDIA documents.
  syevjInfo_t sj;
  CK(cusolverDnCreateSyevjInfo(&sj));
  std::vector<double> s((size_t)16);
  for (int j = 0; j < 4; ++j)
    for (int i = 0; i < 4; ++i) s[(size_t)j * 4 + i] = i == j ? 2.0 + i : 0.3;
  double* dS = upload(s);
  double* dW = upload(std::vector<double>(4));
  int* di = upload(std::vector<int>{-1});
  double* w = upload(std::vector<double>(4096));
  CK(cusolverDnDsyevjBatched(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, 4, dS, 4, dW, w, 4096, di, sj, 1));
  double r = 0;
  const int after = cusolverDnXsyevjGetResidual(h, sj, &r);
  check(after == CUSOLVER_STATUS_NOT_SUPPORTED, "XsyevjGetResidual after syevjBatched: NOT_SUPPORTED", after);
  cusolverDnDestroySyevjInfo(sj);
  for (void* p : {(void*)dS, (void*)dW, (void*)di, (void*)w}) cudaFree(p);
}

template <class T> void all() {
  tridiagonal<T>(false);
  tridiagonal<T>(true);
  bidiagonal<T>();
  inverse<T>(false);
  inverse<T>(true);
  eig_range<T>();
  generalized<T>();
  x_lu<T>();
  x_trtri<T>();
  x_syevdx<T>();
  x_svd<T>();
  x_larft<T>();
}

int main() {
  if (cusolverDnCreate(&h) || cusolverDnCreateParams(&params)) {
    std::printf("FAIL: could not create the cuSOLVER handle\n");
    return 1;
  }
  all<float>();
  all<double>();
  all<cuComplex>();
  all<cuDoubleComplex>();
  geev_left();
  refinement();
  modes();
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(h);
  std::printf(failures ? "FAIL: %d dense solver checks\n" : "PASS: every dense solver check\n", failures);
  return failures ? 1 : 0;
}
