// cuSOLVER's sparse module (cusolverSp), checked by each result's defining
// property and against what an RTX 3060's cuSOLVER (CUDA 13.0) answers:
//
//   csrlsvlu, csrlsvqr, csrlsvchol   A x = b, host and device entry points,
//                                    S/D/C/Z, every reorder; Cholesky reads
//                                    the lower triangle only; singularity
//                                    for a zero column, dependent rows, a
//                                    pivot under the (absolute) tol, a
//                                    matrix that is not positive definite
//   csrlsqvqr                        least squares, the minimum-norm
//                                    solution when A is rank deficient, and
//                                    the column permutation NVIDIA's reports
//   csreigvsi                        shift-inverse iteration, against the
//                                    same iteration done here
//   csreigs                          eigenvalues in a box, counted by Sturm
//                                    sequences
//   csrissym, the reorderings,       structural answers, checked for what
//   csrperm, csrzfd                  defines them
//   csrqrsvBatched                   least squares over a batch
//   refusals                         a non-general matrix type
//                                    (MATRIX_TYPE_NOT_SUPPORTED), reorder 4
//                                    (INVALID_VALUE)
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <numeric>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
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

static cusolverSpHandle_t h;
static cusparseMatDescr_t D;

template <class T> struct Ty;
template <> struct Ty<float> {
  using R = float;
  static constexpr const char* name = "S";
  static constexpr double tol = 2e-5;
  static float make(cdouble v) { return (float)v.real(); }
  static cdouble get(float v) { return v; }
};
template <> struct Ty<double> {
  using R = double;
  static constexpr const char* name = "D";
  static constexpr double tol = 1e-12;
  static double make(cdouble v) { return v.real(); }
  static cdouble get(double v) { return v; }
};
template <> struct Ty<cuComplex> {
  using R = float;
  static constexpr const char* name = "C";
  static constexpr double tol = 2e-5;
  static cuComplex make(cdouble v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
  static cdouble get(cuComplex v) { return {v.x, v.y}; }
};
template <> struct Ty<cuDoubleComplex> {
  using R = double;
  static constexpr const char* name = "Z";
  static constexpr double tol = 1e-12;
  static cuDoubleComplex make(cdouble v) { return make_cuDoubleComplex(v.real(), v.imag()); }
  static cdouble get(cuDoubleComplex v) { return {v.x, v.y}; }
};

#define SP_DISPATCH(fn)                                                                                            \
  template <class... A> static cusolverStatus_t fn(float*, A... a) { return cusolverSpS##fn(a...); }               \
  template <class... A> static cusolverStatus_t fn(double*, A... a) { return cusolverSpD##fn(a...); }              \
  template <class... A> static cusolverStatus_t fn(cuComplex*, A... a) { return cusolverSpC##fn(a...); }           \
  template <class... A> static cusolverStatus_t fn(cuDoubleComplex*, A... a) { return cusolverSpZ##fn(a...); }
SP_DISPATCH(csrlsvluHost)
SP_DISPATCH(csrlsvqr)
SP_DISPATCH(csrlsvqrHost)
SP_DISPATCH(csrlsvchol)
SP_DISPATCH(csrlsvcholHost)
SP_DISPATCH(csrlsqvqrHost)
SP_DISPATCH(csreigvsiHost)
SP_DISPATCH(csreigvsi)
SP_DISPATCH(csrzfdHost)
SP_DISPATCH(csrqrBufferInfoBatched)
SP_DISPATCH(csrqrsvBatched)
#undef SP_DISPATCH

// A sparse matrix as CSR plus its dense (row-major) form.
template <class T> struct Sparse {
  int m, n;
  std::vector<int> off, col;
  std::vector<T> val;
  std::vector<cdouble> dense;
};
template <class T> Sparse<T> from_dense(int m, int n, const std::vector<cdouble>& d) {
  Sparse<T> s{m, n, {0}, {}, {}, std::vector<cdouble>((size_t)m * n)};
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      const cdouble v = d[(size_t)i * n + j];
      if (v == cdouble(0)) continue;
      s.col.push_back(j);
      s.val.push_back(Ty<T>::make(v));
      s.dense[(size_t)i * n + j] = Ty<T>::get(s.val.back());
    }
    s.off.push_back((int)s.col.size());
  }
  return s;
}
static double hash01(int i, int j, int seed) {
  return std::fmod(std::fabs(std::sin(12.9898 * (i + 1) + 78.233 * (j + 1) + seed)) * 43758.5453, 1.0);
}
// A general sparse n x n matrix with a dominant-enough diagonal.
static std::vector<cdouble> general(int n, int seed, bool cplx) {
  std::vector<cdouble> d((size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (i == j || std::abs(i - j) == 1 || hash01(i, j, seed) < 0.08)
        d[(size_t)i * n + j] = cdouble(std::sin(1.0 + i + 3.0 * j) + (i == j ? 4.0 : 0.0),
                                       cplx ? std::cos(2.0 * i - j) : 0.0);
  return d;
}
// Symmetric (Hermitian for complex) positive definite: a weighted Laplacian
// of a sparse graph plus the identity.
static std::vector<cdouble> spd(int n, int seed, bool cplx) {
  std::vector<cdouble> d((size_t)n * n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j)
      if (std::abs(i - j) == 1 || hash01(i, j, seed) < 0.1) {
        const cdouble w(0.5 + hash01(j, i, seed), cplx ? 0.3 * std::sin(i * j + 1.0) : 0.0);
        d[(size_t)i * n + j] = -w;
        d[(size_t)j * n + i] = -std::conj(w);
        d[(size_t)i * n + i] += std::abs(w);
        d[(size_t)j * n + j] += std::abs(w);
      }
  for (int i = 0; i < n; ++i) d[(size_t)i * n + i] += 1.0;
  return d;
}
template <class T> static std::vector<T> rhs(int n) {
  std::vector<T> b;
  for (int i = 0; i < n; ++i) b.push_back(Ty<T>::make(cdouble(std::cos(0.3 * i), 0.5 * std::sin(0.7 * i))));
  return b;
}
template <class T>
static double residual(const std::vector<cdouble>& a, int n, const std::vector<T>& x, const std::vector<T>& b) {
  double r = 0, an = 0, xn = 0;
  for (int i = 0; i < n; ++i) {
    cdouble s = 0;
    for (int j = 0; j < n; ++j) {
      s += a[(size_t)i * n + j] * Ty<T>::get(x[j]);
      an = std::max(an, std::abs(a[(size_t)i * n + j]));
    }
    r = std::max(r, std::abs(s - Ty<T>::get(b[i])));
    xn = std::max(xn, std::abs(Ty<T>::get(x[i])));
  }
  return r / (an * xn * n);
}

template <class T> static void linear_solves() {
  const int n = 40;
  const bool cplx = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
  const auto g = from_dense<T>(n, n, general(n, 3, cplx));
  const auto s = from_dense<T>(n, n, spd(n, 5, cplx));
  // Cholesky reads the lower triangle: give it junk above the diagonal.
  auto junk = spd(n, 5, cplx);
  for (int i = 0; i < n; ++i)
    for (int j = i + 1; j < n; ++j)
      if (junk[(size_t)i * n + j] != cdouble(0) || hash01(i, j, 9) < 0.05) junk[(size_t)i * n + j] = cdouble(7.0, -3.0);
  const auto sj = from_dense<T>(n, n, junk);
  const auto b = rhs<T>(n);
  using R = typename Ty<T>::R;
  const R tol = (R)1e-12;
  T* dummy = nullptr;
  int* doff = upload(g.off);
  int* dcol = upload(g.col);
  T* dval = upload(g.val);
  int* soff = upload(s.off);
  int* scol = upload(s.col);
  T* sval = upload(s.val);
  T* db = upload(b);
  T* dx = upload(std::vector<T>(n));
  double worst[5] = {0, 0, 0, 0, 0};
  bool ok[5] = {true, true, true, true, true};
  for (int reorder = 0; reorder < 4; ++reorder) {
    std::vector<T> x(n);
    int sing = -7;
    CK(csrlsvluHost(dummy, h, n, (int)g.val.size(), D, g.val.data(), g.off.data(), g.col.data(), b.data(), tol,
                    reorder, x.data(), &sing));
    double e = residual(g.dense, n, x, b);
    worst[0] = std::max(worst[0], e);
    ok[0] = ok[0] && sing == -1 && e < Ty<T>::tol;
    sing = -7;
    CK(csrlsvqrHost(dummy, h, n, (int)g.val.size(), D, g.val.data(), g.off.data(), g.col.data(), b.data(), tol,
                    reorder, x.data(), &sing));
    e = residual(g.dense, n, x, b);
    worst[1] = std::max(worst[1], e);
    ok[1] = ok[1] && sing == -1 && e < Ty<T>::tol;
    sing = -7;
    CK(csrlsvqr(dummy, h, n, (int)g.val.size(), D, dval, doff, dcol, db, tol, reorder, dx, &sing));
    e = residual(g.dense, n, download(dx, n), b);
    worst[2] = std::max(worst[2], e);
    ok[2] = ok[2] && sing == -1 && e < Ty<T>::tol;
    sing = -7;
    CK(csrlsvcholHost(dummy, h, n, (int)s.val.size(), D, s.val.data(), s.off.data(), s.col.data(), b.data(), tol,
                      reorder, x.data(), &sing));
    e = residual(s.dense, n, x, b);
    worst[3] = std::max(worst[3], e);
    ok[3] = ok[3] && sing == -1 && e < Ty<T>::tol;
    sing = -7;
    CK(csrlsvchol(dummy, h, n, (int)s.val.size(), D, sval, soff, scol, db, tol, reorder, dx, &sing));
    e = residual(s.dense, n, download(dx, n), b);
    worst[4] = std::max(worst[4], e);
    ok[4] = ok[4] && sing == -1 && e < Ty<T>::tol;
  }
  const char* what[5] = {"csrlsvluHost", "csrlsvqrHost", "csrlsvqr", "csrlsvcholHost", "csrlsvchol"};
  for (int k = 0; k < 5; ++k) {
    char line[160];
    std::snprintf(line, sizeof line, "%s%s, every reorder: A x = b", Ty<T>::name, what[k]);
    check(ok[k], line, worst[k]);
  }
  {  // Above the diagonal is not read (reorder 0: NVIDIA's reads it with reorder 1).
    std::vector<T> x(n);
    int sing = -7;
    CK(csrlsvcholHost(dummy, h, n, (int)sj.val.size(), D, sj.val.data(), sj.off.data(), sj.col.data(), b.data(), tol,
                      0, x.data(), &sing));
    const double e = residual(s.dense, n, x, b);
    char line[160];
    std::snprintf(line, sizeof line, "%scsrlsvcholHost ignores what lies above the diagonal", Ty<T>::name);
    check(sing == -1 && e < Ty<T>::tol, line, e);
  }
  for (void* p : {(void*)doff, (void*)dcol, (void*)dval, (void*)soff, (void*)scol, (void*)sval, (void*)db, (void*)dx})
    cudaFree(p);
}

// singularity: the first index whose pivot is <= tol (absolute), or -1.
static void singularity() {
  const int n = 6;
  std::vector<cdouble> a((size_t)n * n);
  for (int i = 0; i < n; ++i) {
    a[(size_t)i * n + i] = 4.0 + i;
    if (i + 1 < n) a[(size_t)i * n + i + 1] = -1;
    if (i > 1) a[(size_t)i * n + i - 2] = 0.5 * i;
  }
  a[0 * n + 5] = 2;
  a[5 * n + 1] = -3;
  const std::vector<double> b = {1, 2, 3, 4, 5, 6};
  std::vector<double> x(n);
  double* dd = nullptr;
  auto lu = [&](const std::vector<cdouble>& m, double tol) {
    const auto s = from_dense<double>(n, n, m);
    int sing = -7;
    csrlsvluHost(dd, h, n, (int)s.val.size(), D, s.val.data(), s.off.data(), s.col.data(), b.data(), tol, 0,
                 x.data(), &sing);
    return sing;
  };
  auto qr = [&](const std::vector<cdouble>& m, double tol) {
    const auto s = from_dense<double>(n, n, m);
    int sing = -7;
    csrlsvqrHost(dd, h, n, (int)s.val.size(), D, s.val.data(), s.off.data(), s.col.data(), b.data(), tol, 0,
                 x.data(), &sing);
    return sing;
  };
  auto zero_col = a, dep_rows = a, zero_row = a;
  for (int i = 0; i < n; ++i) zero_col[(size_t)i * n + 2] = 0;
  for (int j = 0; j < n; ++j) {
    dep_rows[(size_t)3 * n + j] = dep_rows[(size_t)1 * n + j];
    zero_row[(size_t)4 * n + j] = 0;
  }
  check(lu(zero_col, 1e-12) == 2 && qr(zero_col, 1e-12) == 2, "a zero column: singularity = its index", 0);
  check(lu(dep_rows, 1e-12) == 5 && qr(dep_rows, 1e-12) == 5, "dependent rows: singularity = 5, the last", 0);
  check(lu(zero_row, 1e-12) == 5 && qr(zero_row, 1e-12) == 5, "a zero row: singularity = 5, the last", 0);
  {
    std::vector<cdouble> small((size_t)n * n);
    for (int i = 0; i < n; ++i) small[(size_t)i * n + i] = i == 1 ? 1e-3 : 1.0 + i;
    small[0 * n + 2] = 0.5;
    small[2 * n + 0] = 0.25;
    small[3 * n + 4] = 0.1;
    check(lu(small, 1e-6) == -1 && qr(small, 1e-6) == -1 && lu(small, 1e-2) == 1 && qr(small, 1e-2) == 1,
          "tol is absolute: a pivot of 1e-3 is singular at tol 1e-2 and not at 1e-6", 0);
  }
  {  // Cholesky on a tridiagonal matrix: a negative pivot, a missing diagonal.
    std::vector<cdouble> t((size_t)n * n);
    for (int i = 0; i < n; ++i) {
      t[(size_t)i * n + i] = 4.0 + i;
      if (i) t[(size_t)i * n + i - 1] = t[(size_t)(i - 1) * n + i] = -1;
    }
    auto neg = t, missing = t;
    neg[3 * n + 3] = -5;
    missing[2 * n + 2] = 0;
    auto chol = [&](const std::vector<cdouble>& m) {
      const auto s = from_dense<double>(n, n, m);
      int sing = -7;
      csrlsvcholHost(dd, h, n, (int)s.val.size(), D, s.val.data(), s.off.data(), s.col.data(), b.data(), 1e-12, 0,
                     x.data(), &sing);
      return sing;
    };
    check(chol(t) == -1 && chol(neg) == 3 && chol(missing) == 2,
          "csrlsvchol: singularity at a pivot that is not positive, or a missing diagonal", 0);
  }
  {  // Refusals.
    const auto s = from_dense<double>(n, n, a);
    int sing = -7;
    const int st4 = csrlsvluHost(dd, h, n, (int)s.val.size(), D, s.val.data(), s.off.data(), s.col.data(), b.data(),
                                 1e-12, 4, x.data(), &sing);
    cusparseMatDescr_t sym;
    cusparseCreateMatDescr(&sym);
    cusparseSetMatType(sym, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
    const int st8 = csrlsvcholHost(dd, h, n, (int)s.val.size(), sym, s.val.data(), s.off.data(), s.col.data(),
                                   b.data(), 1e-12, 0, x.data(), &sing);
    cusparseDestroyMatDescr(sym);
    check(st4 == CUSOLVER_STATUS_INVALID_VALUE && st8 == CUSOLVER_STATUS_MATRIX_TYPE_NOT_SUPPORTED,
          "reorder 4 is INVALID_VALUE; a symmetric matrix type MATRIX_TYPE_NOT_SUPPORTED", st4 * 100 + st8);
  }
  {  // One-based indices.
    cusparseMatDescr_t one;
    cusparseCreateMatDescr(&one);
    cusparseSetMatIndexBase(one, CUSPARSE_INDEX_BASE_ONE);
    auto s = from_dense<double>(n, n, a);
    for (int& v : s.off) ++v;
    for (int& v : s.col) ++v;
    int sing = -7;
    csrlsvluHost(dd, h, n, (int)s.val.size(), one, s.val.data(), s.off.data(), s.col.data(), b.data(), 1e-12, 0,
                 x.data(), &sing);
    const double e = residual(s.dense, n, x, b);
    check(sing == -1 && e < 1e-12, "csrlsvluHost with one-based indices", e);
    cusparseDestroyMatDescr(one);
  }
}

template <class T> static void least_squares() {
  const int m = 30, n = 8;
  const bool cplx = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
  using R = typename Ty<T>::R;
  T* dummy = nullptr;
  for (int deficient = 0; deficient < 2; ++deficient) {
    std::vector<cdouble> d((size_t)m * n);
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j)
        if ((i + 2 * j) % 3 != 1)
          d[(size_t)i * n + j] = cdouble(2 * hash01(i, j, 17) - 1, cplx ? 2 * hash01(j, i, 19) - 1 : 0.0);
    if (deficient)
      for (int i = 0; i < m; ++i) d[(size_t)i * n + 5] = d[(size_t)i * n + 2];  // column 5 = column 2
    const auto a = from_dense<T>(m, n, d);
    const auto b = rhs<T>(m);
    std::vector<T> x(n);
    std::vector<int> p(n, -1);
    int rank = -1;
    R mn = -1;
    const R tol = std::is_same_v<R, float> ? (R)1e-4 : (R)1e-10;  // above the precision's rounding of a zero
    CK(csrlsqvqrHost(dummy, h, m, n, (int)a.val.size(), D, a.val.data(), a.off.data(), a.col.data(), b.data(), tol,
                     &rank, x.data(), p.data(), &mn));
    // The normal equations hold, |b - A x| is what min_norm says, and a
    // rank-deficient solution splits equally between the equal columns.
    std::vector<cdouble> r(m);
    double rn = 0, scale = 0;
    for (int i = 0; i < m; ++i) {
      cdouble s = Ty<T>::get(b[i]);
      for (int j = 0; j < n; ++j) s -= a.dense[(size_t)i * n + j] * Ty<T>::get(x[j]);
      r[i] = s;
      rn += std::norm(s);
    }
    double ne = 0;
    for (int j = 0; j < n; ++j) {
      cdouble s = 0;
      for (int i = 0; i < m; ++i) {
        s += std::conj(a.dense[(size_t)i * n + j]) * r[i];
        scale = std::max(scale, std::abs(a.dense[(size_t)i * n + j]));
      }
      ne = std::max(ne, std::abs(s));
    }
    ne /= scale * scale * m;
    // NVIDIA's p: a dependent column changes places with the last column.
    const std::vector<int> want_p = deficient ? std::vector<int>{0, 1, 2, 3, 4, 7, 6, 5}
                                              : std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7};
    const bool ident = p == want_p;
    const bool split = !deficient || std::abs(Ty<T>::get(x[2]) - Ty<T>::get(x[5])) < 1e3 * Ty<T>::tol;
    char line[200];
    std::snprintf(line, sizeof line, "%scsrlsqvqrHost%s: rank %d, A^H (b - A x) = 0, min_norm = |b - A x|, p%s",
                  Ty<T>::name, deficient ? ", rank deficient" : "", rank,
                  deficient ? ", the minimum-norm x" : "");
    check(rank == (deficient ? n - 1 : n) && ident && split && ne < 10 * Ty<T>::tol &&
              std::fabs(mn - std::sqrt(rn)) < 10 * Ty<T>::tol * (1 + std::sqrt(rn)),
          line, ne);
  }
}

// Shift-inverse iteration, done here as NVIDIA's does it: x = x0 / |x0|; at
// each step mu = x^H A x, stop when |A x - mu x| <= tol, else x = (A - mu0
// I)^{-1} x, normalized.
template <class T> static void shift_inverse() {
  const int n = 12;
  const bool cplx = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
  std::vector<cdouble> d((size_t)n * n);
  for (int i = 0; i < n; ++i) {
    d[(size_t)i * n + i] = 2.0 + i;
    if (i) {
      const cdouble w(-1.0, cplx ? 0.25 : 0.0);
      d[(size_t)i * n + i - 1] = w;
      d[(size_t)(i - 1) * n + i] = std::conj(w);
    }
  }
  const auto a = from_dense<T>(n, n, d);
  const cdouble mu0(4.1, 0);
  std::vector<cdouble> m((size_t)n * n);  // (A - mu0 I)^{-1} by Gauss-Jordan
  {
    std::vector<cdouble> w = a.dense;
    for (int i = 0; i < n; ++i) w[(size_t)i * n + i] -= mu0;
    for (int i = 0; i < n; ++i) m[(size_t)i * n + i] = 1;
    for (int c = 0; c < n; ++c) {
      int p = c;
      for (int r = c + 1; r < n; ++r)
        if (std::abs(w[(size_t)r * n + c]) > std::abs(w[(size_t)p * n + c])) p = r;
      for (int k = 0; k < n; ++k) {
        std::swap(w[(size_t)c * n + k], w[(size_t)p * n + k]);
        std::swap(m[(size_t)c * n + k], m[(size_t)p * n + k]);
      }
      const cdouble piv = w[(size_t)c * n + c];
      for (int k = 0; k < n; ++k) {
        w[(size_t)c * n + k] /= piv;
        m[(size_t)c * n + k] /= piv;
      }
      for (int r = 0; r < n; ++r)
        if (r != c) {
          const cdouble f = w[(size_t)r * n + c];
          for (int k = 0; k < n; ++k) {
            w[(size_t)r * n + k] -= f * w[(size_t)c * n + k];
            m[(size_t)r * n + k] -= f * m[(size_t)c * n + k];
          }
        }
    }
  }
  using R = typename Ty<T>::R;
  const double tol = std::is_same_v<R, float> ? 1e-4 : 1e-10;
  std::vector<T> x0(n, Ty<T>::make(1.0));
  T* dummy = nullptr;
  for (int maxite : {0, 1, 2, 7, 100}) {
    std::vector<cdouble> x(n, 1.0 / std::sqrt((double)n));
    cdouble mu = 0;
    for (int it = 0; it < maxite; ++it) {
      std::vector<cdouble> ax(n, 0.0);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) ax[i] += a.dense[(size_t)i * n + j] * x[j];
      mu = 0;
      for (int i = 0; i < n; ++i) mu += std::conj(x[i]) * ax[i];
      double r = 0;
      for (int i = 0; i < n; ++i) r += std::norm(ax[i] - mu * x[i]);
      if (std::sqrt(r) <= tol) break;
      std::vector<cdouble> y(n, 0.0);
      for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) y[i] += m[(size_t)i * n + j] * x[j];
      double nr = 0;
      for (const auto& v : y) nr += std::norm(v);
      for (int i = 0; i < n; ++i) x[i] = y[i] / std::sqrt(nr);
    }
    T mu_got = Ty<T>::make(0.0);
    std::vector<T> x_got(n);
    CK(csreigvsiHost(dummy, h, n, (int)a.val.size(), D, a.val.data(), a.off.data(), a.col.data(), Ty<T>::make(mu0),
                     x0.data(), maxite, (R)tol, &mu_got, x_got.data()));
    // The device entry point, the same.
    int* doff = upload(a.off);
    int* dcol = upload(a.col);
    T* dval = upload(a.val);
    T* dx0 = upload(x0);
    T* dx = upload(std::vector<T>(n));
    T* dmu = upload(std::vector<T>(1));
    CK(csreigvsi(dummy, h, n, (int)a.val.size(), D, dval, doff, dcol, Ty<T>::make(mu0), dx0, maxite, (R)tol, dmu, dx));
    const auto xd = download(dx, n);
    const T mud = download(dmu, 1)[0];
    double e = std::abs(Ty<T>::get(mu_got) - mu) + std::abs(Ty<T>::get(mud) - mu);
    for (int i = 0; i < n; ++i)
      e = std::max(e, std::max(std::abs(Ty<T>::get(x_got[i]) - x[i]), std::abs(Ty<T>::get(xd[i]) - x[i])));
    char line[160];
    std::snprintf(line, sizeof line, "%scsreigvsi and csreigvsiHost, maxite %d: mu and x as the iteration gives them",
                  Ty<T>::name, maxite);
    check(e < (std::is_same_v<R, float> ? 1e-4 : 1e-10), line, e);
    for (void* p : {(void*)doff, (void*)dcol, (void*)dval, (void*)dx0, (void*)dx, (void*)dmu}) cudaFree(p);
  }
}

// csreigsHost against Sturm counts on a symmetric tridiagonal matrix.
static void eigs() {
  const int n = 20;
  std::vector<cdouble> d((size_t)n * n);
  std::vector<double> diag(n), off(n, -1.0);
  for (int i = 0; i < n; ++i) {
    diag[i] = 2.0 + 0.5 * i;
    d[(size_t)i * n + i] = diag[i];
    if (i) d[(size_t)i * n + i - 1] = d[(size_t)(i - 1) * n + i] = -1;
  }
  auto below = [&](double x) {  // eigenvalues < x
    int count = 0;
    double q = diag[0] - x;
    count += q < 0;
    for (int i = 1; i < n; ++i) {
      q = diag[i] - x - 1.0 / q;
      count += q < 0;
    }
    return count;
  };
  const auto a = from_dense<double>(n, n, d);
  bool ok = true;
  const std::vector<std::pair<double, double>> boxes = {{1.05, 4.05}, {0.0, 20.0}, {5.3, 7.7}};
  for (const auto& [lo, hi] : boxes) {
    int num = -1;
    cusolverSpDcsreigsHost(h, n, (int)a.val.size(), D, a.val.data(), a.off.data(), a.col.data(),
                           make_cuDoubleComplex(lo, -1), make_cuDoubleComplex(hi, 1), &num);
    ok = ok && num == below(hi) - below(lo);
  }
  check(ok, "csreigsHost counts the eigenvalues in a box", 0);
}

static void structure() {
  const int n = 10;
  std::vector<cdouble> d((size_t)n * n);
  for (int i = 0; i < n; ++i) {
    d[(size_t)i * n + i] = 1;
    const int j = (i * 3 + 1) % n, k = (i * 7 + 2) % n;
    d[(size_t)i * n + j] = d[(size_t)j * n + i] = 1;
    d[(size_t)i * n + k] = d[(size_t)k * n + i] = 1;
  }
  auto a = from_dense<double>(n, n, d);
  std::vector<int> end(a.off.begin() + 1, a.off.end());
  int sym = -1, unsym = -1;
  cusolverSpXcsrissymHost(h, n, (int)a.val.size(), D, a.off.data(), end.data(), a.col.data(), &sym);
  {
    auto u = d;
    u[0 * n + 9] = 5;
    if (u[9 * n + 0] != cdouble(0)) u[9 * n + 0] = 0;
    const auto b = from_dense<double>(n, n, u);
    std::vector<int> e2(b.off.begin() + 1, b.off.end());
    cusolverSpXcsrissymHost(h, n, (int)b.val.size(), D, b.off.data(), e2.data(), b.col.data(), &unsym);
  }
  check(sym == 1 && unsym == 0, "csrissymHost: 1 for a symmetric pattern, 0 otherwise", 0);

  // Each reordering is a permutation; RCM brings a scrambled band back near its width.
  const int m = 30, band = 2;
  std::vector<int> scramble(m);
  for (int i = 0; i < m; ++i) scramble[i] = (i * 7 + 3) % m;
  std::vector<cdouble> bd((size_t)m * m);
  for (int i = 0; i < m; ++i)
    for (int j = std::max(0, i - band); j <= std::min(m - 1, i + band); ++j)
      bd[(size_t)scramble[i] * m + scramble[j]] = 1;
  const auto bm = from_dense<double>(m, m, bd);
  bool perms = true;
  int rcm_width = -1, scrambled_width = 0;
  for (int i = 0; i < m; ++i)
    for (int k = bm.off[i]; k < bm.off[i + 1]; ++k) scrambled_width = std::max(scrambled_width, std::abs(i - bm.col[k]));
  for (int which = 0; which < 4; ++which) {
    std::vector<int> p(m, -1);
    int st = -1;
    if (which == 0) st = cusolverSpXcsrsymrcmHost(h, m, (int)bm.val.size(), D, bm.off.data(), bm.col.data(), p.data());
    if (which == 1) st = cusolverSpXcsrsymamdHost(h, m, (int)bm.val.size(), D, bm.off.data(), bm.col.data(), p.data());
    if (which == 2) st = cusolverSpXcsrsymmdqHost(h, m, (int)bm.val.size(), D, bm.off.data(), bm.col.data(), p.data());
    if (which == 3)
      st = cusolverSpXcsrmetisndHost(h, m, (int)bm.val.size(), D, bm.off.data(), bm.col.data(), nullptr, p.data());
    std::vector<int> sorted = p;
    std::sort(sorted.begin(), sorted.end());
    std::vector<int> want(m);
    std::iota(want.begin(), want.end(), 0);
    perms = perms && st == 0 && sorted == want;
    if (which == 0) {
      std::vector<int> pinv(m);
      for (int i = 0; i < m; ++i) pinv[p[i]] = i;
      rcm_width = 0;
      for (int i = 0; i < m; ++i)
        for (int k = bm.off[i]; k < bm.off[i + 1]; ++k)
          rcm_width = std::max(rcm_width, std::abs(pinv[i] - pinv[bm.col[k]]));
    }
  }
  char line[160];
  std::snprintf(line, sizeof line, "symrcm, symamd, symmdq, metisnd give permutations; RCM's bandwidth %d (scrambled %d)",
                rcm_width, scrambled_width);
  check(perms && rcm_width <= 2 * band, line, rcm_width);

  // csrpermHost: B = P A Q^T, rows' columns ascending, map carried along.
  {
    std::vector<int> P(n), Q(n);
    for (int i = 0; i < n; ++i) {
      P[i] = (i * 3 + 2) % n;
      Q[i] = (i * 7 + 5) % n;
    }
    std::vector<int> off = a.off, col = a.col, map(a.val.size());
    std::iota(map.begin(), map.end(), 0);
    size_t bytes = 0;
    CK(cusolverSpXcsrperm_bufferSizeHost(h, n, n, (int)a.val.size(), D, off.data(), col.data(), P.data(), Q.data(),
                                         &bytes));
    std::vector<char> buf(std::max<size_t>(bytes, 1));
    CK(cusolverSpXcsrpermHost(h, n, n, (int)a.val.size(), D, off.data(), col.data(), P.data(), Q.data(), map.data(),
                              buf.data()));
    std::vector<int> qinv(n);
    for (int j = 0; j < n; ++j) qinv[Q[j]] = j;
    bool ok = off[0] == 0 && off[n] == (int)a.val.size();
    for (int i = 0; i < n && ok; ++i) {
      ok = off[i + 1] - off[i] == a.off[P[i] + 1] - a.off[P[i]];
      for (int k = off[i]; k < off[i + 1] && ok; ++k) {
        const int src = map[k];
        ok = (k == off[i] || col[k - 1] < col[k]) && src >= a.off[P[i]] && src < a.off[P[i] + 1] &&
             col[k] == qinv[a.col[src]];
      }
    }
    check(ok, "csrpermHost: row i of P A Q^T is row P[i], columns renumbered and sorted, map follows", 0);
  }

  // csrzfdHost: P A has a zero-free diagonal; P[j] is the row of A.
  {
    const int k = 6;
    std::vector<cdouble> z((size_t)k * k), s((size_t)k * k);
    for (int i = 0; i < k; ++i) z[(size_t)i * k + (i + 1) % k] = 1.0 + i;
    z[2 * k + 2] = 5;
    for (int i = 0; i < k; ++i) s[(size_t)i * k + 0] = 1;
    for (int j = 1; j < k; ++j) s[(size_t)0 * k + j] = 1;
    const auto za = from_dense<double>(k, k, z), sa = from_dense<double>(k, k, s);
    std::vector<int> P(k), P2(k);
    int nz = -1, nz2 = -1;
    double* dd = nullptr;
    CK(csrzfdHost(dd, h, k, (int)za.val.size(), D, za.val.data(), za.off.data(), za.col.data(), P.data(), &nz));
    CK(csrzfdHost(dd, h, k, (int)sa.val.size(), D, sa.val.data(), sa.off.data(), sa.col.data(), P2.data(), &nz2));
    check(nz == k && P == std::vector<int>({5, 0, 1, 2, 3, 4}) && nz2 == 2 && P2 == std::vector<int>({1, 0, 2, 3, 4, 5}),
          "csrzfdHost: the transversal NVIDIA's finds, and 2 for a structurally singular matrix", 0);
  }
}

template <class T> static void batched_qr() {
  const int m = 12, n = 5, batch = 3;
  const bool cplx = !std::is_same_v<T, float> && !std::is_same_v<T, double>;
  std::vector<cdouble> pattern((size_t)m * n);
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j)
      if ((i + j) % 3 != 2 || i == j) pattern[(size_t)i * n + j] = 1;
  auto a = from_dense<T>(m, n, pattern);
  const int nnz = (int)a.val.size();
  std::vector<T> vals, bs;
  std::vector<std::vector<cdouble>> dense(batch, std::vector<cdouble>((size_t)m * n));
  for (int b = 0; b < batch; ++b) {
    for (int i = 0; i < m; ++i)
      for (int k = a.off[i]; k < a.off[i + 1]; ++k) {
        const T v = Ty<T>::make(cdouble(std::sin(1.0 + i + 2.0 * a.col[k] + b), cplx ? std::cos(i - b) : 0.0));
        vals.push_back(v);
        dense[b][(size_t)i * n + a.col[k]] = Ty<T>::get(v);
      }
    const auto r = rhs<T>(m + b);
    bs.insert(bs.end(), r.begin() + b, r.end());
  }
  int* doff = upload(a.off);
  int* dcol = upload(a.col);
  T* dval = upload(vals);
  T* db = upload(bs);
  T* dx = upload(std::vector<T>((size_t)n * batch));
  csrqrInfo_t info;
  CK(cusolverSpCreateCsrqrInfo(&info));
  CK(cusolverSpXcsrqrAnalysisBatched(h, m, n, nnz, D, doff, dcol, info));
  size_t internal = 0, work = 0;
  T* dummy = nullptr;
  CK(csrqrBufferInfoBatched(dummy, h, m, n, nnz, D, dval, doff, dcol, batch, info, &internal, &work));
  void* buf = nullptr;
  cudaMalloc(&buf, std::max<size_t>(work, 16));
  CK(csrqrsvBatched(dummy, h, m, n, nnz, D, dval, doff, dcol, db, dx, batch, info, buf));
  const auto x = download(dx, (size_t)n * batch);
  double ne = 0;
  for (int b = 0; b < batch; ++b) {
    std::vector<cdouble> r(m);
    for (int i = 0; i < m; ++i) {
      cdouble s = Ty<T>::get(bs[(size_t)b * m + i]);
      for (int j = 0; j < n; ++j) s -= dense[b][(size_t)i * n + j] * Ty<T>::get(x[(size_t)b * n + j]);
      r[i] = s;
    }
    for (int j = 0; j < n; ++j) {
      cdouble s = 0;
      for (int i = 0; i < m; ++i) s += std::conj(dense[b][(size_t)i * n + j]) * r[i];
      ne = std::max(ne, std::abs(s) / m);
    }
  }
  char line[160];
  std::snprintf(line, sizeof line, "%scsrqrsvBatched: least squares for each of %d systems (A^H r = 0)", Ty<T>::name,
                batch);
  check(ne < 10 * Ty<T>::tol, line, ne);
  cusolverSpDestroyCsrqrInfo(info);
  for (void* p : {(void*)doff, (void*)dcol, (void*)dval, (void*)db, (void*)dx, buf}) cudaFree(p);
}

int main() {
  if (cusolverSpCreate(&h) || cusparseCreateMatDescr(&D)) {
    std::printf("FAIL: cusolverSpCreate\n");
    return 1;
  }
  linear_solves<float>();
  linear_solves<double>();
  linear_solves<cuComplex>();
  linear_solves<cuDoubleComplex>();
  singularity();
  least_squares<float>();
  least_squares<double>();
  least_squares<cuComplex>();
  least_squares<cuDoubleComplex>();
  shift_inverse<float>();
  shift_inverse<double>();
  shift_inverse<cuDoubleComplex>();
  eigs();
  structure();
  batched_qr<double>();
  batched_qr<cuComplex>();
  cusparseDestroyMatDescr(D);
  cusolverSpDestroy(h);
  std::printf(failures ? "FAIL: %d sparse solver checks\n" : "PASS: every sparse solver check\n", failures);
  return failures ? 1 : 0;
}
