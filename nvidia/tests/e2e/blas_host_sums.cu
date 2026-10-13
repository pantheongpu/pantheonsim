// The matrix routines of cuBLAS that compute their sums of products on the host's vector kernels (gemm, herk, syrk,
// her2k, syr2k, hemm, symm and trmm in single and double precision, real and complex), against the loops they replaced.
//
// The loops are std::complex<double> arithmetic, rounded to the caller's precision on the way out (a real routine is
// the same with a zero imaginary part). The kernels give those loops' bits when no operand and no sum is an infinity
// or a NaN, and the routine runs the loops themselves otherwise (a complex product with an infinity in it can come out
// NaN in both parts, which the complex multiply repairs by a rule of its own). So this program compares every result
// bit for bit -- two NaNs count as equal, since which one wins when two meet is the compiler's choice of operand order
// -- over ordinary values, zeros of both signs, subnormals, operands that hold an infinity or a NaN, and values
// that are finite but overflow in the sums, with leading dimensions that pad, every operation and fill mode.
// Not run against the card: the card's order of summation is its own; this program pins this library to itself.
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>

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

// ---- element types through std::complex<double> ----
static cd C(float v) { return {v, 0.0}; }
static cd C(double v) { return {v, 0.0}; }
static cd C(cuComplex v) { return {v.x, v.y}; }
static cd C(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> static T to(cd v);
template <> float to<float>(cd v) { return (float)v.real(); }
template <> double to<double>(cd v) { return v.real(); }
template <> cuComplex to<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> cuDoubleComplex to<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <class T> static constexpr bool is_cplx = std::is_same_v<T, cuComplex> || std::is_same_v<T, cuDoubleComplex>;
template <class T> using Scalar = std::conditional_t<std::is_same_v<T, cuComplex> || std::is_same_v<T, float>, float, double>;

// Bit equality, two NaNs equal.
template <class S> static bool same_part(S x, S y) { return std::memcmp(&x, &y, sizeof(S)) == 0 || (std::isnan(x) && std::isnan(y)); }
static bool same(float a, float b) { return same_part(a, b); }
static bool same(double a, double b) { return same_part(a, b); }
static bool same(cuComplex a, cuComplex b) { return same_part(a.x, b.x) && same_part(a.y, b.y); }
static bool same(cuDoubleComplex a, cuDoubleComplex b) { return same_part(a.x, b.x) && same_part(a.y, b.y); }
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

// ---- operand generation ----
enum class Mix { Plain, Zeros, Subnormal, InfNan, Overflow };
static const char* mix_name(Mix m) {
  switch (m) {
    case Mix::Plain: return "plain";
    case Mix::Zeros: return "zeros";
    case Mix::Subnormal: return "subnormal";
    case Mix::InfNan: return "inf/nan";
    default: return "overflow";
  }
}
template <class T> static std::vector<T> gen(size_t n, Mix mix, std::mt19937_64& rng) {
  using S = Scalar<T>;
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  std::vector<T> v(n);
  auto one = [&]() -> double {
    switch (mix) {
      case Mix::Plain: return u(rng);
      case Mix::Zeros: return rng() % 3 == 0 ? (rng() % 2 ? 0.0 : -0.0) : u(rng);
      case Mix::Subnormal: {
        if (rng() % 4) return u(rng);
        const double sub = std::numeric_limits<S>::denorm_min() * (double)(1 + rng() % 1000);
        return rng() % 2 ? sub : -sub;
      }
      case Mix::InfNan: {
        const uint64_t r = rng() % 90;
        if (r == 0) return std::numeric_limits<double>::infinity();
        if (r == 1) return -std::numeric_limits<double>::infinity();
        if (r == 2) return std::numeric_limits<double>::quiet_NaN();
        return u(rng);
      }
      default: {  // finite, but in double precision the sums overflow (single precision keeps its range in double)
        if (rng() % 6 == 0) {
          const double big = std::is_same_v<S, float> ? 3.0e38 : 1.0e300;
          return rng() % 2 ? big : -big;
        }
        return u(rng);
      }
    }
  };
  for (auto& x : v) x = to<T>(cd(one(), is_cplx<T> ? one() : 0.0));
  return v;
}

// ---- the reference: the loops, in std::complex<double> ----
template <class T> struct Mat {
  std::vector<cd> v;
  int ld;
  cd& operator()(int r, int c) { return v[(size_t)c * ld + r]; }
  cd operator()(int r, int c) const { return v[(size_t)c * ld + r]; }
};
template <class T> static Mat<T> load(const std::vector<T>& h, int ld) {
  Mat<T> m;
  m.ld = ld;
  m.v.resize(h.size());
  for (size_t i = 0; i < h.size(); ++i) m.v[i] = C(h[i]);
  return m;
}
template <class T> static std::vector<T> store(const Mat<T>& m) {
  std::vector<T> h(m.v.size());
  for (size_t i = 0; i < h.size(); ++i) h[i] = to<T>(m.v[i]);
  return h;
}
enum Op { N, Tr, Cj };
template <class M> static cd el(const M& a, Op op, int i, int j) { return op == N ? a(i, j) : (op == Cj ? std::conj(a(j, i)) : a(j, i)); }

template <class T> static void ref_gemm(Op oa, Op ob, int m, int n, int k, cd alpha, const Mat<T>& a, const Mat<T>& b, cd beta, Mat<T>& c) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      cd s = 0;
      for (int p = 0; p < k; ++p) s += el(a, oa, i, p) * el(b, ob, p, j);
      c(i, j) = beta == cd(0) ? alpha * s : alpha * s + beta * c(i, j);
    }
}
// herk / syrk
template <class T> static void ref_rk(bool herm, bool lower, bool trans, int n, int k, cd alpha, const Mat<T>& a, cd beta, Mat<T>& c) {
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd s = 0;
      for (int p = 0; p < k; ++p) {
        const cd x = trans ? (herm ? std::conj(a(p, i)) : a(p, i)) : a(i, p);
        const cd y = trans ? a(p, j) : (herm ? std::conj(a(j, p)) : a(j, p));
        s += x * y;
      }
      c(i, j) = beta == cd(0) ? alpha * s : alpha * s + beta * c(i, j);
      if (herm && i == j) c(i, j) = c(i, j).real();
    }
}
template <class T> static void ref_r2k(bool herm, bool lower, bool trans, int n, int k, cd alpha, const Mat<T>& a, const Mat<T>& b, cd beta, Mat<T>& c) {
  auto op = [&](cd v) { return herm ? std::conj(v) : v; };
  auto elm = [&](const Mat<T>& m, int i, int p) { return trans ? m(p, i) : m(i, p); };
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd s1 = 0, s2 = 0;
      for (int p = 0; p < k; ++p) {
        if (trans && herm) {
          s1 += std::conj(elm(a, i, p)) * elm(b, j, p);
          s2 += std::conj(elm(b, i, p)) * elm(a, j, p);
        } else {
          s1 += elm(a, i, p) * op(elm(b, j, p));
          s2 += elm(b, i, p) * op(elm(a, j, p));
        }
      }
      const cd v = alpha * s1 + op(alpha) * s2;
      c(i, j) = beta == cd(0) ? v : v + beta * c(i, j);
      if (herm && i == j) c(i, j) = c(i, j).real();
    }
}
template <class T> static Mat<T> expand(const Mat<T>& a, int n, bool lower, bool herm) {
  Mat<T> f;
  f.ld = n;
  f.v.resize((size_t)n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      const bool stored = lower ? i >= j : i <= j;
      f(i, j) = stored ? a(i, j) : (herm ? std::conj(a(j, i)) : a(j, i));
      if (herm && i == j) f(i, j) = f(i, j).real();
    }
  return f;
}
template <class T> static void ref_hemm(bool herm, bool left, bool lower, int m, int n, cd alpha, const Mat<T>& a0, const Mat<T>& b, cd beta, Mat<T>& c) {
  const int ka = left ? m : n;
  const Mat<T> a = expand(a0, ka, lower, herm);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      cd s = 0;
      if (left) for (int p = 0; p < m; ++p) s += a(i, p) * b(p, j);
      else for (int p = 0; p < n; ++p) s += b(i, p) * a(p, j);
      c(i, j) = beta == cd(0) ? alpha * s : alpha * s + beta * c(i, j);
    }
}
template <class T> static void ref_trmm(bool left, bool lower, Op op, bool unit, int m, int n, cd alpha, const Mat<T>& a, const Mat<T>& b, Mat<T>& c) {
  auto tri = [&](int i, int j) -> cd {
    const int r = op == N ? i : j, cc = op == N ? j : i;
    if (r == cc && unit) return 1;
    if (lower ? cc > r : cc < r) return 0;
    return op == Cj ? std::conj(a(r, cc)) : a(r, cc);
  };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      cd s = 0;
      if (left) for (int p = 0; p < m; ++p) s += tri(i, p) * b(p, j);
      else for (int p = 0; p < n; ++p) s += b(i, p) * tri(p, j);
      c(i, j) = alpha * s;
    }
}

// ---- the library calls, one overload per type ----
#define GEMM_CALL(T, f) static cublasStatus_t gemm_call(cublasHandle_t h, cublasOperation_t a, cublasOperation_t b, int m, int n, int k, const T* al, const T* A, int lda, const T* B, int ldb, const T* be, T* Cm, int ldc) { return f(h, a, b, m, n, k, al, A, lda, B, ldb, be, Cm, ldc); }
GEMM_CALL(cuComplex, cublasCgemm)
GEMM_CALL(cuDoubleComplex, cublasZgemm)
#define RK_CALL(T, S, f) static cublasStatus_t rk_call(cublasHandle_t h, bool herm, cublasFillMode_t u, cublasOperation_t t, int n, int k, const T* al, const T* A, int lda, const T* be, T* Cm, int ldc) { (void)herm; return f(h, u, t, n, k, al, A, lda, be, Cm, ldc); }
RK_CALL(float, float, cublasSsyrk)
RK_CALL(double, double, cublasDsyrk)
static cublasStatus_t rk_call(cublasHandle_t h, bool herm, cublasFillMode_t u, cublasOperation_t t, int n, int k, const cuComplex* al, const cuComplex* A, int lda, const cuComplex* be, cuComplex* Cm, int ldc) {
  if (herm) { const float a = al->x, b = be->x; return cublasCherk(h, u, t, n, k, &a, A, lda, &b, Cm, ldc); }
  return cublasCsyrk(h, u, t, n, k, al, A, lda, be, Cm, ldc);
}
static cublasStatus_t rk_call(cublasHandle_t h, bool herm, cublasFillMode_t u, cublasOperation_t t, int n, int k, const cuDoubleComplex* al, const cuDoubleComplex* A, int lda, const cuDoubleComplex* be, cuDoubleComplex* Cm, int ldc) {
  if (herm) { const double a = al->x, b = be->x; return cublasZherk(h, u, t, n, k, &a, A, lda, &b, Cm, ldc); }
  return cublasZsyrk(h, u, t, n, k, al, A, lda, be, Cm, ldc);
}
static cublasStatus_t r2k_call(cublasHandle_t h, bool, cublasFillMode_t u, cublasOperation_t t, int n, int k, const float* al, const float* A, int lda, const float* B, int ldb, const float* be, float* Cm, int ldc) { return cublasSsyr2k(h, u, t, n, k, al, A, lda, B, ldb, be, Cm, ldc); }
static cublasStatus_t r2k_call(cublasHandle_t h, bool, cublasFillMode_t u, cublasOperation_t t, int n, int k, const double* al, const double* A, int lda, const double* B, int ldb, const double* be, double* Cm, int ldc) { return cublasDsyr2k(h, u, t, n, k, al, A, lda, B, ldb, be, Cm, ldc); }
static cublasStatus_t r2k_call(cublasHandle_t h, bool herm, cublasFillMode_t u, cublasOperation_t t, int n, int k, const cuComplex* al, const cuComplex* A, int lda, const cuComplex* B, int ldb, const cuComplex* be, cuComplex* Cm, int ldc) {
  if (herm) { const float b = be->x; return cublasCher2k(h, u, t, n, k, al, A, lda, B, ldb, &b, Cm, ldc); }
  return cublasCsyr2k(h, u, t, n, k, al, A, lda, B, ldb, be, Cm, ldc);
}
static cublasStatus_t r2k_call(cublasHandle_t h, bool herm, cublasFillMode_t u, cublasOperation_t t, int n, int k, const cuDoubleComplex* al, const cuDoubleComplex* A, int lda, const cuDoubleComplex* B, int ldb, const cuDoubleComplex* be, cuDoubleComplex* Cm, int ldc) {
  if (herm) { const double b = be->x; return cublasZher2k(h, u, t, n, k, al, A, lda, B, ldb, &b, Cm, ldc); }
  return cublasZsyr2k(h, u, t, n, k, al, A, lda, B, ldb, be, Cm, ldc);
}
#define MM_CALL(T, sym, her) \
  static cublasStatus_t mm_call(cublasHandle_t h, bool herm, cublasSideMode_t s, cublasFillMode_t u, int m, int n, const T* al, const T* A, int lda, const T* B, int ldb, const T* be, T* Cm, int ldc) { return herm ? her(h, s, u, m, n, al, A, lda, B, ldb, be, Cm, ldc) : sym(h, s, u, m, n, al, A, lda, B, ldb, be, Cm, ldc); }
MM_CALL(cuComplex, cublasCsymm, cublasChemm)
MM_CALL(cuDoubleComplex, cublasZsymm, cublasZhemm)
static cublasStatus_t mm_call(cublasHandle_t h, bool, cublasSideMode_t s, cublasFillMode_t u, int m, int n, const float* al, const float* A, int lda, const float* B, int ldb, const float* be, float* Cm, int ldc) { return cublasSsymm(h, s, u, m, n, al, A, lda, B, ldb, be, Cm, ldc); }
static cublasStatus_t mm_call(cublasHandle_t h, bool, cublasSideMode_t s, cublasFillMode_t u, int m, int n, const double* al, const double* A, int lda, const double* B, int ldb, const double* be, double* Cm, int ldc) { return cublasDsymm(h, s, u, m, n, al, A, lda, B, ldb, be, Cm, ldc); }
#define TRMM_CALL(T, f) static cublasStatus_t trmm_call(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int m, int n, const T* al, const T* A, int lda, const T* B, int ldb, T* Cm, int ldc) { return f(h, s, u, t, d, m, n, al, A, lda, B, ldb, Cm, ldc); }
TRMM_CALL(float, cublasStrmm)
TRMM_CALL(double, cublasDtrmm)
TRMM_CALL(cuComplex, cublasCtrmm)
TRMM_CALL(cuDoubleComplex, cublasZtrmm)

static cublasOperation_t cop(Op o) { return o == N ? CUBLAS_OP_N : o == Tr ? CUBLAS_OP_T : CUBLAS_OP_C; }

template <class T> struct Scalars {
  T al, be;
};

// A scalar pair for case number q: ordinary, beta zero (C write-only), awkward values.
template <class T> static Scalars<T> scalars(int q, bool real_beta = false) {
  const cd als[] = {{1.0, 0.0}, {-0.5, 0.25}, {2.5, -1.5}, {0.0, 0.0}, {0.75, 0.0}};
  const cd bes[] = {{0.0, 0.0}, {1.0, 0.0}, {-1.25, 0.5}, {0.5, 0.0}};
  cd a = als[q % 5], b = bes[(q / 2) % 4];
  if (!is_cplx<T>) { a = a.real(); b = b.real(); }
  if (real_beta) b = b.real();
  return {to<T>(a), to<T>(b)};
}

struct Shape {
  int m, n, k;
};

static void check_level3(cublasHandle_t h) {
  std::mt19937_64 rng(2024);
  const Shape shapes[] = {{37, 29, 23}, {1, 40, 90}, {64, 5, 50}, {33, 33, 33}, {20, 65, 17}, {70, 70, 3}, {9, 9, 120}};
  int run = 0;
  auto each_type = [&](auto tag) {
    using T = decltype(tag);
    const char* tn = std::is_same_v<T, float> ? "S" : std::is_same_v<T, double> ? "D" : std::is_same_v<T, cuComplex> ? "C" : "Z";
    char what[200];
    for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::InfNan, Mix::Overflow})
      for (const Shape& s : shapes) {
        const int pad = run % 3;
        ++run;
        const auto sc = scalars<T>(run);
        const cd alpha = C(sc.al), beta = C(sc.be);
        // gemm (complex types)
        if constexpr (is_cplx<T>) {
          for (const Op oa : {N, Tr, Cj})
            for (const Op ob : {N, Cj}) {
              const int lda = (oa == N ? s.m : s.k) + pad, ldb = (ob == N ? s.k : s.n) + pad, ldc = s.m + pad;
              const auto A = gen<T>((size_t)lda * (oa == N ? s.k : s.m), mix, rng), B = gen<T>((size_t)ldb * (ob == N ? s.n : s.k), mix, rng);
              const auto C0 = gen<T>((size_t)ldc * s.n, mix, rng);
              auto ref = load(C0, ldc);
              ref_gemm(oa, ob, s.m, s.n, s.k, alpha, load(A, lda), load(B, ldb), beta, ref);
              T *dA = dev(A), *dB = dev(B), *dC = dev(C0);
              const cublasStatus_t st = gemm_call(h, cop(oa), cop(ob), s.m, s.n, s.k, &sc.al, dA, lda, dB, ldb, &sc.be, dC, ldc);
              std::snprintf(what, sizeof what, "%sgemm op %d%d %dx%dx%d pad %d mix %s", tn, oa, ob, s.m, s.n, s.k, pad, mix_name(mix));
              report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dC, C0.size()), store(ref)), what);
              cudaFree(dA); cudaFree(dB); cudaFree(dC);
            }
        }
        // syrk and herk, rank-2k
        for (const bool herm : {false, true}) {
          if (herm && !is_cplx<T>) continue;
          for (const bool lower : {false, true})
            for (const Op t : {N, Tr, Cj}) {
              if (!herm && t == Cj && is_cplx<T>) continue;   // syrk takes N and T for complex types
              if (herm && t == Tr) continue;                   // herk takes N and C
              const int n = s.m, k = s.k;
              const int lda = (t == N ? n : k) + pad, ldc = n + pad;
              const auto A = gen<T>((size_t)lda * (t == N ? k : n), mix, rng), B = gen<T>((size_t)lda * (t == N ? k : n), mix, rng);
              const auto C0 = gen<T>((size_t)ldc * n, mix, rng);
              auto scl = sc;   // herk takes real scalars
              if (herm) { scl.al = to<T>(cd(C(sc.al).real(), 0)); scl.be = to<T>(cd(C(sc.be).real(), 0)); }
              auto ref = load(C0, ldc);
              ref_rk<T>(herm, lower, t != N, n, k, C(scl.al), load(A, lda), C(scl.be), ref);
              T *dA = dev(A), *dB = dev(B), *dC = dev(C0);
              const cublasFillMode_t u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
              cublasStatus_t st = rk_call(h, herm, u, cop(t), n, k, &scl.al, dA, lda, &scl.be, dC, ldc);
              std::snprintf(what, sizeof what, "%s%srk %s op %d n %d k %d pad %d mix %s", tn, herm ? "he" : "sy", lower ? "lower" : "upper", t, n, k, pad, mix_name(mix));
              report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dC, C0.size()), store(ref)), what);
              // 2k
              const T b2 = scl.be;   // her2k: alpha complex, beta real
              auto ref2 = load(C0, ldc);
              ref_r2k<T>(herm, lower, t != N, n, k, C(sc.al), load(A, lda), load(B, lda), C(b2), ref2);
              cudaMemcpy(dC, C0.data(), C0.size() * sizeof(T), cudaMemcpyHostToDevice);
              st = r2k_call(h, herm, u, cop(t), n, k, &sc.al, dA, lda, dB, lda, &b2, dC, ldc);
              std::snprintf(what, sizeof what, "%s%sr2k %s op %d n %d k %d pad %d mix %s", tn, herm ? "he" : "sy", lower ? "lower" : "upper", t, n, k, pad, mix_name(mix));
              report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dC, C0.size()), store(ref2)), what);
              cudaFree(dA); cudaFree(dB); cudaFree(dC);
            }
        }
        // hemm / symm
        for (const bool herm : {false, true}) {
          if (herm && !is_cplx<T>) continue;
          for (const bool left : {false, true})
            for (const bool lower : {false, true}) {
              const int ka = left ? s.m : s.n, lda = ka + pad, ldb = s.m + pad, ldc = s.m + pad;
              const auto A = gen<T>((size_t)lda * ka, mix, rng), B = gen<T>((size_t)ldb * s.n, mix, rng), C0 = gen<T>((size_t)ldc * s.n, mix, rng);
              auto ref = load(C0, ldc);
              ref_hemm<T>(herm, left, lower, s.m, s.n, alpha, load(A, lda), load(B, ldb), beta, ref);
              T *dA = dev(A), *dB = dev(B), *dC = dev(C0);
              const cublasStatus_t st = mm_call(h, herm, left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, s.m, s.n, &sc.al, dA, lda, dB, ldb, &sc.be, dC, ldc);
              std::snprintf(what, sizeof what, "%s%smm %s %s %dx%d pad %d mix %s", tn, herm ? "he" : "sy", left ? "left" : "right", lower ? "lower" : "upper", s.m, s.n, pad, mix_name(mix));
              report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dC, C0.size()), store(ref)), what);
              cudaFree(dA); cudaFree(dB); cudaFree(dC);
            }
        }
        // trmm
        for (const bool left : {false, true})
          for (const bool lower : {false, true})
            for (const Op t : {N, Tr, Cj})
              for (const bool unit : {false, true}) {
                if ((run + (int)left + 2 * (int)lower + 4 * (int)t + 8 * (int)unit) % 3) continue;   // a third of the combinations per shape
                const int ka = left ? s.m : s.n, lda = ka + pad, ldb = s.m + pad, ldc = s.m + pad;
                const auto A = gen<T>((size_t)lda * ka, mix, rng), B = gen<T>((size_t)ldb * s.n, mix, rng), C0 = gen<T>((size_t)ldc * s.n, mix, rng);
                auto ref = load(C0, ldc);
                ref_trmm<T>(left, lower, t, unit, s.m, s.n, alpha, load(A, lda), load(B, ldb), ref);
                T *dA = dev(A), *dB = dev(B), *dC = dev(C0);
                const cublasStatus_t st = trmm_call(h, left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, cop(t), unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, s.m, s.n, &sc.al, dA, lda, dB, ldb, dC, ldc);
                std::snprintf(what, sizeof what, "%strmm %s %s op %d %s %dx%d pad %d mix %s", tn, left ? "left" : "right", lower ? "lower" : "upper", t, unit ? "unit" : "nonunit", s.m, s.n, pad, mix_name(mix));
                report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dC, C0.size()), store(ref)), what);
                cudaFree(dA); cudaFree(dB); cudaFree(dC);
              }
      }
  };
  each_type(float{});
  each_type(double{});
  each_type(cuComplex{});
  each_type(cuDoubleComplex{});
}

// ---- trsm: the loops as they were (real: cublas_api.cpp's tri_solve, complex: complex_linalg.hpp's) ----
// One vector x of length k, M(i, j) given by `m`, the solve running forward (span [0, i)) or backward (span (i, k)).
// X is std::complex<double> for the complex routines and double for the real ones (which never had an imaginary part).
template <class X, class MF> static void ref_solve(int k, bool forward, bool unit, MF m, X* x, size_t step) {
  auto row = [&](int i, int from, int to) {
    X s = x[(size_t)i * step];
    for (int j = from; j < to; ++j) s -= m(i, j) * x[(size_t)j * step];
    x[(size_t)i * step] = unit ? s : s / m(i, i);
  };
  if (forward) for (int i = 0; i < k; ++i) row(i, 0, i);
  else for (int i = k - 1; i >= 0; --i) row(i, i + 1, k);
}
template <class T> static void ref_trsm(bool left, bool lower, Op op, bool unit, int m, int n, cd alpha, const Mat<T>& a, Mat<T>& b) {
  using X = std::conditional_t<is_cplx<T>, cd, double>;
  auto cvt = [](cd v) -> X { if constexpr (is_cplx<T>) return v; else return v.real(); };
  auto conjx = [](X v) -> X { if constexpr (is_cplx<T>) return std::conj(v); else return v; };
  const int k = left ? m : n;
  std::vector<X> bx(b.v.size());
  for (size_t i = 0; i < bx.size(); ++i) bx[i] = cvt(b.v[i]);
  auto B = [&](int i, int j) -> X& { return bx[(size_t)j * b.ld + i]; };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) B(i, j) *= cvt(alpha);
  // M = op'(A) for the solve of one vector: the left side solves op(A) x = b; the right side solves against the transpose,
  // conjugating a row around a solve with A itself when op is C.
  auto solve = [&](Op sop, bool conj_row, X* x, size_t step) {
    auto M = [&](int i, int j) -> X { return cvt(sop == N ? a(i, j) : (sop == Cj ? std::conj(a(j, i)) : a(j, i))); };
    if (conj_row) for (int i = 0; i < k; ++i) x[i * step] = conjx(x[i * step]);
    ref_solve<X>(k, lower == (sop == N), unit, M, x, step);
    if (conj_row) for (int i = 0; i < k; ++i) x[i * step] = conjx(x[i * step]);
  };
  if (left) {
    for (int j = 0; j < n; ++j) solve(op, false, &B(0, j), 1);
  } else {
    for (int i = 0; i < m; ++i) {
      if (op == Cj) solve(N, true, &B(i, 0), (size_t)b.ld);
      else solve(op == N ? Tr : N, false, &B(i, 0), (size_t)b.ld);
    }
  }
  for (size_t i = 0; i < bx.size(); ++i) b.v[i] = cd(bx[i]);
}
#define TRSM_CALL(T, f) static cublasStatus_t trsm_call(cublasHandle_t h, cublasSideMode_t s, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int m, int n, const T* al, const T* A, int lda, T* B, int ldb) { return f(h, s, u, t, d, m, n, al, A, lda, B, ldb); }
TRSM_CALL(float, cublasStrsm)
TRSM_CALL(double, cublasDtrsm)
TRSM_CALL(cuComplex, cublasCtrsm)
TRSM_CALL(cuDoubleComplex, cublasZtrsm)

static void check_trsm(cublasHandle_t h) {
  std::mt19937_64 rng(555);
  const Shape shapes[] = {{37, 29, 0}, {1, 40, 0}, {64, 5, 0}, {33, 33, 0}, {20, 65, 0}, {9, 120, 0}};
  int run = 0;
  auto each_type = [&](auto tag) {
    using T = decltype(tag);
    const char* tn = std::is_same_v<T, float> ? "S" : std::is_same_v<T, double> ? "D" : std::is_same_v<T, cuComplex> ? "C" : "Z";
    char what[200];
    for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::InfNan, Mix::Overflow})
      for (const Shape& s : shapes)
        for (const bool left : {false, true})
          for (const bool lower : {false, true})
            for (const Op t : {N, Tr, Cj}) {
              if (!is_cplx<T> && t == Cj && (run % 2)) { ++run; continue; }
              const bool unit = (run++ % 3) == 0;
              const int pad = run % 3, ka = left ? s.m : s.n, lda = ka + pad, ldb = s.m + pad;
              const auto sc = scalars<T>(run);
              auto A = gen<T>((size_t)lda * ka, mix, rng);
              // keep the diagonal away from zero (and the plain mixes finite) so the solves are not all NaN
              if (mix != Mix::InfNan && mix != Mix::Overflow)
                for (int i = 0; i < ka; ++i) A[(size_t)i * lda + i] = to<T>(cd(2.0 + (i % 5) * 0.25, is_cplx<T> ? 0.5 : 0.0));
              const auto B0 = gen<T>((size_t)ldb * s.n, mix, rng);
              auto ref = load(B0, ldb);
              ref_trsm<T>(left, lower, t, unit, s.m, s.n, C(sc.al), load(A, lda), ref);
              T *dA = dev(A), *dB = dev(B0);
              const cublasStatus_t st = trsm_call(h, left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER, cop(t), unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT, s.m, s.n, &sc.al, dA, lda, dB, ldb);
              std::snprintf(what, sizeof what, "%strsm %s %s op %d %s %dx%d pad %d mix %s", tn, left ? "left" : "right", lower ? "lower" : "upper", t, unit ? "unit" : "nonunit", s.m, s.n, pad, mix_name(mix));
              report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dB, B0.size()), store(ref)), what);
              cudaFree(dA); cudaFree(dB);
            }
  };
  each_type(float{});
  each_type(double{});
  each_type(cuComplex{});
  each_type(cuDoubleComplex{});
}

// ---- gemv (real): y = alpha op(A) x + beta y, each y[i] one chain in the type's own arithmetic ----
template <class T> static cublasStatus_t gemv_call(cublasHandle_t h, cublasOperation_t t, int m, int n, const T* al, const T* A, int lda, const T* x, int incx, const T* be, T* y, int incy);
template <> cublasStatus_t gemv_call<float>(cublasHandle_t h, cublasOperation_t t, int m, int n, const float* al, const float* A, int lda, const float* x, int incx, const float* be, float* y, int incy) { return cublasSgemv(h, t, m, n, al, A, lda, x, incx, be, y, incy); }
template <> cublasStatus_t gemv_call<double>(cublasHandle_t h, cublasOperation_t t, int m, int n, const double* al, const double* A, int lda, const double* x, int incx, const double* be, double* y, int incy) { return cublasDgemv(h, t, m, n, al, A, lda, x, incx, be, y, incy); }

static size_t velem(int i, int n, int inc) { return inc > 0 ? (size_t)i * inc : (size_t)(n - 1 - i) * -inc; }

template <class T> static void check_gemv_type(cublasHandle_t h, const char* tn) {
  std::mt19937_64 rng(88);
  char what[200];
  for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Subnormal, Mix::InfNan, Mix::Overflow})
    for (const cublasOperation_t t : {CUBLAS_OP_N, CUBLAS_OP_T})
      for (const int incx : {1, -2, 3})
        for (const int incy : {1, -1, 2}) {
          const int m = 70 + (int)(rng() % 9), n = 53 + (int)(rng() % 7), lda = m + (int)(rng() % 3);
          const int xlen = t == CUBLAS_OP_N ? n : m, ylen = t == CUBLAS_OP_N ? m : n;
          const auto A = gen<T>((size_t)lda * n, mix, rng);
          const auto x = gen<T>((size_t)(xlen - 1) * std::abs(incx) + 1, mix, rng), y0 = gen<T>((size_t)(ylen - 1) * std::abs(incy) + 1, Mix::Plain, rng);
          for (const int q : {0, 1, 2}) {
            const T al = q == 0 ? T(1) : T(0.75), be = q == 2 ? T(-0.5) : (q == 1 ? T(1) : T(0));
            std::vector<T> want = y0;
            for (int i = 0; i < ylen; ++i) {
              T acc = 0;
              for (int j = 0; j < xlen; ++j) acc += A[t == CUBLAS_OP_N ? (size_t)i + (size_t)j * lda : (size_t)j + (size_t)i * lda] * x[velem(j, xlen, incx)];
              T& yi = want[velem(i, ylen, incy)];
              yi = be == T(0) ? al * acc : al * acc + be * yi;
            }
            T *dA = dev(A), *dx = dev(x), *dy = dev(y0);
            const cublasStatus_t st = gemv_call<T>(h, t, m, n, &al, dA, lda, dx, incx, &be, dy, incy);
            std::snprintf(what, sizeof what, "%sgemv op %d %dx%d incx %d incy %d beta %g mix %s", tn, (int)t, m, n, incx, incy, (double)be, mix_name(mix));
            report(st == CUBLAS_STATUS_SUCCESS && same_all(host(dy, y0.size()), want), what);
            cudaFree(dA); cudaFree(dx); cudaFree(dy);
          }
        }
}

static void check_gemv(cublasHandle_t h) {
  check_gemv_type<float>(h, "S");
  check_gemv_type<double>(h, "D");
}

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL cublasCreate\n");
    return 1;
  }
  check_level3(h);
  check_trsm(h);
  check_gemv(h);
  cublasDestroy(h);
  std::printf("%d checks, %d failed\n", checks, failures);
  std::printf(failures ? "FAIL\n" : "PASS\n");
  return failures ? 1 : 0;
}
