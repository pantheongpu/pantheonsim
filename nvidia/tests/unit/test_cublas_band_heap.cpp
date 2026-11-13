// The band and packed matrix-vector routines of cuBLAS (cublas_packed.inc: tbmv, tpmv, gbmv, sbmv/hbmv, spmv/hpmv) read
// the band where it is stored and hold no full n x n copy of it (#551). They used to build that copy on every call: SCS
// calls Dtbmv (a diagonal scaling, k = 0) on a vector of 1500 a few dozen times a second, and each call mapped 34 MiB,
// took 8,000 page faults on it and summed 2.25 million products, almost all of them zero; the solve of max_ent took
// 1432 s on GitHub's runners and is 12 s without the copy.
//
// Two things are checked: the heap one call holds is about the band, not n^2; and every output is bit for bit what the
// loops over the full matrix gave (copied below as the reference), over operands with signed zeros, negative values and
// sums whose rounding depends on their order, for every fill mode, operation, diagonal and band width, with an infinity
// in x too (the full loops make NaN of the zeros it meets, and so must the routines).
#include <cublas_v2.h>
#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

#include "alloc_counter.hpp"
#include "vtest.hpp"

namespace {

using cd = std::complex<double>;
enum class Op { N, T, C };

// ---- the reference: the loops over the full matrix, as cublas_packed.inc had them ----

struct Full {
  std::vector<cd> v;
  int ld;
  cd& operator()(int r, int c) { return v[(size_t)c * ld + r]; }
  cd operator()(int r, int c) const { return v[(size_t)c * ld + r]; }
};
cd opel(const Full& a, Op op, int i, int j) {
  if (op == Op::N) return a(i, j);
  return op == Op::C ? std::conj(a(j, i)) : a(j, i);
}
size_t packed_at(bool lower, int n, int i, int j) {
  return lower ? (size_t)i + ((size_t)(2 * n - j - 1) * j) / 2 : (size_t)i + ((size_t)j * (j + 1)) / 2;
}
Full band_full(const std::vector<cd>& band, int lda, int n, int k, bool lower, bool mirror, bool hermitian) {
  Full f{std::vector<cd>((size_t)n * n, 0), n};
  for (int j = 0; j < n; ++j) {
    const int lo = lower ? j : std::max(0, j - k), hi = lower ? std::min(n - 1, j + k) : j;
    for (int i = lo; i <= hi; ++i) {
      cd v = band[(size_t)j * lda + (lower ? i - j : k + i - j)];
      if (hermitian && i == j) v = v.real();
      f(i, j) = v;
      if (mirror && i != j) f(j, i) = hermitian ? std::conj(v) : v;
    }
  }
  return f;
}
Full packed_full(const std::vector<cd>& ap, int n, bool lower, bool mirror, bool hermitian) {
  Full f{std::vector<cd>((size_t)n * n, 0), n};
  for (int j = 0; j < n; ++j)
    for (int i = lower ? j : 0; i < (lower ? n : j + 1); ++i) {
      cd v = ap[packed_at(lower, n, i, j)];
      if (hermitian && i == j) v = v.real();
      f(i, j) = v;
      if (mirror && i != j) f(j, i) = hermitian ? std::conj(v) : v;
    }
  return f;
}
std::vector<cd> ref_mv(const Full& M, Op op, int rows, int cols, cd alpha, const std::vector<cd>& xv, cd beta,
                       std::vector<cd> yv) {
  const int ylen = op == Op::N ? rows : cols, xlen = op == Op::N ? cols : rows;
  if (beta == cd(0)) yv.assign(ylen, 0);
  for (int i = 0; i < ylen; ++i) {
    cd s = 0;
    for (int j = 0; j < xlen; ++j) s += opel(M, op, i, j) * xv[j];
    yv[i] = beta == cd(0) ? alpha * s : alpha * s + beta * yv[i];
  }
  return yv;
}
std::vector<cd> ref_trmv(const Full& a, Op op, bool unit, int n, const std::vector<cd>& xv) {
  std::vector<cd> out(n, 0);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) out[i] += (i == j && unit) ? xv[j] : opel(a, op, i, j) * xv[j];
  return out;
}

// ---- operands ----

// Values whose sums round differently in different orders, with signed zeros and negatives among them.
double pick(std::mt19937_64& rng) {
  static const double fixed[] = {0.0, -0.0, 1.0, -1.0, 1e-17, -3.0};
  const uint64_t r = rng();
  if (r % 5 == 0) return fixed[(r >> 8) % 6];
  return std::ldexp(static_cast<double>(static_cast<int64_t>(r >> 11) % 1000003) / 7.0, static_cast<int>((r >> 3) % 9) - 4);
}
template <class T> T make(std::mt19937_64& rng);
template <> double make<double>(std::mt19937_64& rng) { return pick(rng); }
template <> cuDoubleComplex make<cuDoubleComplex>(std::mt19937_64& rng) {
  const double re = pick(rng);
  return make_cuDoubleComplex(re, pick(rng));
}
cd to_cd(double v) { return cd(v, 0); }
cd to_cd(cuDoubleComplex v) { return cd(v.x, v.y); }
template <class T> std::vector<T> rand_vec(std::mt19937_64& rng, size_t n) {
  std::vector<T> v(n);
  for (auto& e : v) e = make<T>(rng);
  return v;
}
template <class T> std::vector<cd> as_cd(const std::vector<T>& v) {
  std::vector<cd> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = to_cd(v[i]);
  return o;
}
// The logical vector of a strided one (a negative increment starts at the end), and back.
template <class T> std::vector<cd> gather(const std::vector<T>& raw, int n, int inc) {
  const int a = std::abs(inc);
  std::vector<cd> o(n);
  for (int i = 0; i < n; ++i) o[i] = to_cd(raw[(size_t)(inc > 0 ? i : n - 1 - i) * a]);
  return o;
}
// Bit equality of what the routine stored and the reference's answer rounded into T; two NaNs count as equal.
bool same(double got, cd want) {
  const double w = want.real();
  return std::memcmp(&got, &w, 8) == 0 || (std::isnan(got) && std::isnan(w));
}
bool same(cuDoubleComplex got, cd want) { return same(got.x, cd(want.real())) && same(got.y, cd(want.imag())); }
template <class T> bool same_vec(const std::vector<T>& raw, int n, int inc, const std::vector<cd>& want) {
  const int a = std::abs(inc);
  for (int i = 0; i < n; ++i)
    if (!same(raw[(size_t)(inc > 0 ? i : n - 1 - i) * a], want[i])) return false;
  return true;
}

template <class T> struct Dev {
  T* p = nullptr;
  size_t n = 0;
  explicit Dev(const std::vector<T>& h) : n(h.size()) {
    cudaMalloc(reinterpret_cast<void**>(&p), std::max<size_t>(n, 1) * sizeof(T));
    cudaMemcpy(p, h.data(), n * sizeof(T), cudaMemcpyHostToDevice);
  }
  ~Dev() { cudaFree(p); }
  Dev(const Dev&) = delete;
  Dev& operator=(const Dev&) = delete;
  std::vector<T> get() const {
    std::vector<T> h(n);
    cudaMemcpy(h.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
    return h;
  }
};

cublasOperation_t cu(Op op) { return op == Op::N ? CUBLAS_OP_N : op == Op::T ? CUBLAS_OP_T : CUBLAS_OP_C; }
const char* name(Op op) { return op == Op::N ? "N" : op == Op::T ? "T" : "C"; }

cublasStatus_t tbmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n, int k,
                    const double* A, int lda, double* x, int incx) {
  return cublasDtbmv(h, u, t, d, n, k, A, lda, x, incx);
}
cublasStatus_t tbmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n, int k,
                    const cuDoubleComplex* A, int lda, cuDoubleComplex* x, int incx) {
  return cublasZtbmv(h, u, t, d, n, k, A, lda, x, incx);
}
cublasStatus_t tpmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n,
                    const double* AP, double* x, int incx) {
  return cublasDtpmv(h, u, t, d, n, AP, x, incx);
}
cublasStatus_t tpmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n,
                    const cuDoubleComplex* AP, cuDoubleComplex* x, int incx) {
  return cublasZtpmv(h, u, t, d, n, AP, x, incx);
}
cublasStatus_t gbmv(cublasHandle_t h, cublasOperation_t t, int m, int n, int kl, int ku, const double* al, const double* A,
                    int lda, const double* x, int incx, const double* be, double* y, int incy) {
  return cublasDgbmv(h, t, m, n, kl, ku, al, A, lda, x, incx, be, y, incy);
}
cublasStatus_t gbmv(cublasHandle_t h, cublasOperation_t t, int m, int n, int kl, int ku, const cuDoubleComplex* al,
                    const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, int incx, const cuDoubleComplex* be,
                    cuDoubleComplex* y, int incy) {
  return cublasZgbmv(h, t, m, n, kl, ku, al, A, lda, x, incx, be, y, incy);
}
// sbmv/spmv for double, hbmv/hpmv for complex (the routines the symmetric and Hermitian forms share).
cublasStatus_t symbmv(cublasHandle_t h, cublasFillMode_t u, int n, int k, const double* al, const double* A, int lda,
                      const double* x, const double* be, double* y) {
  return cublasDsbmv(h, u, n, k, al, A, lda, x, 1, be, y, 1);
}
cublasStatus_t symbmv(cublasHandle_t h, cublasFillMode_t u, int n, int k, const cuDoubleComplex* al,
                      const cuDoubleComplex* A, int lda, const cuDoubleComplex* x, const cuDoubleComplex* be,
                      cuDoubleComplex* y) {
  return cublasZhbmv(h, u, n, k, al, A, lda, x, 1, be, y, 1);
}
cublasStatus_t sympmv(cublasHandle_t h, cublasFillMode_t u, int n, const double* al, const double* AP, const double* x,
                      const double* be, double* y) {
  return cublasDspmv(h, u, n, al, AP, x, 1, be, y, 1);
}
cublasStatus_t sympmv(cublasHandle_t h, cublasFillMode_t u, int n, const cuDoubleComplex* al, const cuDoubleComplex* AP,
                      const cuDoubleComplex* x, const cuDoubleComplex* be, cuDoubleComplex* y) {
  return cublasZhpmv(h, u, n, al, AP, x, 1, be, y, 1);
}

constexpr bool is_complex(double) { return false; }
constexpr bool is_complex(cuDoubleComplex) { return true; }

// Every routine and form against the reference, over one type.
template <class T> int check_all(cublasHandle_t h, bool with_inf) {
  std::mt19937_64 rng(with_inf ? 77 : 551);
  int bad = 0;
  const T inf = [] {
    if constexpr (std::is_same_v<T, double>) return INFINITY;
    else return make_cuDoubleComplex(INFINITY, 0.0);
  }();
  auto report = [&](bool ok, const std::string& what) {
    if (!ok) {
      ++bad;
      std::printf("  MISMATCH %s%s\n", what.c_str(), with_inf ? " (inf in x)" : "");
    }
  };
  const bool herm = is_complex(T{});
  // tbmv and tpmv
  const int n = 37;
  for (int packed = 0; packed < 2; ++packed)
    for (int k : {0, 1, 5, 40})
      for (int lower = 0; lower < 2; ++lower)
        for (Op op : {Op::N, Op::T, Op::C})
          for (int unit = 0; unit < 2; ++unit)
            for (int incx : {1, -2}) {
              if (packed && k != 0) continue;   // one pass of packed per form
              const int lda = k + 2;
              const auto A = rand_vec<T>(rng, packed ? (size_t)n * (n + 1) / 2 : (size_t)lda * n);
              auto x = rand_vec<T>(rng, (size_t)(n - 1) * std::abs(incx) + 1);
              if (with_inf) x[x.size() / 2] = inf;
              const auto xv = gather(x, n, incx);
              const Full M = packed ? packed_full(as_cd(A), n, lower, false, false)
                                    : band_full(as_cd(A), lda, n, k, lower, false, false);
              const auto want = ref_trmv(M, op, unit, n, xv);
              Dev<T> dA(A), dx(x);
              const auto u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
              const auto d = unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;
              const cublasStatus_t st = packed ? tpmv(h, u, cu(op), d, n, dA.p, dx.p, incx)
                                               : tbmv(h, u, cu(op), d, n, k, dA.p, lda, dx.p, incx);
              report(st == CUBLAS_STATUS_SUCCESS && same_vec(dx.get(), n, incx, want),
                     std::string(packed ? "tpmv" : "tbmv k=" + std::to_string(k)) + (lower ? " lower" : " upper") +
                         " op " + name(op) + (unit ? " unit" : "") + " incx " + std::to_string(incx));
            }
  // gbmv
  const int m = 23, nc = 31;
  for (auto [kl, ku] : {std::pair{2, 4}, std::pair{0, 0}, std::pair{30, 1}})
    for (Op op : {Op::N, Op::T, Op::C})
      for (int beta0 = 0; beta0 < 2; ++beta0) {
        const int lda = kl + ku + 2;
        const int xlen = op == Op::N ? nc : m, ylen = op == Op::N ? m : nc;
        const auto A = rand_vec<T>(rng, (size_t)lda * nc);
        auto x = rand_vec<T>(rng, xlen);
        if (with_inf) x[xlen / 2] = inf;
        const auto y = rand_vec<T>(rng, ylen);
        const T al = make<T>(rng), be = beta0 ? T{} : make<T>(rng);
        Full M{std::vector<cd>((size_t)m * nc, 0), m};
        const auto band = as_cd(A);
        for (int j = 0; j < nc; ++j)
          for (int i = std::max(0, j - ku); i <= std::min(m - 1, j + kl); ++i) M(i, j) = band[(size_t)j * lda + ku + i - j];
        const auto want = ref_mv(M, op, m, nc, to_cd(al), as_cd(x), to_cd(be), as_cd(y));
        Dev<T> dA(A), dx(x), dy(y);
        const cublasStatus_t st = gbmv(h, cu(op), m, nc, kl, ku, &al, dA.p, lda, dx.p, 1, &be, dy.p, 1);
        report(st == CUBLAS_STATUS_SUCCESS && same_vec(dy.get(), ylen, 1, want),
               "gbmv kl=" + std::to_string(kl) + " ku=" + std::to_string(ku) + " op " + name(op) + (beta0 ? " beta 0" : ""));
      }
  // sbmv/hbmv and spmv/hpmv
  for (int packed = 0; packed < 2; ++packed)
    for (int k : {0, 3, 40})
      for (int lower = 0; lower < 2; ++lower) {
        if (packed && k != 0) continue;
        const int lda = k + 1;
        const auto A = rand_vec<T>(rng, packed ? (size_t)n * (n + 1) / 2 : (size_t)lda * n);
        auto x = rand_vec<T>(rng, n);
        if (with_inf) x[n / 2] = inf;
        const auto y = rand_vec<T>(rng, n);
        const T al = make<T>(rng), be = make<T>(rng);
        const Full M = packed ? packed_full(as_cd(A), n, lower, true, herm) : band_full(as_cd(A), lda, n, k, lower, true, herm);
        const auto want = ref_mv(M, Op::N, n, n, to_cd(al), as_cd(x), to_cd(be), as_cd(y));
        Dev<T> dA(A), dx(x), dy(y);
        const auto u = lower ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;
        const cublasStatus_t st = packed ? sympmv(h, u, n, &al, dA.p, dx.p, &be, dy.p)
                                         : symbmv(h, u, n, k, &al, dA.p, lda, dx.p, &be, dy.p);
        report(st == CUBLAS_STATUS_SUCCESS && same_vec(dy.get(), n, 1, want),
               std::string(packed ? (herm ? "hpmv" : "spmv") : (herm ? "hbmv k=" : "sbmv k=") + std::to_string(k)) +
                   (lower ? " lower" : " upper"));
      }
  return bad;
}

constexpr double kMiB = 1048576.0;

// The heap a call holds, beyond what was live before it.
template <class F> size_t peak_of(F call) {
  const size_t base = alloc_counter::live();
  alloc_counter::reset_peak();
  call();
  return alloc_counter::peak() - base;
}

}  // namespace

VTEST(cublas_band_mv_matches_the_full_loops_bit_for_bit) {
  cublasHandle_t h = nullptr;
  VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  VCHECK_EQ(check_all<double>(h, false), 0);
  VCHECK_EQ(check_all<cuDoubleComplex>(h, false), 0);
  VCHECK_EQ(check_all<double>(h, true), 0);
  VCHECK_EQ(check_all<cuDoubleComplex>(h, true), 0);
  cublasDestroy(h);
}

// SCS's call: x = diag(d) x with Dtbmv, k = 0. A full copy of a 3000 x 3000 matrix is 137 MiB as complex<double>; the
// band, the vector and the answer are under 1 MiB.
VTEST(cublas_band_mv_holds_the_band_not_the_matrix) {
  const int n = 3000;
  std::mt19937_64 rng(3);
  cublasHandle_t h = nullptr;
  VCHECK_EQ(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  const auto d = rand_vec<double>(rng, n), x = rand_vec<double>(rng, n), y = rand_vec<double>(rng, n);
  const auto band3 = rand_vec<double>(rng, (size_t)3 * n);
  Dev<double> dd(d), dx(x), dy(y), db(band3);
  const double al = 0.75, be = -1.5;
  cublasStatus_t st = CUBLAS_STATUS_SUCCESS;
  const size_t tb = peak_of([&] { st = cublasDtbmv(h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, n, 0, dd.p, 1, dx.p, 1); });
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  const size_t gb = peak_of([&] { st = cublasDgbmv(h, CUBLAS_OP_T, n, n, 1, 1, &al, db.p, 3, dx.p, 1, &be, dy.p, 1); });
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  const size_t sb = peak_of([&] { st = cublasDsbmv(h, CUBLAS_FILL_MODE_LOWER, n, 2, &al, db.p, 3, dx.p, 1, &be, dy.p, 1); });
  VCHECK_EQ(st, CUBLAS_STATUS_SUCCESS);
  std::printf("  n=%d peak extra heap: tbmv %.2f MiB, gbmv %.2f MiB, sbmv %.2f MiB (the full matrix %.0f MiB)\n", n, tb / kMiB,
              gb / kMiB, sb / kMiB, (double)n * n * 16 / kMiB);
  VCHECK(tb < (4u << 20));
  VCHECK(gb < (4u << 20));
  VCHECK(sb < (4u << 20));
  // and the answer: d .* x
  const auto got = dx.get();
  bool ok = true;
  for (int i = 0; i < n; ++i) {
    const double want = (cd(0) + cd(d[i]) * cd(x[i])).real();   // the sum starts at +0, as the routine's does
    ok = ok && std::memcmp(&got[i], &want, 8) == 0;
  }
  VCHECK(ok);
  cublasDestroy(h);
}

VTEST_MAIN
