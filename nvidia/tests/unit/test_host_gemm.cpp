// The host matrix product behind cuBLAS and cuBLASLt (src/host_gemm.hpp), bit for bit against the plain loops it
// replaced. Each output is one chain acc = acc + a*b over p in order, the product and the sum rounded separately;
// the blocked, vectorised, threaded kernels run many chains side by side and must give every chain's exact bits,
// on both instruction sets, for any thread count, over inputs that break arithmetic (signed zeros, subnormals,
// infinities, NaN, the extremes) and over shapes that cut the work into blocks and chunks of k.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include "../../src/host_gemm.hpp"
#include "vtest.hpp"

namespace {

using vgpu::hostgemm::Isa;

size_t idx(int64_t row, int64_t col, int64_t ld) { return static_cast<size_t>(col * ld + row); }

// The loops as cublas_api.cpp had them before host_gemm.hpp: gemm_host (float, double) and gemm_int8.
template <class T>
void ref_gemm(bool ta, bool tb, int m, int n, int k, T alpha, const std::vector<T>& A, int lda, const std::vector<T>& B,
              int ldb, T beta, std::vector<T>& C, int ldc) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      T acc = T(0);
      for (int p = 0; p < k; ++p) {
        T a = A[ta ? idx(p, i, lda) : idx(i, p, lda)];
        T b = B[tb ? idx(j, p, ldb) : idx(p, j, ldb)];
        acc += a * b;
      }
      T& c = C[idx(i, j, ldc)];
      c = beta == T(0) ? alpha * acc : alpha * acc + beta * c;
    }
}

void ref_int8(bool ta, bool tb, int m, int n, int k, int32_t alpha, const std::vector<int8_t>& A, int lda,
              const std::vector<int8_t>& B, int ldb, int32_t beta, std::vector<int32_t>& C, int ldc) {
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      uint32_t acc = 0;
      for (int p = 0; p < k; ++p)
        acc += static_cast<uint32_t>(static_cast<int32_t>(A[ta ? idx(p, i, lda) : idx(i, p, lda)])) *
               static_cast<uint32_t>(static_cast<int32_t>(B[tb ? idx(j, p, ldb) : idx(p, j, ldb)]));
      int32_t& c = C[idx(i, j, ldc)];
      c = static_cast<int32_t>(beta == 0 ? static_cast<uint32_t>(alpha) * acc
                                         : static_cast<uint32_t>(alpha) * acc +
                                               static_cast<uint32_t>(beta) * static_cast<uint32_t>(c));
    }
}

template <class T>
void fast_gemm(Isa isa, unsigned threads, bool ta, bool tb, int m, int n, int k, T alpha, const std::vector<T>& A,
               int lda, const std::vector<T>& B, int ldb, T beta, std::vector<T>& C, int ldc) {
  T* const c0 = C.data();
  vgpu::hostgemm::gemm<T, T>(
      ta, tb, m, n, k, A.data(), lda, B.data(), ldb,
      [=](int64_t i, int64_t j, T acc) {
        T& c = c0[idx(i, j, ldc)];
        c = beta == T(0) ? alpha * acc : alpha * acc + beta * c;
      },
      isa, threads);
}

void fast_int8(Isa isa, unsigned threads, bool ta, bool tb, int m, int n, int k, int32_t alpha,
               const std::vector<int8_t>& A, int lda, const std::vector<int8_t>& B, int ldb, int32_t beta,
               std::vector<int32_t>& C, int ldc) {
  int32_t* const c0 = C.data();
  vgpu::hostgemm::gemm<uint32_t, int8_t>(
      ta, tb, m, n, k, A.data(), lda, B.data(), ldb,
      [=](int64_t i, int64_t j, uint32_t acc) {
        int32_t& c = c0[idx(i, j, ldc)];
        c = static_cast<int32_t>(beta == 0 ? static_cast<uint32_t>(alpha) * acc
                                           : static_cast<uint32_t>(alpha) * acc +
                                                 static_cast<uint32_t>(beta) * static_cast<uint32_t>(c));
      },
      isa, threads);
}

// ---- inputs ----

template <class T>
struct Special;
template <>
struct Special<float> {
  static std::vector<float> values() {
    using L = std::numeric_limits<float>;
    return {0.0f,  -0.0f, 1.0f,  -1.0f, L::denorm_min(), -L::denorm_min(), L::min(), -L::min(), L::max(), -L::max(),
            L::infinity(), -L::infinity(), L::quiet_NaN(), 3.0f, 0.1f, 1e-20f, 1e20f, 0.5f, 16777217.0f};
  }
};
template <>
struct Special<double> {
  static std::vector<double> values() {
    using L = std::numeric_limits<double>;
    return {0.0,  -0.0, 1.0,  -1.0, L::denorm_min(), -L::denorm_min(), L::min(), -L::min(), L::max(), -L::max(),
            L::infinity(), -L::infinity(), L::quiet_NaN(), 3.0, 0.1, 1e-200, 1e200, 0.5, 9007199254740993.0};
  }
};

enum class Mix { Plain, Zeros, Special, SpecialSparse };

template <class T>
std::vector<T> fill(size_t n, Mix mix, std::mt19937_64& rng) {
  std::vector<T> v(n);
  std::uniform_real_distribution<double> u(-2.0, 2.0);
  const auto sp = Special<T>::values();
  for (auto& x : v) {
    switch (mix) {
      case Mix::Plain: x = static_cast<T>(u(rng)); break;
      case Mix::Zeros: x = rng() % 3 == 0 ? (rng() % 2 ? T(0) : T(-0.0)) : static_cast<T>(u(rng)); break;
      case Mix::Special: x = rng() % 2 ? sp[rng() % sp.size()] : static_cast<T>(u(rng)); break;
      case Mix::SpecialSparse: x = rng() % 40 == 0 ? sp[rng() % sp.size()] : static_cast<T>(u(rng)); break;
    }
  }
  return v;
}

template <class T>
bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

struct Shape {
  int m, n, k;
};

// Shapes that matter: under the naive cutoff, one tile, partial tiles in each direction, k across the chunk size
// (256), several blocks per direction (128 x 192 and the halved blocks threads choose), and long thin ones.
const Shape kShapes[] = {{1, 1, 1},     {3, 5, 7},      {16, 6, 17},    {17, 7, 16},    {33, 13, 40},   {64, 64, 64},
                         {65, 67, 63},  {100, 90, 255}, {100, 90, 256}, {100, 90, 257}, {129, 193, 300}, {257, 130, 513},
                         {300, 250, 70}, {1, 400, 200}, {400, 1, 200},  {7, 3, 1500},   {200, 200, 3}};

// One case against every instruction set and thread count (the reference is computed once).
template <class T>
int run_float_case(bool ta, bool tb, const Shape& s, Mix mix, int pad, T alpha, T beta, std::mt19937_64& rng, int* variants) {
  const int lda = (ta ? s.k : s.m) + pad, ldb = (tb ? s.n : s.k) + pad, ldc = s.m + pad;
  const size_t na = static_cast<size_t>(lda) * (ta ? s.m : s.k), nb = static_cast<size_t>(ldb) * (tb ? s.k : s.n),
               nc = static_cast<size_t>(ldc) * s.n;
  const auto A = fill<T>(na, mix, rng), B = fill<T>(nb, mix, rng);
  auto C0 = fill<T>(nc, mix, rng);
  auto Cr = C0;
  ref_gemm<T>(ta, tb, s.m, s.n, s.k, alpha, A, lda, B, ldb, beta, Cr, ldc);
  size_t bad = 0;
  for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
    if (isa == Isa::Avx2 && !vgpu::hostgemm::avx2_available()) continue;
    for (const unsigned threads : {1u, 3u}) {
      auto Cf = C0;
      fast_gemm<T>(isa, threads, ta, tb, s.m, s.n, s.k, alpha, A, lda, B, ldb, beta, Cf, ldc);
      ++*variants;
      // Exact, except that two NaNs count as equal: which NaN wins when two meet (one in an operand, one an inf*0 made) is the
      // compiler's choice of operand order, which differs between any two copies of the same source line (see host_gemm.hpp).
      for (size_t i = 0; i < nc; ++i) {
        if (std::memcmp(&Cr[i], &Cf[i], sizeof(T)) == 0) continue;
        if (std::isnan(Cr[i]) && std::isnan(Cf[i])) continue;
        ++bad;
      }
    }
  }
  return static_cast<int>(bad);
}

template <class T>
void sweep_float(const char* name) {
  std::mt19937_64 rng(12345);
  int cases = 0, variants = 0;
  size_t bad_total = 0;
  for (int t = 0; t < 4; ++t)
    for (const Shape& s : kShapes)
      for (const Mix mix : {Mix::Plain, Mix::Zeros, Mix::Special, Mix::SpecialSparse}) {
        const bool ta = t & 1, tb = t & 2;
        // alpha and beta: ordinary, zero beta (C write-only), and awkward values.
        const T al = (cases % 5 == 0 || cases % 15 == 3) ? T(1) : (cases % 5 == 1) ? T(-0.5) : (cases % 5 == 2) ? T(0) : T(2.5);
        const T be = (cases % 3 == 0 || cases % 15 == 3) ? T(0) : (cases % 3 == 1) ? T(1) : T(-1.25);
        const int pad = cases % 4 == 0 ? 0 : cases % 4;
        bad_total += run_float_case<T>(ta, tb, s, mix, pad, al, be, rng, &variants);
        ++cases;
      }
  std::printf("  %s: %d cases x instruction sets and thread counts (%d runs), %zu elements differ\n", name, cases, variants, bad_total);
  VCHECK_EQ(bad_total, static_cast<size_t>(0));
}

}  // namespace

VTEST(host_gemm_float_matches_the_loops) { sweep_float<float>("float"); }
VTEST(host_gemm_double_matches_the_loops) { sweep_float<double>("double"); }

VTEST(host_gemm_int8_matches_the_loops) {
  std::mt19937_64 rng(777);
  int cases = 0;
  for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
    if (isa == Isa::Avx2 && !vgpu::hostgemm::avx2_available()) continue;
    for (const unsigned threads : {1u, 3u})
      for (int t = 0; t < 4; ++t)
        for (const Shape& s : kShapes) {
          const bool ta = t & 1, tb = t & 2;
          const int lda = (ta ? s.k : s.m) + 1, ldb = (tb ? s.n : s.k) + 2, ldc = s.m + 3;
          std::vector<int8_t> A(static_cast<size_t>(lda) * (ta ? s.m : s.k)), B(static_cast<size_t>(ldb) * (tb ? s.k : s.n));
          // The extremes of int8 (-128 included) in about a quarter of the entries, so sums reach 2^31 and wrap.
          for (auto& x : A) x = rng() % 4 == 0 ? (rng() % 2 ? int8_t(-128) : int8_t(127)) : static_cast<int8_t>(rng());
          for (auto& x : B) x = rng() % 4 == 0 ? (rng() % 2 ? int8_t(-128) : int8_t(127)) : static_cast<int8_t>(rng());
          std::vector<int32_t> C0(static_cast<size_t>(ldc) * s.n);
          for (auto& x : C0) x = static_cast<int32_t>(rng());
          const int32_t al = cases % 3 == 0 ? 1 : cases % 3 == 1 ? -3 : std::numeric_limits<int32_t>::min();
          const int32_t be = cases % 2 == 0 ? 0 : 7;
          auto Cr = C0, Cf = C0;
          ref_int8(ta, tb, s.m, s.n, s.k, al, A, lda, B, ldb, be, Cr, ldc);
          fast_int8(isa, threads, ta, tb, s.m, s.n, s.k, al, A, lda, B, ldb, be, Cf, ldc);
          VCHECK(same_bits(Cr, Cf));
          ++cases;
        }
  }
  std::printf("  int8: %d cases\n", cases);
}

VTEST_MAIN
