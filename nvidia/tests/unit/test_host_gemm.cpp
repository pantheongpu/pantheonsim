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
#include "alloc_counter.hpp"
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

// ---- operands that are converted as they are packed (gemm_with and loaders) ----
//
// cuBLASLt and cuBLAS keep a bf16, fp16 or fp32 matrix as the bytes the device holds and convert each element when a
// panel is packed. The reference is the old way: widen the whole matrix first, then the plain loops.

uint16_t bf16_of(float f) {   // exact for the values made below
  uint32_t x;
  std::memcpy(&x, &f, 4);
  return static_cast<uint16_t>(x >> 16);
}
double bf16_val(uint16_t b) {
  const uint32_t x = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
double half_val(uint16_t h) {   // the library's reference decode (with ldexp)
  const int e = (h >> 10) & 0x1f, m = h & 0x3ff;
  const double v = e == 0 ? std::ldexp(static_cast<double>(m), -24)
                   : e == 31 ? (m ? NAN : INFINITY)
                             : std::ldexp(static_cast<double>(m | 0x400), e - 25);
  return (h & 0x8000) ? -v : v;
}

enum class Kind { Bf16, Half, F32 };

struct Stored {   // a matrix as its raw bytes, and widened
  std::vector<uint8_t> raw;
  std::vector<double> wide;
};

Stored make_stored(Kind kind, size_t n, Mix mix, std::mt19937_64& rng) {
  Stored s;
  s.wide.resize(n);
  s.raw.resize(n * (kind == Kind::F32 ? 4 : 2));
  for (size_t i = 0; i < n; ++i) {
    const bool wild = (mix == Mix::Special && rng() % 2) || (mix == Mix::SpecialSparse && rng() % 40 == 0);
    const bool zero = mix == Mix::Zeros && rng() % 3 == 0;
    if (kind == Kind::F32) {
      float f = static_cast<float>((static_cast<int>(rng() % 2001) - 1000) / 256.0);
      if (wild) { const uint32_t b = static_cast<uint32_t>(rng()); std::memcpy(&f, &b, 4); }
      if (zero) f = rng() % 2 ? 0.0f : -0.0f;
      std::memcpy(s.raw.data() + i * 4, &f, 4);
      s.wide[i] = f;
    } else {
      uint16_t u;
      if (kind == Kind::Bf16) u = bf16_of(static_cast<float>((static_cast<int>(rng() % 511) - 255) / 64.0));
      else u = static_cast<uint16_t>(((rng() % 2) << 15) | ((10 + rng() % 11) << 10) | (rng() % 1024));   // 2^-5 .. 2^5
      if (wild) u = static_cast<uint16_t>(rng());
      if (zero) u = rng() % 2 ? 0 : 0x8000;
      std::memcpy(s.raw.data() + i * 2, &u, 2);
      s.wide[i] = kind == Kind::Bf16 ? bf16_val(u) : half_val(u);
    }
  }
  return s;
}

template <class T>
struct RawLoader {   // what cublaslt_api.cpp's loaders do, for T = double and float
  const uint8_t* raw;
  Kind kind;
  T operator()(int64_t idx, int64_t, int64_t) const {
    if (kind == Kind::F32) { float f; std::memcpy(&f, raw + idx * 4, 4); return static_cast<T>(f); }
    uint16_t u;
    std::memcpy(&u, raw + idx * 2, 2);
    return static_cast<T>(kind == Kind::Bf16 ? bf16_val(u) : half_val(u));
  }
};
// A's scale depends on the element's position, as a block-scaled operand's does.
double scale_at(int64_t i, int64_t p) { return 0.25 + static_cast<double>((i * 7 + p * 3) % 5) * 0.1; }
template <class L>
struct Scaled {
  L inner;
  double operator()(int64_t idx, int64_t i, int64_t p) const { return inner(idx, i, p) * scale_at(i, p); }
};
template <class L>
struct Shift {   // B seen from column j0 on
  L inner;
  int64_t off;
  auto operator()(int64_t idx, int64_t j, int64_t p) const { return inner(idx + off, j, p); }
};

bool same_or_both_nan(double a, double b) { return std::memcmp(&a, &b, 8) == 0 || (std::isnan(a) && std::isnan(b)); }
bool same_or_both_nan(float a, float b) { return std::memcmp(&a, &b, 4) == 0 || (std::isnan(a) && std::isnan(b)); }

template <class T>
size_t run_loader_case(Kind ka, Kind kb, bool scaled, bool ta, bool tb, const Shape& s, Mix mix, int pad, std::mt19937_64& rng,
                       int* variants) {
  const int lda = (ta ? s.k : s.m) + pad, ldb = (tb ? s.n : s.k) + pad;
  const size_t na = static_cast<size_t>(lda) * (ta ? s.m : s.k), nb = static_cast<size_t>(ldb) * (tb ? s.k : s.n);
  const Stored A = make_stored(ka, na, mix, rng), B = make_stored(kb, nb, mix, rng);
  // The reference: widen first (A times its scale, where it has one), then the old loops.
  std::vector<T> Aw(na), Bw(nb);
  for (int64_t c = 0; c < (ta ? s.m : s.k); ++c)
    for (int64_t r = 0; r < (ta ? s.k : s.m); ++r) {
      const int64_t i = ta ? c : r, p = ta ? r : c;
      const double a = A.wide[static_cast<size_t>(r + c * lda)];
      Aw[static_cast<size_t>(r + c * lda)] = static_cast<T>(scaled ? a * scale_at(i, p) : a);
    }
  for (size_t i = 0; i < nb; ++i) Bw[i] = static_cast<T>(B.wide[i]);
  std::vector<T> Cr(static_cast<size_t>(s.m) * s.n, T(0));
  ref_gemm<T>(ta, tb, s.m, s.n, s.k, T(1), Aw, lda, Bw, ldb, T(0), Cr, s.m);
  size_t bad = 0;
  for (const Isa isa : {Isa::Generic, Isa::Avx2}) {
    if (isa == Isa::Avx2 && !vgpu::hostgemm::avx2_available()) continue;
    for (const unsigned threads : {1u, 3u}) {
      std::vector<T> Cf(Cr.size(), T(7));
      T* const out = Cf.data();
      const int64_t m = s.m;
      const RawLoader<T> la{A.raw.data(), ka}, lb{B.raw.data(), kb};
      auto epi = [=](int64_t i, int64_t j, T acc) { out[static_cast<size_t>(i + j * m)] = acc; };
      if (scaled) vgpu::hostgemm::gemm_with<T>(ta, tb, s.m, s.n, s.k, Scaled<RawLoader<T>>{la}, lda, lb, ldb, epi, isa, threads);
      else vgpu::hostgemm::gemm_with<T>(ta, tb, s.m, s.n, s.k, la, lda, lb, ldb, epi, isa, threads);
      ++*variants;
      for (size_t i = 0; i < Cr.size(); ++i) bad += !same_or_both_nan(Cr[i], Cf[i]);
      // The same sums a strip of columns at a time (B seen from the strip's first column), as cuBLASLt takes them.
      for (const int64_t w : {int64_t(1), int64_t(5), int64_t(33)}) {
        std::vector<T> Cs(Cr.size(), T(7));
        T* const so = Cs.data();
        for (int64_t j0 = 0; j0 < s.n; j0 += w) {
          const int64_t cols = std::min<int64_t>(w, s.n - j0);
          auto sepi = [=](int64_t i, int64_t j, T acc) { so[static_cast<size_t>(i + (j0 + j) * m)] = acc; };
          const Shift<RawLoader<T>> sb{lb, tb ? j0 : j0 * ldb};
          if (scaled) vgpu::hostgemm::gemm_with<T>(ta, tb, s.m, cols, s.k, Scaled<RawLoader<T>>{la}, lda, sb, ldb, sepi, isa, threads);
          else vgpu::hostgemm::gemm_with<T>(ta, tb, s.m, cols, s.k, la, lda, sb, ldb, sepi, isa, threads);
        }
        ++*variants;
        for (size_t i = 0; i < Cr.size(); ++i) bad += !same_or_both_nan(Cr[i], Cs[i]);
      }
    }
  }
  return bad;
}

template <class T>
void sweep_loaders(const char* name) {
  std::mt19937_64 rng(4242);
  int cases = 0, variants = 0;
  size_t bad = 0;
  const std::vector<Shape> shapes = [] {
    std::vector<Shape> v(std::begin(kShapes), std::end(kShapes));
    v.push_back({9, 300, 1});     // K = 1
    v.push_back({130, 70, 1});
    v.push_back({257, 3, 2});
    return v;
  }();
  const Kind kinds[] = {Kind::Bf16, Kind::Half, Kind::F32};
  for (int t = 0; t < 4; ++t)
    for (const Shape& s : shapes)
      for (const Mix mix : {Mix::Plain, Mix::Special, Mix::SpecialSparse}) {
        const Kind ka = kinds[cases % 3], kb = (cases % 4 == 0) ? kinds[(cases / 3) % 3] : ka;   // mostly the same type, some mixed
        bad += run_loader_case<T>(ka, kb, cases % 5 == 0, t & 1, t & 2, s, mix, cases % 3, rng, &variants);
        ++cases;
      }
  std::printf("  %s loaders: %d cases (%d runs incl. strips), %zu elements differ\n", name, cases, variants, bad);
  VCHECK_EQ(bad, static_cast<size_t>(0));
}

}  // namespace

VTEST(host_gemm_loaders_double_match_widened_copies) { sweep_loaders<double>("double"); }
VTEST(host_gemm_loaders_float_match_widened_copies) { sweep_loaders<float>("float"); }

// The point of the loaders: converting a panel at a time holds a few panels, not a matrix. 1024 x 1024 x 1024 from
// bf16 would have been 8 MiB each for A and B widened and 8 MiB for the sums; here the whole call may allocate little.
VTEST(host_gemm_loaders_hold_only_panels) {
  const int64_t n = 1024;
  std::mt19937_64 rng(99);
  const Stored A = make_stored(Kind::Bf16, static_cast<size_t>(n * n), Mix::Plain, rng), B = make_stored(Kind::Bf16, static_cast<size_t>(n * n), Mix::Plain, rng);
  std::vector<double> C(static_cast<size_t>(n * n), 0.0);
  double* const out = C.data();
  const RawLoader<double> la{A.raw.data(), Kind::Bf16}, lb{B.raw.data(), Kind::Bf16};
  const size_t base = alloc_counter::live();
  alloc_counter::reset_peak();
  vgpu::hostgemm::gemm_with<double>(false, false, n, n, n, la, n, lb, n, [=](int64_t i, int64_t j, double acc) { out[i + j * n] = acc; },
                                    Isa::Auto, 2);
  const size_t extra = alloc_counter::peak() - base;
  std::printf("  1024^3 bf16 -> double sums, 2 threads: peak extra heap %.2f MiB\n", extra / 1048576.0);
  VCHECK(extra < (size_t(4) << 20));   // a matrix widened would be 8 MiB
  // and the answer is the plain loop's
  std::vector<double> Aw(A.wide), Bw(B.wide), Cr(C.size(), 0.0);
  for (int64_t j = 0; j < n; j += 97)
    for (int64_t i = 0; i < n; i += 89) {
      double acc = 0.0;
      for (int64_t p = 0; p < n; ++p) acc += Aw[static_cast<size_t>(i + p * n)] * Bw[static_cast<size_t>(p + j * n)];
      VCHECK(std::memcmp(&acc, &C[static_cast<size_t>(i + j * n)], 8) == 0);
    }
}

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
