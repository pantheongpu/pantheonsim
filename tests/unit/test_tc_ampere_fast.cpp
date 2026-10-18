// The sm_86 tensor-core sum's vector kernels (include/vgpu/exec/tc_ampere.hpp) against the FROZEN earlier form of the same header
// (tests/unit/tc86_v0.hpp, a copy taken before the kernel was made faster): bit for bit, over every shape and type the engines use, with
// random operands over wide and narrow exponent ranges and a set of adversarial ones (zeros of both signs, subnormals, infinities and
// NaNs, the largest finite values, powers of two, cancelling pairs, one huge term among tiny ones, accumulators at, above and below the
// products' size, and operands with arbitrary 32-bit patterns). The speed-up must change no result bit.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "tc86_v0.hpp"
#include "vgpu/exec/tc_ampere.hpp"
#include "vtest.hpp"

namespace {

using vgpu::simd::f16_to_f32;
namespace now = vgpu::exec::tc86;
namespace old = vgpu::exec::tc86_v0;

float f(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }

struct Gen {
  std::mt19937_64 rng;
  explicit Gen(uint64_t seed) : rng(seed) {}
  uint32_t u() { return static_cast<uint32_t>(rng()); }

  // One finite or special operand of format fmt (0 f16, 1 bf16, 2 tf32), by class.
  float operand(int fmt, int cls, bool wide) {
    const uint32_t h = u();
    const uint32_t sign = h & 0x80000000u;
    switch (cls) {
      case 0:   // a normal number, any exponent (wide) or a window around 1 (narrow)
      case 1: {
        if (fmt == 0) {
          const uint32_t e = wide ? 1 + (h >> 8) % 30 : 10 + (h >> 8) % 10;
          return f16_to_f32(static_cast<uint16_t>((sign >> 16) | (e << 10) | ((h >> 20) & 0x3ffu)));
        }
        const uint32_t e = wide ? 1 + (h >> 8) % 254 : 112 + (h >> 8) % 32;
        const uint32_t m = fmt == 1 ? ((h >> 20) & 0x7fu) << 16 : ((h >> 20) & 0x3ffu) << 13;
        return f(sign | (e << 23) | m);
      }
      case 2: return sign ? -0.0f : 0.0f;
      case 3: {   // a subnormal of the format (f16: a half subnormal; bf16 and tf32: a float subnormal)
        if (fmt == 0) return f16_to_f32(static_cast<uint16_t>((sign >> 16) | ((h >> 8) & 0x3ffu)));
        const uint32_t m = fmt == 1 ? ((h >> 8) & 0x7fu) << 16 : ((h >> 8) & 0x3ffu) << 13;
        return f(sign | m);
      }
      case 4: {   // an infinity or a NaN
        static const uint32_t sp[] = {0x7f800000u, 0xff800000u, 0x7fc00000u, 0xffc00001u, 0x7f800001u};
        return f(sp[(h >> 8) % 5]);
      }
      case 5: return fmt == 0 ? (sign ? -65504.0f : 65504.0f) : f(sign | (fmt == 1 ? 0x7f7f0000u : 0x7f7fe000u));   // the largest finite
      case 6: return f(sign | ((127u + (h >> 8) % 41 - 20) << 23));                                            // +-2^k
      case 7: return f(h);                                                                                      // any 32-bit pattern
      default: return f(sign | (127u << 23));                                                                  // +-1
    }
  }

  float acc(int cls, int exp_lo, int exp_hi, bool f16acc) {
    const uint32_t h = u();
    switch (cls) {
      case 0: return 0.0f;
      case 1: return -0.0f;
      case 2: return f(h & 0x807fffffu);   // a float subnormal
      case 3: return f(0x7f800000u | (h & 0x80000000u));
      case 4: return f(0x7fc00000u | (h & 0x803fffffu));
      case 5: return f((h & 0x80000000u) | 0x7f7fffffu);
      default: {
        const int e = exp_lo + static_cast<int>((h >> 8) % static_cast<uint32_t>(exp_hi - exp_lo + 1));
        const float v = f((h & 0x80000000u) | (static_cast<uint32_t>(std::clamp(e, 1, 254)) << 23) | (h & 0x7fffffu));
        return f16acc ? f16_to_f32(vgpu::simd::f32_to_f16_bits(v)) : v;
      }
    }
  }
};

struct Shape {
  unsigned M, N, K, chunk;
};
// The shapes of the engines: mma.sync m16n8k16 / m16n8k8 / m16n8k4 (tf32), wmma 16x16x16 and 16x16x8, and the largest the PTX path takes.
const Shape kShapes[] = {{16, 8, 16, 8}, {16, 8, 8, 8}, {16, 8, 8, 4}, {16, 8, 4, 4}, {16, 16, 16, 8}, {16, 16, 8, 8},
                         {16, 16, 8, 4}, {8, 32, 16, 8}, {32, 8, 16, 8}, {32, 32, 16, 8}, {16, 24, 16, 8}, {4, 8, 12, 4}, {16, 8, 12, 8}, {7, 8, 16, 8}, {5, 16, 8, 4}, {1, 8, 8, 8}};

struct Totals {
  long elements = 0, mismatches = 0;
};

// Draw a problem (a "mode" picks how adversarial), run old and new, compare every element's bits.
void one_round(Gen& g, int round, const Shape& sh, int fmt, bool f16acc, Totals& tot) {
  const int emin = fmt == 0 ? now::kEminF16 : now::kEminWide;
  const bool wide = g.u() & 1;
  const int mode = static_cast<int>(g.u() % 8);
  std::vector<float> A(size_t{sh.M} * sh.K), B(size_t{sh.K} * sh.N), C(size_t{sh.M} * sh.N);
  // The class weights: mode 0 plain random; 1 zeros heavy; 2 subnormal heavy; 3 specials; 4 extremes (largest, powers of two);
  // 5 raw bit patterns; 6 cancelling pairs; 7 one huge among tiny.
  const auto cls = [&]() -> int {
    const uint32_t r = g.u() % 100;
    switch (mode) {
      case 1: return r < 50 ? 2 : r < 55 ? 3 : 0;
      case 2: return r < 40 ? 3 : r < 55 ? 2 : r < 60 ? 6 : 0;
      case 3: return r < 12 ? 4 : r < 30 ? 2 : r < 35 ? 3 : 0;
      case 4: return r < 30 ? 5 : r < 60 ? 6 : r < 70 ? 8 : 0;
      case 5: return r < 50 ? 7 : 0;
      default: return r < 4 ? 2 : r < 6 ? 3 : r < 8 ? 6 : 0;
    }
  };
  for (float& x : A) x = g.operand(fmt, cls(), wide);
  for (float& x : B) x = g.operand(fmt, cls(), wide);
  if (mode == 6) {   // cancelling pairs: a, -a against equal b, in adjacent k, inside and across chunk borders
    for (unsigned i = 0; i < sh.M; ++i)
      for (unsigned k = 0; k + 1 < sh.K; k += 2 + g.u() % 2) {
        A[i * sh.K + k + 1] = -A[i * sh.K + k];
        for (unsigned j = 0; j < sh.N; ++j) B[(k + 1) * sh.N + j] = B[k * sh.N + j];
      }
  }
  if (mode == 7) {   // one huge term in each row, the rest tiny
    for (unsigned i = 0; i < sh.M; ++i) {
      for (unsigned k = 0; k < sh.K; ++k) A[i * sh.K + k] = g.operand(fmt, 3, wide) + g.operand(fmt, 3, wide);
      A[i * sh.K + g.u() % sh.K] = g.operand(fmt, 5, wide);
    }
  }
  const int ac_mode = static_cast<int>(g.u() % 4);   // accumulators: random classes, or sized against the products
  for (float& x : C) {
    const uint32_t r = g.u() % 100;
    const int c = r < 8 ? 0 : r < 12 ? 1 : r < 18 ? 2 : r < 20 ? 3 : r < 22 ? 4 : r < 25 ? 5 : 9;
    if (mode == 3 || mode == 5 || ac_mode == 0 || c != 9) x = g.acc(c, 1, 254, f16acc);
    else if (ac_mode == 1) x = g.acc(9, 100, 150, f16acc);      // near the products' size
    else if (ac_mode == 2) x = g.acc(9, 60, 130, f16acc);       // below it
    else x = g.acc(9, 120, 254, f16acc);                        // above it
  }
  std::vector<float> D0(C.size()), D1(C.size()), D2(C.size());
  old::mma(sh.M, sh.N, sh.K, sh.chunk, emin, f16acc, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D0.data(), sh.N);
  now::mma(sh.M, sh.N, sh.K, sh.chunk, emin, f16acc, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D1.data(), sh.N);
#ifdef VGPU_TC86_AVX2
  if (!f16acc && sh.N % 8 == 0 && __builtin_cpu_supports("avx2"))
    now::mma_avx2(sh.M, sh.N, sh.K, sh.chunk, emin, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D2.data(), sh.N);
  else
    D2 = D1;
#else
  D2 = D1;
#endif
  for (size_t e = 0; e < D0.size(); ++e) {
    ++tot.elements;
    if (bits(D0[e]) != bits(D1[e]) || bits(D0[e]) != bits(D2[e])) {
      if (++tot.mismatches <= 5) {
        char buf[300];
        std::snprintf(buf, sizeof buf, "round %d shape %ux%ux%u chunk %u fmt %d f16acc %d mode %d element %zu: v0 0x%08x, dispatch 0x%08x, avx2 0x%08x", round, sh.M, sh.N,
                      sh.K, sh.chunk, fmt, f16acc, mode, e, bits(D0[e]), bits(D1[e]), bits(D2[e]));
        std::fprintf(stderr, "%s\n", buf);
      }
    }
  }
}

long rounds_from_env(long dflt) {
  const char* v = std::getenv("TC86_FAST_ROUNDS");
  return v ? std::atol(v) : dflt;
}

}  // namespace

VTEST(vector_kernels_equal_the_frozen_form) {
  Gen g(0x86a11ce);
  Totals tot;
  const long rounds = rounds_from_env(120000);
  for (long round = 0; round < rounds; ++round) {
    const Shape& sh = kShapes[round % (sizeof kShapes / sizeof kShapes[0])];
    const int fmt = static_cast<int>((round / 13) % 3);
    // (a K that is not a multiple of the chunk takes the generic vector form; both are held to the frozen form)
    one_round(g, static_cast<int>(round), sh, fmt, fmt == 0 && (round / 39) % 4 == 3, tot);
  }
  std::printf("  %ld rounds, %ld elements compared bit for bit\n", rounds, tot.elements);
  VCHECK_EQ(tot.mismatches, 0L);
  VCHECK(tot.elements > 1000000);
}

#ifdef VGPU_TC86_AVX2
// The vector conversion of a sum to f32 against the scalar one (detail::to_f32_rz), on sums just below and above every power of two (the
// all-ones significands that a nearest-even conversion of the magnitude would round up), and random ones, for all frames g.
__attribute__((target("avx2"))) static void check_sums(long& checked, long& bad) {
  Gen gen(0x5ca1e);
  std::vector<int32_t> sums;
  for (int n = 0; n <= 30; ++n)
    for (int d = -40; d <= 40; ++d) {
      const int64_t v = (int64_t{1} << n) + d;
      if (v > 0 && v < (int64_t{1} << 31)) {
        sums.push_back(static_cast<int32_t>(v));
        sums.push_back(static_cast<int32_t>(-v));
      }
    }
  for (int n = 24; n <= 30; ++n)   // all-ones significands with a half or more below the 24 bits kept, and sparse low bits
    for (int low = 0; low < 128; ++low) {
      const int64_t v = (((int64_t{1} << n) - 1) & ~int64_t{127}) | low;
      sums.push_back(static_cast<int32_t>(v));
      sums.push_back(static_cast<int32_t>(-v));
    }
  for (int i = 0; i < 200000; ++i) sums.push_back(static_cast<int32_t>(gen.u()) >> (gen.u() % 31));
  while (sums.size() % 8) sums.push_back(0);
  for (int pass = 0; pass < 12; ++pass)
    for (size_t x = 0; x < sums.size(); x += 8) {
      alignas(32) int32_t S[8], g[8];
      for (int l = 0; l < 8; ++l) {
        S[l] = sums[x + l];
        g[l] = pass < 6 ? -30 + pass * 40 + static_cast<int>(gen.u() % 40) - 20 : static_cast<int>(gen.u() % 400) - 200;
      }
      __m256i res, snz, inr;
      now::sum_to_f32_avx2(_mm256_load_si256(reinterpret_cast<const __m256i*>(S)), _mm256_load_si256(reinterpret_cast<const __m256i*>(g)), _mm256_set1_epi32(0xff),
                           _mm256_set1_epi32(0x7fffff), res, snz, inr);
      alignas(32) uint32_t r[8], nz[8], in[8];
      _mm256_store_si256(reinterpret_cast<__m256i*>(r), res);
      _mm256_store_si256(reinterpret_cast<__m256i*>(nz), snz);
      _mm256_store_si256(reinterpret_cast<__m256i*>(in), inr);
      for (int l = 0; l < 8; ++l) {
        ++checked;
        const uint32_t want = bits(now::detail::to_f32_rz(S[l], g[l]));
        const uint32_t e = want >> 23 & 0xff;
        // (A lane the vector code calls out of range goes to the scalar conversion, which may still round to a normal number: the
        // vector code only has to be right where it answers.)
        const bool answered = S[l] != 0 && in[l] != 0;
        const bool ok = (nz[l] != 0) == (S[l] != 0) && (!answered || (r[l] == want && e >= 1 && e <= 254)) && (answered || r[l] == 0);
        if (!ok && ++bad <= 5) std::fprintf(stderr, "S %d g %d: vector 0x%08x (nz %u in %u), scalar 0x%08x\n", S[l], g[l], r[l], nz[l], in[l], want);
      }
    }
}

VTEST(sum_to_f32_equals_the_scalar_conversion) {
  if (!__builtin_cpu_supports("avx2")) return;
  long checked = 0, bad = 0;
  check_sums(checked, bad);
  std::printf("  %ld sums converted\n", checked);
  VCHECK_EQ(bad, 0L);
}
#endif

VTEST_MAIN
