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
                         {16, 16, 8, 4}, {8, 32, 16, 8}, {32, 8, 16, 8}, {32, 32, 16, 8}, {16, 24, 16, 8}, {4, 8, 12, 4}, {16, 8, 12, 8}, {7, 8, 16, 8}, {5, 16, 8, 4}, {1, 8, 8, 8}, {6, 8, 12, 4}, {3, 16, 16, 8}, {12, 16, 12, 4}};

struct Totals {
  long elements = 0, mismatches = 0;
  long k512 = 0, k512_taken = 0;   // calls the 512-bit float-product kernel was held to the 256-bit one on, and how many it took
};

#ifdef VGPU_TC86_AVX2
// The 512-bit float-product kernel (mma_avx512_chunked_f) against the 256-bit one (mma_avx2_chunked_f) and the frozen form, called directly (the
// dispatch picks it only at the AVX-512 level): the same answer to "taken or not", nothing written when not taken, and when taken the frozen
// form's bits in every element. Skipped on a CPU without AVX-512 (and for the shapes the 512-bit kernel leaves to the 256-bit one).
template <unsigned NC, bool ACC16>
void check_512(int round, const Shape& sh, int fmt, int mode, int emin, const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C,
               const std::vector<float>& D0, Totals& tot) {
  const uint32_t sentinel = 0xdeadbeefu;
  std::vector<float> D3(C.size(), f(sentinel)), D4(C.size(), f(sentinel));
  const bool t2 = now::mma_avx2_chunked_f<NC, ACC16>(sh.M, sh.N, sh.K, emin, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D3.data(), sh.N);
  const bool t5 = now::mma_avx512_chunked_f<NC, ACC16>(sh.M, sh.N, sh.K, emin, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D4.data(), sh.N);
  ++tot.k512;
  tot.k512_taken += t5;
  bool bad = t2 != t5;
  for (size_t e = 0; e < D0.size(); ++e) {
    if (t5 && bits(D4[e]) != bits(D0[e])) bad = true;
    if (!t5 && bits(D4[e]) != sentinel) bad = true;   // (a declined call writes nothing)
    if (t2 && bits(D3[e]) != bits(D4[e])) bad = true;
  }
  if (bad && ++tot.mismatches <= 5)
    std::fprintf(stderr, "round %d shape %ux%ux%u chunk %u fmt %d acc16 %d mode %d: avx2 kernel %s, avx512 kernel %s, or their bits differ\n", round, sh.M, sh.N, sh.K, NC, fmt, ACC16, mode,
                 t2 ? "took" : "declined", t5 ? "took" : "declined");
}

// The edge tests' extra check: the 512-bit kernel says what the 256-bit one said and, taken, gives the frozen form's bits.
void edge_512(bool took, unsigned chunk, unsigned M, unsigned N, unsigned K, int emin, const std::vector<float>& A, const std::vector<float>& B, const std::vector<float>& C,
              const std::vector<float>& D0, const char* what) {
  if (!vgpu::simd::supported(vgpu::simd::Level::Avx512) || !now::mma_avx512_f_shape(M, N)) return;
  std::vector<float> D5(D0.size(), f(0xdeadbeefu));
  const bool t5 = chunk == 8 ? now::mma_avx512_chunked_f<8, false>(M, N, K, emin, A.data(), K, B.data(), N, C.data(), N, D5.data(), N)
                             : now::mma_avx512_chunked_f<4, false>(M, N, K, emin, A.data(), K, B.data(), N, C.data(), N, D5.data(), N);
  if (t5 != took) throw vtest::Failure(std::string(what) + ": the 512-bit float kernel " + (t5 ? "took" : "declined") + " a call the 256-bit one " + (took ? "took" : "declined"));
  for (size_t e = 0; e < D0.size(); ++e)
    if (t5 ? bits(D5[e]) != bits(D0[e]) : bits(D5[e]) != 0xdeadbeefu) throw vtest::Failure(std::string(what) + ": the 512-bit float kernel's element " + std::to_string(e) + " differs");
}
#endif

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
  else if (f16acc && sh.N % 8 == 0 && sh.K % sh.chunk == 0 && __builtin_cpu_supports("avx2"))
    now::mma_avx2_acc<true>(sh.M, sh.N, sh.K, sh.chunk, emin, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D2.data(), sh.N);
  else
    D2 = D1;
#else
  D2 = D1;
#endif
#ifdef VGPU_TC86_AVX2
  if (vgpu::simd::supported(vgpu::simd::Level::Avx512) && now::mma_avx512_f_shape(sh.M, sh.N) && sh.K % sh.chunk == 0 && sh.K <= 16 && sh.M <= 32 && sh.N <= 32) {
    if (sh.chunk == 8)
      f16acc ? check_512<8, true>(round, sh, fmt, mode, emin, A, B, C, D0, tot) : check_512<8, false>(round, sh, fmt, mode, emin, A, B, C, D0, tot);
    else if (sh.chunk == 4)
      f16acc ? check_512<4, true>(round, sh, fmt, mode, emin, A, B, C, D0, tot) : check_512<4, false>(round, sh, fmt, mode, emin, A, B, C, D0, tot);
  }
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
    one_round(g, static_cast<int>(round), sh, fmt, (round / 39) % 4 == 3, tot);
  }
  std::printf("  %ld rounds, %ld elements compared bit for bit; the 512-bit float kernel on %ld of them (%ld taken)\n", rounds, tot.elements, tot.k512, tot.k512_taken);
  VCHECK_EQ(tot.mismatches, 0L);
  VCHECK(tot.elements > 1000000);
  if (vgpu::simd::supported(vgpu::simd::Level::Avx512)) {
    VCHECK(tot.k512 > 20000);
    VCHECK(tot.k512_taken > 5000);
  }
}

#ifdef VGPU_TC86_AVX2
// The float-product kernel (mma_avx2_chunked_f) is taken only when every product of the call is an exact f32 between 2^-126 and 2^127, which
// the operands' exponent fields say up front: the nonzero ones add to at least 128 (smallest + smallest) and at most 379 (largest + largest), and
// no operand is a float subnormal with a significand. At the edges of that rule, and past them, it must say yes or no as the rule does, and the
// result must be the frozen form's whichever kernel ran -- with accumulators from nothing to the largest, and the lanes whose scaling power of two
// is not a float (a subnormal accumulator beside nothing) finished by the scalar rule.
VTEST(float_product_kernel_edges_equal_the_frozen_form) {
  if (!__builtin_cpu_supports("avx2")) {
    std::printf("  [ SKIP ] this CPU has no AVX2\n");
    return;
  }
  Gen g(0xed9e5);
  long taken = 0, declined = 0, elements = 0;
  struct Edge {
    int ea_lo, ea_hi, eb_lo, eb_hi;   // exponent fields of A's and B's elements (drawn from [lo, hi])
    bool want;                        // whether the kernel takes it
  };
  const Edge edges[] = {
      {64, 64, 64, 64, true},      {64, 64, 63, 63, false},   {64, 64, 63, 64, false},  {1, 1, 127, 127, true},     {1, 1, 126, 126, false},
      {190, 190, 189, 189, true},  {190, 190, 190, 190, false}, {127, 127, 127, 127, true}, {100, 140, 110, 150, true}, {120, 130, 1, 254, false},
      {1, 254, 1, 254, false},     {127, 127, 252, 252, true},  {127, 127, 253, 253, false}, {250, 250, 129, 129, true}, {254, 254, 125, 125, true},
  };
  for (const Edge& edge : edges)
    for (int fmt = 1; fmt <= 2; ++fmt)
      for (int accmode = 0; accmode < 7; ++accmode) {
        const unsigned M = 16, N = 8, K = 16, chunk = fmt == 2 ? 4 : 8;
        std::vector<float> A(M * K), B(K * N), C(M * N), D0(M * N), D1(M * N), D2(M * N);
        const auto draw = [&](int lo, int hi) {
          const uint32_t h = g.u();
          const uint32_t e = static_cast<uint32_t>(lo + static_cast<int>((h >> 8) % static_cast<uint32_t>(hi - lo + 1)));
          const uint32_t m = fmt == 1 ? ((h >> 20) & 0x7fu) << 16 : ((h >> 20) & 0x3ffu) << 13;
          return f((h & 0x80000000u) | (e << 23) | m);
        };
        for (float& x : A) x = (g.u() % 11 == 0) ? 0.0f : draw(edge.ea_lo, edge.ea_hi);
        for (float& x : B) x = (g.u() % 11 == 0) ? -0.0f : draw(edge.eb_lo, edge.eb_hi);
        // (one operand at each end of the range, so that the extremes are present and not merely possible)
        A[0] = draw(edge.ea_lo, edge.ea_lo);
        A[K + 1] = draw(edge.ea_hi, edge.ea_hi);
        B[0] = draw(edge.eb_lo, edge.eb_lo);
        B[N + 1] = draw(edge.eb_hi, edge.eb_hi);
        for (float& x : C) {
          const uint32_t h = g.u();
          switch (accmode) {
            case 0: x = 0.0f; break;
            case 1: x = f(h & 0x807fffffu); break;                                                  // a subnormal
            case 2: x = f((h & 0x80000000u) | ((1u + (h >> 8) % 20) << 23) | (h & 0x7fffffu)); break;   // tiny normals
            case 3: x = f((h & 0x80000000u) | ((235u + (h >> 8) % 20) << 23) | (h & 0x7fffffu)); break;  // huge
            case 4: x = f((h & 0x80000000u) | ((100u + (h >> 8) % 60) << 23) | (h & 0x7fffffu)); break;  // about the products
            case 5: x = (h & 1) ? -0.0f : 0.0f; break;
            default: x = f((h & 0x80000000u) | ((1u + (h >> 8) % 254) << 23) | (h & 0x7fffffu));
          }
        }
        const int emin = now::kEminWide;
        old::mma(M, N, K, chunk, emin, false, A.data(), K, B.data(), N, C.data(), N, D0.data(), N);
        now::mma(M, N, K, chunk, emin, false, A.data(), K, B.data(), N, C.data(), N, D1.data(), N);
        const bool took = fmt == 1 ? now::mma_avx2_chunked_f<8, false>(M, N, K, emin, A.data(), K, B.data(), N, C.data(), N, D2.data(), N)
                                   : now::mma_avx2_chunked_f<4, false>(M, N, K, emin, A.data(), K, B.data(), N, C.data(), N, D2.data(), N);
        took ? ++taken : ++declined;
        edge_512(took, chunk, M, N, K, emin, A, B, C, D0, "edge exponents");
        if (took != edge.want)
          throw vtest::Failure("exponents A " + std::to_string(edge.ea_lo) + ".." + std::to_string(edge.ea_hi) + ", B " + std::to_string(edge.eb_lo) + ".." +
                               std::to_string(edge.eb_hi) + " fmt " + std::to_string(fmt) + ": the float kernel " + (took ? "took" : "declined") + " the call");
        for (size_t e = 0; e < D0.size(); ++e) {
          ++elements;
          if (bits(D0[e]) != bits(D1[e]) || (took && bits(D0[e]) != bits(D2[e])))
            throw vtest::Failure("exponents A " + std::to_string(edge.ea_lo) + ".." + std::to_string(edge.ea_hi) + ", B " + std::to_string(edge.eb_lo) + ".." +
                                 std::to_string(edge.eb_hi) + " fmt " + std::to_string(fmt) + " acc " + std::to_string(accmode) + " element " + std::to_string(e));
        }
      }
  // A bf16 float subnormal with a significand, anywhere, sends the call to the integer kernel.
  for (int fmt = 1; fmt <= 2; ++fmt)
    for (int where = 0; where < 4; ++where) {
      const unsigned M = 16, N = 8, K = 16, chunk = fmt == 2 ? 4 : 8;
      std::vector<float> A(M * K), B(K * N), C(M * N, 1.0f), D0(M * N), D1(M * N), D2(M * N);
      for (float& x : A) x = f((127u << 23) | ((g.u() & 0x7fu) << 16));
      for (float& x : B) x = f((127u << 23) | ((g.u() & 0x7fu) << 16));
      (where < 2 ? A : B)[where % 2 ? 100 : 3] = f(0x00200000u);   // a subnormal with a bit in the kept ten
      const int emin = now::kEminWide;
      old::mma(M, N, K, chunk, emin, false, A.data(), K, B.data(), N, C.data(), N, D0.data(), N);
      now::mma(M, N, K, chunk, emin, false, A.data(), K, B.data(), N, C.data(), N, D1.data(), N);
      const bool took = fmt == 1 ? now::mma_avx2_chunked_f<8, false>(M, N, K, emin, A.data(), K, B.data(), N, C.data(), N, D2.data(), N)
                                 : now::mma_avx2_chunked_f<4, false>(M, N, K, emin, A.data(), K, B.data(), N, C.data(), N, D2.data(), N);
      if (took) throw vtest::Failure("the float kernel took a call with a float subnormal operand");
      edge_512(took, chunk, M, N, K, emin, A, B, C, D0, "float subnormal");
      for (size_t e = 0; e < D0.size(); ++e)
        if (bits(D0[e]) != bits(D1[e])) throw vtest::Failure("a float subnormal operand: result differs from the frozen form");
    }
  // tf32 words with their low 13 bits set (the tensor core ignores them), in the shapes whose K is not a multiple of eight: the tail of the
  // decode, and an infinity or a NaN among moderate operands (which sends the call to the integer kernel).
  for (unsigned K : {4u, 12u})
    for (int special = 0; special < 3; ++special)
      for (int rep = 0; rep < 6; ++rep) {
        const unsigned M = 16, N = 8;
        std::vector<float> A(M * K), B(K * N), C(M * N), D0(M * N), D1(M * N), D2(M * N);
        for (float& x : A) x = f((g.u() & 0x80000000u) | ((120u + g.u() % 10) << 23) | (g.u() & 0x7fffffu));
        for (float& x : B) x = f((g.u() & 0x80000000u) | ((120u + g.u() % 10) << 23) | (g.u() & 0x7fffffu));
        for (float& x : C) x = f((g.u() & 0x80000000u) | ((100u + g.u() % 40) << 23) | (g.u() & 0x7fffffu));
        if (special == 1) A[g.u() % A.size()] = f(0x7f800000u);
        if (special == 2) B[g.u() % B.size()] = f(0x7fc00000u);
        old::mma(M, N, K, 4, now::kEminWide, false, A.data(), K, B.data(), N, C.data(), N, D0.data(), N);
        now::mma(M, N, K, 4, now::kEminWide, false, A.data(), K, B.data(), N, C.data(), N, D1.data(), N);
        const bool took = now::mma_avx2_chunked_f<4, false>(M, N, K, now::kEminWide, A.data(), K, B.data(), N, C.data(), N, D2.data(), N);
        edge_512(took, 4, M, N, K, now::kEminWide, A, B, C, D0, "tf32 tail shape");
        if (took == (special != 0)) throw vtest::Failure("tf32 tail shape: the float kernel " + std::string(took ? "took" : "declined") + " a call with" + (special ? "" : "out") + " an infinity or a NaN");
        for (size_t e = 0; e < D0.size(); ++e)
          if (bits(D0[e]) != bits(D1[e]) || (took && bits(D0[e]) != bits(D2[e]))) throw vtest::Failure("tf32 tail shape K " + std::to_string(K) + ": result differs from the frozen form");
      }
  std::printf("  %ld calls taken by the float kernel, %ld declined, %ld elements compared bit for bit\n", taken, declined, elements);
  VCHECK(taken > 40);
  VCHECK(declined > 40);
}
#endif

#ifdef VGPU_TC86_AVX2
// The vector conversion of a sum to f32 against the scalar one (detail::to_f32_rz), on sums just below and above every power of two (the
// all-ones significands that a nearest-even conversion of the magnitude would round up), and random ones, for all frames g.
static std::vector<int32_t> sums_for_f32(Gen& gen) {
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
  return sums;
}

__attribute__((target("avx2"))) static void check_sums(long& checked, long& bad) {
  Gen gen(0x5ca1e);
  std::vector<int32_t> sums = sums_for_f32(gen);
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

static std::vector<int32_t> sums_for_f16(Gen& gen) {
  std::vector<int32_t> sums;
  for (int n = 0; n <= 30; ++n)
    for (int d = -40; d <= 40; ++d) {
      const int64_t v = (int64_t{1} << n) + d;
      if (v > 0 && v < (int64_t{1} << 31)) {
        sums.push_back(static_cast<int32_t>(v));
        sums.push_back(static_cast<int32_t>(-v));
      }
    }
  // exact ties and their neighbours on the f16 grid: odd and even significands, half a step, a step and a half
  for (int sig = 1000; sig <= 2100; ++sig)
    for (int sh = 1; sh <= 18; ++sh)
      for (int adj = -1; adj <= 1; ++adj) {
        const int64_t v = ((int64_t{sig} << sh) | (int64_t{1} << (sh - 1))) + adj;
        if (v > 0 && v < (int64_t{1} << 31)) {
          sums.push_back(static_cast<int32_t>(v));
          sums.push_back(static_cast<int32_t>(-v));
        }
      }
  for (int i = 0; i < 200000; ++i) sums.push_back(static_cast<int32_t>(gen.u()) >> (gen.u() % 31));
  return sums;
}

__attribute__((target("avx2"))) static void check_sums16(long& checked, long& bad, long& answered) {
  Gen gen(0xf16);
  std::vector<int32_t> sums = sums_for_f16(gen);
  while (sums.size() % 8) sums.push_back(0);
  for (int pass = 0; pass < 14; ++pass)
    for (size_t x = 0; x < sums.size(); x += 8) {
      alignas(32) int32_t S[8], g[8];
      for (int l = 0; l < 8; ++l) {
        S[l] = sums[x + l];
        g[l] = pass < 10 ? -50 + pass * 8 + static_cast<int>(gen.u() % 16) - 8 : static_cast<int>(gen.u() % 400) - 200;
      }
      __m256i res, snz, inr;
      now::sum_to_f16_avx2(_mm256_load_si256(reinterpret_cast<const __m256i*>(S)), _mm256_load_si256(reinterpret_cast<const __m256i*>(g)), res, snz, inr);
      alignas(32) uint32_t r[8], nz[8], in[8];
      _mm256_store_si256(reinterpret_cast<__m256i*>(r), res);
      _mm256_store_si256(reinterpret_cast<__m256i*>(nz), snz);
      _mm256_store_si256(reinterpret_cast<__m256i*>(in), inr);
      for (int l = 0; l < 8; ++l) {
        ++checked;
        const uint32_t want = bits(now::detail::to_f16_rn(S[l], g[l]));
        const bool ans = S[l] != 0 && in[l] != 0;
        if (ans) ++answered;
        const bool ok = (nz[l] != 0) == (S[l] != 0) && (S[l] == 0 ? r[l] == 0 : (!ans || r[l] == want));
        if (!ok && ++bad <= 5) std::fprintf(stderr, "S %d g %d: vector 0x%08x (nz %u in %u), scalar 0x%08x\n", S[l], g[l], r[l], nz[l], in[l], want);
      }
    }
}

// The same two checks on the sixteen-lane conversions (sum_to_f32_avx512, sum_to_f16_avx512), over the same sums and frames.
__attribute__((target("avx2,avx512f,avx512bw,avx512dq,avx512vl"))) static void check_sums512(long& checked, long& bad, long& answered, bool f16) {
  Gen gen(f16 ? 0xf16 : 0x5ca1e);
  std::vector<int32_t> sums = f16 ? sums_for_f16(gen) : sums_for_f32(gen);
  while (sums.size() % 16) sums.push_back(0);
  for (int pass = 0; pass < (f16 ? 14 : 12); ++pass)
    for (size_t x = 0; x < sums.size(); x += 16) {
      alignas(64) int32_t S[16], g[16];
      for (int l = 0; l < 16; ++l) {
        S[l] = sums[x + l];
        g[l] = f16 ? (pass < 10 ? -50 + pass * 8 + static_cast<int>(gen.u() % 16) - 8 : static_cast<int>(gen.u() % 400) - 200)
                   : (pass < 6 ? -30 + pass * 40 + static_cast<int>(gen.u() % 40) - 20 : static_cast<int>(gen.u() % 400) - 200);
      }
      __m512i res;
      __mmask16 snz, inr;
      if (f16)
        now::sum_to_f16_avx512(_mm512_load_si512(S), _mm512_load_si512(g), res, snz, inr);
      else
        now::sum_to_f32_avx512(_mm512_load_si512(S), _mm512_load_si512(g), res, snz, inr);
      alignas(64) uint32_t r[16];
      _mm512_store_si512(r, res);
      for (int l = 0; l < 16; ++l) {
        ++checked;
        const uint32_t want = bits(f16 ? now::detail::to_f16_rn(S[l], g[l]) : now::detail::to_f32_rz(S[l], g[l]));
        const bool nz = snz >> l & 1, in = inr >> l & 1;
        const bool ans = S[l] != 0 && in;
        if (ans) ++answered;
        bool ok;
        if (f16) {
          ok = nz == (S[l] != 0) && (S[l] == 0 ? r[l] == 0 : (!ans || r[l] == want));
        } else {
          const uint32_t e = want >> 23 & 0xff;
          ok = nz == (S[l] != 0) && (!ans || (r[l] == want && e >= 1 && e <= 254)) && (ans || r[l] == 0);
        }
        if (!ok && ++bad <= 5) std::fprintf(stderr, "512-bit f%d: S %d g %d: vector 0x%08x (nz %d in %d), scalar 0x%08x\n", f16 ? 16 : 32, S[l], g[l], r[l], nz, in, want);
      }
    }
}

VTEST(sum_to_f16_f32_512_equal_the_scalar_conversion) {
  if (!vgpu::simd::supported(vgpu::simd::Level::Avx512)) {
    std::printf("  [ SKIP ] this CPU has no AVX-512\n");
    return;
  }
  for (const bool f16 : {true, false}) {
    long checked = 0, bad = 0, answered = 0;
    check_sums512(checked, bad, answered, f16);
    std::printf("  f%d: %ld sums converted, %ld answered by the vector code\n", f16 ? 16 : 32, checked, answered);
    VCHECK_EQ(bad, 0L);
    VCHECK(answered > checked / 2);
  }
}

VTEST(sum_to_f16_equals_the_scalar_conversion) {
  if (!__builtin_cpu_supports("avx2")) return;
  long checked = 0, bad = 0, answered = 0;
  check_sums16(checked, bad, answered);
  std::printf("  %ld sums converted, %ld answered by the vector code\n", checked, answered);
  VCHECK_EQ(bad, 0L);
  VCHECK(answered > checked / 2);   // (the vector code answers every sum a chunk can give)
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
