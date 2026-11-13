// The scalar reference of the simulator's MUFU results (include/vgpu/host_gpu_ref.hpp) is the simulator's, and the IEEE division idiom's refinement gives the
// correctly rounded quotient whatever the last bit of the reciprocal. Both are what the card comparison (tools/probes/hostgpu_sfu_exhaustive.cu) relies on;
// neither needs a card. VGPU_EXHAUSTIVE=1 runs the whole space (every float for the SIMD kernels, every mantissa of the divisor with 24 dividends for the
// refinement); the default is a stride that finishes in seconds under the sanitizers.
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "vgpu/host_gpu_ref.hpp"
#include "vgpu/simd/simd.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace vgpu::hostgpu;

namespace {
bool exhaustive() {
  const char* e = std::getenv("VGPU_EXHAUSTIVE");
  return e && *e && *e != '0';
}
std::string hex(uint32_t v) {
  char b[16];
  std::snprintf(b, sizeof b, "%08x", v);
  return b;
}
}  // namespace

// The pinned values: what an RTX 3060 and the simulator agree on for the easy points.
VTEST(reference_special_values) {
  const auto bits = [](float f) { return std::bit_cast<uint32_t>(f); };
  VCHECK(ref::mufu(ref::Fn::Rcp, bits(1.0f)) == bits(1.0f));
  VCHECK(ref::mufu(ref::Fn::Rcp, bits(4.0f)) == bits(0.25f));
  VCHECK(ref::mufu(ref::Fn::Rcp, 0x00000000u) == 0x7f800000u);                       // 1/+0 = +inf
  VCHECK(ref::mufu(ref::Fn::Rcp, 0x80000000u) == 0xff800000u);                       // 1/-0 = -inf
  VCHECK(ref::mufu(ref::Fn::Rcp, 0x00400000u) == 0x7f800000u);                       // a subnormal reads as zero
  VCHECK(ref::mufu(ref::Fn::Rcp, 0x7f800000u) == 0x00000000u);                       // 1/inf = 0
  VCHECK(ref::mufu(ref::Fn::Rcp, 0x7fc00001u) == 0x7fffffffu);                       // any NaN is the canonical one
  VCHECK(ref::mufu(ref::Fn::Sqrt, bits(-1.0f)) == 0x7fffffffu);
  VCHECK(ref::mufu(ref::Fn::Sqrt, bits(4.0f)) == bits(2.0f));
  VCHECK(ref::mufu(ref::Fn::Rsq, bits(4.0f)) == bits(0.5f));
  VCHECK(ref::mufu(ref::Fn::Ex2, bits(3.0f)) == bits(8.0f));
  VCHECK(ref::mufu(ref::Fn::Lg2, bits(8.0f)) == bits(3.0f));
  VCHECK(ref::mufu(ref::Fn::Lg2, 0u) == 0xff800000u);                                 // log2(0) = -inf
  VCHECK(ref::mufu(ref::Fn::Tanh, 0x00000001u) == 0x00000001u);                       // tanh keeps subnormals (an RTX 3060's tanh of 2^-149 is 2^-149)
  VCHECK(ref::mufu(ref::Fn::Ex2, 0xc3000000u) == 0u);                                 // 2^-128 flushes to +0
  VCHECK(ref::ulp_distance(0x3f800000u, 0x3f800001u) == 1);
  VCHECK(ref::ulp_distance(0x3f800000u, 0xbf800000u) == UINT32_MAX);
  VCHECK(ref::ulp_distance(0x7fffffffu, 0xffc00000u) == 0);
  VCHECK(ref::ulp_distance(0x80000000u, 0x00000000u) == 0);
  VCHECK(ref::fdiv_rn(bits(1.0f), bits(3.0f)) == bits(1.0f / 3.0f));
  VCHECK(ref::fdiv_rn(0u, 0u) == 0x7fffffffu);
  // The IEEE forms of the card probe's sqrtrn and rcprn ops (sp43): correctly rounded, subnormals kept, NaN canonical.
  VCHECK(ref::fsqrt_rn(bits(2.0f)) == bits(std::sqrt(2.0f)));
  VCHECK(ref::fsqrt_rn(bits(4.0f)) == bits(2.0f));
  VCHECK(ref::fsqrt_rn(0x00000001u) == bits(std::sqrt(std::bit_cast<float>(0x00000001u))));   // a subnormal is not flushed
  VCHECK(ref::fsqrt_rn(bits(-1.0f)) == 0x7fffffffu);
  VCHECK(ref::fsqrt_rn(0x80000000u) == 0x80000000u);                                          // sqrt(-0) = -0
  VCHECK(ref::frcp_rn(bits(3.0f)) == bits(1.0f / 3.0f));
  VCHECK(ref::frcp_rn(0x00000001u) == 0x7f800000u);                                           // 1 / 2^-149 overflows
  VCHECK(ref::frcp_rn(0x00400000u) == 0x7f000000u);                                           // 1 / 2^-127 = 2^127: a subnormal divisor is not read as zero (the MUFU form says +inf)
  VCHECK(ref::frcp_rn(0x7fc00001u) == 0x7fffffffu);
}

// The reference is the simulator's SIMD kernel, bit for bit, at every level this CPU has: RCP, RSQ, SQRT over a spread of every exponent and mantissa shape
// and over the special patterns. (A mutation of the reference's flush or its NaN, or of the kernel, fails here.)
VTEST(reference_equals_the_simulators_kernels) {
  std::mt19937 rng(12345);
  const uint64_t n = exhaustive() ? (1ull << 32) : (1ull << 20);
  const uint64_t step = (1ull << 32) / n;
  std::vector<uint32_t> special;   // every exponent with a few mantissas, both signs
  for (uint32_t e = 0; e < 256; ++e)
    for (uint32_t m : {0u, 1u, 0x400000u, 0x7fffffu, 0x123456u})
      for (uint32_t sgn : {0u, 0x80000000u}) special.push_back(sgn | e << 23 | m);
  while (special.size() % 32) special.push_back(0x3f800000u);
  struct Counts {
    uint64_t bad = 0;
    uint32_t in = 0, got = 0, want = 0;
  };
  for (int li = 0; li < simd::kLevels; ++li) {
    const auto lv = static_cast<simd::Level>(li);
    if (!simd::supported(lv)) continue;
    const auto& k = simd::kernels_for(lv);
    Counts c[3];
    const std::pair<unsigned, ref::Fn> kinds[3] = {{4, ref::Fn::Rcp}, {5, ref::Fn::Rsq}, {8, ref::Fn::Sqrt}};
    alignas(32) uint32_t in[32], r[32];
    const auto check = [&](const uint32_t* x) {
      for (int t = 0; t < 3; ++t) {
        k.mufu(kinds[t].first, r, x);
        for (int l = 0; l < 32; ++l) {
          const uint32_t want = ref::mufu(kinds[t].second, x[l]);
          if (r[l] != want && !c[t].bad++) c[t] = {c[t].bad, x[l], r[l], want};
        }
      }
    };
    for (uint64_t i = 0; i < n; i += 32) {
      for (int l = 0; l < 32; ++l) in[l] = static_cast<uint32_t>((i + l) * step) ^ (exhaustive() ? 0u : static_cast<uint32_t>(rng() & (step - 1)));
      check(in);
    }
    for (size_t i = 0; i < special.size(); i += 32) check(&special[i]);
    for (int t = 0; t < 3; ++t)
      if (c[t].bad)
        throw vtest::Failure(std::string("level ") + simd::level_name(lv) + " MUFU kind " + std::to_string(kinds[t].first) + ": " + std::to_string(c[t].bad) +
                             " differ, first input 0x" + hex(c[t].in) + " kernel 0x" + hex(c[t].got) + " reference 0x" + hex(c[t].want));
  }
}

// ptxas's fast path for a / b (sm_86): the reciprocal approximation r, one Newton step on it, the quotient estimate, its residual and the correction. Done
// with the fused multiply-add the card has.
static uint32_t idiom_fast_path(uint32_t a_bits, uint32_t b_bits, uint32_t r_bits) {
  const float a = std::bit_cast<float>(a_bits), b = std::bit_cast<float>(b_bits), r = std::bit_cast<float>(r_bits);
  const float e = std::fmaf(-b, r, 1.0f);           // FFMA R9, -R3, R8, 1
  const float r2 = std::fmaf(r, e, r);              // FFMA R9, R8, R9, R8
  const float q0 = std::fmaf(a, r2, 0.0f);          // FFMA R8, R0, R9, RZ
  const float res = std::fmaf(-b, q0, a);           // FFMA R10, -R3, R8, R0
  const float q = std::fmaf(r2, res, q0);           // FFMA R9, R9, R10, R8
  return std::bit_cast<uint32_t>(q);
}

// For a divisor and a dividend whose quotient is a normal number and neither is near the range's ends (the operands FCHK lets through to the fast path:
// the others take the slow path, which the card run covers), the result must be the correctly rounded quotient for the simulator's reciprocal: that has to be
// exactly right, always. For a reciprocal one ulp either side of it (the card's is within one ulp of the correctly rounded one on 13% of mantissas) the
// refinement is right except at a handful of points: the exhaustive run (VGPU_EXHAUSTIVE=1, 1.0e9 cases) finds exactly one, a = 1.0, b = 0x3fffffff, whose
// quotient 0.50000003 sits next to a power of two. Whether the card's own reciprocal avoids that point is what tools/probes/hostgpu_sfu_exhaustive.cu div
// answers on a card; until it has, the idiom stays behind VGPU_HOST_GPU_DIV. So: zero failures for the simulator's reciprocal, and no more than one case in a
// million for the one-ulp-off ones.
VTEST(division_refinement_is_correctly_rounded_for_the_simulators_reciprocal) {
  const uint32_t stride = exhaustive() ? 1 : 97;
  const uint32_t a_mants[] = {0x000000u, 0x7fffffu, 0x400000u, 0x123457u, 0x2aaaaau, 0x555555u, 0x7ffffeu, 0x000001u};
  const uint32_t a_exps[] = {127, 100, 150, 127 + 60, 127 - 60};
  uint64_t bad[3] = {0, 0, 0}, total = 0;
  uint32_t first_a = 0, first_b = 0;
  int first_variant = -1;
  for (uint32_t bm = 0; bm < (1u << 23); bm += stride) {
    for (uint32_t be : {127u, 127u + 40, 127u - 40}) {
      const uint32_t b = be << 23 | bm;
      const uint32_t exact_r = ref::mufu(ref::Fn::Rcp, b);
      const uint32_t rs[3] = {exact_r, exact_r + 1, exact_r - 1};
      for (uint32_t ae : a_exps)
        for (uint32_t am : a_mants) {
          const uint32_t a = ae << 23 | am;
          const uint32_t want = ref::fdiv_rn(a, b);
          if ((want & 0x7f800000u) == 0 || (want & 0x7f800000u) == 0x7f800000u) continue;   // the quotient must be a normal number
          ++total;
          for (int v = 0; v < 3; ++v)
            if (idiom_fast_path(a, b, rs[v]) != want && !bad[v]++ && first_variant < 0) { first_a = a; first_b = b; first_variant = v; }
        }
    }
  }
  VCHECK(total > 100000);
  if (bad[0] || bad[1] > total / 1000000 || bad[2] > total / 1000000)
    throw vtest::Failure("fast path differs from the correctly rounded quotient: simulator's reciprocal " + std::to_string(bad[0]) + ", +1 ulp " + std::to_string(bad[1]) +
                         ", -1 ulp " + std::to_string(bad[2]) + " of " + std::to_string(total) + "; first a=0x" + hex(first_a) + " b=0x" + hex(first_b) + " variant " +
                         std::to_string(first_variant));
}

// The check above would pass for any refinement that ignores r: show it can fail. A reciprocal that is 5% off is not repaired by the one Newton step
// plus correction (this is why the card's 1-ulp bound matters), so the refinement test has teeth.
VTEST(division_refinement_test_can_fail) {
  uint64_t bad = 0, total = 0;
  for (uint32_t bm = 0; bm < (1u << 23); bm += 4093) {
    const uint32_t b = 127u << 23 | bm, a = 0x3f9e3779u;
    const uint32_t r = ref::mufu(ref::Fn::Rcp, b) + 400000;   // 5% off
    ++total;
    bad += idiom_fast_path(a, b, r) != ref::fdiv_rn(a, b);
  }
  VCHECK(total > 1000);
  VCHECK(bad > 0);
}

// ---- the integer division idiom (sp43; docs/performance.md "Round sp43", finding 3) ------------------------------------------------------------------------
// Most MUFU.RCP of the refused kernels are not float divisions at all: they are the first step of ptxas's 32-bit integer `/` and `%` by a run-time divisor
// (HeCBench mask, matrixT, md5hash, Rodinia backprop: I2F.RP, MUFU.RCP, IADD3 +0xffffffe, F2I.FTZ.U32.TRUNC.NTZ, a Newton step with IMAD.HI.U32, the
// quotient estimate, and two corrections). The result is the exact quotient whatever the last bits of the reciprocal are, as long as the estimate is an
// underestimate of 2^32 / y that is less than 2 off. This is the sequence of `mask-cuda`'s kernels (cuobjdump of sm_86, ptxas 13.0) as code:
//   R4 = I2F.RP(y)          the float above y (round toward +inf)
//   R2 = MUFU.RCP(R4)       the card's reciprocal (the simulator's: correctly rounded; the card's: within one ulp)
//   R4 = R2 + 0xffffffe     as integer bits: times 2^32 and a little under (two units in the 23rd place)
//   q0 = F2I.FTZ.U32.TRUNC.NTZ(R4)
//   t  = (-q0) * y          low 32 bits
//   q1 = IMAD.HI.U32(q0, t, {q0:0})  = q0 + (q0 * t >> 32)       (the 64-bit addend is the pair whose high word is q0)
//   q  = IMAD.HI.U32(q1, x, RZ);  r = x - q * y;  if (!(y > r)) { r -= y; ++q; }  if (r >= y) ++q;
// What is proven here, with no card: for EVERY divisor y (VGPU_EXHAUSTIVE=1; a spread of 1.05e6 by default) the estimate q1 satisfies q1 * y <= 2^32 (never
// over) and (q1 + 2) * y > 2^32 (less than 2 under), for the simulator's reciprocal and for one that is up to 1 ulp off either way (the card's bound is 1 ulp of
// the exact value, the simulator's 0.5 ulp, so 2 covers any pair). Then q = floor(x * q1 / 2^32) is at most the true quotient and at most 2 below it for
// every x < 2^32 (x / y - x * q1 / 2^32 < 2 x / 2^32 < 2), which the two corrections repair: the result is x / y exactly on the card (C semantics) and in
// the simulator, whatever the reciprocal's low bits. The test is the analysis; whether a kernel's instructions really are this chain is the recognizer's job
// (not written: see the doc, and tools/probes/hostgpu_mufu_card_experiment.sh for the card run that gates it).
namespace {
uint32_t idiv_q0(uint32_t y, int ulps) {
  float f = static_cast<float>(y);                                    // I2F.RP: round toward +inf
  if (static_cast<double>(f) < static_cast<double>(y)) f = std::nextafter(f, INFINITY);
  const uint32_t r = ref::mufu(ref::Fn::Rcp, std::bit_cast<uint32_t>(f)) + static_cast<uint32_t>(ulps);   // the reciprocal, shifted by whole ulps (a normal positive float)
  const float g = std::bit_cast<float>(r + 0x0ffffffeu);
  return g >= 4294967296.0f ? 0xffffffffu : static_cast<uint32_t>(g);   // F2I.FTZ.U32.TRUNC (a float below 2^32 truncates; the instruction saturates above)
}
uint32_t idiv_q1(uint32_t y, int ulps) {
  const uint32_t q0 = idiv_q0(y, ulps);
  const uint32_t t = (0u - q0) * y;
  return q0 + static_cast<uint32_t>((static_cast<uint64_t>(q0) * t) >> 32);
}
uint32_t idiv_model(uint32_t x, uint32_t y, int ulps) {
  const uint32_t q1 = idiv_q1(y, ulps);
  uint32_t q = static_cast<uint32_t>((static_cast<uint64_t>(q1) * x) >> 32);
  uint32_t r = x - q * y;
  if (!(y > r)) { r -= y; ++q; }
  if (r >= y) ++q;
  return q;
}
// 0 when the estimate for divisor y is an underestimate less than 2 off.
bool idiv_estimate_ok(uint32_t y, int ulps) {
  const uint32_t q1 = idiv_q1(y, ulps);
  const unsigned __int128 two32 = static_cast<unsigned __int128>(1) << 32;
  return static_cast<unsigned __int128>(q1) * y <= two32 && static_cast<unsigned __int128>(q1 + 2ull) * y > two32;
}
}  // namespace

VTEST(integer_division_estimate_is_a_close_underestimate_for_every_divisor) {
  const uint32_t step = exhaustive() ? 1 : 4093;
  uint64_t bad[3] = {0, 0, 0}, tested = 0;
  uint32_t first = 0;
  const auto check = [&](uint32_t y) {
    ++tested;
    for (int u = -1; u <= 1; ++u)
      if (!idiv_estimate_ok(y, u) && !bad[u + 1]++ && !first) first = y;
  };
  for (uint64_t y = 1; y < (1ull << 32); y += step) check(static_cast<uint32_t>(y));
  for (int b = 0; b < 32; ++b)   // every power of two and its neighbours, where the float rounding and the exponent change
    for (int d = -3; d <= 3; ++d) {
      const int64_t y = (int64_t{1} << b) + d;
      if (y >= 1 && y < (int64_t{1} << 32)) check(static_cast<uint32_t>(y));
    }
  for (uint32_t y : {0xffffffffu, 0xfffffffeu, 0x80000000u, 0x7fffffffu, 3u, 5u, 7u, 10u, 641u}) check(y);
  VCHECK(tested > 1000000);
  std::string report;
  for (int u = 0; u < 3; ++u)
    if (bad[u]) report += " reciprocal " + std::to_string(u - 1) + " ulps off: " + std::to_string(bad[u]) + " divisors;";
  if (!report.empty()) throw vtest::Failure("the estimate is not a close underestimate:" + report + " of " + std::to_string(tested) + ", first y=0x" + hex(first));
}

VTEST(integer_division_with_a_reciprocal_up_to_1_ulp_off_is_exact) {
  std::mt19937_64 rng(7);
  const uint32_t edges[] = {0u, 1u, 2u, 3u, 0x7fffffffu, 0x80000000u, 0x80000001u, 0xfffffffeu, 0xffffffffu, 12345u, 0x01234567u};
  uint64_t n = 0;
  const auto check = [&](uint32_t x, uint32_t y) {
    if (!y) return;
    for (int u = -1; u <= 1; ++u) {
      ++n;
      const uint32_t got = idiv_model(x, y, u);
      if (got != x / y) throw vtest::Failure("x=0x" + hex(x) + " y=0x" + hex(y) + ": quotient 0x" + hex(got) + " instead of 0x" + hex(x / y) + " with the reciprocal " + std::to_string(u) + " ulps off");
    }
  };
  for (uint32_t x : edges)
    for (uint32_t y : edges) check(x, y);
  const uint64_t pairs = exhaustive() ? 400000000ull : 1000000ull;
  for (uint64_t i = 0; i < pairs; ++i) {
    const uint64_t r = rng();
    uint32_t x = static_cast<uint32_t>(r), y = static_cast<uint32_t>(r >> 32);
    switch (i % 4) {   // divisors of every magnitude, and dividends next to multiples of the divisor (where a quotient is one off)
      case 1: y >>= (r >> 59) & 31; break;
      case 2: if (y) { const uint32_t k = static_cast<uint32_t>(rng()) % (0xffffffffu / y + 1); x = k * y + (static_cast<uint32_t>(r) % 3) - 1; } break;
      case 3: y = (y >> 24) | 1u; break;
      default: break;
    }
    check(x, y);
  }
  VCHECK(n > 3000000);
}

// The proof above has teeth: a reciprocal 100 ulps off, an estimate pushed 4 above, and a reciprocal 2 ulps off (the margin of the offset is one ulp) all fail the bounds.
VTEST(integer_division_bounds_can_fail) {
  uint64_t far = 0, over = 0, two = 0;
  for (uint64_t y = 3; y < (1ull << 32); y += 99991) {
    const uint32_t yy = static_cast<uint32_t>(y);
    if (!idiv_estimate_ok(yy, 100) || !idiv_estimate_ok(yy, -100)) ++far;
    over += static_cast<unsigned __int128>(idiv_q1(yy, 0) + 4ull) * yy > (static_cast<unsigned __int128>(1) << 32);
    if (!idiv_estimate_ok(yy, 2)) ++two;
  }
  for (uint64_t y = 3; y < (1ull << 32); y += 1021) two += !idiv_estimate_ok(static_cast<uint32_t>(y), 2);
  VCHECK(far > 0);
  VCHECK(over > 0);
  VCHECK(two > 0);
}

// ---- the square root idiom (include/vgpu/host_gpu_idiom.hpp, Round sp68): nvcc's sqrt.rn for sm_86 as scalar code ----
// Fast path (x - 0x0d000000 <= 0x727fffff unsigned: positive floats from 2^-101 up): y = MUFU.RSQ(x), s = y x (FMUL.FTZ), h = y / 2 (FMUL.FTZ),
// d = fma(-s, s, x), r = fma(d, h, s). Slow path for the other positive finite floats: the same on x 2^64 (an FFMA, exact), the result times 2^-32
// (FMUL.FTZ). `k` moves the reciprocal square root k ulp away from the simulator's (the card's differs from it by up to 2 ulp on 10.7% of inputs, sp44).
namespace {
uint32_t sqrt_idiom_model(uint32_t x, int k) {
  const auto f = [](uint32_t b) { return std::bit_cast<float>(b); };
  const auto ftz = [](float v) { return ref::flush(v); };
  if (x - 0x0d000000u <= 0x727fffffu) {
    const float xf = f(x), y = f(ref::mufu(ref::Fn::Rsq, x) + static_cast<uint32_t>(k));
    const float s = ftz(y * xf), h = ftz(y * 0.5f);
    return std::bit_cast<uint32_t>(std::fmaf(std::fmaf(-s, s, xf), h, s));
  }
  const float sc = std::fmaf(f(x), 18446744073709551616.0f, 0.0f);
  const float y = f(ref::mufu(ref::Fn::Rsq, std::bit_cast<uint32_t>(sc)) + static_cast<uint32_t>(k));
  const float t = ftz(sc * y), h = ftz(y * 0.5f);
  const float z = std::fmaf(std::fmaf(t, -t, sc), h, t);
  return std::bit_cast<uint32_t>(ftz(z * 2.3283064365386962891e-10f));
}
}  // namespace

VTEST(sqrt_refinement_is_correctly_rounded_for_the_simulators_reciprocal_square_root) {
  // every positive finite float with VGPU_EXHAUSTIVE=1 (2^31 inputs, about a minute), else a stride (zero, negatives, infinities and NaNs never reach
  // the refinement: the slow path returns them before its MUFU)
  const uint64_t stride = exhaustive() ? 1 : 4099;
  uint64_t wrong = 0, tested = 0;
  uint32_t first = 0;
  for (uint64_t x = 1; x < 0x7f800000u; x += stride, ++tested)
    if (sqrt_idiom_model(static_cast<uint32_t>(x), 0) != ref::fsqrt_rn(static_cast<uint32_t>(x)) && !wrong++) first = static_cast<uint32_t>(x);
  for (uint32_t x : {0x00000001u, 0x007fffffu, 0x00800000u, 0x0cffffffu, 0x0d000000u, 0x3f800000u, 0x7f7fffffu, 0x0d6e9372u, 0x7f6e9372u, 0x0d4d142au})
    wrong += sqrt_idiom_model(x, 0) != ref::fsqrt_rn(x);
  if (wrong) std::fprintf(stderr, "  %llu of %llu wrong, first 0x%s\n", static_cast<unsigned long long>(wrong), static_cast<unsigned long long>(tested), hex(first).c_str());
  VCHECK_EQ(wrong, uint64_t{0});
}

VTEST(sqrt_refinement_is_not_exact_for_every_reciprocal_square_root_one_ulp_off) {
  // Unlike the division (exact for a reciprocal one ulp either way), the square root's refinement gives a wrong root for a few inputs when the reciprocal
  // square root is one ulp from the simulator's: of the 2^31 positive floats, 383 at -1 ulp and 255 at +1 (1,401 and 1,273 at 2 ulp; the program
  // docs/perf-design/data/sp68/sqrt_model.cpp ran them all). So the idiom is exact on a card only if its MUFU.RSQ is right where these inputs need it,
  // which the card run checks (sp44: the A10G's was, on all 2^32 inputs); a card of another architecture needs its own run.
  for (uint32_t x : {0x0d6e9372u, 0x3f6e9372u, 0x7f6e9372u, 0x007749b9u}) {
    VCHECK(sqrt_idiom_model(x, 0) == ref::fsqrt_rn(x));
    VCHECK(sqrt_idiom_model(x, +1) != ref::fsqrt_rn(x));
  }
  for (uint32_t x : {0x0d4d142au, 0x00668a15u}) {
    VCHECK(sqrt_idiom_model(x, 0) == ref::fsqrt_rn(x));
    VCHECK(sqrt_idiom_model(x, -1) != ref::fsqrt_rn(x));
  }
}

VTEST_MAIN
