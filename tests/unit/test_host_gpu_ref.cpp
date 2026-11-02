// The scalar reference of the simulator's MUFU results (include/vgpu/host_gpu_ref.hpp) is the simulator's, and the IEEE division idiom's refinement gives the
// correctly rounded quotient whatever the last bit of the reciprocal. Both are what the card comparison (tools/probes/hostgpu_sfu_exhaustive.cu) relies on;
// neither needs a card. VGPU_EXHAUSTIVE=1 runs the whole space (every float for the SIMD kernels, every mantissa of the divisor with 24 dividends for the
// refinement); the default is a stride that finishes in seconds under the sanitizers.
#include <bit>
#include <cmath>
#include <cstdint>
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

VTEST_MAIN
