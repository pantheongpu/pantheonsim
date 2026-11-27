// Monotonicity of the multi-term adder inside the tensor cores (issue #664; Mikaitis, "Monotonicity of Multi-Term Floating-Point Adders",
// arXiv 2304.01407). The paper's point: a block adder that aligns every term to the largest exponent and truncates what falls off is not
// monotonic. Raise one input so that it passes a power of two and the alignment of the others moves up one place; the bits they lose can
// outweigh the increase, and the result goes DOWN. The effect needs the cut to be coarser than one unit of the raised term, so it exists for
// every block width and every alignment grid the rules here use. Two checks, on the rule of tc_ampere.hpp with every (block width, grid)
// the profiles name (Volta 4/24, Ampere 8/25 and tf32 4/25, Hopper 16/26 and tf32 8/26):
//   * a_decrease_exists: an accumulator c below 2 and the same c = 2 (one unit in the last place higher), with the other terms of the chunk
//     just under the coarser cut: the result with the larger accumulator is smaller. This is the paper's adversarial pattern, found by a
//     scan, so a change of the alignment (a rounding adder, a wider grid) would make the check fail and say so;
//   * monotone_at_fixed_top: with the largest term fixed (so the alignment does not move), raising any other product never lowers the
//     result: each truncated term is monotone in its input and so is their sum. Random chunks, f32 result.
// No card is involved: the numbers are the rules' own, which cards of ours measured (docs/numerics/per-profile.md).
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

#include "vgpu/exec/tc_ampere.hpp"
#include "vtest.hpp"

using namespace vgpu::exec;

namespace {

struct Rule {
  const char* name;
  unsigned chunk;
  int grid;
};
const Rule kRules[] = {{"volta 4/24", 4, 24}, {"ampere 8/25", 8, 25}, {"ampere tf32 4/25", 4, 25}, {"hopper 16/26", 16, 26}, {"hopper tf32 8/26", 8, 26}};

// One dot product of `chunk` products and an f32 accumulator under the rule: a[k] * 1 for every k.
float dot(const Rule& r, const float* a, float c) {
  float b[16];
  for (unsigned k = 0; k < r.chunk; ++k) b[k] = 1.0f;
  float d = 0;
  tc86::mma_wide(1, 1, r.chunk, r.chunk, r.grid, -126, false, a, r.chunk, b, 1, &c, 1, &d, 1);
  return d;
}

}  // namespace

VTEST(a_decrease_exists_for_every_block_width) {
  for (const Rule& r : kRules) {
    const float c_low = 2.0f - std::ldexp(1.0f, -23), c_high = 2.0f;   // c_high is the next f32 up
    bool found = false;
    float a[16];
    for (int k = r.grid - 3; k <= r.grid + 8 && !found; ++k)
      for (float m : {1.0f, 1.25f, 1.5f, 1.75f}) {
        for (unsigned i = 0; i < r.chunk; ++i) a[i] = m * std::ldexp(1.0f, -k);
        const float lo = dot(r, a, c_low), hi = dot(r, a, c_high);
        if (hi < lo) found = true;   // the larger accumulator gives the smaller result
      }
    if (!found) throw vtest::Failure(std::string(r.name) + ": no input pair in the scan makes the result fall as the accumulator rises");
  }
}

VTEST(monotone_at_fixed_top) {
  std::mt19937_64 rng(0x4d6f6e6f746f6e65ull);
  for (const Rule& r : kRules) {
    for (int trial = 0; trial < 20000; ++trial) {
      float a[16];
      // Products are multiples of 2^-12 below 1/4 in magnitude, exact; the accumulator in [1, 2) keeps the largest term and so the alignment.
      for (unsigned i = 0; i < r.chunk; ++i) a[i] = static_cast<float>(static_cast<int>(rng() % 2047) - 1023) * std::ldexp(1.0f, -12) / 4.0f;
      const float c = 1.0f + static_cast<float>(rng() % (1u << 20)) * std::ldexp(1.0f, -20);
      const unsigned k = static_cast<unsigned>(rng() % r.chunk);
      float raised[16];
      for (unsigned i = 0; i < r.chunk; ++i) raised[i] = a[i];
      raised[k] = a[k] + static_cast<float>(1 + rng() % 64) * std::ldexp(1.0f, -14);
      if (std::fabs(raised[k]) >= 0.25f) continue;
      const float d0 = dot(r, a, c), d1 = dot(r, raised, c);
      if (d1 < d0) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "%s trial %d: raising product %u lowered the result 0x%08x -> 0x%08x", r.name, trial, k, std::bit_cast<uint32_t>(d0),
                      std::bit_cast<uint32_t>(d1));
        throw vtest::Failure(buf);
      }
    }
  }
}

VTEST_MAIN
