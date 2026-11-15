// The sum an RTX 3060's (sm_86) tensor core makes of f16, bf16 and tf32 products (include/vgpu/exec/tc_ampere.hpp): chunks of 8
// (tf32: 4) exact products plus the accumulator, aligned to the largest frame position, truncated at 2^(top - 25), added exactly,
// then truncated to f32 (rounded to f16 for an f16 accumulator).
//
//   * card_vectors: 40 results measured on the card (wmma and mma.sync), each one a result that the sequential f32 sum this
//     simulator used before gets wrong (tc86_card_vectors.inc);
//   * agrees_with_an_independent_form: the same rule written the other way round (value arithmetic in double with the host's
//     directed rounding, not integer alignment), over random operands of every format, narrow and wide exponents, subnormals;
//   * special_values: the zero's sign, an infinity carried from one chunk to the next, a NaN, as measured on the card.
// The engines' own sums (PTX wmma and mma, the SASS HMMA kernel and its general form) are checked against the card by
// nvidia/tests/e2e/float_half_forms.cu and simd_lane_forms.cu.
#include <cfenv>
#include <cmath>
#include <limits>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "vgpu/exec/tc_ampere.hpp"
#include "vtest.hpp"

using namespace vgpu::exec;

namespace {

float f(uint32_t u) { return std::bit_cast<float>(u); }
uint32_t bits(float x) { return std::bit_cast<uint32_t>(x); }

struct Vec {
  int fmt;      // 0 f16, 1 bf16, 2 tf32 (the exponent floor)
  unsigned K, chunk;
  int acc16;
  uint32_t a[16], b[16];
  uint32_t c, d;
};
const Vec kVectors[] = {
#include "tc86_card_vectors.inc"
};

#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
// The rule again, by value: each term truncated toward zero to a multiple of 2^(top - 25) in double (exact: the scaled terms are
// integers below 2^27), summed in double (exact), converted to f32 with the host rounding toward zero, or to f16 to nearest.
float reference(const float* a, const float* b, unsigned K, unsigned chunk, int emin, bool acc16, float c, int grid = 25) {
  const auto expo = [&](float v) { return std::max(std::ilogb(v), emin); };
  float acc = c;
  for (unsigned k0 = 0; k0 < K; k0 += chunk) {
    int top = INT32_MIN;
    for (unsigned t = 0; t < chunk; ++t)
      if (a[k0 + t] != 0 && b[k0 + t] != 0) top = std::max(top, expo(a[k0 + t]) + expo(b[k0 + t]) + 1);
    // (An accumulator that overflowed in an earlier chunk is an infinity: ilogb gives INT_MAX and +1 overflows. It has taken part as
    // INT32_MIN, which is what the wrapped sum was.)
    // (A subnormal accumulator counts at its format's smallest normal exponent; under a zero one the chunk's position has a floor.)
    if (acc != 0) top = std::max(top, std::isinf(acc) ? INT32_MIN : std::max(std::ilogb(acc) + 1, acc16 ? -13 : -125));
    else if (top != INT32_MIN) top = std::max(top, (acc16 ? 6 : -106) - grid);
    if (top == INT32_MIN) {
      acc = 0.0f;
      continue;
    }
    const auto q = [&](double v) { return std::trunc(std::ldexp(v, grid - top)); };   // in units of the grid
    double sum = q(acc);
    for (unsigned t = 0; t < chunk; ++t) sum += q(static_cast<double>(a[k0 + t]) * static_cast<double>(b[k0 + t]));
    const double v = std::ldexp(sum, top - grid);
    if (std::fabs(v) >= 0x1p128) {   // past the f32 range (and the f16 range, below): the infinity (checked against the card chunk by chunk)
      acc = std::signbit(v) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      continue;
    }
    if (acc16) {
      acc = vgpu::simd::f16_to_f32(vgpu::simd::f64_to_f16_bits(v));
      if (acc == 0) acc = 0.0f;
    } else {
      const int saved = std::fegetround();
      std::fesetround(FE_TOWARDZERO);
      volatile double vv = v;
      acc = static_cast<float>(vv);
      std::fesetround(saved);
      if (acc == 0) acc = 0.0f;
    }
  }
  return acc;
}
#pragma GCC pop_options

}  // namespace

VTEST(card_vectors) {
  int checked = 0;
  for (const Vec& v : kVectors) {
    float a[16], b[16], d;
    for (unsigned k = 0; k < v.K; ++k) {
      a[k] = f(v.a[k]);
      b[k] = f(v.b[k]);
    }
    tc86::mma(1, 1, v.K, v.chunk, v.fmt == 0 ? tc86::kEminF16 : tc86::kEminWide, v.acc16 != 0, a, v.K, b, 1, reinterpret_cast<const float*>(&v.c), 1, &d, 1);
    if (bits(d) != v.d) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "vector %d (fmt %d K %u): got 0x%08x, the card gave 0x%08x", checked, v.fmt, v.K, bits(d), v.d);
      throw vtest::Failure(buf);
    }
    ++checked;
  }
  VCHECK_EQ(checked, 40);
}

VTEST(agrees_with_an_independent_form) {
  std::mt19937 rng(86);
  const auto operand = [&](int fmt, bool wide) {
    const uint32_t h = static_cast<uint32_t>(rng());
    if (fmt == 0) {   // f16: any finite half, narrow or wide exponents, subnormals included
      const uint32_t e = wide ? (h >> 20) % 31 : 10 + (h >> 20) % 10;
      return vgpu::simd::f16_to_f32(static_cast<uint16_t>(((h & 0x8000u) | (e << 10) | ((h >> 1) & 0x3ffu))));
    }
    if (fmt == 1) {   // bf16: exponent field 0..254
      const uint32_t e = wide ? (h >> 20) % 255 : 115 + (h >> 20) % 24;
      return f(((h & 0x8000u) | (e << 7) | ((h >> 1) & 0x7fu)) << 16);
    }
    const uint32_t e = wide ? (h >> 20) % 255 : 115 + (h >> 20) % 24;   // tf32: 10 mantissa bits
    return f((h & 0x80000000u) | (e << 23) | ((h & 0x3ffu) << 13));
  };
  long checked = 0;
  for (int round = 0; round < 6000; ++round) {
    const int fmt = round % 3;
    const bool wide = (round / 3) % 2 == 1, acc16 = fmt == 0 && (round / 6) % 2 == 1;
    const unsigned K = fmt == 2 ? 8 : 16, chunk = fmt == 2 ? 4 : 8;
    float A[16 * 16], B[16 * 8], C[16 * 8], D[16 * 8];
    for (unsigned x = 0; x < 16 * K; ++x) A[x] = round % 7 == 0 && x % 5 == 0 ? 0.0f : operand(fmt, wide);
    for (unsigned x = 0; x < K * 8; ++x) B[x] = operand(fmt, wide);
    for (unsigned x = 0; x < 16 * 8; ++x) {
      const float c = fmt == 0 && acc16 ? operand(0, wide) : f((static_cast<uint32_t>(rng()) & 0x807fffffu) | ((100u + rng() % 60u) << 23));
      C[x] = round % 11 == 0 && x % 3 == 0 ? 0.0f : c;
    }
    tc86::mma(16, 8, K, chunk, fmt == 0 ? tc86::kEminF16 : tc86::kEminWide, acc16, A, K, B, 8, C, 8, D, 8);
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned j = 0; j < 8; ++j) {
        float a[16], b[16];
        for (unsigned k = 0; k < K; ++k) {
          a[k] = A[i * K + k];
          b[k] = B[k * 8 + j];
        }
        const float want = reference(a, b, K, chunk, fmt == 0 ? tc86::kEminF16 : tc86::kEminWide, acc16, C[i * 8 + j]);
        const float got = D[i * 8 + j];
        // (Past the f32 range the reference is the host's conversion and so is the kernel's ldexp: both give the infinity.)
        if (bits(got) != bits(want) && !(std::isnan(got) && std::isnan(want))) {
          char buf[200];
          std::snprintf(buf, sizeof buf, "round %d fmt %d acc16 %d element (%u, %u): got 0x%08x, the other form gives 0x%08x", round, fmt, acc16, i, j, bits(got),
                        bits(want));
          throw vtest::Failure(buf);
        }
        ++checked;
      }
  }
  VCHECK(checked > 100000);
}

VTEST(special_values) {
  const float inf = std::numeric_limits<float>::infinity(), nan = std::numeric_limits<float>::quiet_NaN();
  const auto one = [&](std::initializer_list<float> a, std::initializer_list<float> b, float c, unsigned K = 8, unsigned chunk = 8, bool a16 = false) {
    float x[16] = {}, y[16] = {}, d;
    unsigned i = 0;
    for (float v : a) x[i++] = v;
    i = 0;
    for (float v : b) y[i++] = v;
    tc86::mma(1, 1, K, chunk, tc86::kEminF16, a16, x, K, y, 1, &c, 1, &d, 1);
    return d;
  };
  // A zero sum is +0 whatever the signs of the terms (an RTX 3060: -0 products with c = -0 give +0).
  VCHECK_EQ(bits(one({-0.0f, -0.0f}, {1.0f, 1.0f}, -0.0f)), 0u);
  VCHECK_EQ(bits(one({1.0f, -1.0f}, {1.0f, 1.0f}, -0.0f)), 0u);
  VCHECK_EQ(bits(one({-1.0f}, {1.0f}, 1.0f)), 0u);
  // The accumulator alone is kept whole, subnormal or not.
  VCHECK_EQ(bits(one({0.0f}, {1.0f}, f(1u))), 1u);
  VCHECK_EQ(bits(one({0.0f, 5.9604645e-8f}, {1.0f, 5.9604645e-8f}, f(1u))), 0x27800000u);   // 2^-48 swallows 2^-149
  // Infinities and NaNs: IEEE on exact products.
  VCHECK_EQ(bits(one({inf}, {1.0f}, 0.0f)), 0x7f800000u);
  VCHECK_EQ(bits(one({inf}, {0.0f}, 0.0f)), 0x7fffffffu);
  VCHECK_EQ(bits(one({inf, -inf}, {1.0f, 1.0f}, 0.0f)), 0x7fffffffu);
  VCHECK_EQ(bits(one({1.0f}, {1.0f}, nan)), 0x7fffffffu);
  VCHECK_EQ(bits(one({1.0f}, {1.0f}, -inf)), 0xff800000u);
  // A finite sum that overflows is an infinity for the next chunk: a chunk of 8 whose products sum past 2^128 (bf16 range), then a
  // chunk with -inf is a NaN; with only finite terms it stays the infinity however big they are.
  const float big = f(0x7f7f0000u);   // bf16 max
  VCHECK_EQ(bits(one({big, big}, {big, big}, 0.0f, 16, 8)), 0x7f800000u);
  float a[16] = {}, b[16] = {}, c = 0, d;
  a[0] = big; b[0] = big;            // chunk 0: +big^2 -> +inf
  a[9] = -inf; b[9] = 1.0f;          // chunk 1: -inf
  tc86::mma(1, 1, 16, 8, tc86::kEminWide, false, a, 16, b, 1, &c, 1, &d, 1);
  VCHECK_EQ(bits(d), 0x7fffffffu);
  a[9] = -big; b[9] = big;           // chunk 1: a finite -big^2, however big: the infinity stays
  tc86::mma(1, 1, 16, 8, tc86::kEminWide, false, a, 16, b, 1, &c, 1, &d, 1);
  VCHECK_EQ(bits(d), 0x7f800000u);
}


// ---- The rule with other numbers (tc86::mma_wide), and the H100's ------------------------------------------------------------------

const Vec kHopperVectors[] = {
#include "tc90_card_vectors.inc"
};

// 48 results measured on an H100 PCIe (wmma f16 with f32 and f16 accumulators, bf16, tf32; finite operands), each one a result
// the RTX 3060's rule gets wrong: the Hopper rule gives the card's bits, and the Ampere rule and the sequential sum do not.
VTEST(hopper_card_vectors) {
  int checked = 0, ampere_differs = 0;
  for (const Vec& v : kHopperVectors) {
    float a[16], b[16], d, d_ampere;
    for (unsigned k = 0; k < v.K; ++k) {
      a[k] = f(v.a[k]);
      b[k] = f(v.b[k]);
    }
    const int emin = v.fmt == 0 ? tc86::kEminF16 : tc86::kEminWide;
    // (v.chunk: the tf32 vectors are wmma 16x16x8 results, two m16n8k4 on the card, so chunks of 4)
    tc86::mma(tc86::kHopperRule, v.fmt == 2, 1, 1, v.K, emin, v.acc16 != 0, a, v.K, b, 1, reinterpret_cast<const float*>(&v.c), 1, &d, 1, v.chunk);
    tc86::mma(tc86::kAmpereRule, v.fmt == 2, 1, 1, v.K, emin, v.acc16 != 0, a, v.K, b, 1, reinterpret_cast<const float*>(&v.c), 1, &d_ampere, 1, v.chunk);
    if (bits(d) != v.d) {
      char buf[160];
      std::snprintf(buf, sizeof buf, "H100 vector %d (fmt %d K %u acc16 %d): got 0x%08x, the card gave 0x%08x", checked, v.fmt, v.K, v.acc16, bits(d), v.d);
      throw vtest::Failure(buf);
    }
    ampere_differs += bits(d_ampere) != v.d;
    ++checked;
  }
  VCHECK_EQ(checked, 48);
  VCHECK_EQ(ampere_differs, 48);
}

// The general form with the RTX 3060's numbers is the RTX 3060's sum: the 40 card vectors, and random tiles against mma_scalar.
VTEST(the_general_form_with_amperes_numbers_is_the_ampere_sum) {
  for (const Vec& v : kVectors) {
    float a[16], b[16], d;
    for (unsigned k = 0; k < v.K; ++k) a[k] = f(v.a[k]), b[k] = f(v.b[k]);
    tc86::mma_wide(1, 1, v.K, v.chunk, 25, v.fmt == 0 ? tc86::kEminF16 : tc86::kEminWide, v.acc16 != 0, a, v.K, b, 1, reinterpret_cast<const float*>(&v.c), 1, &d, 1);
    VCHECK_EQ(bits(d), v.d);
  }
  std::mt19937 rng(90);
  const float inf = std::numeric_limits<float>::infinity();
  long checked = 0;
  for (int round = 0; round < 3000; ++round) {
    const int fmt = round % 3;
    const bool acc16 = fmt == 0 && (round / 3) % 2 == 1;
    const unsigned K = fmt == 2 ? 8 : 16, chunk = fmt == 2 ? 4 : 8;
    float A[16 * 16], B[16 * 8], C[16 * 8], D[16 * 8], W[16 * 8];
    const auto val = [&] {
      const uint32_t h = static_cast<uint32_t>(rng());
      if (h % 97 == 0) return h & 1 ? inf : -inf;
      if (h % 89 == 0) return std::numeric_limits<float>::quiet_NaN();
      if (h % 13 == 0) return 0.0f;
      if (fmt == 0) return vgpu::simd::f16_to_f32(static_cast<uint16_t>((h & 0x8000u) | (((h >> 20) % 31) << 10) | ((h >> 1) & 0x3ffu)));
      return f((h & 0x80000000u) | (((h >> 20) % 255) << 23) | ((h & (fmt == 1 ? 0x7fu : 0x3ffu)) << (fmt == 1 ? 16 : 13)));
    };
    for (float& x : A) x = val();
    for (float& x : B) x = val();
    for (float& x : C) x = acc16 ? vgpu::simd::f16_to_f32(static_cast<uint16_t>(rng() & 0xfbffu)) : f(static_cast<uint32_t>(rng()));
    const int emin = fmt == 0 ? tc86::kEminF16 : tc86::kEminWide;
    tc86::mma_scalar(16, 8, K, chunk, emin, acc16, A, K, B, 8, C, 8, D, 8);
    tc86::mma_wide(16, 8, K, chunk, 25, emin, acc16, A, K, B, 8, C, 8, W, 8);
    for (unsigned x = 0; x < 16 * 8; ++x, ++checked)
      if (bits(D[x]) != bits(W[x])) {
        char buf[160];
        std::snprintf(buf, sizeof buf, "round %d fmt %d acc16 %d element %u: the general form gives 0x%08x, mma_scalar 0x%08x", round, fmt, acc16, x, bits(W[x]), bits(D[x]));
        throw vtest::Failure(buf);
      }
  }
  VCHECK(checked > 300000);
}

// The Hopper numbers against the rule written the other way round (value arithmetic in double), over random finite operands.
VTEST(the_hopper_rule_agrees_with_an_independent_form) {
  std::mt19937 rng(9026);
  long checked = 0, differs_from_ampere = 0;
  for (int round = 0; round < 3000; ++round) {
    const int fmt = round % 3;
    const bool wide = (round / 3) % 2 == 1, acc16 = fmt == 0 && (round / 6) % 2 == 1;
    const unsigned K = fmt == 2 ? ((round / 12) % 2 ? 4 : 8) : (round / 12) % 2 ? 8 : 16, chunk = fmt == 2 ? 8 : 16;
    const auto operand = [&] {
      const uint32_t h = static_cast<uint32_t>(rng());
      if (fmt == 0) return vgpu::simd::f16_to_f32(static_cast<uint16_t>((h & 0x8000u) | ((wide ? (h >> 20) % 31 : 10 + (h >> 20) % 10) << 10) | ((h >> 1) & 0x3ffu)));
      const uint32_t e = wide ? (h >> 20) % 255 : 115 + (h >> 20) % 24;
      return fmt == 1 ? f(((h & 0x8000u) | (e << 7) | ((h >> 1) & 0x7fu)) << 16) : f((h & 0x80000000u) | (e << 23) | ((h & 0x3ffu) << 13));
    };
    float A[16 * 16], B[16 * 8], C[16 * 8], D[16 * 8], Da[16 * 8];
    for (unsigned x = 0; x < 16 * K; ++x) A[x] = round % 7 == 0 && x % 5 == 0 ? 0.0f : operand();
    for (unsigned x = 0; x < K * 8; ++x) B[x] = operand();
    for (unsigned x = 0; x < 16 * 8; ++x)
      C[x] = round % 11 == 0 && x % 3 == 0 ? 0.0f
             : acc16 ? vgpu::simd::f16_to_f32(static_cast<uint16_t>(rng() & 0x7bffu))
                     : f((static_cast<uint32_t>(rng()) & 0x807fffffu) | ((100u + rng() % 60u) << 23));
    const int emin = fmt == 0 ? tc86::kEminF16 : tc86::kEminWide;
    tc86::mma(tc86::kHopperRule, fmt == 2, 16, 8, K, emin, acc16, A, K, B, 8, C, 8, D, 8);
    tc86::mma(tc86::kAmpereRule, fmt == 2, 16, 8, K, emin, acc16, A, K, B, 8, C, 8, Da, 8);
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned j = 0; j < 8; ++j) {
        float a[16] = {}, b[16] = {};
        for (unsigned k = 0; k < K; ++k) a[k] = A[i * K + k], b[k] = B[k * 8 + j];
        // (the reference walks whole chunks: pad K = 8 to one chunk of 16 with zero products, which take no part)
        const float want = reference(a, b, chunk, chunk, emin, acc16, C[i * 8 + j], 26);
        const float got = D[i * 8 + j];
        if (bits(got) != bits(want) && !(std::isnan(got) && std::isnan(want))) {
          char buf[200];
          std::snprintf(buf, sizeof buf, "round %d fmt %d acc16 %d K %u element (%u, %u): got 0x%08x, the other form gives 0x%08x", round, fmt, acc16, K, i, j, bits(got), bits(want));
          throw vtest::Failure(buf);
        }
        differs_from_ampere += bits(got) != bits(Da[i * 8 + j]);
        ++checked;
      }
  }
  VCHECK(checked > 300000);
  VCHECK(differs_from_ampere > 1000);   // (the two rules are not the same rule)
}

// The four corners in the subnormal range, which the probes that first measured the rule did not reach. S1 and S2 are the two inputs
// that tell the floor under a zero accumulator (and, for S2, the cut of a subnormal f32 result) apart: an RTX 3060 gives 0x0000 and
// 0x00000041 (run 2026-11-15, nvcc 13.0), where the rule without them gave 0x0001 and 0x00000042. The others are the published
// models' answers (MMA-Sim, Xie et al., arXiv 2511.10909; Khattak and Mikaitis, arXiv 2512.07004), not yet run on a card of ours.
// Through the dispatcher (the vector kernels must hand these tiles to the scalar code) and through the scalar code itself.
VTEST(the_subnormal_corners) {
  const auto h = [](uint16_t b) { return vgpu::simd::f16_to_f32(b); };
  const auto bf = [](uint16_t b) { return f(static_cast<uint32_t>(b) << 16); };
  struct Case {
    const char* name;
    int emin;
    bool acc16;
    std::vector<float> a, b;
    float c;
    uint32_t ampere, hopper;   // an f16 result as the f16's bits
  };
  const std::vector<Case> cases = {
      {"S1: floor under a zero f16 accumulator (RTX 3060: 0x0000)", tc86::kEminF16, true, {h(0x1000), h(0x0001)}, {h(0x0400), h(0x0004)}, 0.0f, 0x0000u, 0x0001u},
      {"S2: floor and subnormal cut, bf16 (RTX 3060: 0x00000041)", tc86::kEminWide, false, {bf(0x1C81), bf(0x1A7F)}, {bf(0x1B01), bf(0x1A00)}, 0.0f, 0x00000041u, 0x00000042u},
      {"R1: an f32 result in the subnormal range is cut, not rounded", tc86::kEminWide, false,
       {f(0x9d3c0000u), f(0x8d1d0000u), f(0x014a0000u), f(0x19930000u), f(0x9c640000u), f(0x036f0000u), f(0x94a60000u), f(0x16c80000u)},
       {f(0x8e3e0000u), f(0x81620000u), f(0x973e0000u), f(0x84330000u), f(0xa27f0000u), f(0x095d0000u), f(0x10d80000u), f(0xa1d80000u)}, f(0x004ebd07u), 0x006b1f35u, 0u},
      {"R2: a zero accumulator and tiny products: +0", tc86::kEminWide, false,
       {f(0x123e0000u), f(0x0fe80000u), f(0x8d0b0000u), f(0x82610000u), f(0x0ac10000u), f(0x0b330000u), f(0x93fd0000u), f(0x20ee0000u)},
       {f(0x8e9a0000u), f(0x02650000u), f(0x86eb0000u), f(0x911a0000u), f(0x126a0000u), f(0x9d2c0000u), f(0x80020000u), f(0x01850000u)}, 0.0f, 0x00000000u, 0u},
      {"R3: a subnormal accumulator counts at 2^-126", tc86::kEminWide, false,
       {bf(0x836c), bf(0x0768), bf(0x182e), bf(0x83a3), bf(0x19f0), bf(0x8132), bf(0x814c), bf(0x0cc3)},
       {bf(0x1eeb), bf(0x099b), bf(0x9eff), bf(0x2345), bf(0x850d), bf(0x893a), bf(0x21de), bf(0x0c3e)}, f(0x00411ff8u), 0x00411fcdu, 0u},
      {"R4: a negative sum that rounds to zero is +0 (f16 accumulator)", tc86::kEminF16, true,
       {f(0x3c87c000u), f(0x3b3f0000u), f(0x38650000u), f(0x39afa000u), f(0xb991c000u), f(0x38c2a000u), f(0xbbeb6000u), f(0x37b08000u)},
       {f(0x3aa6a000u), f(0xbb728000u), f(0xbce7e000u), f(0xbac20000u), f(0x3b624000u), f(0xb5a80000u), f(0x3a860000u), f(0x3b194000u)}, 0.0f, 0x0000u, 0u},
  };
  const auto result_bits = [](float d, bool acc16) { return acc16 ? static_cast<uint32_t>(vgpu::simd::f64_to_f16_bits(static_cast<double>(d))) : bits(d); };
  for (const Case& c : cases) {
    // One element, and the same element in every place of a 16 x 8 tile (the shape the vector kernels take).
    float a1[8] = {}, b1[8] = {}, d;
    for (size_t k = 0; k < c.a.size(); ++k) a1[k] = c.a[k], b1[k] = c.b[k];
    tc86::mma_scalar(1, 1, 8, 8, c.emin, c.acc16, a1, 8, b1, 1, &c.c, 1, &d, 1);
    if (result_bits(d, c.acc16) != c.ampere) throw vtest::Failure(std::string(c.name) + ": the scalar rule gives " + std::to_string(result_bits(d, c.acc16)));
    float A[16 * 8], B[8 * 8], C[16 * 8], D[16 * 8];
    for (unsigned i = 0; i < 16; ++i)
      for (unsigned k = 0; k < 8; ++k) A[i * 8 + k] = a1[k];
    for (unsigned k = 0; k < 8; ++k)
      for (unsigned j = 0; j < 8; ++j) B[k * 8 + j] = b1[k];
    for (float& x : C) x = c.c;
    tc86::mma(tc86::kAmpereRule, false, 16, 8, 8, c.emin, c.acc16, A, 8, B, 8, C, 8, D, 8);
    for (unsigned x = 0; x < 16 * 8; ++x)
      if (result_bits(D[x], c.acc16) != c.ampere) throw vtest::Failure(std::string(c.name) + ": the dispatcher gives " + std::to_string(result_bits(D[x], c.acc16)));
    if (c.hopper != 0 || c.name[0] == 'S') {   // (S1, S2: an H100's floors are one lower, so it keeps what the Ampere cards cut off)
      tc86::mma(tc86::kHopperRule, false, 16, 8, 8, c.emin, c.acc16, A, 8, B, 8, C, 8, D, 8);
      if (result_bits(D[0], c.acc16) != c.hopper) throw vtest::Failure(std::string(c.name) + ": the Hopper rule gives " + std::to_string(result_bits(D[0], c.acc16)));
    }
  }
  // No -0 comes out of a finite chunk, f32 or f16, on either rule.
  VCHECK_EQ(bits(tc86::detail::to_f32_rz(-1, -200)), 0u);
  VCHECK_EQ(bits(tc86::detail::to_f16_rn(-1, -60)), 0u);
  VCHECK_EQ(bits(tc86::detail::to_f32_rz(-3, -150)), 0x80000001u);   // -1.5 * 2^-149 is cut to -2^-149, not rounded to -2^-148
  VCHECK_EQ(bits(tc86::detail::to_f32_rz64(-3, -150)), 0x80000001u);
}

// The FTTN paper's tests (Li et al., CCGrid 2024, arXiv 2403.00232) as fixed vectors on element (1,1) of D = A x B + C, f16 operands,
// by each rule: what its programs print on the cards. One extra bit and truncation on both; blocks of 8 on an RTX 3060, a T4, an A10G,
// an A100, an L4 and an L40S, of 16 on an H100; and its motivating GEMM (section V: D = C - A x B, K = 2^13) is 0 on the first and
// 191.875 on the second, as the paper reports for the A100 and the H100 and as our cards print.
VTEST(the_fttn_tests_by_rule) {
  const auto d11 = [](const tc86::SumRule& r, std::initializer_list<std::pair<int, std::pair<float, float>>> terms, float c) {
    float a[16] = {}, b[16] = {}, d;
    for (const auto& t : terms) a[t.first] = t.second.first, b[t.first] = t.second.second;
    tc86::mma(r, false, 1, 1, 16, tc86::kEminF16, false, a, 16, b, 1, &c, 1, &d, 1);
    return bits(d);
  };
  const float half_ulp_a = 0x1p-10f, half_ulp_b = 0x1p-14f;   // their product is 2^-24: half an ulp of 1
  for (const tc86::SumRule* r : {&tc86::kAmpereRule, &tc86::kHopperRule}) {
    VCHECK_EQ(d11(*r, {{0, {0x1p-24f, 1.0f}}}, 0.0f), bits(0x1p-24f));                                   // T_si_no: a subnormal half
    VCHECK_EQ(d11(*r, {}, 0x1p-127f), bits(0x1p-127f));                                                  // T_sa: a subnormal C
    VCHECK_EQ(d11(*r, {{0, {0x1p-14f, 0x1p-9f + 0x1p-10f}}, {1, {1.0f, 2.0f}}}, 0.0f), bits(2.0f));       // T_rnd_dir: toward zero
    VCHECK_EQ(d11(*r, {{0, {0x1p-14f, -0x1p-9f - 0x1p-10f}}, {1, {1.0f, -2.0f}}}, 0.0f), bits(-2.0f));
    VCHECK_EQ(d11(*r, {{0, {1.0f, 1.0f}}}, -std::nextafterf(1.0f, 0.0f)), bits(0x1p-24f));               // T_1_bit: one extra bit
    // T_blk_fma_width: 1 in C and two half ulps, one at k = 0 and one at k = j. In one block they make an ulp; apart, each is cut off.
    for (int j = 1; j < 16; ++j) {
      const bool one_block = j < static_cast<int>(r->chunk);
      VCHECK_EQ(d11(*r, {{0, {half_ulp_a, half_ulp_b}}, {j, {half_ulp_a, half_ulp_b}}}, 1.0f), one_block ? 0x3f800001u : 0x3f800000u);
    }
  }
  // T_3_bits_fin_rnd under truncation: 1.25 - (2^-3 + 2^-25). One extra bit: the 2^-25 is cut off, 1.125. Two: it is kept, and the
  // sum is cut to 1.125 - 2^-23. (FTTN prints "Only one extra bit" on the Ampere-rule cards and "2 or 3 extra bits" on an H100.)
  VCHECK_EQ(d11(tc86::kAmpereRule, {{0, {1.0f, 1.25f}}}, -(0x1p-3f + 0x1p-25f)), 0x3f900000u);
  VCHECK_EQ(d11(tc86::kHopperRule, {{0, {1.0f, 1.25f}}}, -(0x1p-3f + 0x1p-25f)), 0x3f8fffffu);
  // The motivating example: 512 tiles of K = 16 into one accumulator from zero, then D = C - accumulator.
  const auto motivating = [](const tc86::SumRule& r) {
    float acc = 0.0f;
    for (int t = 0; t < 512; ++t) {
      float a[16], b[16];
      for (int k = 0; k < 16; ++k) {
        const int kk = t * 16 + k;
        a[k] = kk == 0 ? 0x1p10f : kk % 2 ? 0x1p-2f : 0x1p-3f;
        b[k] = kk == 0 ? 0x1p10f : -0x1p-3f;
      }
      float d;
      tc86::mma(r, false, 1, 1, 16, tc86::kEminF16, false, a, 16, b, 1, &acc, 1, &d, 1);
      acc = d;
    }
    return 0x1p20f - acc;
  };
  VCHECK_EQ(bits(motivating(tc86::kAmpereRule)), 0u);              // V100, A100 in the paper; RTX 3060, T4, A10G, A100, L4, L40S here
  VCHECK_EQ(bits(motivating(tc86::kHopperRule)), 0x433fe000u);     // 191.875: the paper's H100 and ours
}

#ifdef VGPU_TC86_AVX2
// The vector code against the scalar code, bit for bit: every shape the engines use, finite operands over wide exponent ranges with
// zeros and subnormals, and operands and accumulators that are infinities and NaNs (those lanes are finished by the scalar code).
VTEST(avx2_equals_scalar) {
  if (!__builtin_cpu_supports("avx2")) return;
  std::mt19937 rng(11);
  struct Shape { unsigned M, N, K, chunk; };
  const Shape shapes[] = {{16, 8, 16, 8}, {16, 8, 8, 8}, {16, 16, 16, 8}, {16, 16, 8, 4}, {8, 32, 16, 8}, {32, 8, 16, 8}, {16, 8, 4, 4}};
  long checked = 0;
  for (int round = 0; round < 3000; ++round) {
    const Shape sh = shapes[round % 7];
    const int fmt = round % 3;
    const bool wide = (round / 3) % 2 == 1, special = (round / 6) % 5 == 0;
    const auto operand = [&]() -> float {
      const uint32_t h = static_cast<uint32_t>(rng());
      if (special && h % 17 == 0) {
        static const float sp[] = {0.0f, -0.0f, std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                   std::numeric_limits<float>::quiet_NaN(), 1.0f, -1.0f};
        return sp[(h >> 8) % 7];
      }
      if (h % 13 == 0) return 0.0f;
      if (fmt == 0) {
        const uint32_t e = wide ? (h >> 20) % 31 : 8 + (h >> 20) % 14;
        return vgpu::simd::f16_to_f32(static_cast<uint16_t>(((h & 0x8000u) | (e << 10) | ((h >> 1) & 0x3ffu))));
      }
      if (fmt == 1) {
        const uint32_t e = wide ? (h >> 20) % 255 : 110 + (h >> 20) % 40;
        return f(((h & 0x8000u) | (e << 7) | ((h >> 1) & 0x7fu)) << 16);
      }
      const uint32_t e = wide ? (h >> 20) % 255 : 110 + (h >> 20) % 40;
      return f((h & 0x80000000u) | (e << 23) | ((h & 0x3ffu) << 13));
    };
    std::vector<float> A(sh.M * sh.K), B(sh.K * sh.N), C(sh.M * sh.N), D1(sh.M * sh.N), D2(sh.M * sh.N);
    for (float& x : A) x = operand();
    for (float& x : B) x = operand();
    for (float& x : C) {
      const uint32_t h = static_cast<uint32_t>(rng());
      x = special && h % 11 == 0 ? std::numeric_limits<float>::infinity()
          : h % 9 == 0          ? 0.0f
                                : f((h & 0x807fffffu) | ((wide ? 1u + (h >> 8) % 254 : 100u + (h >> 8) % 60) << 23));
    }
    const int emin = fmt == 0 ? tc86::kEminF16 : tc86::kEminWide;
    tc86::mma_scalar(sh.M, sh.N, sh.K, sh.chunk, emin, false, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D1.data(), sh.N);
    tc86::mma_avx2(sh.M, sh.N, sh.K, sh.chunk, emin, A.data(), sh.K, B.data(), sh.N, C.data(), sh.N, D2.data(), sh.N);
    for (size_t e = 0; e < D1.size(); ++e) {
      ++checked;
      if (bits(D1[e]) != bits(D2[e])) {
        char buf[200];
        std::snprintf(buf, sizeof buf, "round %d shape %ux%ux%u fmt %d element %zu: scalar 0x%08x, avx2 0x%08x", round, sh.M, sh.N, sh.K, fmt, e, bits(D1[e]),
                      bits(D2[e]));
        throw vtest::Failure(buf);
      }
    }
  }
  VCHECK(checked > 100000);
}
#endif

VTEST_MAIN
