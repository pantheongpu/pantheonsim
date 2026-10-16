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
#include <random>
#include <string>
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
float reference(const float* a, const float* b, unsigned K, unsigned chunk, int emin, bool acc16, float c) {
  const auto expo = [&](float v) { return std::max(std::ilogb(v), emin); };
  float acc = c;
  for (unsigned k0 = 0; k0 < K; k0 += chunk) {
    int top = INT32_MIN;
    for (unsigned t = 0; t < chunk; ++t)
      if (a[k0 + t] != 0 && b[k0 + t] != 0) top = std::max(top, expo(a[k0 + t]) + expo(b[k0 + t]) + 1);
    // (An accumulator that overflowed in an earlier chunk is an infinity: ilogb gives INT_MAX and +1 overflows. It has taken part as
    // INT32_MIN, which is what the wrapped sum was.)
    if (acc != 0) top = std::max(top, std::isinf(acc) ? INT32_MIN : std::ilogb(acc) + 1);
    if (top == INT32_MIN) {
      acc = 0.0f;
      continue;
    }
    const auto q = [&](double v) { return std::trunc(std::ldexp(v, 25 - top)); };   // in units of the grid
    double sum = q(acc);
    for (unsigned t = 0; t < chunk; ++t) sum += q(static_cast<double>(a[k0 + t]) * static_cast<double>(b[k0 + t]));
    const double v = std::ldexp(sum, top - 25);
    if (std::fabs(v) >= 0x1p128) {   // past the f32 range (and the f16 range, below): the infinity (checked against the card chunk by chunk)
      acc = std::signbit(v) ? -std::numeric_limits<float>::infinity() : std::numeric_limits<float>::infinity();
      continue;
    }
    if (acc16) {
      acc = vgpu::simd::f16_to_f32(vgpu::simd::f64_to_f16_bits(v));
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
