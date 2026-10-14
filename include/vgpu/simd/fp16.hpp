// binary16 and bfloat16 conversions on the bits, for the hot paths of both
// engines. Each one gives the bits the compiler's own _Float16 conversion
// gives (libgcc's __extendhfsf2, __truncsfhf2, __truncdfhf2 on a baseline
// x86-64 build), for every input, NaN payloads included: the unit test
// tests/unit/test_simd.cpp checks all 65536 halves and, for the narrowing
// ones, the rounding boundaries and a large random sample.
//
// Why not the compiler's: a baseline build has no F16C, so every _Float16
// conversion is a call into libgcc -- 33% of omni_virus's time before this.
#pragma once

#include <bit>
#include <cstdint>

namespace vgpu::simd {

// binary16 -> binary32, exact. A NaN keeps its payload shifted up and is
// quieted, as the hardware conversion (VCVTPH2PS) and libgcc both do.
inline uint32_t f16_to_f32_bits(uint16_t h) {
  const uint32_t sign = (static_cast<uint32_t>(h) & 0x8000u) << 16;
  const uint32_t e = (h >> 10) & 0x1Fu;
  uint32_t m = h & 0x3FFu;
  if (e == 0x1F) return sign | 0x7F800000u | (m << 13) | (m ? 0x00400000u : 0u);
  if (e == 0) {
    if (m == 0) return sign;
    // A subnormal half is m * 2^-24: normalise it. m has its top bit at
    // 31 - clz(m); moving that bit to bit 10 takes clz(m) - 21 places.
    const int sh = std::countl_zero(m) - 21;
    m = (m << sh) & 0x3FFu;
    return sign | (static_cast<uint32_t>(113 - sh) << 23) | (m << 13);
  }
  return sign | ((e + 112u) << 23) | (m << 13);
}
inline float f16_to_f32(uint16_t h) { return std::bit_cast<float>(f16_to_f32_bits(h)); }

// binary32 -> binary16, to nearest even. Overflow gives infinity; a NaN keeps
// the top bits of its payload with the quiet bit set.
inline uint16_t f32_to_f16_bits(float f) {
  const uint32_t u = std::bit_cast<uint32_t>(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t mag = u & 0x7FFFFFFFu;
  if (mag > 0x7F800000u) return static_cast<uint16_t>(sign | 0x7C00u | 0x0200u | ((mag >> 13) & 0x3FFu));
  if (mag >= 0x477FF000u) return static_cast<uint16_t>(sign | 0x7C00u);   // >= 65520, or infinity
  if (mag >= 0x38800000u) {   // a normal half: exponent >= -14
    // Re-bias the exponent (127 -> 15) and round the 13 dropped bits to even;
    // a carry out of the mantissa lands in the exponent, which is the right answer.
    const uint32_t r = mag - 0x38000000u;
    return static_cast<uint16_t>(sign | ((r + 0x0FFFu + ((r >> 13) & 1u)) >> 13));
  }
  if (mag < 0x33000001u) return static_cast<uint16_t>(sign);   // <= 2^-25: half the smallest subnormal rounds to zero
  // A subnormal result: the 24-bit significand shifted right to the half's grid.
  const uint32_t e = mag >> 23;                       // 102..112
  const uint32_t sig = (mag & 0x7FFFFFu) | 0x800000u;
  const uint32_t shift = 126 - e;                     // 14..24
  const uint32_t q = sig >> shift, rem = sig & ((1u << shift) - 1u), half = 1u << (shift - 1);
  return static_cast<uint16_t>(sign | (q + (rem > half || (rem == half && (q & 1u)) ? 1u : 0u)));
}

// binary64 -> binary16 in one rounding (to nearest even), as _Float16(double)
// does -- not through float, which would round twice.
inline uint16_t f64_to_f16_bits(double d) {
  const uint64_t u = std::bit_cast<uint64_t>(d);
  const uint32_t sign = static_cast<uint32_t>(u >> 48) & 0x8000u;
  const uint64_t mag = u & 0x7FFFFFFFFFFFFFFFull;
  if (mag > 0x7FF0000000000000ull)
    return static_cast<uint16_t>(sign | 0x7C00u | 0x0200u | static_cast<uint32_t>((mag >> 42) & 0x3FFu));
  if (mag >= 0x40EFFE0000000000ull) return static_cast<uint16_t>(sign | 0x7C00u);   // >= 65520, or infinity
  const int e = static_cast<int>(mag >> 52) - 1023;
  if (e >= -14) {   // a normal half before rounding
    const uint64_t r = mag - (static_cast<uint64_t>(1023 - 15) << 52);   // exponent re-biased
    const uint64_t rest = r & ((1ull << 42) - 1);
    uint64_t v = r >> 42;
    if (rest > (1ull << 41) || (rest == (1ull << 41) && (v & 1))) ++v;
    return static_cast<uint16_t>(sign | static_cast<uint32_t>(v));
  }
  if (e < -25) return static_cast<uint16_t>(sign);   // below 2^-25, and every double subnormal: rounds to zero
  // A subnormal half (e = -25 reaches the smallest one by rounding up): the
  // 53-bit significand shifted right to the half's grid.
  const uint64_t sig = (mag & ((1ull << 52) - 1)) | (1ull << 52);
  const unsigned shift = static_cast<unsigned>(28 - e);   // 42 + (-14 - e)
  const uint64_t q = sig >> shift, rem = sig & ((1ull << shift) - 1), half = 1ull << (shift - 1);
  return static_cast<uint16_t>(sign | static_cast<uint32_t>(q + ((rem > half || (rem == half && (q & 1))) ? 1u : 0u)));
}

// bfloat16 is the top half of a binary32.
inline float bf16_to_f32(uint16_t h) { return std::bit_cast<float>(static_cast<uint32_t>(h) << 16); }
// binary32 -> bfloat16 to nearest even; a NaN keeps its top payload bits with the quiet bit set.
inline uint16_t f32_to_bf16_bits(float f) {
  uint32_t b = std::bit_cast<uint32_t>(f);
  if (f != f) return static_cast<uint16_t>((b >> 16) | 0x40u);
  b += 0x7FFFu + ((b >> 16) & 1u);
  return static_cast<uint16_t>(b >> 16);
}

}  // namespace vgpu::simd
