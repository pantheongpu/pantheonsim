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


// ---- directed and single rounding of an exact value to f16 or bf16 (used by the SASS executor's conversions) ----
// A finite or infinite value to the nearest representable of a format with `mb` mantissa bits and `eb` exponent bits, in SASS rounding
// mode `rnd` (0 to nearest even, 1 toward -inf, 2 toward +inf, 3 toward zero); the integer significand `S` has its top bit at bit 63
// and stands for 1.xxx * 2^E.
inline uint32_t narrow_core(bool sign, uint64_t S, int E, unsigned mb, unsigned eb, unsigned rnd) {
  const int bias = (1 << (eb - 1)) - 1, emin = 1 - bias;
  const uint32_t inf_bits = ((1u << eb) - 1) << mb;
  const int shift = static_cast<int>(63 - mb) + (E < emin ? emin - E : 0);
  unsigned __int128 q, rem, half;
  const unsigned __int128 R = S;
  if (shift >= 127) { q = 0; rem = 1; half = static_cast<unsigned __int128>(1) << 126; }
  else { q = R >> shift; rem = R & ((static_cast<unsigned __int128>(1) << shift) - 1); half = static_cast<unsigned __int128>(1) << (shift - 1); }
  bool up;
  switch (rnd) {
    case 0: up = rem > half || (rem == half && (q & 1)); break;
    case 1: up = sign && rem != 0; break;
    case 2: up = !sign && rem != 0; break;
    default: up = false; break;
  }
  q += up;
  uint32_t bits;
  if (E < emin) bits = static_cast<uint32_t>(q);   // a subnormal (a carry into 1 << mb is the smallest normal)
  else bits = static_cast<uint32_t>(((E + bias) << mb) + (static_cast<int64_t>(q) - (int64_t{1} << mb)));
  if (E > bias + 1 || bits >= inf_bits) {   // past the format's range
    const bool to_inf = rnd == 0 || (rnd == 1 && sign) || (rnd == 2 && !sign);
    bits = to_inf ? inf_bits : inf_bits - 1;
  }
  return (sign ? 1u << (mb + eb) : 0u) | bits;
}
inline uint32_t narrow_from_double(double x, unsigned dsize, unsigned rnd) {   // x not a NaN; dsize 1 f16, 4 bf16
  const unsigned mb = dsize == 1 ? 10 : 7, eb = dsize == 1 ? 5 : 8;
  const uint64_t b = std::bit_cast<uint64_t>(x);
  const bool sign = b >> 63;
  const int e = static_cast<int>((b >> 52) & 0x7ff);
  uint64_t m = b & ((uint64_t{1} << 52) - 1);
  if (e == 0x7ff) return (sign ? 1u << (mb + eb) : 0u) | (((1u << eb) - 1) << mb);
  if (e == 0 && m == 0) return sign ? 1u << (mb + eb) : 0u;
  int E;
  uint64_t S;
  if (e == 0) {
    const int z = std::countl_zero(m);
    S = m << z;
    E = (63 - z) - 1074;   // m * 2^-1074, whose leading bit is bit 63 - z of m
  } else {
    S = (m | (uint64_t{1} << 52)) << 11;
    E = e - 1023;
  }
  return narrow_core(sign, S, E, mb, eb, rnd);
}

// a * b + c for two bfloat16 (or float, or half) values a and b and any double c, rounded to double TO ODD: the exact result when it
// fits, else whichever of the two neighbouring doubles has an odd last mantissa bit. A later rounding to a narrower format of that is the
// rounding of the exact value (the double has more than two bits beyond any narrow format's): rounding to nearest in between could land
// on the narrow format's tie and break it the wrong way (an RTX 3060: -(1.5 * 2^k) + 1.5 for bf16 operands is below the tie by far less than
// a double's ulp). The product a * b is exact in a double; the sum's rounding error comes from TwoSum.
inline double fma_round_odd(double a, double b, double c) {
  const double p = a * b;
  const double s = p + c;
  if (!(s == s) || s - s != 0) return s;   // a NaN or an infinity
  const double bv = s - p;
  const double err = (p - (s - bv)) + (c - bv);
  if (err == 0) return s;
  uint64_t u = std::bit_cast<uint64_t>(s);
  if (u & 1) return s;
  // toward the exact value, which is s + err
  if ((err > 0) == (s > 0)) ++u; else --u;
  if (u == 0x8000000000000000ull) return std::bit_cast<double>(uint64_t{0x8000000000000001ull});   // (s was -0: not reachable with err != 0)
  return std::bit_cast<double>(u);
}
}  // namespace vgpu::simd
