// The three instructions a float division is built from -- v_div_scale, v_div_fmas and v_div_fixup, in single and double precision --
// as plain functions of the operands, so the executor and a unit test run the same code.
//
// The sequence the compiler writes (LLVM's SITargetLowering::LowerFDIV32 and LowerFDIV64) is
//     Ds = div_scale(D, D, N)       Ns = div_scale(N, D, N)       r = rcp(Ds)
//     (Newton-Raphson steps on r, then q = Ns * r and residuals)
//     q = div_fmas(residual, r, q, VCC of Ns)                     div_fixup(q, D, N)
// and lands on the correctly rounded quotient N / D. What each step does is the CDNA3 guide's pseudocode (12.11, V_DIV_SCALE_F32,
// V_DIV_SCALE_F64, V_DIV_FMAS_F32, V_DIV_FMAS_F64, V_DIV_FIXUP_F32, V_DIV_FIXUP_F64), apart from the places listed here, where the
// text cannot be what the hardware does and the sequence above only works one way:
//   * The guide has v_div_fmas multiply by 2^32 (f32) and 2^64 (f64) where v_div_scale scales by 2^64 and 2^128; every ISA guide
//     (GCN3 to CDNA5, RDNA2 to RDNA4) says so. Taken literally no pairing of the two divides correctly, and a fixed positive factor
//     cannot undo both of the scalings v_div_scale makes (the denominator up, the numerator up). LLVM's own workaround for the SI
//     chips whose v_div_scale condition output is unusable (SIISelLowering.cpp, LowerFDIV64: "Scale = XOR(CmpNum, CmpDen)", where
//     each compare says that operand was left unscaled) shows what VCC means: exactly one of the two operands was scaled, so the
//     quotient is off by a power of two and needs undoing. Which way is not in VCC. ASSUMED here: v_div_fmas reads it from the
//     quotient estimate it is given as S2 -- if that is below 1.0 the numerator was scaled up and the result is brought back down by
//     2^64 (2^128), otherwise the denominator was scaled up and it comes back up. The factors are 2^64 and 2^128, the ones v_div_scale
//     scales by.
//   * The result is scaled before it is rounded (the guide: "scale the exponent of the result"), so a quotient that lands in the
//     denormal range is rounded once. ASSUMED, as a second rounding would make the sequence inexact for those quotients.
//   * v_div_scale's branch for a denominator whose reciprocal is denormal and whose quotient is denormal scales the denominator by
//     ldexp(.., 64) in the guide. A denominator that large (above 2^126) overflows when scaled up, so this is read as scaling it
//     down, 2^-64, with VCC set (the quotient estimate then comes out 2^64 too large and the post-scale in v_div_fmas undoes it).
//   * "1.0 / S1.f32 == DENORM.f64" is read as "the f32 reciprocal of S1 is denormal" (|S1| > 2^126), as the f64 form says it.
//   * A value v_div_scale writes when none of its branches does is S0 itself (VCC set or not): the compiler's sequence needs the
//     operand it is not scaling to come back unchanged.
//   * The guide's v_div_fixup tests "exponent(S1) == 255" for OVERFLOW, which cannot be it (a denominator of 255 is an infinity or a NaN,
//     taken above). The quotient certainly overflows when the exponents are 129 (f32) or 1025 (f64) or more apart, and then the
//     scaled sequence has produced a NaN or an infinity of its own accord; that is the test used. UNDERFLOW_F32/F64 are +0.0 (the
//     quotient rounds to zero when the exponent difference is below -150 / -1075), OVERFLOW the infinity.
// The result is checked against the host's own division, bit for bit, by amd/tests/unit/test_gcn_divide.cpp and by the compiled
// divisions of amd/tests/hipcc/divide.cpp.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace vgpu::amd::gcn::divide {

template <typename T>
struct Fp;
template <>
struct Fp<float> {
  using Bits = uint32_t;
  static constexpr int kMantissa = 23, kBias = 127, kMaxExp = 255, kMaxExpBits = 8;
  static constexpr int kScale = 64;       // v_div_scale's factor, and v_div_fmas's
  static constexpr int kFar = 96;         // exponent distance at which the quotient is near the largest finite value
  static constexpr int kTiny = 23;        // a numerator whose exponent is this small or less
  static constexpr int kMinExp = -126;    // the smallest normal exponent
  static constexpr int kUnderflow = -150; // below this (in exponent difference) the quotient rounds to zero
  static constexpr int kOverflow = 129;   // at or above this it overflows
};
template <>
struct Fp<double> {
  using Bits = uint64_t;
  static constexpr int kMantissa = 52, kBias = 1023, kMaxExp = 2047, kMaxExpBits = 11;
  static constexpr int kScale = 128;
  static constexpr int kFar = 768;
  static constexpr int kTiny = 53;
  static constexpr int kMinExp = -1022;
  static constexpr int kUnderflow = -1075;
  static constexpr int kOverflow = 1025;
};

template <typename T>
inline typename Fp<T>::Bits bits_of(T v) {
  typename Fp<T>::Bits b;
  std::memcpy(&b, &v, sizeof b);
  return b;
}
template <typename T>
inline T from_bits(typename Fp<T>::Bits b) {
  T v;
  std::memcpy(&v, &b, sizeof v);
  return v;
}

// The biased exponent field: 0 for zero and denormals, kMaxExp for infinities and NaNs.
template <typename T>
inline int exponent_field(T v) {
  return static_cast<int>((bits_of(v) >> Fp<T>::kMantissa) & Fp<T>::kMaxExp);
}
template <typename T>
inline bool is_denormal(T v) {
  return std::fpclassify(v) == FP_SUBNORMAL;
}
template <typename T>
inline bool sign_of(T v) {
  return std::signbit(v);
}

// |v| has a unit exponent and a mantissa in [1, 2): v = m * 2^e. Not for zero, infinities or NaNs.
template <typename T>
inline void split(T v, int* e, T* m) {
  *e = std::ilogb(v);
  *m = std::scalbn(std::fabs(v), -*e);
}

// Whether the reciprocal of d is a denormal: |d| above 2^-kMinExp, and finite.
template <typename T>
inline bool reciprocal_is_denormal(T d) {
  if (!std::isfinite(d) || d == 0) return false;
  int e;
  T m;
  split(d, &e, &m);
  return e > -Fp<T>::kMinExp || (e == -Fp<T>::kMinExp && m > T(1));
}

// Whether n / d is below the smallest normal number but not zero in the exact sense (both finite, nonzero), judged without dividing.
template <typename T>
inline bool quotient_is_denormal(T n, T d) {
  if (!std::isfinite(n) || !std::isfinite(d) || n == 0 || d == 0) return false;
  int en, ed;
  T mn, md;
  split(n, &en, &mn);
  split(d, &ed, &md);
  const int diff = en - ed;
  return diff < Fp<T>::kMinExp || (diff == Fp<T>::kMinExp && mn < md);
}

template <typename T>
struct Scaled {
  T value;
  bool vcc;
};

// v_div_scale(S0, S1 = denominator, S2 = numerator), the reference: written with the library's classification and scaling calls. The tier
// and the interpreter run div_scale below, which gives these results (bits and condition) for every operand, with bit arithmetic
// where all three of the numbers involved are normal; test_gcn_divide holds the two against each other.
template <typename T>
inline Scaled<T> div_scale_reference(T s0, T s1, T s2) {
  using F = Fp<T>;
  Scaled<T> r{s0, false};
  if (s2 == 0 || s1 == 0) {
    r.value = std::numeric_limits<T>::quiet_NaN();
    return r;
  }
  const int e2 = exponent_field(s2), e1 = exponent_field(s1);
  if (e2 - e1 >= F::kFar) {   // N / D near the largest finite value: only the denominator is scaled
    r.vcc = true;
    if (s0 == s1) r.value = std::ldexp(s0, F::kScale);
  } else if (is_denormal(s1)) {
    r.value = std::ldexp(s0, F::kScale);
  } else {
    const bool rcp_denormal = reciprocal_is_denormal(s1), quotient_denormal = quotient_is_denormal(s2, s1);
    if (rcp_denormal && quotient_denormal) {   // (the guide scales up here; see the header)
      r.vcc = true;
      if (s0 == s1) r.value = std::ldexp(s0, -F::kScale);
    } else if (rcp_denormal) {
      r.value = std::ldexp(s0, -F::kScale);
    } else if (quotient_denormal) {   // only the numerator is scaled
      r.vcc = true;
      if (s0 == s2) r.value = std::ldexp(s0, F::kScale);
    } else if (e2 <= F::kTiny) {   // a tiny numerator
      r.value = std::ldexp(s0, F::kScale);
    }
  }
  return r;
}

// x * 2^k for a normal x and a result that is normal too: the exponent field moves, the rest is untouched (exactly what ldexp gives).
// Anything else -- zero, a denormal, an infinity or a NaN, an exponent that leaves the normal range -- is the library's call.
template <typename T>
inline T scale_by_pow2(T x, int k) {
  using F = Fp<T>;
  using B = typename F::Bits;
  const int ef = exponent_field(x);
  if (ef >= 1 && ef < F::kMaxExp && ef + k >= 1 && ef + k < F::kMaxExp)
    return from_bits<T>(bits_of(x) + static_cast<B>(static_cast<int64_t>(k) * (int64_t{1} << F::kMantissa)));
  return std::ldexp(x, k);
}

// v_div_scale(S0, S1 = denominator, S2 = numerator). The same as div_scale_reference. When both S1 and S2 are normal numbers (the
// usual case, and the only one where the reference needed no special handling) every test on them is a test of the exponent and
// mantissa fields: the reference's classification, ilogb and scalbn calls were most of a division's time in the real programs.
template <typename T>
inline Scaled<T> div_scale(T s0, T s1, T s2) {
  using F = Fp<T>;
  using B = typename F::Bits;
  const int e2 = exponent_field(s2), e1 = exponent_field(s1);
  if (e1 < 1 || e1 >= F::kMaxExp || e2 < 1 || e2 >= F::kMaxExp) return div_scale_reference(s0, s1, s2);
  Scaled<T> r{s0, false};
  const B mant_mask = (B{1} << F::kMantissa) - 1;
  if (e2 - e1 >= F::kFar) {
    r.vcc = true;
    if (s0 == s1) r.value = scale_by_pow2(s0, F::kScale);
    return r;
  }
  // (S1 is not a denormal.) Its reciprocal is: exponent above 2^-kMinExp, or equal to it with a mantissa above 1.
  const int d1 = e1 - F::kBias;
  const bool rcp_denormal = d1 > -F::kMinExp || (d1 == -F::kMinExp && (bits_of(s1) & mant_mask) != 0);
  // The quotient is below the smallest normal: the exponents differ by less than kMinExp, or by that and the numerator's mantissa
  // is below the denominator's.
  const int diff = e2 - e1;
  const bool quotient_denormal = diff < F::kMinExp || (diff == F::kMinExp && (bits_of(s2) & mant_mask) < (bits_of(s1) & mant_mask));
  if (rcp_denormal && quotient_denormal) {
    r.vcc = true;
    if (s0 == s1) r.value = scale_by_pow2(s0, -F::kScale);
  } else if (rcp_denormal) {
    r.value = scale_by_pow2(s0, -F::kScale);
  } else if (quotient_denormal) {
    r.vcc = true;
    if (s0 == s2) r.value = scale_by_pow2(s0, F::kScale);
  } else if (e2 <= F::kTiny) {
    r.value = scale_by_pow2(s0, F::kScale);
  }
  return r;
}

// fma(a, b, c) * 2^k with one rounding, to the format T. Exact: the operands are integers times powers of two, summed in 128 bits with
// whatever falls outside the window kept as a sticky bit, and rounded to nearest even at the target's precision, where a denormal
// has fewer bits.
template <typename T>
inline T scaled_fma(T a, T b, T c, int k) {
  using F = Fp<T>;
  using U = unsigned __int128;
  if (!std::isfinite(a) || !std::isfinite(b) || !std::isfinite(c)) return std::ldexp(std::fma(a, b, c), k);
  // value = sign * m * 2^e for each of a*b (up to 106 bits) and c.
  auto parts = [](T v, uint64_t* m, int* e) {
    int ex;
    const T frac = std::frexp(std::fabs(v), &ex);
    *m = static_cast<uint64_t>(std::ldexp(frac, F::kMantissa + 1));
    *e = ex - (F::kMantissa + 1);
    if (v == 0) *m = 0;
  };
  uint64_t ma, mb, mc;
  int ea, eb, ec;
  parts(a, &ma, &ea);
  parts(b, &mb, &eb);
  parts(c, &mc, &ec);
  U p = static_cast<U>(ma) * mb;
  int ep = ea + eb;
  const bool neg_p = sign_of(a) != sign_of(b), neg_c = sign_of(c);
  U cc = mc;
  auto bitlen = [](U x) {
    int n = 0;
    while (x) {
      ++n;
      x >>= 1;
    }
    return n;
  };
  if (p == 0 && cc == 0) return std::ldexp(std::fma(a, b, c), k);   // (a zero, with its sign)
  // Put the larger by magnitude (leading bit position) in a 120-bit window; the other is shifted to the same exponent, right shifts
  // keeping a sticky bit.
  const int top_p = p ? ep + bitlen(p) : std::numeric_limits<int>::min() / 2;
  const int top_c = cc ? ec + bitlen(cc) : std::numeric_limits<int>::min() / 2;
  const int top = top_p > top_c ? top_p : top_c;
  const int e = top - 120;   // the window's unit: 2^e
  auto align = [&](U m, int ex, bool* sticky) -> U {
    if (m == 0) return 0;
    const int sh = ex - e;
    if (sh >= 0) return m << sh;
    if (-sh >= 128) {
      *sticky = true;
      return 0;
    }
    const U kept = m >> -sh;
    if ((kept << -sh) != m) *sticky = true;
    return kept;
  };
  bool st_p = false, st_c = false;
  const U ap = align(p, ep, &st_p), ac = align(cc, ec, &st_c);
  // Magnitude of the sum, its sign, and the lost fraction (always in the direction of the smaller operand's sign).
  bool neg;
  U mag;
  if (neg_p == neg_c) {
    neg = neg_p;
    mag = ap + ac;
    if (st_p || st_c) mag |= 1;   // the lost bits are far below the rounding position: they only break ties
  } else {
    // ap and ac have opposite signs; the lost bits belong to one operand (only the smaller one is ever shifted out).
    const bool p_big = ap >= ac;
    neg = p_big ? neg_p : neg_c;
    mag = p_big ? ap - ac : ac - ap;
    const bool lost_from_big = p_big ? st_p : st_c, lost_from_small = p_big ? st_c : st_p;
    // A bit lost from the larger one makes the true magnitude larger; from the smaller, smaller.
    if (lost_from_big && !lost_from_small) mag |= 1;
    else if (lost_from_small && !lost_from_big) mag -= 1;
  }
  if (mag == 0) {
    // An exact zero: +0 unless both the product and the addend are negative zeros (round to nearest).
    return (neg_p && neg_c) ? -T(0) : T(0);
  }
  // value = mag * 2^e * 2^k. Round it to T.
  const int len = bitlen(mag);
  const int target_lsb = (F::kMinExp - F::kMantissa);   // the exponent of a denormal's unit in place
  int lsb = e + k + len - (F::kMantissa + 1);          // the exponent of the lowest bit kept, at full precision
  if (lsb < target_lsb) lsb = target_lsb;
  const int drop = lsb - (e + k);   // bits of mag to drop
  U kept;
  if (drop <= 0) {
    kept = mag << -drop;
  } else if (drop > len + 1) {
    kept = 0;
  } else {
    kept = drop >= 128 ? 0 : mag >> drop;
    const U half = static_cast<U>(1) << (drop - 1);
    const U rest = drop >= 128 ? mag : mag & ((static_cast<U>(1) << drop) - 1);
    if (rest > half || (rest == half && (kept & 1))) ++kept;
  }
  T result = std::ldexp(static_cast<T>(static_cast<uint64_t>(kept)), lsb);
  return neg ? -result : result;
}

// v_div_fmas(S0, S1, S2) with the condition bit of this lane.
template <typename T>
inline T div_fmas(T s0, T s1, T s2, bool vcc) {
  using F = Fp<T>;
  if (!vcc) return std::fma(s0, s1, s2);
  // (ASSUMED, see the header) a quotient estimate below one means the numerator was scaled up.
  const int k = exponent_field(s2) >= F::kBias ? F::kScale : -F::kScale;
  return scaled_fma(s0, s1, s2, k);
}

// v_div_fixup(S0 = quotient, S1 = denominator, S2 = numerator).
template <typename T>
inline T div_fixup(T s0, T s1, T s2) {
  using F = Fp<T>;
  const bool sign_out = sign_of(s1) != sign_of(s2);
  const T inf = std::numeric_limits<T>::infinity();
  // The guide's 0/0 and inf/inf: a quiet NaN with the sign bit set (0xffc00000, 0xfff8000000000000).
  const T quiet_nan = from_bits<T>(static_cast<typename F::Bits>((typename F::Bits{1} << (F::kMantissa + F::kMaxExpBits)) |
                                                                  (static_cast<typename F::Bits>(F::kMaxExp) << F::kMantissa) |
                                                                  (typename F::Bits{1} << (F::kMantissa - 1))));
  auto quieted = [](T v) {
    return from_bits<T>(bits_of(v) | (typename F::Bits{1} << (F::kMantissa - 1)));
  };
  if (std::isnan(s2)) return quieted(s2);
  if (std::isnan(s1)) return quieted(s1);
  if (s1 == 0 && s2 == 0) return quiet_nan;
  if (std::isinf(s1) && std::isinf(s2)) return quiet_nan;
  if (s1 == 0 || std::isinf(s2)) return sign_out ? -inf : inf;
  if (std::isinf(s1) || s2 == 0) return sign_out ? -T(0) : T(0);
  if (exponent_field(s2) - exponent_field(s1) < F::kUnderflow) return sign_out ? -T(0) : T(0);
  if (exponent_field(s2) - exponent_field(s1) >= F::kOverflow) return sign_out ? -inf : inf;
  return sign_out ? -std::fabs(s0) : std::fabs(s0);
}

// ldexpf(x, k) for any x and any k (v_ldexp_f32): for a normal x and a result that is normal too the exponent field moves and nothing else
// does -- exact, with no rounding, which is what libm gives -- and everything else (zero, a denormal, an infinity, a NaN, a result that leaves
// the normal range) is libm's call. The sum is made in 64 bits: k is any 32-bit integer.
[[gnu::always_inline]] inline bool ldexp_f32_exact(uint32_t bits, int32_t k) {
  const uint32_t ef = (bits >> 23) & 255u;
  const int64_t en = static_cast<int64_t>(ef) + k;
  return ef - 1u < 254u && en >= 1 && en <= 254;
}
// (The bits of the answer where ldexp_f32_exact holds.)
[[gnu::always_inline]] inline uint32_t ldexp_f32_shift(uint32_t bits, int32_t k) { return bits + (static_cast<uint32_t>(k) << 23); }
// ldexpf for ANY x and k, in integers (libm's answer, bit for bit): zero and infinity come back as they are, a NaN comes back quieted (libm
// returns x + x), an overflow is the infinity of x's sign, and a result below the normal range is x * 2^k rounded once, to nearest even, to a
// denormal or to zero (libm multiplies by a power of two, which rounds once; so does this). A denormal x is first normalized.
[[gnu::always_inline]] inline uint32_t ldexp_f32_bits(uint32_t bits, int32_t k) {
  const uint32_t sign = bits & 0x80000000u, ef = (bits >> 23) & 255u, frac = bits & 0x7FFFFFu;
  if (ef == 255u) return frac ? (bits | 0x400000u) : bits;
  if (ef == 0 && frac == 0) return bits;
  uint32_t m;
  int64_t e;
  if (ef) {
    m = frac | 0x800000u, e = ef;
  } else {
    const int sh = __builtin_clz(frac) - 8;   // (the leading one of a denormal moves to bit 23)
    m = frac << sh, e = 1 - sh;
  }
  e += k;
  if (e >= 255) return sign | 0x7F800000u;
  if (e >= 1) return sign | static_cast<uint32_t>(e) << 23 | (m & 0x7FFFFFu);
  const int64_t shift = 1 - e;   // the result is m / 2^shift in units of the smallest denormal
  if (shift >= 26) return sign;
  const uint32_t rem = m & ((1u << shift) - 1u), half = 1u << (shift - 1);
  uint32_t q = m >> shift;
  if (rem > half || (rem == half && (q & 1u))) ++q;   // (a carry into 0x800000 is the smallest normal number: the right bits)
  return sign | q;
}
[[gnu::always_inline]] inline float ldexp_f32(float x, int32_t k) {
  const uint32_t bits = bits_of(x);
  if (__builtin_expect(ldexp_f32_exact(bits, k), 1)) return from_bits<float>(ldexp_f32_shift(bits, k));
  return __builtin_ldexpf(x, k);
}

// ---- Rows of binary32 lanes (the tier's wave-wide forms)
//
// A wave's v_div_scale_f32 / v_div_fixup_f32 on the lanes where every operand is an ordinary number and nothing is scaled or special-cased:
// the answer is a few bit operations on the exponent fields, the same for every lane, so a row of them is tested and computed without a branch
// and a row with one lane outside it takes the lane-by-lane functions above (nothing is written by these before the verdict). Each returns
// true when EVERY one of the `n` lanes (active or not) is in the plain case, which is cheap to be conservative about: a lane that is not
// makes the caller do the lane-by-lane work, which is the reference. test_gcn_divide holds both against div_scale / div_fixup.

// v_div_scale_f32 (S0, S1 = denominator, S2 = numerator): in the plain case the value is S0 and the condition is clear. Plain means that
// S1 and S2 are normal numbers, their exponents are less than 96 apart, the reciprocal of S1 and the quotient are normal, and S2's exponent
// is above 23 (the branches of div_scale that leave S0 unchanged and VCC clear).
[[gnu::always_inline]] inline bool div_scale_row_plain_f32(const uint32_t* s1, const uint32_t* s2, unsigned n) {
  uint32_t bad = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t e1 = (s1[i] >> 23) & 255u, e2 = (s2[i] >> 23) & 255u;
    const uint32_t m1 = s1[i] & 0x7FFFFFu, m2 = s2[i] & 0x7FFFFFu;
    const int32_t diff = static_cast<int32_t>(e2) - static_cast<int32_t>(e1);
    uint32_t b = (e1 - 1u >= 254u) | (e2 - 1u >= 254u);
    b |= diff >= Fp<float>::kFar;
    b |= (e1 >= 254u) | ((e1 == 253u) & (m1 != 0));                                           // the reciprocal is a denormal
    b |= (diff < Fp<float>::kMinExp) | ((diff == Fp<float>::kMinExp) & (m2 < m1));            // the quotient is
    b |= e2 <= static_cast<uint32_t>(Fp<float>::kTiny);
    bad |= b;
  }
  return bad == 0;
}

// v_div_fixup_f32 (S0 = quotient, S1 = denominator, S2 = numerator): in the plain case (S1 and S2 normal numbers whose exponents are within
// the range that neither underflows nor overflows) the answer is |S0| with the sign of S1 xor S2. Writes the row to `r`; the result is of
// no use unless true is returned.
[[gnu::always_inline]] inline bool div_fixup_row_plain_f32(const uint32_t* s0, const uint32_t* s1, const uint32_t* s2, uint32_t* r, unsigned n) {
  uint32_t bad = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t e1 = (s1[i] >> 23) & 255u, e2 = (s2[i] >> 23) & 255u;
    const int32_t diff = static_cast<int32_t>(e2) - static_cast<int32_t>(e1);
    bad |= (e1 - 1u >= 254u) | (e2 - 1u >= 254u) | (diff < Fp<float>::kUnderflow) | (diff >= Fp<float>::kOverflow);
    r[i] = (s0[i] & 0x7FFFFFFFu) | ((s1[i] ^ s2[i]) & 0x80000000u);
  }
  return bad == 0;
}

// ---- Rows of binary64 lanes
//
// The same two tests for v_div_scale_f64 / v_div_fixup_f64, with the double's fields (11 exponent bits, 52 mantissa bits). Unlike the f32 rows
// only the ACTIVE lanes (`ex`, bit i for lane i) are asked: a lane the instruction does not write holds whatever its register held (zero
// in a diverged wave, which would make every row of it special), and nothing of it is read back. `ex` has no bit at or above `n` (the tier's rows
// are 32 lanes only when the high half of EXEC is clear). true: every active lane is plain.

// v_div_scale_f64 (S0, S1 = denominator, S2 = numerator): the value is S0 and the condition is clear when S1 and S2 are normal numbers whose
// exponents are less than 768 apart, the reciprocal of S1 and the quotient are normal, and S2's exponent is above 53.
[[gnu::always_inline]] inline bool div_scale_row_plain_f64(const uint64_t* s1, const uint64_t* s2, uint64_t ex, unsigned n) {
  using F = Fp<double>;
  constexpr uint64_t kMant = (uint64_t{1} << F::kMantissa) - 1;
  // First pass on the high words alone (32-bit lanes, which the vector units do whole): the exponent fields are all it needs, except at
  // the two places where the mantissas decide (the reciprocal's exponent just below the largest, the quotient's just at the denormal edge):
  // those lanes are only counted, and looked at in full below.
  uint32_t bad = 0, edge = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t e1 = (static_cast<uint32_t>(s1[i] >> 32) >> 20) & 2047u, e2 = (static_cast<uint32_t>(s2[i] >> 32) >> 20) & 2047u;
    const int32_t diff = static_cast<int32_t>(e2) - static_cast<int32_t>(e1);
    uint32_t b = (e1 - 1u >= F::kMaxExp - 1u) | (e2 - 1u >= F::kMaxExp - 1u);
    b |= diff >= F::kFar;
    b |= e1 >= F::kMaxExp - 1u;                    // the reciprocal is a denormal
    b |= diff < F::kMinExp;                        // the quotient is
    b |= e2 <= static_cast<uint32_t>(F::kTiny);
    const uint32_t act = 0u - static_cast<uint32_t>((ex >> i) & 1);
    bad |= b & act;
    edge |= ((e1 == F::kMaxExp - 2u) | (diff == F::kMinExp)) & act;
  }
  if (bad) return false;
  if (!edge) return true;
  for (uint64_t m = ex; m; m &= m - 1) {
    const unsigned i = static_cast<unsigned>(__builtin_ctzll(m));
    const uint64_t e1 = (s1[i] >> F::kMantissa) & F::kMaxExp, e2 = (s2[i] >> F::kMantissa) & F::kMaxExp;
    const uint64_t m1 = s1[i] & kMant, m2 = s2[i] & kMant;
    const int64_t diff = static_cast<int64_t>(e2) - static_cast<int64_t>(e1);
    if ((e1 == F::kMaxExp - 2u && m1 != 0) || (diff == F::kMinExp && m2 < m1)) return false;
  }
  return true;
}

// v_div_fixup_f64 (S0 = quotient, S1 = denominator, S2 = numerator): for normal S1 and S2 whose exponents are within the range that neither
// underflows (-1075) nor overflows (1025) the answer is |S0| with the sign of S1 xor S2. Writes the row to `r`; of no use unless true.
[[gnu::always_inline]] inline bool div_fixup_row_plain_f64(const uint64_t* s0, const uint64_t* s1, const uint64_t* s2, uint64_t* r,
                                                           uint64_t ex, unsigned n) {
  using F = Fp<double>;
  uint32_t bad = 0;
  for (unsigned i = 0; i < n; ++i) {
    const uint32_t e1 = (static_cast<uint32_t>(s1[i] >> 32) >> 20) & 2047u, e2 = (static_cast<uint32_t>(s2[i] >> 32) >> 20) & 2047u;
    const int32_t diff = static_cast<int32_t>(e2) - static_cast<int32_t>(e1);
    const uint32_t b = (e1 - 1u >= F::kMaxExp - 1u) | (e2 - 1u >= F::kMaxExp - 1u) | (diff < F::kUnderflow) | (diff >= F::kOverflow);
    bad |= b & (0u - static_cast<uint32_t>((ex >> i) & 1));
    r[i] = (s0[i] & 0x7FFFFFFFFFFFFFFFull) | ((s1[i] ^ s2[i]) & 0x8000000000000000ull);
  }
  return bad == 0;
}

}  // namespace vgpu::amd::gcn::divide
