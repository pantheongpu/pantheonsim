// SASS's LEA on the 32 lanes of a warp: (a << sh) + b, the form every 64-bit address calculation is made of -- LEA for the low word with the carry
// out in a predicate, LEA.HI.X for the high word (the shifted value is the 64-bit (top:a), the carry comes in from the predicate). Four lanes to a
// vector with SSE2, where it is there; lea_reference is the lane-by-lane rule, which the tier's handler used to be, and the test (tests/unit/test_sass_lea.cpp)
// holds the vector code to.
//
// The rule for each lane:
//   shifted = hi ? ((a | tp << 32) << sh) >> 32 : a << sh        (tp: the third source's lane, or a's sign extended when there is none)
//   sum     = shifted + b + (carry in) + (bneg && b == 0 ? 2^32 : 0)   (a negated b that came out zero: -0 is ~0 + 1, which carries out)
//   result  = low 32 bits of sum, carry out = whether sum reached 2^32
// sh is the instruction's 5-bit shift, 0 to 31.
#pragma once

#include <cstdint>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vgpu::exec::sasslea {

using Mask = uint32_t;

struct Form {
  bool hi = false;      // .HI: the shifted value is (top : a), and the high word of the shifted result is taken
  bool bneg = false;    // b is a negated operand
  unsigned sh = 0;      // 0 to 31
};

// The lane-by-lane rule. `top` is null when the high word is the sign of a. Returns the carry-out mask (all lanes).
inline Mask lea_reference(const Form& f, uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* top, Mask cin) {
  Mask carry = 0;
  for (unsigned l = 0; l < 32; ++l) {
    uint32_t shifted;
    if (!f.hi) {
      shifted = f.sh >= 32 ? 0 : a[l] << f.sh;
    } else {
      const uint32_t tp = top ? top[l] : static_cast<uint32_t>(static_cast<int32_t>(a[l]) >> 31);
      shifted = static_cast<uint32_t>(((a[l] | (static_cast<uint64_t>(tp) << 32)) << f.sh) >> 32);
    }
    uint64_t sum = static_cast<uint64_t>(shifted) + b[l] + ((cin >> l) & 1);
    if (f.bneg && b[l] == 0) sum += uint64_t{1} << 32;
    r[l] = static_cast<uint32_t>(sum);
    carry |= static_cast<Mask>(sum >> 32 != 0) << l;
  }
  return carry;
}

#if defined(__SSE2__)
namespace detail {

// x < y as unsigned numbers, all ones in the lanes where it holds (SSE2 has only the signed compare).
inline __m128i ult(__m128i x, __m128i y) {
  const __m128i bias = _mm_set1_epi32(static_cast<int>(0x80000000u));
  return _mm_cmpgt_epi32(_mm_xor_si128(y, bias), _mm_xor_si128(x, bias));
}

template <bool Hi>
inline Mask lea_t(const Form& f, uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* top, Mask cin) {
  const __m128i shl = _mm_cvtsi32_si128(static_cast<int>(f.sh)), shr = _mm_cvtsi32_si128(static_cast<int>(32 - f.sh));   // (a count of 32 gives zero)
  const __m128i lane_bit = _mm_set_epi32(8, 4, 2, 1);
  Mask carry = 0;
  for (unsigned i = 0; i < 32; i += 4) {
    const __m128i x = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i));
    const __m128i y = _mm_loadu_si128(reinterpret_cast<const __m128i*>(b + i));
    __m128i shifted;
    if constexpr (Hi) {
      const __m128i tp = top ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(top + i)) : _mm_srai_epi32(x, 31);
      shifted = _mm_or_si128(_mm_sll_epi32(tp, shl), _mm_srl_epi32(x, shr));
    } else {
      shifted = _mm_sll_epi32(x, shl);
    }
    const __m128i s1 = _mm_add_epi32(shifted, y);
    __m128i out = ult(s1, shifted);   // the carry of shifted + b
    const __m128i cin_mask = _mm_cmpeq_epi32(_mm_and_si128(_mm_set1_epi32(static_cast<int>((cin >> i) & 0xfu)), lane_bit), lane_bit);
    const __m128i s2 = _mm_sub_epi32(s1, cin_mask);   // + the carry in (the mask is -1 where it is set)
    out = _mm_or_si128(out, ult(s2, s1));              // (the carry in made the sum wrap: s1 was all ones)
    if (f.bneg) out = _mm_or_si128(out, _mm_cmpeq_epi32(y, _mm_setzero_si128()));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(r + i), s2);
    carry |= static_cast<Mask>(_mm_movemask_ps(_mm_castsi128_ps(out))) << i;
  }
  return carry;
}

}  // namespace detail
#endif

// The same as lea_reference. r may be a, b or top (each lane is read before it is written).
inline Mask lea_lanes(const Form& f, uint32_t* r, const uint32_t* a, const uint32_t* b, const uint32_t* top, Mask cin) {
#if defined(__SSE2__)
  if (f.sh < 32) return f.hi ? detail::lea_t<true>(f, r, a, b, top, cin) : detail::lea_t<false>(f, r, a, b, top, cin);
#endif
  return lea_reference(f, r, a, b, top, cin);
}

}  // namespace vgpu::exec::sasslea
