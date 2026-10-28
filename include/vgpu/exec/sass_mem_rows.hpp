// The SASS tier's memory handlers (src/sass/tier_ops.inc h_global, tier_global16) ask of a full warp's address rows whether the
// lanes are on consecutive elements, before anything else is done with them. Done on the registers' rows, two or four lanes to a vector, the
// answer needs no array of 64-bit addresses (32 stores) nor the second pass that compared it: the array is built only for an access that
// is not one of these. They are written with SSE2 where it is there and tested against the lane-by-lane definitions
// (tests/unit/test_sass_mem_rows.cpp).
#pragma once

#include <cstdint>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vgpu::exec::sassmem {

// Whether the lanes `first` .. `first + count - 1` of a global access are on consecutive elements of `bytes` bytes: lane l's address is
// (lo[l] | hi[l] << 32) + add (`hi` null: just lo[l]; `lo` null: `add` for every lane, which is a run only for one lane), and `g0` is
// the first lane's. A single pass that cannot stop early, as the definition does not.
inline bool global_run(const uint32_t* lo, const uint32_t* hi, uint64_t add, unsigned first, unsigned count, unsigned bytes,
                       uint64_t& g0) {
  g0 = add;
  if (!lo) return count == 1;
  g0 = (hi ? (static_cast<uint64_t>(hi[first]) << 32) | lo[first] : uint64_t{lo[first]}) + add;
  uint64_t differ = 0;
  unsigned i = 0;
#if defined(__SSE2__) && defined(__x86_64__)
  const __m128i vadd = _mm_set1_epi64x(static_cast<long long>(add));
  const __m128i step = _mm_set1_epi64x(static_cast<long long>(uint64_t{2} * bytes));
  const __m128i zero = _mm_setzero_si128();
  __m128i expect = _mm_set_epi64x(static_cast<long long>(g0 + bytes), static_cast<long long>(g0));
  __m128i acc = zero;
  for (; i + 2 <= count; i += 2) {
    const __m128i l = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(lo + first + i));
    const __m128i h = hi ? _mm_loadl_epi64(reinterpret_cast<const __m128i*>(hi + first + i)) : zero;
    const __m128i a = _mm_add_epi64(_mm_unpacklo_epi32(l, h), vadd);
    acc = _mm_or_si128(acc, _mm_xor_si128(a, expect));
    expect = _mm_add_epi64(expect, step);
  }
  differ = static_cast<uint64_t>(_mm_cvtsi128_si64(acc)) | static_cast<uint64_t>(_mm_cvtsi128_si64(_mm_unpackhi_epi64(acc, acc)));
#endif
  for (; i < count; ++i) {
    const uint64_t a = (hi ? (static_cast<uint64_t>(hi[first + i]) << 32) | lo[first + i] : uint64_t{lo[first + i]}) + add;
    differ |= a ^ (g0 + uint64_t{i} * bytes);
  }
  return differ == 0;
}

}  // namespace vgpu::exec::sassmem
