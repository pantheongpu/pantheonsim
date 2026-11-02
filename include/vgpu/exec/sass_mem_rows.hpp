// The SASS tier's memory handlers (src/sass/tier_ops.inc h_global, tier_global16) ask of a full warp's address rows whether the
// lanes are on consecutive elements, before anything else is done with them. Done on the registers' rows, two or four lanes to a vector, the
// answer needs no array of 64-bit addresses (32 stores) nor the second pass that compared it: the array is built only for an access that
// is not one of these. They are written with SSE2 where it is there and tested against the lane-by-lane definitions
// (tests/unit/test_sass_mem_rows.cpp).
#pragma once

#include <algorithm>
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

// The addresses of a full warp's global access (a base that is a pair of rows, a single row or none, plus `add`), and in the same pass
// whether some lane is in the 4 GiB from `window`, the constant window: the result is below 2^32 when one is (it is 0), else all ones.
// (A pass over the stored addresses that asked this alone waited for the stores. The scalar form kept the smallest distance above the
// window, a compare and a conditional move per lane; the callers only ask whether it is below 2^32.)
// With a window that starts on a multiple of 4 GiB (the constant window does) a lane is in it exactly when its high word is the window's,
// so the pass is two lanes to a vector: the low and high words interleaved, `add` added, the pair stored, and its high words compared
// with the window's; the compares are ORed and looked at once at the end. Any other window takes the scalar loop.
inline uint64_t full_warp_addresses(const uint32_t* lo, const uint32_t* hi, uint64_t add, uint64_t window, uint64_t* addrs) {
#if defined(__SSE2__) && defined(__x86_64__)
  if (lo && (window & 0xffffffffu) == 0) {
    const __m128i vadd = _mm_set1_epi64x(static_cast<long long>(add));
    const __m128i want = _mm_set1_epi32(static_cast<int>(static_cast<uint32_t>(window >> 32)));
    const __m128i zero = _mm_setzero_si128();
    __m128i in = zero;
    for (unsigned l = 0; l < 32; l += 2) {
      const __m128i a = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(lo + l));
      const __m128i b = hi ? _mm_loadl_epi64(reinterpret_cast<const __m128i*>(hi + l)) : zero;
      const __m128i v = _mm_add_epi64(_mm_unpacklo_epi32(a, b), vadd);
      _mm_storeu_si128(reinterpret_cast<__m128i*>(addrs + l), v);
      in = _mm_or_si128(in, _mm_cmpeq_epi32(v, want));
    }
    // (the high words of the two lanes are dwords 1 and 3)
    return (_mm_movemask_ps(_mm_castsi128_ps(in)) & 0xA) ? 0 : ~uint64_t{0};
  }
#endif
  uint64_t nearest = ~uint64_t{0};
  const auto lane = [&](unsigned l, uint64_t a) {
    addrs[l] = a;
    nearest = std::min(nearest, a - window);
  };
  if (lo && hi) {
    for (unsigned l = 0; l < 32; ++l) lane(l, (lo[l] | (static_cast<uint64_t>(hi[l]) << 32)) + add);
  } else if (lo) {
    for (unsigned l = 0; l < 32; ++l) lane(l, uint64_t{lo[l]} + add);
  } else {
    for (unsigned l = 0; l < 32; ++l) lane(l, add);
  }
  return nearest;
}


// The data of a full warp's global store of 1, 2, 4 or 8 bytes as 32 values for the memory manager's batch store: lane l's word of the
// register (the low `bytes` bytes of it for 1 and 2) and, for 8, the next register's word above it; `d` null (RZ) stores zeros. The
// registers are rows of 32 words (`d` the first register's, the second 32 words on). Four lanes to a vector with SSE2: the words of the
// two rows interleaved into 64-bit values.
inline void full_warp_store_values(const uint32_t* d, unsigned bytes, uint64_t* fv) {
  if (!d) {
    for (unsigned l = 0; l < 32; ++l) fv[l] = 0;
    return;
  }
#if defined(__SSE2__) && defined(__x86_64__)
  const __m128i keep = _mm_set1_epi32(bytes == 1 ? 0xff : bytes == 2 ? 0xffff : -1);
  const __m128i zero = _mm_setzero_si128();
  for (unsigned l = 0; l < 32; l += 4) {
    const __m128i a = _mm_and_si128(_mm_loadu_si128(reinterpret_cast<const __m128i*>(d + l)), keep);
    const __m128i b = bytes == 8 ? _mm_loadu_si128(reinterpret_cast<const __m128i*>(d + 32 + l)) : zero;
    _mm_storeu_si128(reinterpret_cast<__m128i*>(fv + l), _mm_unpacklo_epi32(a, b));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(fv + l + 2), _mm_unpackhi_epi32(a, b));
  }
#else
  for (unsigned l = 0; l < 32; ++l) {
    const uint32_t v0 = d[l], v1 = bytes == 8 ? d[32 + l] : 0u;
    fv[l] = bytes == 8 ? (static_cast<uint64_t>(v1) << 32 | v0) : bytes == 4 ? v0 : (v0 & ((1u << (8 * bytes)) - 1));
  }
#endif
}

}  // namespace vgpu::exec::sassmem
