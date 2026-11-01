// Whether every active lane of a shared- or local-memory access is inside the array the SASS executor's fast path moves
// bytes straight into (src/sass/exec_ops.inc exec_mem, and the tier's h_shared / h_local in tier_ops.inc).
//
// The fast path is taken only when ALL the active lanes are inside; if any one is not, the slow loop runs and finds the lane
// that faults. Testing lane by lane with a short-circuit made a branch per lane, and the branches are what the checks below
// avoid: the bits that say "another block's window" are ORed together and the highest offset is kept, and the answer is made
// once. They are the same test: every lane is in this block's window iff no lane has other bits, and every lane's last byte
// is inside the array iff the highest lane's is. (tests/unit/test_memcount_forms.cpp compares them with the per-lane test.)
#pragma once

#include <algorithm>
#include <cstdint>
#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vgpu::exec {

// A shared-memory address is the block's rank in its cluster in bits 24 and up, and the offset in its array below.
struct SharedWindowCheck {
  uint64_t other = 0;     // the rank bits that differ from this block's, ORed over the lanes
  uint64_t highest = 0;   // the highest offset
  void add(uint64_t address, uint64_t rank) {
    other |= (address >> 24) ^ rank;
    highest = std::max<uint64_t>(highest, address & 0xffffff);
  }
  // The 32 addresses of a whole warp (32-bit, as the shared window's are): add() of each, written as two reductions over a plain array so that
  // the compiler vectors them (the loop of add() calls has a chain of 32 dependent maxima).
  void add_warp(const uint32_t* a, uint32_t rank) {
    uint32_t o = 0, h = 0;
    for (unsigned l = 0; l < 32; ++l) {
      o |= (a[l] >> 24) ^ rank;
      h = std::max<uint32_t>(h, a[l] & 0xffffffu);
    }
    other |= o;
    highest = std::max<uint64_t>(highest, h);
  }
  // `any_lane`: whether the mask had a lane (no lane at all is trivially inside).
  bool inside(uint64_t array_size, uint32_t bytes, bool any_lane) const {
    return !any_lane || (other == 0 && highest + bytes <= array_size);
  }
};

// SharedWindowCheck::add_warp + inside() for the lanes of `active` (any subset of a warp), in one pass and without a maximum (SSE2 has
// no unsigned 32-bit maximum, and the maximum was most of the work): a lane is inside when its rank bits are the block's and
// `size - bytes - offset` is not negative. The offset is under 2^24, so the difference needs no more than 32 bits once the array size
// is clamped (a size of 2^25 or more holds every offset's last byte, as it does unclamped). The same answer as
// SharedWindowCheck over the active lanes (tests/unit/test_memcount_forms.cpp compares them).
inline bool shared_window_inside_warp(const uint32_t* a, uint32_t active, uint32_t rank, uint64_t size, uint32_t bytes) {
  if (!active) return true;
  if (rank > 255) return false;   // (the rank bits are eight at most: no address names such a block)
  const int32_t limit = static_cast<int32_t>(std::min<uint64_t>(size, uint64_t{1} << 25)) - static_cast<int32_t>(bytes);
#if defined(__SSE2__)
  const __m128i vrank = _mm_set1_epi32(static_cast<int>(rank << 24)), vlim = _mm_set1_epi32(limit), low24 = _mm_set1_epi32(0xffffff);
  const __m128i bit = _mm_set_epi32(8, 4, 2, 1);
  __m128i acc = _mm_setzero_si128();
  for (unsigned g = 0; g < 8; ++g) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(a) + g);
    const __m128i other = _mm_srli_epi32(_mm_xor_si128(v, vrank), 24);
    const __m128i past = _mm_srli_epi32(_mm_sub_epi32(vlim, _mm_and_si128(v, low24)), 31);
    __m128i bad = _mm_or_si128(other, past);
    if (active != 0xffffffffu) {
      const __m128i sel = _mm_and_si128(_mm_set1_epi32(static_cast<int>(active >> (4 * g))), bit);
      bad = _mm_and_si128(bad, _mm_cmpeq_epi32(sel, bit));
    }
    acc = _mm_or_si128(acc, bad);
  }
  return _mm_movemask_epi8(_mm_cmpeq_epi32(acc, _mm_setzero_si128())) == 0xffff;
#else
  uint32_t bad = 0;
  for (unsigned l = 0; l < 32; ++l) {
    if (!(active >> l & 1)) continue;
    bad |= ((a[l] >> 24) ^ rank) | static_cast<uint32_t>((limit - static_cast<int32_t>(a[l] & 0xffffffu)) < 0);
  }
  return bad == 0;
#endif
}

// A local-memory offset is inside the lane's frame when the access's last byte is.
struct LocalWindowCheck {
  uint64_t highest = 0;
  void add(uint64_t offset) { highest = std::max(highest, offset); }
  bool inside(uint64_t frame_size, uint32_t bytes, bool any_lane) const {
    return !any_lane || highest + bytes <= frame_size;
  }
};

}  // namespace vgpu::exec
