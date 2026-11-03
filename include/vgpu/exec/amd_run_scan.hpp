// How many of a wave's lanes, from the first, make a run of consecutive addresses (amd/src/gcn_exec.cpp Machine::memory_run): lane k's
// address is the first lane's plus k * stride. The addresses are in the registers' rows, a pair of them (low and high words) or one
// (a scalar base and a 32-bit offset a lane), and the definition is the loop that compares lane after lane and stops at the first that
// differs (run_scan_reference, which the unit test holds this to). Done four lanes to a vector (SSE2) when the low words cannot wrap
// round 2^32 over the lanes asked about -- the usual case, with the high word the same for all of them -- and lane by lane otherwise.
#pragma once

#include <algorithm>
#include <cstdint>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

namespace vgpu::exec::amdrun {

// The definition. `lo` and `hi` are the rows from the first lane on (`hi` null: lane k's address is `base` + lo[k], the scalar-base
// form, where `first` is lo[0] and `base` is not used by the comparison); `first` is the first lane's address with the instruction's
// offset taken off again (for the pair form) or lo[0] (for the base form). Returns the number of lanes, 1 to `reach`, whose
// address is first + k * stride (mod 2^64) -- the first always counts.
inline uint32_t run_scan_reference(const uint32_t* lo, const uint32_t* hi, uint32_t reach, uint64_t first, uint32_t stride) {
  uint32_t count = 1;
  uint64_t next = first;
  if (hi) {
    for (; count < reach && ((static_cast<uint64_t>(hi[count]) << 32 | lo[count]) == (next += stride)); ++count) {
    }
  } else {
    for (; count < reach && (uint64_t{lo[count]} == (next += stride)); ++count) {
    }
  }
  return count;
}

inline uint32_t run_scan(const uint32_t* lo, const uint32_t* hi, uint32_t reach, uint64_t first, uint32_t stride) {
#if defined(__SSE2__) && defined(__x86_64__)
  // (The expected low word of lane k is first's + k * stride with nothing carried into the high word: no wrap over the lanes asked about,
  // and for the base form the sum is the lane's whole value, `first` being the first lane's, so it must not wrap either. Four lanes a vector, from lane 0.)
  const uint64_t low0 = hi ? (first & 0xFFFFFFFFu) : first;
  if (reach >= 4 && low0 + uint64_t{reach - 1} * stride < (uint64_t{1} << 32)) {
    const uint32_t f = static_cast<uint32_t>(low0), s = stride;
    __m128i expect_lo = _mm_setr_epi32(static_cast<int>(f), static_cast<int>(f + s), static_cast<int>(f + 2 * s), static_cast<int>(f + 3 * s));
    const __m128i step = _mm_set1_epi32(static_cast<int>(4 * s));
    const __m128i expect_hi = _mm_set1_epi32(static_cast<int>(static_cast<uint32_t>(first >> 32)));
    uint32_t i = 0;
    for (; i + 4 <= reach; i += 4) {
      const __m128i l = _mm_loadu_si128(reinterpret_cast<const __m128i*>(lo + i));
      __m128i same = _mm_cmpeq_epi32(l, expect_lo);
      if (hi) same = _mm_and_si128(same, _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(hi + i)), expect_hi));
      const unsigned bits = static_cast<unsigned>(_mm_movemask_ps(_mm_castsi128_ps(same)));
      if (bits != 0xF) return std::max<uint32_t>(1, i + static_cast<uint32_t>(__builtin_ctz(~bits)));
      expect_lo = _mm_add_epi32(expect_lo, step);
    }
    for (; i < reach; ++i)
      if (lo[i] != f + i * s || (hi && hi[i] != static_cast<uint32_t>(first >> 32))) return std::max<uint32_t>(1, i);
    return reach;
  }
#endif
  return run_scan_reference(lo, hi, reach, first, stride);
}

// Whether two neighbouring entries of a list of 64-bit addresses are equal (a load's lane that follows a lane with the same address takes
// that lane's value, Machine::wave_scalars: the usual wave has no such pair, and the pass that copies the values then need not run).
// `any_adjacent_equal_reference` is the definition; the vector form compares two entries to their predecessors at a time (SSE2).
inline bool any_adjacent_equal_reference(const uint64_t* a, uint32_t n) {
  for (uint32_t i = 1; i < n; ++i)
    if (a[i] == a[i - 1]) return true;
  return false;
}

inline bool any_adjacent_equal(const uint64_t* a, uint32_t n) {
#if defined(__SSE2__) && defined(__x86_64__)
  __m128i acc = _mm_setzero_si128();
  uint32_t i = 1;
  for (; i + 2 <= n; i += 2) {
    const __m128i eq = _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)),
                                       _mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i - 1)));
    acc = _mm_or_si128(acc, _mm_and_si128(eq, _mm_shuffle_epi32(eq, _MM_SHUFFLE(2, 3, 0, 1))));  // (both halves of a 64-bit entry equal)
  }
  bool any = _mm_movemask_epi8(acc) != 0;
  for (; i < n; ++i) any |= a[i] == a[i - 1];
  return any;
#else
  return any_adjacent_equal_reference(a, n);
#endif
}

// Whether every entry equals the first (a wave's lanes all loading one address: a uniform value read through vector registers). The
// definition is `all_equal_reference`; the vector form compares two entries to the first at a time (SSE2).
inline bool all_equal_reference(const uint64_t* a, uint32_t n) {
  for (uint32_t i = 1; i < n; ++i)
    if (a[i] != a[0]) return false;
  return true;
}

inline bool all_equal(const uint64_t* a, uint32_t n) {
#if defined(__SSE2__) && defined(__x86_64__)
  if (n < 2) return true;
  const __m128i first = _mm_set1_epi64x(static_cast<long long>(a[0]));
  __m128i acc = _mm_set1_epi32(-1);
  uint32_t i = 0;
  for (; i + 2 <= n; i += 2)
    acc = _mm_and_si128(acc, _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(a + i)), first));
  bool same = _mm_movemask_epi8(acc) == 0xFFFF;
  for (; i < n; ++i) same &= a[i] == a[0];
  return same;
#else
  return all_equal_reference(a, n);
#endif
}

}  // namespace vgpu::exec::amdrun
