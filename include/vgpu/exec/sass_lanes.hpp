// The SASS executor's work on the 32 lanes of a warp that is not arithmetic: which lanes stand at an address, the lowest address among
// some lanes, and the state of lanes that wait at a barrier. They are written with SSE2 where it is there and tested against the
// lane-by-lane definitions (tests/unit/test_sass_lanes.cpp). The warp is any type with the fields these name: `pc[32]` (uint64_t),
// `waiting` (a lane mask), `wait_kind[32]` (a byte-sized enum whose zero is "not waiting") and `wait_arg[32]` (uint32_t).
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>
#include <type_traits>

#if defined(__SSE2__)
#include <emmintrin.h>
#endif

#include "vgpu/simd/simd.hpp"

namespace vgpu::exec::sasslanes {

using Mask = uint32_t;
constexpr Mask kAll = ~Mask{0};

// The lanes whose program counter is `p`, compared four to a movemask where SSE2 is there.
inline Mask lanes_at(const uint64_t* pc, uint64_t p) {
  Mask eq = 0;
#if defined(__SSE2__)
  const __m128i want = _mm_set1_epi64x(static_cast<long long>(p));
  for (unsigned i = 0; i < 32; i += 4) {
    const __m128i a = _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&pc[i])), want);
    const __m128i b = _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&pc[i + 2])), want);
    // The low halves of the four lanes' addresses, and their high halves: a lane is equal when both are.
    const __m128 lo = _mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _MM_SHUFFLE(2, 0, 2, 0));
    const __m128 hi = _mm_shuffle_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b), _MM_SHUFFLE(3, 1, 3, 1));
    eq |= static_cast<Mask>(_mm_movemask_ps(_mm_and_ps(lo, hi))) << i;
  }
#else
  for (unsigned i = 0; i < 32; ++i) eq |= static_cast<Mask>(pc[i] == p) << i;
#endif
  return eq;
}

// The lowest address among the runnable lanes' (`first` is the first one's): the same minimum the plain loop finds. With many lanes
// running, as four independent chains over all 32 lanes, a lane that does not run counting as the largest address (an all-ones OR,
// no branch on the lane's bit: those were mispredicted in a diverged warp); with few, the loop over just those lanes.
inline uint64_t lowest_pc(const uint64_t* pc, Mask run, uint64_t first) {
  if (simd::popcount32(run) <= 5) {
    uint64_t lo = first;
    for (Mask m = run & (run - 1); m; m &= m - 1) lo = std::min(lo, pc[std::countr_zero(m)]);
    return lo;
  }
  const auto pcv = [&](unsigned l) { return pc[l] | ((static_cast<uint64_t>((run >> l) & 1)) - 1); };
  uint64_t a = ~uint64_t{0}, b = a, c = a, d = a;
  for (unsigned l = 0; l < 32; l += 4) {
    a = std::min(a, pcv(l));
    b = std::min(b, pcv(l + 1));
    c = std::min(c, pcv(l + 2));
    d = std::min(d, pcv(l + 3));
  }
  return std::min(std::min(a, b), std::min(c, d));
}


// The group the scheduler runs next: the lowest address among the runnable lanes' (`run`, not zero) in `lo`, and the runnable lanes standing there.
// Runnable lanes that all stand at one address (the usual case) are found by comparing the 32 addresses with the first one's; a diverged
// warp's lowest address and its lanes come from pick_group_diverged (src/simd/lanes.cpp: one pass of 256-bit compares with AVX2).
// pick_group_reference is the lanes_at/lowest_pc definition, for the test.
Mask pick_group_diverged(const uint64_t* pc, Mask run, uint64_t& lo);
Mask pick_group_reference(const uint64_t* pc, Mask run, uint64_t& lo);
inline Mask pick_group(const uint64_t* pc, Mask run, uint64_t& lo) {
  const uint64_t first = pc[std::countr_zero(run)];
  if ((run & ~lanes_at(pc, first)) == 0) {
    lo = first;
    return run;
  }
  return pick_group_diverged(pc, run, lo);
}

// The lanes of `group` stand at address `v` from now on, and (select_pcs) the lanes of `take` at `target` and the others of `group` at `next`:
// the addresses a branch sets. With AVX2 four lanes to a store under a mask, not a loop over the lanes (whose length no predictor learns in
// a diverged warp); src/simd/lanes.cpp. The lanes outside `group` keep theirs.
void set_pcs_partial(uint64_t* pc, Mask group, uint64_t v);
void select_pcs(uint64_t* pc, Mask group, Mask take, uint64_t target, uint64_t next);

// The lanes of `ex` start to wait with `kind` on `arg`.
template <class W, class K>
inline void wait(W& w, Mask ex, K kind, unsigned arg) {
#if defined(__SSE2__)
  if (ex == kAll) {
    const __m128i k = _mm_set1_epi8(static_cast<char>(kind)), a = _mm_set1_epi32(static_cast<int>(arg));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[0]), k);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[16]), k);
    for (unsigned i = 0; i < 32; i += 4) _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_arg[i]), a);
    w.waiting |= ex;
    return;
  }
#endif
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    w.wait_kind[l] = kind;
    w.wait_arg[l] = arg;
  }
  w.waiting |= ex;
}

// The lanes of `w` that wait with `kind` on `arg` (the two arrays compared a vector at a time instead of looked at lane by lane).
template <class W, class K>
inline Mask waiting_on(const W& w, K kind, unsigned arg) {
  Mask arrived = w.waiting;
  if (!arrived) return 0;
#if defined(__SSE2__)
  const __m128i k = _mm_set1_epi8(static_cast<char>(kind));
  const uint32_t kinds = static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.wait_kind[0])), k))) |
                         static_cast<uint32_t>(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.wait_kind[16])), k))) << 16;
  const __m128i want = _mm_set1_epi32(static_cast<int>(arg));
  uint32_t args = 0;
  for (unsigned i = 0; i < 8; ++i)
    args |= static_cast<uint32_t>(_mm_movemask_ps(_mm_castsi128_ps(
                _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.wait_arg[4 * i])), want)))) << (4 * i);
  return arrived & kinds & args;
#else
  Mask match = 0;
  for (Mask m = arrived; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    if (w.wait_kind[l] == kind && w.wait_arg[l] == arg) match |= Mask{1} << l;
  }
  return match;
#endif
}

// The lanes of `m` (waiting) stop waiting and go on past the instruction they wait at.
template <class W>
inline void wake(W& w, Mask m) {
  if (!m) return;
  w.waiting &= ~m;
#if defined(__SSE2__)
  if (m == kAll) {   // a whole warp: the kinds cleared and the 32 addresses moved on a vector at a time
    const __m128i zero = _mm_setzero_si128(), step = _mm_set1_epi64x(16);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[0]), zero);
    _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.wait_kind[16]), zero);
    for (unsigned i = 0; i < 32; i += 2)
      _mm_storeu_si128(reinterpret_cast<__m128i*>(&w.pc[i]), _mm_add_epi64(_mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.pc[i])), step));
    return;
  }
#endif
  for (; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    w.wait_kind[l] = std::remove_reference_t<decltype(w.wait_kind[0])>{};   // (not waiting)
    w.pc[l] += 16;
  }
}


// `v` into the words of the lanes of `ex` (a row of 32), the others left alone: the whole warp is a broadcast and eight stores in place of a
// lane search and a store a lane (S2R of a value the same for every lane: a block index, the block's size).
inline void fill_lanes(uint32_t* d, Mask ex, uint32_t v) {
  if (ex == kAll) {
    for (unsigned l = 0; l < 32; ++l) d[l] = v;
    return;
  }
  for (Mask m = ex; m; m &= m - 1) d[static_cast<unsigned>(std::countr_zero(m))] = v;
}

// The lane numbers into the words of the lanes of `ex` (S2R SR_LANEID): the whole warp is a constant row.
inline void lane_ids(uint32_t* d, Mask ex) {
  if (ex == kAll) {
    for (unsigned l = 0; l < 32; ++l) d[l] = l;
    return;
  }
  for (Mask m = ex; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    d[l] = l;
  }
}

// Whether all 32 words of a row are the same, and then that word.
inline bool row_uniform(const uint32_t* r, uint32_t& v) {
#if defined(__SSE2__)
  const __m128i first = _mm_set1_epi32(static_cast<int>(r[0]));
  __m128i eq = _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r)), first);
  for (unsigned i = 1; i < 8; ++i) eq = _mm_and_si128(eq, _mm_cmpeq_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r) + i), first));
  v = r[0];
  return _mm_movemask_epi8(eq) == 0xffff;
#else
  v = r[0];
  for (unsigned l = 1; l < 32; ++l)
    if (r[l] != v) return false;
  return true;
#endif
}

}  // namespace vgpu::exec::sasslanes
