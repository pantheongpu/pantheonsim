// The counters a warp's memory access feeds: how many 32-byte sectors its lanes
// touch, and how many passes shared memory's banks need to serve it.
//
// Both engines count these for every global, local and shared access of every
// kernel -- they are what `vgpu counters` and the CUPTI metrics report -- so
// they sit on the path of every memory instruction. The first versions compared
// every lane's sectors with every other's, zero-filled a few kilobytes of
// scratch to do it, and for shared memory did that once per bank: a shared load
// cost a thousand comparisons and 64 KB of stores. These give the same numbers,
// lane for lane, without the scratch or the all-pairs comparison.
// tests/unit/test_memfast.cpp holds the earlier code verbatim and requires the
// same answer from both over millions of random accesses.
//
// The two engines differ in one small way and each keeps its own rule: the PTX
// interpreter counts the words (address + k * 4) / 4 for k * 4 < bytes (an
// unaligned 4-byte access counts one word, the one its address is in), the SASS
// executor counts the words the bytes actually cover (an unaligned one counts
// two).
//
// The earlier code is kept below as the *_reference functions. Anything outside
// the shapes the fast versions were written for (an access of more than 128
// bytes, a warp of more than 64 lanes, a range of sectors too long for the
// scratch) is counted by them, and so is everything when VGPU_FASTCOUNT=0.
//
// The shapes kernels make most -- lanes on consecutive elements, on a fixed
// stride, all on one element, and a thread-private array's few sectors -- are
// counted by closed forms first (sectors_progression / sectors_monotone /
// sectors_window / sectors_hash, bank_passes_progression): each reads the
// lanes' addresses once, proves the shape holds for every active lane, and
// gives up (the general counting below then runs from the start) the moment
// one lane does not fit. A closed form is only ever used where it is shown
// here to equal the counting it stands for, and tests/unit/test_memcount_forms.cpp
// compares every entry point with the definition of what it counts over
// exhaustive small patterns and random warps, and test_memfast.cpp with the
// earlier code.
#pragma once

#include <algorithm>
#include <array>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "vgpu/simd/simd.hpp"

namespace vgpu::exec::memcount {

// Whether the fast versions are used (not when VGPU_FASTCOUNT=0).
inline bool fast() {
  static const bool on = [] {
    const char* v = std::getenv("VGPU_FASTCOUNT");
    return !(v && v[0] == '0' && v[1] == '\0');
  }();
  return on;
}

// ---- the earlier counting, as it stood ------------------------------------------

// Distinct 32-byte sectors in [addr, addr + span) over the lanes of `mask`, the
// interpreter's way: the first `cap` (at most 512) are remembered, and a sector
// past that is not counted.
inline uint32_t sectors_reference(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span,
                                  size_t cap) {
  std::array<uint64_t, 512> sectors{};
  if (cap > sectors.size()) cap = sectors.size();
  uint32_t distinct = 0;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    if (!(mask & (uint64_t{1} << lane))) continue;
    const uint64_t first = addrs[lane] / 32;
    const uint64_t last = (addrs[lane] + (span ? span - 1 : 0)) / 32;
    for (uint64_t sec = first; sec <= last; ++sec) {
      bool seen = false;
      for (uint32_t i = 0; i < distinct; ++i)
        if (sectors[i] == sec) {
          seen = true;
          break;
        }
      if (!seen && distinct < cap) sectors[distinct++] = sec;
    }
  }
  return distinct;
}

// The SASS executor's: every sector of every lane, sorted and de-duplicated.
inline uint64_t sass_sectors_reference(const uint64_t* addr, uint64_t mask, unsigned lanes, uint32_t bytes) {
  std::vector<uint64_t> sectors;
  for (unsigned l = 0; l < lanes; ++l)
    if (mask & (uint64_t{1} << l))
      for (uint64_t s = addr[l] / 32; s <= (addr[l] + bytes - 1) / 32; ++s) sectors.push_back(s);
  std::sort(sectors.begin(), sectors.end());
  return static_cast<uint64_t>(std::unique(sectors.begin(), sectors.end()) - sectors.begin());
}

// ---- the same numbers, without the scratch ---------------------------------------

// ---- closed forms for the shapes kernels make most --------------------------------

namespace detail {

constexpr uint64_t lane_mask(unsigned lanes) {
  return lanes >= 64 ? ~uint64_t{0} : (uint64_t{1} << lanes) - 1;
}

// (Without -mpopcnt the builtin is a call into libgcc.)
inline unsigned popcount64(uint64_t x) {
  x = x - ((x >> 1) & 0x5555'5555'5555'5555ull);
  x = (x & 0x3333'3333'3333'3333ull) + ((x >> 2) & 0x3333'3333'3333'3333ull);
  x = (x + (x >> 4)) & 0x0F0F'0F0F'0F0F'0F0Full;
  return static_cast<unsigned>((x * 0x0101'0101'0101'0101ull) >> 56);
}

// Calls fn(address) for each active lane in lane order, stopping (false) at the first call that returns false. The
// active lanes are `m` (not zero); when they are consecutive, which is the usual case (a full warp, a tail, a
// reduction's low half), the lanes are walked in a plain counted loop.
template <class Fn>
inline bool for_each_address(const uint64_t* addrs, uint64_t m, Fn&& fn) {
  const unsigned lo = static_cast<unsigned>(std::countr_zero(m));
  const uint64_t run = m >> lo;
  if ((run & (run + 1)) == 0) {
    const unsigned n = 64 - static_cast<unsigned>(std::countl_zero(run));   // (run is 2^n - 1)
    for (unsigned i = 0; i < n; ++i)
      if (!fn(addrs[lo + i])) return false;
    return true;
  }
  for (; m; m &= m - 1)
    if (!fn(addrs[std::countr_zero(m)])) return false;
  return true;
}

// ---- lanes on a fixed stride

// Two 64-bit lanes at a time (GCC's generic vectors: SSE2 where there is any, plain code elsewhere).
typedef uint64_t V2 __attribute__((vector_size(16)));
typedef uint64_t V2u __attribute__((vector_size(16), aligned(8), may_alias));
inline V2 load_v2(const uint64_t* p) { return *reinterpret_cast<const V2u*>(p); }

// Whether addrs[lo + i] == addrs[lo] + i * stride (modulo 2^64) for i < n. The first four lanes alone are compared
// first, so that a warp that is not on a stride is turned away after two steps.
inline bool on_stride(const uint64_t* addrs, unsigned lo, unsigned n, uint64_t stride) {
  const uint64_t a0 = addrs[lo];
  V2 expect = {a0, a0 + stride};
  const V2 step = {stride + stride, stride + stride};
  V2 diff = {0, 0};
  const uint64_t* p = addrs + lo;
  unsigned i = 0;
  const unsigned head = n < 4 ? n & ~1u : 4;
  for (; i < head; i += 2, expect += step) diff |= load_v2(p + i) ^ expect;
  if (diff[0] | diff[1]) return false;
  for (; i + 2 <= n; i += 2, expect += step) diff |= load_v2(p + i) ^ expect;
  uint64_t rest = diff[0] | diff[1];
  if (i < n) rest |= p[i] ^ expect[0];   // (an odd count: the last lane)
  return rest == 0;
}

// Whether the active lanes `m` (at least one) are consecutive and their addresses a fixed stride apart, the stride
// under 2^32 either way and no address wrapped or within `margin` bytes of the top of the space; then the stride in
// bytes (a signed number; zero for a single lane), the lowest lane and the count. With the endpoints in range and
// the stride small, no lane in between has wrapped either, so an address and its word or sector follow the same
// arithmetic as the lanes do.
template <bool LastFirst = false>
inline bool lane_progression(const uint64_t* addrs, uint64_t m, uint64_t margin, int64_t& stride, unsigned& lo,
                             unsigned& n) {
  lo = static_cast<unsigned>(std::countr_zero(m));
  const uint64_t run = m >> lo;
  if (run & (run + 1)) return false;   // (a gap among the active lanes)
  n = 64 - static_cast<unsigned>(std::countl_zero(run));   // (run is 2^n - 1)
  const uint64_t top = ~uint64_t{0} - margin;
  const uint64_t a0 = addrs[lo];
  if (a0 > top) return false;
  if (n == 1) {
    stride = 0;
    return true;
  }
  const int64_t s = static_cast<int64_t>(addrs[lo + 1] - a0);
  if (s > (int64_t{1} << 32) || s < -(int64_t{1} << 32)) return false;
  // (LastFirst: the last lane first, so that a warp of rows -- a run on one stride, then a jump -- is turned away without
  // walking its lanes; one more comparison for the warps that are on the stride, which the bank counting takes for the
  // warps it can remember and the sector counting, whose warps nearly always are, does without)
  if (LastFirst && addrs[lo + n - 1] != a0 + uint64_t{n - 1} * static_cast<uint64_t>(s)) return false;
  if (!on_stride(addrs, lo, n, static_cast<uint64_t>(s))) return false;
  const uint64_t last = addrs[lo + n - 1];
  if (last > top || (s >= 0 ? last < a0 : last > a0)) return false;
  stride = s;
  return true;
}

// ---- sectors

// Addresses within this many bytes of the top of the space are not given to the closed forms for accesses of up to
// 32 bytes: adding the width wraps, which the general loop counts as no sectors at all.
constexpr uint64_t kSectorTop = ~uint64_t{0} - 31;

// The sectors of lanes on a fixed stride (lane_progression), each lane covering `span` bytes. Two cases have a closed
// form:
//  - a stride (either way) of at most the width plus 31 bytes, so that the gap between one access and the next is
//    under a sector: the lanes' accesses cover every sector from the lowest address's to the highest access's last
//    byte, because a sector that none of them touched would have to lie wholly inside a gap. (One lane, and every lane
//    on the same bytes, are this with a stride of zero.)
//  - a stride that is a multiple of 32 and at least as many sectors as one access touches: every lane touches the same
//    number c of sectors (the same offset into the sector) and no two lanes share one, so n * c.
// Anything else (false) goes to the counts that walk the lanes.
inline bool sectors_progression(const uint64_t* addrs, uint64_t m, uint64_t span, uint64_t& out) {
  if (span > 4096) return false;
  int64_t s;
  unsigned lo, n;
  if (!lane_progression(addrs, m, span, s, lo, n)) return false;
  const uint64_t a0 = addrs[lo], last = addrs[lo + n - 1];
  const uint64_t us = s < 0 ? 0 - static_cast<uint64_t>(s) : static_cast<uint64_t>(s);
  if (us <= span + 31) {
    const uint64_t low = s >= 0 ? a0 : last, high = s >= 0 ? last : a0;
    out = ((high + span - 1) >> 5) - (low >> 5) + 1;
    return true;
  }
  if ((us & 31) == 0) {
    const uint64_t one = ((a0 + span - 1) >> 5) - (a0 >> 5) + 1;   // (the sectors of the first lane's access)
    if (us / 32 >= one) {
      out = uint64_t{n} * one;
      return true;
    }
  }
  return false;
}

// Distinct sectors of accesses of 1 to 32 bytes (so each lane covers its sector and at most the next), when the
// lanes' sectors never go backwards in lane order (Ascending), or never go forwards (!Ascending): coalesced,
// strided and broadcast accesses. Lane k covers the sectors [f, e]; given that f is not below the last sector e'
// the lanes before it reached, it adds e - f new sectors, and one more unless f == e' (the sector it starts in is
// the one the last lane ended in). That is the count the general loop makes while its list is still ascending,
// without the list. Gives up (false) on the first lane that does not fit.
template <bool Ascending>
inline bool sectors_monotone(const uint64_t* addrs, uint64_t m, uint64_t span, uint64_t& out) {
  const uint64_t extra = span - 1;
  uint64_t n = 0, prev = 0;
  bool first = true;
  const bool ok = for_each_address(addrs, m, [&](uint64_t a) {
    if (a > kSectorTop) return false;
    const uint64_t f = a >> 5, e = (a + extra) >> 5;
    if (first) {
      first = false;
      n = e - f + 1;
      prev = Ascending ? e : f;
      return true;
    }
    if (Ascending) {
      if (f < prev) return false;
      n += (e - f) + (f != prev);
      prev = e;
    } else {
      if (e > prev) return false;
      n += (e - f) + (e != prev);
      prev = f;
    }
    return true;
  });
  out = n;
  return ok;
}

// The same accesses in any order, when every sector lies within the 64 that start 24 before the first lane's (768
// bytes below its address, 1.2 KB above): a thread-private array indexed by a value of each lane's own lands in a
// few sectors in no order. The sectors are the set bits of one word.
inline bool sectors_window(const uint64_t* addrs, uint64_t m, uint64_t span, uint64_t& out) {
  const uint64_t extra = span - 1;
  const uint64_t base = (addrs[std::countr_zero(m)] >> 5) - 24;   // (wraps below zero; the offsets below wrap with it)
  uint64_t bits = 0;
  const bool ok = for_each_address(addrs, m, [&](uint64_t a) {
    if (a > kSectorTop) return false;
    const uint64_t f = a >> 5, e = (a + extra) >> 5;
    const uint64_t d = f - base;
    if (d > 62) return false;
    const uint64_t bit = uint64_t{1} << d;
    bits |= bit | (bit << (e - f));
    return true;
  });
  out = popcount64(bits);
  return ok;
}

// Any order and any spread, for up to 32 active lanes of 1 to 32 bytes (so at most 64 sectors): each sector is
// looked up in a table of 128 slots (open addressing, so it is never more than half full), which does not need
// clearing, because a bitmap says which slots hold a key. This is the gather: every lane in a sector of its own.
inline bool sectors_hash(const uint64_t* addrs, uint64_t m, uint64_t span, uint64_t& out) {
  if (popcount64(m) > 32) return false;
  const uint64_t extra = span - 1;
  uint64_t keys[128];
  uint64_t used[2] = {0, 0};
  uint64_t n = 0;
  const bool ok = for_each_address(addrs, m, [&](uint64_t a) {
    if (a > kSectorTop) return false;
    const uint64_t f = a >> 5, e = (a + extra) >> 5;
    for (uint64_t sec = f; sec <= e; ++sec) {
      unsigned h = static_cast<unsigned>((sec * 0x9E3779B97F4A7C15ull) >> 57);
      for (;;) {
        const uint64_t bit = uint64_t{1} << (h & 63);
        if (!(used[h >> 6] & bit)) {
          used[h >> 6] |= bit;
          keys[h] = sec;
          ++n;
          break;
        }
        if (keys[h] == sec) break;
        h = (h + 1) & 127;
      }
    }
    return true;
  });
  out = n;
  return ok;
}

// The closed forms, in the order that fits the shapes best: lanes on a stride by arithmetic alone; lanes that only go
// forward (strided, a broadcast, a thread's few bytes) by one comparison each; a warp that stays near its first lane's
// sector in no order goes to the window (judged from its first and last active lane, so that one that spreads wide is
// not walked twice); one that only goes backward is next; and the rest are looked up one sector at a time.
[[gnu::noinline]] inline bool sectors_fast(const uint64_t* addrs, uint64_t m, uint64_t span, uint64_t& out) {
  if (sectors_progression(addrs, m, span, out)) return true;
  if (span > 32) return false;
  const uint64_t f0 = addrs[std::countr_zero(m)] >> 5, f1 = addrs[63 - std::countl_zero(m)] >> 5;
  const bool near = f1 - f0 + 24 <= 62;
  // With the AVX2 window the near warps come first: one pass over the lanes four at a time takes them in any order, where
  // the forward walk below gives up at the first lane that goes back (a thread-private array's index wraps half way).
  if (near)
    if (const simd::SectorWindowFn window = simd::sector_window_kernel(); window && window(addrs, m, span, out)) return true;
  if (sectors_monotone<true>(addrs, m, span, out)) return true;
  if (near && sectors_window(addrs, m, span, out)) return true;
  // (Only a warp whose second lane is below its first can go backward all the way.)
  const uint64_t m2 = m & (m - 1);
  if (m2 && (addrs[std::countr_zero(m2)] >> 5) < f0 && sectors_monotone<false>(addrs, m, span, out)) return true;
  return sectors_hash(addrs, m, span, out);
}

}  // namespace detail

// The sectors of a run: `total` bytes from `g0` -- the accesses of lanes on consecutive elements, which the
// caller has found by comparing each lane's address with g0 + k * width. They cover [g0, g0 + total) with no gap,
// so the count is the span of sectors. False (the caller counts the lanes' addresses) when the closed form is off
// (VGPU_FASTCOUNT=0), the run is empty, or it would run off the top of the address space.
inline bool run_sectors(uint64_t g0, uint64_t total, uint64_t& out) {
  if (!fast() || total == 0 || g0 + total - 1 < g0) return false;
  out = ((g0 + total - 1) >> 5) - (g0 >> 5) + 1;
  return true;
}

inline uint32_t distinct_sectors(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span, size_t cap) {
  // A cap of SIZE_MAX is the SASS executor's, which counts every sector.
  const auto reference = [&]() -> uint32_t {
    return cap == SIZE_MAX ? static_cast<uint32_t>(sass_sectors_reference(addrs, mask, lanes, static_cast<uint32_t>(span)))
                           : sectors_reference(addrs, mask, lanes, span, cap);
  };
  if (!fast()) return reference();
  if (span >= 1 && lanes <= 64) {   // the closed forms (each gives up on what it is not made for)
    const uint64_t m = mask & detail::lane_mask(lanes);
    if (!m) return 0;
    uint64_t n;
    if (detail::sectors_fast(addrs, m, span, n)) return static_cast<uint32_t>(std::min<uint64_t>(n, cap));
  }
  constexpr size_t kBuf = 1024;
  uint64_t buf[kBuf];
  size_t n = 0;
  // The sectors seen so far, each once. While the lanes' sectors come in strictly ascending order (coalesced
  // and strided accesses) a sector is new unless it repeats the last one, which takes one comparison; the
  // first lane that goes backwards switches to searching the list, newest first, for each sector after it
  // (a gather, or a private array indexed by a value of its own, lands in a few sectors in no order: sorting
  // those at full length cost more than the search).
  bool ascending = true;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    if (!(mask & (uint64_t{1} << lane))) continue;
    const uint64_t first = addrs[lane] / 32;
    const uint64_t last = (addrs[lane] + (span ? span - 1 : 0)) / 32;
    if (last < first) continue;   // the address wrapped: the plain loop runs no iterations either
    if (last - first >= kBuf - n) return reference();
    for (uint64_t sec = first; sec <= last; ++sec) {
      if (ascending) {
        if (n && buf[n - 1] == sec) continue;
        if (!n || buf[n - 1] < sec) {
          buf[n++] = sec;
          continue;
        }
        ascending = false;
      }
      size_t i = n;
      while (i > 0 && buf[i - 1] != sec) --i;
      if (i == 0) buf[n++] = sec;
    }
  }
  // The reference stops remembering at `cap`, and a sector it did not keep is
  // never counted: the count saturates there.
  return static_cast<uint32_t>(std::min(n, cap));
}

// distinct_sectors for the SASS executor (every sector counted: no cap), with the closed forms in a function of their own: the list
// that distinct_sectors falls back on is 8 KB on the stack, and a function that inlines it probes every page of that frame on entry
// (two probes, the stack check) -- which every warp that a closed form counts paid, 30% of the counting for a strided gather. The
// warps the closed forms give up on go to `distinct_sectors_listed`, which is distinct_sectors in a function that owns the frame.
[[gnu::noinline]] inline uint32_t distinct_sectors_listed(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span) {
  return distinct_sectors(addrs, mask, lanes, span, SIZE_MAX);
}
inline uint32_t distinct_sectors_uncapped(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span) {
  if (fast() && span >= 1 && lanes <= 64) {
    const uint64_t m = mask & detail::lane_mask(lanes);
    if (!m) return 0;
    uint64_t n;
    if (detail::sectors_fast(addrs, m, span, n)) return static_cast<uint32_t>(n);   // (min(n, SIZE_MAX))
  }
  return distinct_sectors_listed(addrs, mask, lanes, span);
}

namespace detail {
// Distinct words per bank. At most 64 lanes each put at most one word in a bank
// (PTX: up to 32 consecutive words) or two (SASS, 17 words) -- the callers keep
// to shapes where a bank cannot hold more than 64.
struct BankTable {
  uint8_t count[32] = {};   // (a bank holds at most 64 words)
  uint64_t word[32][64];
  void add(uint64_t w) {
    const unsigned b = static_cast<unsigned>(w % 32);
    const uint32_t c = count[b];
    for (uint32_t i = 0; i < c; ++i)
      if (word[b][i] == w) return;
    word[b][c] = w;
    count[b] = static_cast<uint8_t>(c + 1);
  }
  uint32_t worst() const {
    uint32_t m = 1;
    for (uint32_t c : count) m = c > m ? c : m;
    return m;
  }
};
}  // namespace detail

// ---- bank passes

namespace detail {

// The bank passes of lanes on a fixed stride (lane_progression), each covering `nw` consecutive words from its
// address's word: lane i covers the words W + i*sw + j for j < nw, sw = stride / 4. Four cases have a closed form,
// each the most distinct words any bank is asked for:
//  - every lane on the same words (stride 0): nw <= 32 consecutive words, one to a bank: 1 pass.
//  - a stride under four bytes: the lanes' words are a run of consecutive words (a lane's bytes start at most three
//    after the last's, so no whole word is skipped), T of them, all different and spread over the 32 banks as evenly
//    as they go: ceil(T / 32).
//  - the lanes' runs end to end or overlapping (1 <= |sw| <= nw): their union is the (n - 1) * |sw| + nw consecutive
//    words from the first lane's to the last's, all different: ceil(that / 32).
//  - one word a lane (nw = 1) on any other stride that is a multiple of four: lanes i and j share a bank when
//    (i - j) * sw is a multiple of 32, i.e. of p = 32 / gcd(|sw|, 32), and then they are on different words (sw is
//    not zero); the lanes fall into the p classes of i mod p and the biggest holds ceil(n / p).
// Anything else (false) takes the general counting. `sass` is whether a lane's words are those its bytes cover
// (`arg` bytes of them), as the SASS executor counts, or the interpreter's ceil(arg / 4) from its address's word.
template <bool Sass>
inline bool bank_passes_progression(const uint64_t* addrs, uint64_t m, uint32_t arg, uint32_t& out) {
  int64_t stride;
  unsigned lo, n;
  if (!lane_progression<true>(addrs, m, 256, stride, lo, n)) return false;
  const uint64_t first = addrs[lo], last = addrs[lo + n - 1];
  // A lane's words, as the number from its address's word on (the SASS rule depends on the byte in the word).
  const auto words_at = [&](uint64_t addr) -> uint32_t {
    return Sass ? static_cast<uint32_t>(((addr & 3) + arg + 3) >> 2) : (arg + 3) / 4;
  };
  if (n == 1 || stride == 0) {
    out = 1;   // (at most 32 consecutive words, one to a bank)
    return true;
  }
  if (stride > -4 && stride < 4) {
    const uint64_t low = stride > 0 ? first : last, high = stride > 0 ? last : first;
    const uint64_t wmin = low >> 2;
    const uint64_t wmax = Sass ? (high + arg - 1) >> 2 : (high >> 2) + words_at(high) - 1;
    out = static_cast<uint32_t>((wmax - wmin + 1 + 31) / 32);
    return true;
  }
  if (stride & 3) return false;
  const uint32_t nw = words_at(first);   // (the same at every lane: the byte in the word does not change)
  const int64_t sw = stride / 4;
  const uint64_t absw = sw < 0 ? 0 - static_cast<uint64_t>(sw) : static_cast<uint64_t>(sw);
  if (absw <= nw) {
    out = static_cast<uint32_t>(((uint64_t{n} - 1) * absw + nw + 31) / 32);
    return true;
  }
  if (nw == 1) {
    const uint64_t g = std::min<uint64_t>(32, absw & (0 - absw));   // gcd(|sw|, 32)
    const uint64_t p = 32 / g;
    out = static_cast<uint32_t>((n + p - 1) / p);
    return true;
  }
  return false;
}

}  // namespace detail

// The interpreter's passes: the most distinct words any one of 32 banks is
// asked for, over the words (address + k * 4) / 4 for k * 4 < span.
inline uint32_t ptx_bank_passes_reference(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span) {
  uint32_t worst = 1;
  for (uint32_t bank = 0; bank < 32; ++bank) {
    std::array<uint64_t, 256> words{};
    uint32_t distinct = 0;
    for (unsigned lane = 0; lane < lanes; ++lane) {
      if (!(mask & (uint64_t{1} << lane))) continue;
      for (uint64_t off = 0; off < span; off += 4) {
        const uint64_t word = (addrs[lane] + off) / 4;
        if (word % 32 != bank) continue;
        bool seen = false;
        for (uint32_t i = 0; i < distinct; ++i)
          if (words[i] == word) {
            seen = true;
            break;
          }
        if (!seen && distinct < words.size()) words[distinct++] = word;
      }
    }
    if (distinct > worst) worst = distinct;
  }
  return worst;
}

inline uint32_t ptx_bank_passes(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span) {
  if (!fast() || span > 128 || lanes > 64) return ptx_bank_passes_reference(addrs, mask, lanes, span);
  if (span) {
    const uint64_t m = mask & detail::lane_mask(lanes);
    uint32_t out;
    if (m && detail::bank_passes_progression<false>(addrs, m, static_cast<uint32_t>(span), out)) return out;
  }
  detail::BankTable t;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    if (!(mask & (uint64_t{1} << lane))) continue;
    for (uint64_t off = 0; off < span; off += 4) t.add((addrs[lane] + off) / 4);
  }
  return t.worst();
}

// The SASS executor's passes, over the words the bytes cover.
inline uint32_t sass_bank_passes_reference(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint32_t bytes) {
  std::vector<uint64_t> words[32];
  for (unsigned l = 0; l < lanes; ++l) {
    if (!(mask & (uint64_t{1} << l))) continue;
    for (uint64_t a = addrs[l] & ~3ull; a < addrs[l] + bytes; a += 4) words[(a / 4) % 32].push_back(a / 4);
  }
  uint64_t passes = 1;
  for (auto& v : words) {
    std::sort(v.begin(), v.end());
    passes = std::max<uint64_t>(passes, static_cast<uint64_t>(std::unique(v.begin(), v.end()) - v.begin()));
  }
  return static_cast<uint32_t>(passes);
}

// The counting that is left once the closed forms have given up: the words of every active lane, by bank.
inline uint32_t sass_bank_passes_table(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint32_t bytes) {
  detail::BankTable t;
  if (bytes - 1 < 4 && lanes <= 32) {   // (1 to 4 bytes: a 0-byte access covers no word of an aligned address)
    // Up to a word's worth of bytes a lane covers its address's word, and the next one when it does not end in it: a
    // counted walk of the active lanes, a lane at a time.
    const uint64_t m = mask & detail::lane_mask(lanes);
    if (m) {
      const bool ok = detail::for_each_address(addrs, m, [&](uint64_t a) {
        if (a > UINT64_MAX - 256) return false;   // (as below)
        t.add(a >> 2);
        if ((a & 3) + bytes > 4) t.add((a >> 2) + 1);
        return true;
      });
      if (!ok) return sass_bank_passes_reference(addrs, mask, lanes, bytes);
    }
    return t.worst();
  }
  for (unsigned l = 0; l < lanes; ++l) {
    if (!(mask & (uint64_t{1} << l))) continue;
    // A range that runs off the top of the address space makes the loop below
    // wrap and count on from zero; the reference does the same, slowly, and
    // nothing a device address can be reaches it.
    if (addrs[l] > UINT64_MAX - 256) return sass_bank_passes_reference(addrs, mask, lanes, bytes);
    for (uint64_t a = addrs[l] & ~3ull; a < addrs[l] + bytes; a += 4) t.add(a / 4);
  }
  return t.worst();
}

inline uint32_t sass_bank_passes(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint32_t bytes) {
  if (!fast() || bytes > 64 || lanes > 32) return sass_bank_passes_reference(addrs, mask, lanes, bytes);
  if (bytes) {
    const uint64_t m = mask & detail::lane_mask(lanes);
    uint32_t out;
    if (m && detail::bank_passes_progression<true>(addrs, m, bytes, out)) return out;
  }
  return sass_bank_passes_table(addrs, mask, lanes, bytes);
}

// ---- bank passes remembered by shape

// The passes of a warp that is on no stride (two rows of a tile, a pair of words every four, a group of lanes on one
// word) are the same for every warp of that shape wherever it sits: the passes depend on which of the lanes' words are
// the same and which banks they are in, and moving every address by a whole number of words moves the banks round
// together and keeps the same words the same. So a warp's shape -- the width and each lane's address less the first
// active lane's word, a lane that does not run being taken at that word, which adds none -- is the key of a small table
// of the passes already counted. A hit is one comparison of the 256 bytes of offsets; a miss counts as before
// (sass_bank_passes_table) and stores. The table is the host thread's (one per Block, which the thread keeps), so it
// needs no lock. A kernel whose shapes do not repeat (a gather, a data-dependent sort) only pays for the keys it
// builds, so when under a third of the last kWindow lookups hit the table is left alone for kSkip calls.
struct BankMemo {
  static constexpr unsigned kEntries = 32;
  static constexpr uint32_t kWindow = 64;
  static constexpr uint32_t kSkip = 1024;
  struct Entry {
    uint32_t bytes = 0;    // (zero: empty)
    uint32_t passes = 0;
    uint64_t rel[32] = {};
  };
  Entry e[kEntries];
  uint32_t lookups = 0, hits = 0;   // in this window
  uint32_t skip = 0;                // calls still to go without the table
};

// sass_bank_passes, with the table in front of the counting (everything the closed forms count is still counted by them).
inline uint32_t sass_bank_passes_memo(BankMemo& memo, const uint64_t* addrs, uint64_t mask, unsigned lanes, uint32_t bytes) {
  if (!fast() || bytes == 0 || bytes > 64 || lanes != 32) return sass_bank_passes(addrs, mask, lanes, bytes);
  const uint64_t m = mask & 0xFFFFFFFFull;
  if (!m) return sass_bank_passes(addrs, mask, lanes, bytes);
  uint32_t out;
  if (detail::bank_passes_progression<true>(addrs, m, bytes, out)) return out;
  if (memo.skip) {
    --memo.skip;
    return sass_bank_passes_table(addrs, mask, lanes, bytes);
  }
  // The key: each active lane's address less the first active lane's word, as the 64-bit number it is (a lane that does
  // not run is taken at that word). Moving every address by a multiple of four bytes changes no lane's words but their
  // numbers, so keys that are equal are shapes that differ by such a move -- as long as neither warp has an address
  // near the top of the space, where the move would wrap (those are counted, not remembered).
  const uint64_t base = addrs[std::countr_zero(m)] & ~3ull;
  uint64_t held[32];
  const uint64_t* src = addrs;
  if (m != 0xFFFFFFFFull) {
    for (unsigned l = 0; l < 32; ++l) held[l] = (m >> l & 1) ? addrs[l] : base;
    src = held;
  }
  uint32_t h = bytes * 0x01000193u;
  h ^= static_cast<uint32_t>(src[1] - base) * 0x9E3779B1u ^ static_cast<uint32_t>(src[3] - base) * 0x85EBCA77u ^
       static_cast<uint32_t>(src[6] - base) * 0xC2B2AE3Du ^ static_cast<uint32_t>(src[10] - base) * 0x27D4EB2Fu ^
       static_cast<uint32_t>(src[15] - base) * 0x165667B1u ^ static_cast<uint32_t>(src[21] - base) * 0x9E3779B1u ^
       static_cast<uint32_t>(src[26] - base) * 0x85EBCA77u ^ static_cast<uint32_t>(src[31] - base) * 0xC2B2AE3Du;
  h ^= h >> 15;
  BankMemo::Entry& e = memo.e[h % BankMemo::kEntries];
  const detail::V2 basev = {base, base};
  detail::V2 diff = {static_cast<uint64_t>(e.bytes ^ bytes), 0};
  detail::V2 high = basev;
  for (unsigned i = 0; i < 16; ++i) {
    const detail::V2 a = detail::load_v2(src + 2 * i);
    high |= a;
    diff |= (a - basev) ^ detail::load_v2(e.rel + 2 * i);
  }
  if ((high[0] | high[1]) >> 62) return sass_bank_passes_table(addrs, mask, lanes, bytes);
  const bool hit = (diff[0] | diff[1]) == 0;
  uint32_t passes;
  if (hit) {
    passes = e.passes;
  } else {
    passes = sass_bank_passes_table(addrs, mask, lanes, bytes);
    e.bytes = bytes;
    e.passes = passes;
    for (unsigned l = 0; l < 32; ++l) e.rel[l] = src[l] - base;
  }
  memo.hits += hit;
  if (++memo.lookups == BankMemo::kWindow) {
    if (memo.hits * 3 < BankMemo::kWindow) memo.skip = BankMemo::kSkip;
    memo.lookups = memo.hits = 0;
  }
  return passes;
}

}  // namespace vgpu::exec::memcount
