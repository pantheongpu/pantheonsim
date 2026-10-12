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
#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <vector>

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

inline uint32_t distinct_sectors(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint64_t span, size_t cap) {
  // A cap of SIZE_MAX is the SASS executor's, which counts every sector.
  const auto reference = [&]() -> uint32_t {
    return cap == SIZE_MAX ? static_cast<uint32_t>(sass_sectors_reference(addrs, mask, lanes, static_cast<uint32_t>(span)))
                           : sectors_reference(addrs, mask, lanes, span, cap);
  };
  if (!fast()) return reference();
  constexpr size_t kBuf = 1024;
  uint64_t buf[kBuf];
  size_t n = 0;
  bool ascending = true;
  for (unsigned lane = 0; lane < lanes; ++lane) {
    if (!(mask & (uint64_t{1} << lane))) continue;
    const uint64_t first = addrs[lane] / 32;
    const uint64_t last = (addrs[lane] + (span ? span - 1 : 0)) / 32;
    if (last < first) continue;   // the address wrapped: the plain loop runs no iterations either
    if (last - first >= kBuf - n) return reference();
    for (uint64_t sec = first; sec <= last; ++sec) {
      if (n && buf[n - 1] > sec) ascending = false;
      buf[n++] = sec;
    }
  }
  // Coalesced accesses put the sectors in order already; a gather needs sorting.
  if (!ascending) std::sort(buf, buf + n);
  size_t distinct = n ? 1 : 0;
  for (size_t i = 1; i < n; ++i) distinct += buf[i] != buf[i - 1];
  // The reference stops remembering at `cap`, and a sector it did not keep is
  // never counted: the count saturates there.
  return static_cast<uint32_t>(std::min(distinct, cap));
}

namespace detail {
// Distinct words per bank. At most 64 lanes each put at most one word in a bank
// (PTX: up to 32 consecutive words) or two (SASS, 17 words) -- the callers keep
// to shapes where a bank cannot hold more than 64.
struct BankTable {
  uint32_t count[32] = {};
  uint64_t word[32][64];
  void add(uint64_t w) {
    const unsigned b = static_cast<unsigned>(w % 32);
    for (uint32_t i = 0; i < count[b]; ++i)
      if (word[b][i] == w) return;
    word[b][count[b]++] = w;
  }
  uint32_t worst() const {
    uint32_t m = 1;
    for (uint32_t c : count) m = c > m ? c : m;
    return m;
  }
};
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

inline uint32_t sass_bank_passes(const uint64_t* addrs, uint64_t mask, unsigned lanes, uint32_t bytes) {
  if (!fast() || bytes > 64 || lanes > 32) return sass_bank_passes_reference(addrs, mask, lanes, bytes);
  detail::BankTable t;
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

}  // namespace vgpu::exec::memcount
