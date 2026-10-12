// The two numbers a warp-level memory access is counted by, from its lanes'
// addresses: the distinct 32-byte sectors it touches and the shared-memory
// bank passes it needs. Both used to build std::vectors on every access (one
// for the sectors, thirty-two for the banks); here they work in fixed arrays on
// the stack. The answers are the same by construction -- each function gathers
// the same values the vectors held and counts the same distinct ones --
// and tests/unit/test_sass_access_count.cpp holds the vector version beside
// these and compares them over random and awkward access patterns.
#pragma once

#include <algorithm>
#include <bit>
#include <cstdint>

namespace vgpu::sass {

// Most sectors one access of at most 32 bytes can cover per lane is 2; room
// for 3 keeps an unaligned 33..64-byte access (none exists today) correct too.
constexpr unsigned kMaxSectorsPerLane = 3;
// Most 4-byte words one lane's access of at most 32 bytes covers (an unaligned
// 32-byte access covers 9, always fewer than the 32 banks).
constexpr unsigned kMaxWordsPerLane = 9;

// Number of distinct 32-byte sectors touched by `bytes` bytes at each address
// of the lanes in `lanes`. `bytes` is at most 32.
//
// The distinct sectors are collected as they come, each tested against the ones
// already seen: a warp's lanes mostly land in the same few sectors (a coalesced
// access touches 4 to 8), so the scans are short, and sorting every access is
// not needed to count.
inline uint64_t distinct_sectors_scan(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  uint64_t seen[32 * kMaxSectorsPerLane];
  unsigned n = 0;
  uint64_t prev = ~uint64_t{0};   // the sector last added or matched: neighbouring lanes usually share it
  bool have_prev = false;
  for (uint32_t m = lanes; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    // The same range the vector version walked, including its empty range when
    // the access wraps the address space.
    const uint64_t first = addr[l] / 32, last = (addr[l] + bytes - 1) / 32;
    for (uint64_t s = first; s <= last && s < first + kMaxSectorsPerLane; ++s) {
      if (have_prev && s == prev) continue;
      bool found = false;
      for (unsigned i = n; i-- > 0;)
        if (seen[i] == s) {
          found = true;
          break;
        }
      if (!found) seen[n++] = s;
      prev = s;
      have_prev = true;
    }
  }
  return n;
}

// The same count by a bitmap when the sectors lie within 256 of each other (8 KiB -- every coalesced
// access, and most strided ones): one pass finds the lowest sector, a second sets a bit for each
// sector touched, and the answer is the bits set. No data-dependent branches, so no mispredictions
// on a scattered pattern. Wider spreads take the scan above.
inline uint64_t distinct_sectors(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  uint64_t lo = ~uint64_t{0}, hi = 0;
  for (uint32_t m = lanes; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t first = addr[l] / 32, last = (addr[l] + bytes - 1) / 32;
    if (last < first) continue;   // the access wraps the address space: the scan counts nothing for it
    lo = std::min(lo, first);
    hi = std::max(hi, std::min(last, first + (kMaxSectorsPerLane - 1)));
  }
  if (lo > hi) return 0;
  if (hi - lo >= 256) return distinct_sectors_scan(addr, lanes, bytes);
  uint64_t bits[4] = {};
  for (uint32_t m = lanes; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t first = addr[l] / 32, last = (addr[l] + bytes - 1) / 32;
    if (last < first) continue;
    const uint64_t end = std::min(last, first + (kMaxSectorsPerLane - 1));
    for (uint64_t s = first; s <= end; ++s) {
      const uint64_t rel = s - lo;
      bits[rel >> 6] |= uint64_t{1} << (rel & 63);
    }
  }
  uint64_t n = 0;
  for (uint64_t b : bits) {
    b = b - ((b >> 1) & 0x5555555555555555ull);
    b = (b & 0x3333333333333333ull) + ((b >> 2) & 0x3333333333333333ull);
    b = (b + (b >> 4)) & 0x0f0f0f0f0f0f0f0full;
    n += (b * 0x0101010101010101ull) >> 56;
  }
  return n;
}

// Passes a shared-memory access takes through the 32 banks of 4 bytes: lanes
// on different words of one bank take a pass each, lanes on one word share.
// `bytes` is at most 32. Returns at least 1 for a non-empty access.
//
// Each bank keeps the distinct words seen in it (a lane's words are
// consecutive and fewer than 32, so a lane adds at most one per bank, 32 in all
// per bank); the answer is the largest list. A conflict-free access leaves one
// word in each bank, so the scans are almost always empty or a single compare.
inline uint64_t bank_passes(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  uint64_t seen[32][32];
  unsigned count[32] = {};
  uint64_t passes = 1;
  for (uint32_t m = lanes; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    unsigned k = 0;
    for (uint64_t a = addr[l] & ~3ull; a < addr[l] + bytes && k < kMaxWordsPerLane; a += 4, ++k) {
      const uint64_t word = a / 4;
      const unsigned bank = static_cast<unsigned>(word % 32);
      bool found = false;
      for (unsigned i = 0; i < count[bank]; ++i)
        if (seen[bank][i] == word) {
          found = true;
          break;
        }
      if (!found) {
        seen[bank][count[bank]++] = word;
        passes = std::max<uint64_t>(passes, count[bank]);
      }
    }
  }
  return passes;
}

}  // namespace vgpu::sass
