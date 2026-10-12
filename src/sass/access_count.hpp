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
// 32-byte access covers 9).
constexpr unsigned kMaxWordsPerLane = 9;

// Number of distinct 32-byte sectors touched by `bytes` bytes at each address
// of the lanes in `lanes`. `bytes` is at most 32.
inline uint64_t distinct_sectors(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  uint64_t sec[32 * kMaxSectorsPerLane];
  unsigned n = 0;
  for (uint32_t m = lanes; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    // The same range the vector version walked, including its empty range when
    // the access wraps the address space.
    const uint64_t first = addr[l] / 32, last = (addr[l] + bytes - 1) / 32;
    for (uint64_t s = first; s <= last && s < first + kMaxSectorsPerLane; ++s) sec[n++] = s;
  }
  std::sort(sec, sec + n);
  return static_cast<uint64_t>(std::unique(sec, sec + n) - sec);
}

// Passes a shared-memory access takes through the 32 banks of 4 bytes: lanes
// on different words of one bank take a pass each, lanes on one word share.
// `bytes` is at most 32. Returns at least 1 for a non-empty access.
inline uint64_t bank_passes(const uint64_t* addr, uint32_t lanes, uint32_t bytes) {
  uint64_t words[32 * kMaxWordsPerLane];
  unsigned n = 0;
  for (uint32_t m = lanes; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    unsigned k = 0;
    for (uint64_t a = addr[l] & ~3ull; a < addr[l] + bytes && k < kMaxWordsPerLane; a += 4, ++k) words[n++] = a / 4;
  }
  std::sort(words, words + n);
  const uint64_t* end = std::unique(words, words + n);
  unsigned per_bank[32] = {};
  uint64_t passes = 1;
  for (const uint64_t* p = words; p != end; ++p) passes = std::max<uint64_t>(passes, ++per_bank[*p % 32]);
  return passes;
}

}  // namespace vgpu::sass
