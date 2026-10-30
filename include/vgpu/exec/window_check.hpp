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

// A local-memory offset is inside the lane's frame when the access's last byte is.
struct LocalWindowCheck {
  uint64_t highest = 0;
  void add(uint64_t offset) { highest = std::max(highest, offset); }
  bool inside(uint64_t frame_size, uint32_t bytes, bool any_lane) const {
    return !any_lane || highest + bytes <= frame_size;
  }
};

}  // namespace vgpu::exec
