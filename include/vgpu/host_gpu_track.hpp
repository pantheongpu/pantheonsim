// Host-GPU mode: the table of allocations the mode tracks (src/runtime/host_gpu.cpp), kept apart from the driver so that it is unit tested on any machine
// (tests/unit/test_host_gpu_track.cpp).
//
// An entry is one allocation's residency (which 64 KiB chunks the simulator holds, the card holds, or both) and a pointer to the allocation's own count of chunks
// whose card copy is valid (MemoryManager::card_slot). The entry must not outlive its allocation: the pointer points into the memory manager's record of it,
// which a free recycles, and the launch path reads every entry's range (to push what the simulator wrote). `sync_tracked` is what makes the table the live
// allocations again before each launch that may use it, whatever happened to them since.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <map>
#include <utility>
#include <vector>

#include "vgpu/host_gpu.hpp"
#include "vgpu/memory.hpp"

namespace vgpu::hostgpu {

struct Tracked {
  uint64_t base = 0, size = 0;
  Residency r;
  std::atomic<uint32_t>* slot = nullptr;
};
using TrackedMap = std::map<uint64_t, Tracked>;

// Makes `res` hold exactly the allocations of `ranges` (the live ones, from MemoryManager::live_ranges): an entry whose allocation is gone, has another size
// now, or is another record of the memory manager under the same address is dropped (it is not the allocation it was made for: nothing of its card copy
// can be valid, because a free of an allocation with one goes through the gate, which drops the entry itself), and an allocation with no entry gets one with
// every chunk the simulator's. Returns the number of entries dropped.
//
// Why a sweep and not only the gate's drop on a free: a launch registers every live allocation even when the size rule then keeps it in the simulator, and
// the allocations of such a launch have no card copy, so a free of one used to skip the gate (it was asked only while some allocation held a copy). The
// entry stayed, with a pointer into a freed record, and the next launch that went to the card pushed the "host chunks" of the dead allocation and read
// freed memory (HeCBench hadamard-cuda: "illegal memory access" at the next synchronisation).
inline size_t sync_tracked(TrackedMap& res, const MemoryManager& mem, const std::vector<std::pair<uint64_t, uint64_t>>& ranges) {
  size_t dropped = 0;
  // (`ranges` is in address order, as live_ranges gives it: a search per entry, no table built per launch)
  const auto size_at = [&](uint64_t base, uint64_t* size) {
    const auto l = std::lower_bound(ranges.begin(), ranges.end(), base, [](const std::pair<uint64_t, uint64_t>& r, uint64_t b) { return r.first < b; });
    if (l == ranges.end() || l->first != base) return false;
    *size = l->second;
    return true;
  };
  for (auto it = res.begin(); it != res.end();) {
    uint64_t size = 0;
    if (!size_at(it->first, &size) || size != it->second.size || mem.card_slot(it->first) != it->second.slot) {
      it = res.erase(it);
      ++dropped;
    } else {
      ++it;
    }
  }
  for (const auto& [b, s] : ranges) {
    if (res.count(b)) continue;
    Tracked a;
    a.base = b;
    a.size = s;
    a.r = Residency((s + kChunkSize - 1) / kChunkSize);
    a.slot = mem.card_slot(b);
    if (!a.slot) continue;
    res[b] = std::move(a);
  }
  return dropped;
}

}  // namespace vgpu::hostgpu
