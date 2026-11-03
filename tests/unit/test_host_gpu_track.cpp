// The table of allocations the host-GPU mode tracks (vgpu/host_gpu_track.hpp), with no card and no CUDA. HeCBench hadamard-cuda failed with "illegal memory access"
// in most runs of the persistent mode: a launch the size rule refused had registered the allocations of the program (the card path starts from every live
// one), the program freed them while none held a card copy (a free asked the gate only then), and the entries stayed, pointing into freed records. The next
// launch that did go to the card pushed the "host chunks" of those entries and read freed memory. Each test is one way an entry outlives its allocation.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/host_gpu_track.hpp"
#include "vgpu/memory.hpp"
#include "vtest.hpp"

using namespace vgpu;
using namespace vgpu::hostgpu;

namespace {
// The runtime's gate, as far as a free goes (host_gpu.cpp Impl::before_free): the entry is dropped and the allocation's count goes to zero.
struct DroppingGate : CardGate {
  TrackedMap* res = nullptr;
  std::vector<uint64_t> frees;
  void before_access(uint64_t, uint64_t, bool) override {}
  void before_free(uint64_t base) override {
    frees.push_back(base);
    if (base == UINT64_MAX) { res->clear(); return; }
    res->erase(base);
  }
};

// What the launch path does with every tracked range before a launch goes to the card: reads the simulator's bytes of the chunks it has to push.
void push_loop_reads(MemoryManager& mm, const TrackedMap& res) {
  std::vector<uint8_t> buf;
  for (const auto& [b, a] : res)
    for (uint64_t c : a.r.host_chunks()) {
      const uint64_t off = c * kChunkSize, n = std::min<uint64_t>(kChunkSize, a.size - off);
      buf.resize(n);
      mm.read(a.base + off, buf.data(), n);
    }
}

void check_table_is_the_live_allocations(const MemoryManager& mm, const TrackedMap& res) {
  const auto live = mm.live_ranges();
  VCHECK_EQ(res.size(), live.size());
  for (const auto& [b, s] : live) {
    const auto it = res.find(b);
    VCHECK(it != res.end());
    VCHECK_EQ(it->second.size, s);
    VCHECK(it->second.slot == mm.card_slot(b));
  }
}
}  // namespace

VTEST(a_new_allocation_starts_with_every_chunk_on_the_simulator) {
  MemoryManager mm(64 << 20);
  TrackedMap res;
  const uint64_t a = mm.alloc(3 * kChunkSize + 100);
  VCHECK_EQ(sync_tracked(res, mm, mm.live_ranges()), size_t{0});
  check_table_is_the_live_allocations(mm, res);
  VCHECK_EQ(res.at(a).r.chunks(), uint64_t{4});
  VCHECK_EQ(res.at(a).r.count(Where::Host), uint64_t{4});
  VCHECK_EQ(res.at(a).r.card_valid(), uint64_t{0});
}

VTEST(entries_of_freed_allocations_are_dropped_when_no_gate_was_told) {
  MemoryManager mm(256 << 20);   // (no gate: the free is not announced to the table, as before the memory manager told the gate of every free)
  TrackedMap res;
  const uint64_t a = mm.alloc(1 << 20), b = mm.alloc(2 << 20);
  VCHECK_EQ(sync_tracked(res, mm, mm.live_ranges()), size_t{0});   // a launch that the size rule then refused
  VCHECK_EQ(res.size(), size_t{2});
  mm.free(a);
  mm.free(b);
  const uint64_t c = mm.alloc(3 << 20), d = mm.alloc(64 << 10);
  // Without the sweep the two entries stay, and the launch's pushes read freed (or reused) memory.
  VCHECK_EQ(sync_tracked(res, mm, mm.live_ranges()), size_t{2});
  check_table_is_the_live_allocations(mm, res);
  VCHECK(res.count(c) && res.count(d));
  push_loop_reads(mm, res);   // every range of the table is readable: nothing throws
}

VTEST(a_stale_entry_reads_freed_memory_in_the_push_loop) {
  // The hazard itself, so that the test above is known to be the one that matters: a table that was not swept holds a range the memory manager refuses.
  MemoryManager mm(64 << 20);
  TrackedMap res;
  const uint64_t a = mm.alloc(1 << 20);
  sync_tracked(res, mm, mm.live_ranges());
  const uint64_t keep = mm.alloc(1 << 20);
  mm.free(a);
  const auto err = VCAPTURE(Error, push_loop_reads(mm, res));
  VCHECK(err.code() == Err::UseAfterFree || err.code() == Err::InvalidPointer || err.code() == Err::OutOfBounds);
  (void)keep;
}

VTEST(an_entry_whose_allocation_record_changed_is_not_trusted) {
  MemoryManager mm(64 << 20);
  TrackedMap res;
  const uint64_t a = mm.alloc(1 << 20);
  sync_tracked(res, mm, mm.live_ranges());
  std::atomic<uint32_t> other{0};
  res.at(a).slot = &other;   // the memory manager's record under this address is not the one the entry points into (a recycled one)
  VCHECK_EQ(sync_tracked(res, mm, mm.live_ranges()), size_t{1});
  VCHECK(res.at(a).slot == mm.card_slot(a));
  check_table_is_the_live_allocations(mm, res);
}

VTEST(an_entry_of_another_size_is_replaced_and_starts_on_the_simulator) {
  MemoryManager mm(64 << 20);
  TrackedMap res;
  const uint64_t a = mm.alloc(1 << 20);
  sync_tracked(res, mm, mm.live_ranges());
  res.at(a).r.card_launched();   // a card copy of the old allocation
  res.at(a).size = 2 << 20;      // (as if the entry had been made for a bigger one at this address)
  VCHECK_EQ(sync_tracked(res, mm, mm.live_ranges()), size_t{1});
  VCHECK_EQ(res.at(a).size, uint64_t{1 << 20});
  VCHECK_EQ(res.at(a).r.card_valid(), uint64_t{0});
}

VTEST(a_free_the_memory_manager_announces_keeps_the_table_exact) {
  // Both halves together, as the runtime has them: the gate drops the entry on the free, the sweep finds nothing left to do.
  MemoryManager mm(256 << 20);
  TrackedMap res;
  DroppingGate gate;
  gate.res = &res;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20), b = mm.alloc(1 << 20);
  sync_tracked(res, mm, mm.live_ranges());
  VCHECK_EQ(mm.card_resident_count().load(), 0);   // nothing holds a card copy: the case the free used to skip the gate in
  mm.free(a);
  VCHECK(res.count(a) == 0);
  VCHECK_EQ(gate.frees.size(), size_t{1});
  mm.free(b);
  VCHECK(res.empty());
  const uint64_t c = mm.alloc(1 << 20);
  VCHECK_EQ(sync_tracked(res, mm, mm.live_ranges()), size_t{0});
  check_table_is_the_live_allocations(mm, res);
  mm.free_all();   // a reset drops everything
  VCHECK(res.empty());
  (void)c;
}

VTEST(exporting_an_allocation_drops_its_entry) {
  MemoryManager mm(64 << 20);
  TrackedMap res;
  DroppingGate gate;
  gate.res = &res;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20);
  sync_tracked(res, mm, mm.live_ranges());
  const std::string path = "/tmp/vgpu-host-gpu-track-share-" + std::to_string(getpid());
  mm.share(a, path);   // the allocation leaves the memory manager's table for a file
  VCHECK(res.count(a) == 0);
  mm.free(a);
  ::unlink(path.c_str());
}

VTEST_MAIN
