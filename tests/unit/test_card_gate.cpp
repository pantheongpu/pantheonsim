// The memory manager's side of host-GPU residency (vgpu::CardGate, docs/perf-design/09-host-gpu-spike.md section 4): an allocation whose card
// copy is valid is out of every thread's lookup memory and every access to it asks the gate first. No card and no CUDA: the gate is a fake that
// counts the questions. Each assertion is one way a stale card copy would go unseen -- a remembered allocation, a run access that skips the lookup,
// a free that the gate is not told of -- and each fails when its guard is removed.
#include <atomic>
#include <cstdint>
#include <unistd.h>

#include <string>
#include <vector>

#include "vgpu/memory.hpp"
#include "vtest.hpp"

using vgpu::CardGate;
using vgpu::MemoryManager;

namespace {
struct FakeGate : CardGate {
  struct Call {
    uint64_t addr, len;
    bool write;
  };
  std::vector<Call> accesses;
  std::vector<uint64_t> frees;
  void before_access(uint64_t addr, uint64_t len, bool write) override { accesses.push_back({addr, len, write}); }
  void before_free(uint64_t base) override { frees.push_back(base); }
};

// What the runtime does when an allocation's card copy becomes valid.
void make_card_valid(MemoryManager& mm, uint64_t base, uint32_t chunks) {
  mm.card_slot(base)->store(chunks);
  mm.card_resident_count().fetch_add(1);
  mm.card_void_lookups({base});
}
}  // namespace

VTEST(a_remembered_allocation_is_forgotten_when_its_card_copy_becomes_valid) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20);
  mm.store_scalar(a, 4, 7);
  for (int i = 0; i < 4; ++i) VCHECK_EQ(mm.load_scalar(a, 4), uint64_t{7});   // the lookup is remembered now: the gate is not asked
  VCHECK(gate.accesses.empty());
  make_card_valid(mm, a, 16);
  VCHECK_EQ(mm.load_scalar(a, 4), uint64_t{7});
  VCHECK_EQ(gate.accesses.size(), size_t{1});   // the remembered answer was voided: this access took resolve()
  VCHECK_EQ(mm.load_scalar(a + 64, 4), uint64_t{0});
  VCHECK_EQ(gate.accesses.size(), size_t{2});   // and the allocation is not remembered again while its card copy is valid
  mm.store_scalar(a + 128, 4, 9);
  VCHECK_EQ(gate.accesses.size(), size_t{3});
  VCHECK(gate.accesses.back().write);
}

VTEST(run_accesses_decline_an_allocation_with_a_card_copy) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20);
  std::vector<uint32_t> in(64, 5), out(64, 0);
  VCHECK(mm.store_run(a, in.data(), 256, 4));   // a kernel's run access takes the lookup directly, as it always did
  VCHECK(mm.load_run(a, out.data(), 256, 4));
  make_card_valid(mm, a, 16);
  VCHECK(!mm.store_run(a, in.data(), 256, 4));  // declined: the caller goes the general way, which asks the gate
  VCHECK(!mm.load_run(a, out.data(), 256, 4));
}

VTEST(host_copies_ask_the_gate_with_their_range_and_direction) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20);
  std::vector<uint8_t> buf(4096, 3);
  make_card_valid(mm, a, 16);
  mm.write(a + 100, buf.data(), 4096);
  mm.read(a + 8192, buf.data(), 512);
  const uint8_t pat = 0x5a;
  mm.fill(a + 70000, &pat, 1, 1000);
  VCHECK_EQ(gate.accesses.size(), size_t{3});
  VCHECK(gate.accesses[0].write && gate.accesses[0].addr == a + 100 && gate.accesses[0].len == 4096);
  VCHECK(!gate.accesses[1].write && gate.accesses[1].addr == a + 8192 && gate.accesses[1].len == 512);
  VCHECK(gate.accesses[2].write && gate.accesses[2].addr == a + 70000 && gate.accesses[2].len == 1000);
}

VTEST(a_free_tells_the_gate_before_the_allocation_goes) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20), b = mm.alloc(1 << 20);
  make_card_valid(mm, a, 16);
  mm.free(a);
  VCHECK_EQ(gate.frees.size(), size_t{1});
  VCHECK_EQ(gate.frees[0], a);
  mm.free_all();
  VCHECK(gate.frees.size() >= 2);
  bool all = false;
  for (uint64_t f : gate.frees) all = all || f == UINT64_MAX;
  VCHECK(all);   // a reset tells the gate to drop everything
  (void)b;
}

VTEST(exporting_an_allocation_brings_the_card_copy_back_first) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20);
  make_card_valid(mm, a, 16);
  const std::string path = "/tmp/vgpu-card-gate-share-" + std::to_string(getpid());
  mm.share(a, path);   // the exported file is the allocation from here on (cudaIpcGetMemHandle)
  VCHECK(!gate.accesses.empty());
  VCHECK(gate.accesses[0].addr == a && !gate.accesses[0].write);
  mm.free(a);
  ::unlink(path.c_str());
}

VTEST(nothing_is_asked_while_no_allocation_has_a_card_copy) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);   // a gate is set, but the count of resident allocations is zero: the access paths do not look at it
  const uint64_t a = mm.alloc(1 << 20);
  std::vector<uint8_t> buf(100, 1);
  mm.write(a, buf.data(), 100);
  mm.read(a, buf.data(), 100);
  mm.store_scalar(a, 4, 1);
  mm.free(a);
  VCHECK(gate.accesses.empty() && gate.frees.empty());
}

VTEST(a_zero_count_makes_the_allocation_ordinary_again) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20);
  make_card_valid(mm, a, 16);
  VCHECK_EQ(mm.load_scalar(a, 4), uint64_t{0});
  const size_t asked = gate.accesses.size();
  mm.card_slot(a)->store(0);   // the gate brought everything back
  mm.card_resident_count().fetch_sub(1);
  for (int i = 0; i < 3; ++i) VCHECK_EQ(mm.load_scalar(a, 4), uint64_t{0});
  VCHECK_EQ(gate.accesses.size(), asked);   // remembered again, and the gate is not asked
  VCHECK(mm.card_slot(a + 1) == nullptr);   // only a base is a slot
}

VTEST_MAIN
