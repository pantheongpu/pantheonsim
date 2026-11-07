// The memory manager's side of host-GPU residency (vgpu::CardGate, docs/perf-design/09-host-gpu-spike.md section 4): an allocation whose card
// copy is valid is out of every thread's lookup memory and every access to it asks the gate first. No card and no CUDA: the gate is a fake that
// counts the questions. Each assertion is one way a stale card copy would go unseen -- a remembered allocation, a run access that skips the lookup,
// a free that the gate is not told of -- and each fails when its guard is removed.
#include <atomic>
#include <cstdio>
#include <functional>
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
  VCHECK(!gate.frees.empty() && gate.frees[0] == a);   // the allocation leaves the table for a file: the gate drops what it tracks of it
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
  VCHECK(gate.accesses.empty());
  mm.free(a);
  VCHECK(gate.accesses.empty());   // (a free is not an access)
}

// The gate tracks allocations that hold no card copy too (a launch the size rule refused registers every live allocation), so it is told of every free:
// an entry that outlives its allocation points into a freed record, and the next launch that goes to the card reads the freed range (HeCBench hadamard-cuda,
// "illegal memory access" in most runs). Before, a free asked the gate only while some allocation held a card copy.
VTEST(a_free_tells_the_gate_while_no_allocation_has_a_card_copy) {
  MemoryManager mm(64 << 20);
  FakeGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20), b = mm.alloc(1 << 20);
  VCHECK_EQ(mm.card_resident_count().load(), 0);
  mm.free(a);
  VCHECK_EQ(gate.frees.size(), size_t{1});
  VCHECK_EQ(gate.frees[0], a);
  mm.free_all();   // the reset, and the sweep of what is left (b)
  bool all = false, second = false;
  for (uint64_t f : gate.frees) {
    all = all || f == UINT64_MAX;
    second = second || f == b;
  }
  VCHECK(all && second);
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


// ---- every way into an allocation's bytes (the hook inventory of docs/perf-design/10-host-gpu-hooks.md) ----------------------------------------------
// One case per public MemoryManager entry that reads or writes device bytes. For an allocation whose card copy is valid, each must either decline
// (return false having done nothing: the caller then takes a path that asks) or ask the gate for the range it touches, with the right direction. A new
// entry point that forgets the gate is a silent stale read, so the table is the checklist; each row fails when its guard is removed.
namespace {
struct PathCase {
  const char* name;
  bool writes;                                   // the direction the gate must be told when the call goes ahead
  std::function<bool(MemoryManager&, uint64_t)> call;   // true: the call went ahead; false: it declined
};
}  // namespace

VTEST(every_memory_manager_entry_asks_the_gate_or_declines_for_a_card_valid_allocation) {
  alignas(16) uint64_t out[64] = {};
  uint64_t vals[32];
  for (auto& v : vals) v = 3;
  const uint8_t pat = 0x11;
  const std::vector<PathCase> cases = {
      {"write", true, [&](MemoryManager& m, uint64_t a) { m.write(a, vals, 64); return true; }},
      {"read", false, [&](MemoryManager& m, uint64_t a) { m.read(a, out, 64); return true; }},
      {"fill", true, [&](MemoryManager& m, uint64_t a) { m.fill(a, &pat, 1, 4096); return true; }},
      {"load_scalar", false, [&](MemoryManager& m, uint64_t a) { (void)m.load_scalar(a, 4); return true; }},
      {"store_scalar", true, [&](MemoryManager& m, uint64_t a) { m.store_scalar(a, 4, 1); return true; }},
      {"load_run", false, [&](MemoryManager& m, uint64_t a) { return m.load_run(a, out, 256, 4); }},
      {"store_run", true, [&](MemoryManager& m, uint64_t a) { return m.store_run(a, vals, 128, 4); }},
      {"load_batch", false, [&](MemoryManager& m, uint64_t a) {
         uint64_t ad[4] = {a, a + 4, a + 8, a + 12};
         return m.load_batch(ad, 4, 4, out);
       }},
      {"load_broadcast", false, [&](MemoryManager& m, uint64_t a) { return m.load_broadcast(a, 4, 4, out); }},
      {"store_batch", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t ad[4] = {a, a + 4, a + 8, a + 12};
         return m.store_batch(ad, 4, 4, vals);
       }},
      {"store_strided", true, [&](MemoryManager& m, uint64_t a) { return m.store_strided(a, 4, 4, 4, vals); }},
      {"batch_window", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t ad[4] = {a, a + 4, a + 8, a + 12};
         MemoryManager::BatchWindow w;
         return m.batch_window(ad, 4, 4, &w);
       }},
      {"atomic_word_ptrs", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t ad[2] = {a, a + 4};
         uint8_t* p[2];
         return m.atomic_word_ptrs(ad, 2, 4, p);
       }},
      {"rmw_batch", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t ad[2] = {a, a + 4}, olds[2];
         size_t done = 0;
         return m.rmw_batch(ad, 2, 4, [](size_t, uint64_t o) { return o + 1; }, olds, &done, /*lock_free=*/true);
       }},
      {"rmw_scalar", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t old;
         return m.rmw_scalar(a, 4, [](uint64_t o) { return o + 1; }, &old);
       }},
      {"atomic_rmw", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t old;
         return m.atomic_rmw(a, 4, [](uint64_t o) { return o + 1; }, &old);
       }},
      {"rmw_if_changed", true, [&](MemoryManager& m, uint64_t a) {
         uint64_t old;
         return m.rmw_if_changed(a, 4, [](uint64_t o) { return o + 1; }, &old);
       }},
      {"load_fields", false, [&](MemoryManager& m, uint64_t a) {
         const uint8_t sz[2] = {4, 4};
         return m.load_fields(a, sz, 2, out);
       }},
      {"load_texels", false, [&](MemoryManager& m, uint64_t a) {
         uint64_t ad[2] = {a, a + 16};
         return m.load_texels(ad, 2, 4, 1, out);
       }},
      {"load_quad", false, [&](MemoryManager& m, uint64_t a) { m.load_quad(a, out); return true; }},
      {"store_quad", true, [&](MemoryManager& m, uint64_t a) { m.store_quad(a, vals); return true; }},
      {"load_quad_run", false, [&](MemoryManager& m, uint64_t a) { return m.load_quad_run(a, 2, out); }},
      {"store_quad_run", true, [&](MemoryManager& m, uint64_t a) { return m.store_quad_run(a, 2, vals); }},
      {"copy_to (source)", false, [&](MemoryManager& m, uint64_t a) {
         MemoryManager other(64 << 20, 1);
         const uint64_t d = other.alloc(1 << 20);
         return m.copy_to(other, d, a, 4096);
       }},
      {"copy_to (destination)", true, [&](MemoryManager& m, uint64_t a) {
         MemoryManager other(64 << 20, 1);
         const uint64_t s = other.alloc(1 << 20);
         return other.copy_to(m, a, s, 4096);
       }},
  };
  int ran = 0, declined = 0;
  for (const auto& c : cases) {
    MemoryManager mm(64 << 20);
    FakeGate gate;
    mm.set_card_gate(&gate);
    const uint64_t a = mm.alloc(1 << 20);
    make_card_valid(mm, a, 16);
    gate.accesses.clear();
    const bool went = c.call(mm, a);
    if (!went) {
      ++declined;
      // (a fast path that declines hands the caller to the general path, which asks: the declines are why the lookup memory excludes a card-resident allocation)
      if (std::string(c.name) == "write" || std::string(c.name) == "read" || std::string(c.name) == "fill" || std::string(c.name) == "load_scalar" ||
          std::string(c.name) == "store_scalar") {
        std::fprintf(stderr, "entry %s declined: it has no other path\n", c.name);
        VCHECK(false);
      }
      continue;
    }
    ++ran;
    bool asked = false;
    for (const auto& x : gate.accesses)
      if (x.addr >= a && x.addr < a + (1 << 20) && (x.write == c.writes || x.write)) asked = true;
    if (!asked) std::fprintf(stderr, "entry %s went ahead on a card-valid allocation without asking the gate\n", c.name);
    VCHECK(asked);
    // The direction: a call that writes must say so (a read-direction ask for a write leaves the card copy stale).
    if (c.writes) {
      bool wrote = false;
      for (const auto& x : gate.accesses) wrote = wrote || x.write;
      if (!wrote) std::fprintf(stderr, "entry %s writes but told the gate it reads\n", c.name);
      VCHECK(wrote);
    }
  }
  std::fprintf(stderr, "card gate paths: %d asked the gate, %d declined\n", ran, declined);
  VCHECK(ran >= 5);   // the five general entries (the others decline) at least
}

// ---- device memory outside the allocation table is a reason to keep launches off the card ----------------------------------------------------------
// The card copy is the table of live allocations. A VMM mapping, an IPC region or a block of the kernel heap holds device bytes that are not in it, so a
// pointer to one, from an argument or from memory, would read what the card never got. card_blocker() is how the launch path knows.
VTEST(card_blocker_names_device_memory_outside_the_allocation_table) {
  MemoryManager mm(256 << 20);
  VCHECK(mm.card_blocker() == nullptr);
  const uint64_t a = mm.alloc(1 << 20);
  VCHECK(mm.card_blocker() == nullptr);   // ordinary allocations are the table
  // VMM: a mapping holds bytes of its own
  const uint64_t gran = MemoryManager::kVmmGranularity;
  const uint64_t va = mm.reserve(gran, gran);
  const uint64_t h = mm.create_handle(gran);
  mm.map(va, gran, 0, h);
  VCHECK(mm.card_blocker() != nullptr);
  mm.unmap(va, gran);
  mm.release_handle(h);
  mm.address_free(va, gran);
  VCHECK(mm.card_blocker() == nullptr);   // and the card may be used again once it is gone
  // IPC: exported memory leaves the table for a file
  const std::string path = "/tmp/vgpu-card-blocker-share-" + std::to_string(getpid());
  mm.share(a, path);
  VCHECK(mm.card_blocker() != nullptr);
  mm.free(a);
  ::unlink(path.c_str());
  VCHECK(mm.card_blocker() == nullptr);
  // the kernel heap
  const uint64_t p = mm.heap_alloc(256, 1 << 20);
  VCHECK(p != 0);
  VCHECK(mm.card_blocker() != nullptr);
  VCHECK(mm.heap_free(p));
  VCHECK(mm.card_blocker() == nullptr);
}

VTEST_MAIN
