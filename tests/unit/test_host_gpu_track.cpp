// The table of allocations the host-GPU mode tracks (vgpu/host_gpu_track.hpp), with no card and no CUDA. HeCBench hadamard-cuda failed with "illegal memory access"
// in most runs of the persistent mode: a launch the size rule refused had registered the allocations of the program (the card path starts from every live
// one), the program freed them while none held a card copy (a free asked the gate only then), and the entries stayed, pointing into freed records. The next
// launch that did go to the card pushed the "host chunks" of those entries and read freed memory. Each test is one way an entry outlives its allocation.
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <random>
#include <shared_mutex>
#include <thread>
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

// ---- the gate's access path (gate_access): settled accesses take the shared lock and change nothing; the rest take the exclusive lock (sp43) ----
// PolyBench 2DCONV with VGPU_HOST_GPU_VERIFY=1 and 128 MiB live ran for more than 8 CPU-minutes: every simulator access took one exclusive mutex and counted
// the chunks of the allocation twice (Round sp42 finding 6).
namespace {
constexpr uint64_t kChunks128MiB = (128ull << 20) / kChunkSize;

struct GateWorld {
  std::atomic<uint32_t> slot{0};
  GateLock mu;   // (the production type: Runtime::Impl::mu)
  TrackedMap res;
  std::atomic<uint64_t> pulled{0}, updates{0};
  explicit GateWorld(uint64_t chunks = kChunks128MiB, Where state = Where::Card) {
    Tracked t;
    t.base = 0x1000000;
    t.size = chunks * kChunkSize;
    t.r = Residency(chunks);
    for (uint64_t c = 0; c < chunks; ++c) t.r.set(c, state);
    t.slot = &slot;
    slot.store(static_cast<uint32_t>(t.r.card_valid()));
    res[t.base] = std::move(t);
  }
  void access(uint64_t chunk, bool write, uint64_t len = 8) {
    gate_access(mu, res, 0x1000000 + chunk * kChunkSize, len, write,
                [&](Tracked&, const std::vector<uint64_t>& idx) { pulled += idx.size(); },
                [&](Tracked& a) { ++updates; a.slot->store(static_cast<uint32_t>(a.r.card_valid())); });
  }
  const Residency& r() { return res.begin()->second.r; }
};

// What the gate did before: one exclusive mutex, and every chunk counted twice after every access.
struct LegacyGate {
  std::atomic<uint32_t> slot{1};
  std::mutex mu;
  Residency r{kChunks128MiB};
  LegacyGate() { r.card_launched(); }
  void access(uint64_t chunk, bool write) {
    std::lock_guard<std::mutex> g(mu);
    if (write) r.sim_write(chunk, chunk, true);
    else r.sim_read(chunk, chunk);
    uint64_t card = 0, both = 0;
    for (uint64_t c = 0; c < r.chunks(); ++c) {
      card += r.at(c) == Where::Card;
      both += r.at(c) == Where::Both;
    }
    slot.store(static_cast<uint32_t>(card + both));
  }
};

template <class F>
double run_threads(unsigned threads, F&& body) {
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<std::thread> th;
  for (unsigned t = 0; t < threads; ++t) th.emplace_back([&body, t] { body(t); });
  for (auto& x : th) x.join();
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}
}  // namespace

VTEST(a_settled_access_changes_nothing_and_asks_for_nothing) {
  GateWorld w(8, Where::Both);
  for (uint64_t c = 0; c < 8; ++c) w.access(c, false);   // reads of Both chunks
  VCHECK_EQ(w.updates.load(), uint64_t{0});
  VCHECK_EQ(w.pulled.load(), uint64_t{0});
  GateWorld h(8, Where::Host);
  h.slot.store(1);   // (as if another chunk were valid on the card, so the gate is consulted)
  for (uint64_t c = 0; c < 8; ++c) { h.access(c, false); h.access(c, true); }   // reads and writes of the simulator's own chunks
  VCHECK_EQ(h.updates.load(), uint64_t{0});
  VCHECK_EQ(h.pulled.load(), uint64_t{0});
}

VTEST(a_read_of_a_card_chunk_pulls_it_once_and_makes_it_shared) {
  GateWorld w(8, Where::Card);
  w.access(3, false);
  VCHECK_EQ(w.pulled.load(), uint64_t{1});
  VCHECK_EQ(w.updates.load(), uint64_t{1});
  VCHECK(w.r().at(3) == Where::Both);
  VCHECK_EQ(w.slot.load(), uint32_t{8});   // still valid on the card
  w.access(3, false);                      // settled now
  VCHECK_EQ(w.pulled.load(), uint64_t{1});
  VCHECK_EQ(w.updates.load(), uint64_t{1});
}

VTEST(a_write_makes_the_chunk_the_simulators_and_drops_its_card_copy) {
  GateWorld w(8, Where::Both);
  w.access(5, true);
  VCHECK(w.r().at(5) == Where::Host);
  VCHECK_EQ(w.slot.load(), uint32_t{7});
  VCHECK_EQ(w.pulled.load(), uint64_t{0});   // a Both chunk has nothing to pull
  GateWorld c(8, Where::Card);
  c.access(2, true, 8);                      // a partial write to a chunk only the card has: its bytes come first
  VCHECK_EQ(c.pulled.load(), uint64_t{1});
  VCHECK(c.r().at(2) == Where::Host);
  c.access(4, true, kChunkSize);             // a whole-chunk write: nothing to read first
  VCHECK_EQ(c.pulled.load(), uint64_t{1});
  VCHECK(c.r().at(4) == Where::Host);
  VCHECK_EQ(c.slot.load(), uint32_t{6});
}

VTEST(an_access_outside_every_entry_or_to_an_entry_without_a_card_copy_is_ignored) {
  GateWorld w(8, Where::Card);
  gate_access(w.mu, w.res, 0x10, 8, false, [&](Tracked&, const std::vector<uint64_t>&) { w.pulled++; }, [&](Tracked&) { w.updates++; });   // below every entry
  gate_access(w.mu, w.res, 0x1000000 + 8 * kChunkSize, 8, false, [&](Tracked&, const std::vector<uint64_t>&) { w.pulled++; }, [&](Tracked&) { w.updates++; });   // just past
  w.slot.store(0);
  w.access(1, false);
  VCHECK_EQ(w.pulled.load(), uint64_t{0});
  VCHECK_EQ(w.updates.load(), uint64_t{0});
}

VTEST(threads_reading_card_chunks_pull_each_chunk_exactly_once) {   // a verify run's first pass over a card-valid allocation, four simulator threads
  GateWorld w(512, Where::Card);
  run_threads(4, [&](unsigned t) {
    std::mt19937 rng(t + 1);
    for (uint64_t i = 0; i < 20000; ++i) w.access(rng() % 512, false);
    for (uint64_t c = 0; c < 512; ++c) w.access(c, false);   // (every chunk was asked for by every thread)
  });
  VCHECK_EQ(w.pulled.load(), uint64_t{512});
  VCHECK_EQ(w.r().count(Where::Card), uint64_t{0});
  VCHECK_EQ(w.r().count(Where::Both), uint64_t{512});
  VCHECK_EQ(w.slot.load(), uint32_t{512});
}

VTEST(threads_mixing_reads_and_writes_leave_counts_and_the_slot_consistent) {
  GateWorld w(256, Where::Card);
  run_threads(4, [&](unsigned t) {
    std::mt19937 rng(100 + t);
    for (uint64_t i = 0; i < 40000; ++i) w.access(rng() % 256, rng() % 4 == 0, rng() % 2 ? 8 : kChunkSize);
  });
  const Residency& r = w.r();
  uint64_t n[3] = {0, 0, 0};
  for (uint64_t c = 0; c < r.chunks(); ++c) ++n[static_cast<unsigned>(r.at(c))];
  VCHECK_EQ(r.count(Where::Host), n[0]);
  VCHECK_EQ(r.count(Where::Card), n[1]);
  VCHECK_EQ(r.count(Where::Both), n[2]);
  // The slot is the count of card-valid chunks after the last change of any chunk (a change always ends in `update`).
  VCHECK_EQ(w.slot.load(), static_cast<uint32_t>(n[1] + n[2]));
}

VTEST(the_gate_with_128_mib_live_is_much_cheaper_than_the_exclusive_scan_it_replaced) {
  // Four threads read an allocation of 128 MiB (2,048 chunks), all of it valid on the card, the way a verify run's simulator launch does.
  constexpr uint64_t kAccesses = 40000;
  GateWorld w;
  const double now = run_threads(4, [&](unsigned t) {
    std::mt19937 rng(t + 1);
    for (uint64_t i = 0; i < kAccesses; ++i) w.access(rng() % kChunks128MiB, false);
  });
  LegacyGate old;
  const double before = run_threads(4, [&](unsigned t) {
    std::mt19937 rng(t + 1);
    for (uint64_t i = 0; i < kAccesses; ++i) old.access(rng() % kChunks128MiB, false);
  });
  std::printf("gate, 4 threads x %llu reads of 128 MiB: %.3f s now, %.3f s before (%.0fx)\n", static_cast<unsigned long long>(kAccesses), now, before, before / now);
  VCHECK(now * 5 < before);
}

// ---- sp45: what a thread remembers of settled chunks (GateMemory) must never change what the gate does ----
// Two worlds take the same random sequence of accesses and of exclusive sections that put chunks back on the card (what a launch on the card does between the
// simulator's launches): one with the memory on, one with every access taking the lock. Pulls, updates and every chunk state must agree after every step.
VTEST(a_thread_that_remembers_settled_chunks_does_what_the_locking_gate_does) {
  for (unsigned seed = 1; seed <= 20; ++seed) {
    GateWorld fast(64, Where::Card), slow(64, Where::Card);
    slow.mu.cache_enabled = false;
    std::mt19937 rng(seed);
    for (unsigned step = 0; step < 4000; ++step) {
      const unsigned op = rng() % 40;
      if (op == 0) {   // a launch on the card: every chunk is the card's again (an exclusive section, as in Runtime::Impl)
        for (GateWorld* w : {&fast, &slow}) {
          std::lock_guard<GateLock> g(w->mu);
          w->res.begin()->second.r.card_launched();
          w->res.begin()->second.slot->store(64);
        }
      } else if (op == 1) {   // the simulator's launch begins: the chunks the card alone holds come back (exclusive too)
        for (GateWorld* w : {&fast, &slow}) {
          std::lock_guard<GateLock> g(w->mu);
          w->pulled += w->res.begin()->second.r.sim_launch().size();
        }
      } else {
        const uint64_t chunk = rng() % 64;
        const bool write = rng() % 3 == 0;
        const uint64_t len = rng() % 4 == 0 ? kChunkSize : 8;
        fast.access(chunk, write, len);
        slow.access(chunk, write, len);
      }
      VCHECK_EQ(fast.pulled.load(), slow.pulled.load());
      VCHECK_EQ(fast.updates.load(), slow.updates.load());
      VCHECK_EQ(fast.slot.load(), slow.slot.load());
      for (uint64_t c = 0; c < 64; ++c) VCHECK(fast.r().at(c) == slow.r().at(c));
    }
  }
}

VTEST(the_generation_changes_with_every_exclusive_section_and_is_odd_inside_one) {
  GateLock a, b;
  VCHECK(a.generation() != b.generation());   // two locks never share a number
  const uint64_t g0 = a.generation();
  VCHECK((g0 & 1) == 0);
  a.lock();
  VCHECK((a.generation() & 1) == 1);
  a.lock();
  a.unlock();
  VCHECK((a.generation() & 1) == 1);   // the outer section is still open
  a.unlock();
  VCHECK((a.generation() & 1) == 0);
  VCHECK(a.generation() != g0);
}

// ---- sp45: the scalar access path while a card gate is in play asks the gate on EVERY access to a card-valid allocation, remembered or not ----
namespace {
struct CountingGate : CardGate {
  std::atomic<uint64_t> accesses{0}, writes{0};
  void before_access(uint64_t, uint64_t, bool write) override { ++accesses; if (write) ++writes; }
  void before_free(uint64_t) override {}
};
}  // namespace

VTEST(a_scalar_access_to_a_card_valid_allocation_asks_the_gate_every_time_and_reads_the_right_bytes) {
  MemoryManager mm(256 << 20);
  CountingGate gate;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(1 << 20), b = mm.alloc(1 << 20), plain = mm.alloc(1 << 20);
  // (another thread fills the memory, before any card copy: no gate; this thread's memory of recent lookups stays empty, as it is when a copy is made by a launch)
  std::thread([&] {
    for (uint64_t base : {a, b, plain})
      for (uint64_t i = 0; i < 64; ++i) mm.store_scalar(base + i * 4, 4, (base / 4096 + i) & 0xffffffffu);
  }).join();
  VCHECK_EQ(gate.accesses.load(), uint64_t{0});
  for (uint64_t base : {a, b}) {   // two allocations hold a card copy
    mm.card_slot(base)->store(1);
    mm.card_resident_count().fetch_add(1);
  }
  uint64_t expected = 0;
  for (unsigned round = 0; round < 50; ++round)
    for (uint64_t i = 0; i < 64; ++i) {
      VCHECK_EQ(mm.load_scalar(a + i * 4, 4), (a / 4096 + i) & 0xffffffffu);   // alternating allocations: a remembered answer for the wrong one must not be used
      VCHECK_EQ(mm.load_scalar(b + i * 4, 4), (b / 4096 + i) & 0xffffffffu);
      VCHECK_EQ(mm.load_scalar(plain + i * 4, 4), (plain / 4096 + i) & 0xffffffffu);
      expected += 2;
    }
  VCHECK_EQ(gate.accesses.load(), expected);   // one per access to a card-valid allocation, none for the other one
  for (uint64_t base : {a, b})   // runs of accesses to one allocation, the case a thread's memory of the last one answers
    for (unsigned i = 0; i < 200; ++i) {
      VCHECK_EQ(mm.load_scalar(base + (i % 64) * 4, 4), (base / 4096 + i % 64) & 0xffffffffu);
      ++expected;
    }
  VCHECK_EQ(gate.accesses.load(), expected);
  mm.store_scalar(a + 8, 4, 77);
  VCHECK_EQ(gate.writes.load(), uint64_t{1});
  VCHECK_EQ(mm.load_scalar(a + 8, 4), uint64_t{77});
  // The card copy goes away: the allocation is an ordinary one again and the gate is not asked.
  mm.card_slot(a)->store(0);
  mm.card_resident_count().fetch_sub(1);
  const uint64_t before = gate.accesses.load();
  for (unsigned i = 0; i < 100; ++i) VCHECK_EQ(mm.load_scalar(a + 8, 4), uint64_t{77});
  VCHECK_EQ(gate.accesses.load(), before);
  // A free voids what a thread remembers of the allocation: the address is an error now, not the old record.
  VCHECK_EQ(mm.load_scalar(b + 8, 4), (b / 4096 + 2) & 0xffffffffu);   // (b is the allocation this thread remembers)
  mm.free(b);
  bool threw = false;
  try { mm.load_scalar(b + 8, 4); } catch (const Error&) { threw = true; }
  VCHECK(threw);
}

// ---- sp67: verify mode compares chunk by chunk and gives the card back the chunks that agree (compare_with_card, keep_verified_chunks) ----
namespace {
std::vector<uint8_t> pattern(uint64_t n, uint8_t seed) {
  std::vector<uint8_t> v(n);
  for (uint64_t i = 0; i < n; ++i) v[i] = static_cast<uint8_t>(seed + i * 131 + (i >> 9));
  return v;
}

// The runtime's update_slot: the entry's count of card-valid chunks, and the memory manager's count of allocations that have one.
void publish(Tracked& a, MemoryManager& mm) {
  const uint32_t v = static_cast<uint32_t>(a.r.card_valid());
  const uint32_t old = a.slot->exchange(v);
  if (old == 0 && v > 0) mm.card_resident_count().fetch_add(1);
  if (old > 0 && v == 0) mm.card_resident_count().fetch_sub(1);
}

// The runtime's gate, as far as accesses go: gate_access on the table, counting the accesses it is asked about.
struct TrackGate : CardGate {
  GateLock mu;
  TrackedMap* res = nullptr;
  MemoryManager* mm = nullptr;
  std::atomic<uint64_t> asked{0}, pulled{0};
  void before_access(uint64_t addr, uint64_t len, bool write) override {
    ++asked;
    gate_access(mu, *res, addr, len, write, [&](Tracked&, const std::vector<uint64_t>& idx) { pulled += idx.size(); }, [&](Tracked& a) { publish(a, *mm); });
  }
  void before_free(uint64_t base) override {
    if (base == UINT64_MAX) res->clear();
    else res->erase(base);
  }
};
}  // namespace

VTEST(compare_with_card_lists_the_chunks_that_agree_and_hands_over_the_others_with_their_bytes) {
  MemoryManager mm(64 << 20);
  const uint64_t sa = 3 * kChunkSize + 100, sb = kChunkSize;   // (a's last chunk is 100 bytes)
  const uint64_t a = mm.alloc(sa), b = mm.alloc(sb);
  const auto pa = pattern(sa, 7), pb = pattern(sb, 9);
  mm.write(a, pa.data(), sa);
  mm.write(b, pb.data(), sb);
  const std::vector<std::pair<uint64_t, uint64_t>> ranges = {{a, sa}, {b, sb}};
  std::vector<std::vector<uint8_t>> card = {pa, pb};
  card[0][kChunkSize + 5] ^= 1;   // chunk 1 of a
  card[0][sa - 1] ^= 0x80;        // the last byte of a's short last chunk
  std::vector<uint8_t> scratch;
  std::vector<std::vector<uint64_t>> equal;
  std::vector<std::pair<uint64_t, uint64_t>> same, differ;
  bool bytes_right = true;
  compare_with_card(
      mm, ranges, card, scratch, equal, [&](size_t, uint64_t addr, uint64_t n) { same.emplace_back(addr, n); },
      [&](size_t i, uint64_t addr, const uint8_t* c, const uint8_t* s, uint64_t n) {
        differ.emplace_back(addr, n);
        const std::vector<uint8_t>& want = i == 0 ? pa : pb;
        bytes_right = bytes_right && std::memcmp(s, want.data() + (addr - ranges[i].first), n) == 0 && c == card[i].data() + (addr - ranges[i].first);
      });
  VCHECK(equal.size() == 2);
  VCHECK(equal[0] == (std::vector<uint64_t>{0, 2}));
  VCHECK(equal[1] == (std::vector<uint64_t>{0}));
  VCHECK(same == (std::vector<std::pair<uint64_t, uint64_t>>{{a, kChunkSize}, {a + 2 * kChunkSize, kChunkSize}, {b, kChunkSize}}));
  VCHECK(differ == (std::vector<std::pair<uint64_t, uint64_t>>{{a + kChunkSize, kChunkSize}, {a + 3 * kChunkSize, 100}}));
  VCHECK(bytes_right);   // the simulator's bytes of the chunk, and the card's own
  // A second compare with the same scratch and lists starts clean.
  card[0] = pa;
  compare_with_card(mm, ranges, card, scratch, equal, [](size_t, uint64_t, uint64_t) {}, [](size_t, uint64_t, const uint8_t*, const uint8_t*, uint64_t) {});
  VCHECK(equal[0] == (std::vector<uint64_t>{0, 1, 2, 3}));
}

VTEST(keep_verified_chunks_gives_the_card_only_the_simulators_chunks_that_agreed) {
  MemoryManager mm(64 << 20);
  TrackedMap res;
  const uint64_t a = mm.alloc(8 * kChunkSize), b = mm.alloc(2 * kChunkSize);
  sync_tracked(res, mm, mm.live_ranges());
  res.at(a).r.set(5, Where::Card);   // (not the simulator's: the compare does not make it anything else)
  publish(res.at(a), mm);
  unsigned updates = 0;
  const auto update = [&](Tracked& t) { ++updates; publish(t, mm); };
  const std::vector<std::pair<uint64_t, uint64_t>> ranges = {{a, 8 * kChunkSize}, {b, 2 * kChunkSize}};
  const std::vector<std::vector<uint64_t>> equal = {{0, 1, 5, 7}, {}};
  VCHECK_EQ(keep_verified_chunks(res, mm, ranges, equal, update), uint64_t{3});
  const Residency& r = res.at(a).r;
  VCHECK(r.at(0) == Where::Both && r.at(1) == Where::Both && r.at(7) == Where::Both);
  VCHECK(r.at(2) == Where::Host && r.at(3) == Where::Host && r.at(4) == Where::Host && r.at(6) == Where::Host);
  VCHECK(r.at(5) == Where::Card);
  VCHECK(r.host_chunks() == (std::vector<uint64_t>{2, 3, 4, 6}));   // what the next launch on the card pushes
  VCHECK_EQ(updates, 1u);
  VCHECK_EQ(res.at(a).slot->load(), uint32_t{4});
  VCHECK_EQ(res.at(b).r.card_valid(), uint64_t{0});   // nothing agreed: untouched
  VCHECK_EQ(keep_verified_chunks(res, mm, ranges, equal, update), uint64_t{0});   // already Both: nothing changes, nothing is published
  VCHECK_EQ(updates, 1u);
  // A range that is not the entry's allocation (another size at the address) is left alone.
  const std::vector<std::pair<uint64_t, uint64_t>> other = {{b, 3 * kChunkSize}};
  VCHECK_EQ(keep_verified_chunks(res, mm, other, std::vector<std::vector<uint64_t>>{{0, 1}}, update), uint64_t{0});
  VCHECK_EQ(res.at(b).r.card_valid(), uint64_t{0});
}

VTEST(chunks_kept_after_a_verify_are_guarded_by_the_gate_like_any_card_copy) {
  MemoryManager mm(64 << 20);
  TrackedMap res;
  TrackGate gate;
  gate.res = &res;
  gate.mm = &mm;
  mm.set_card_gate(&gate);
  const uint64_t a = mm.alloc(4 * kChunkSize);
  for (uint64_t i = 0; i < 4 * kChunkSize; i += 4) mm.store_scalar(a + i, 4, i);
  sync_tracked(res, mm, mm.live_ranges());
  // This thread remembers the allocation (its lookup memory): the simulator's launch ran on it with no card copy, through the fast paths.
  VCHECK_EQ(mm.load_scalar(a + 16, 4), uint64_t{16});
  VCHECK_EQ(gate.asked.load(), uint64_t{0});
  // The compare: every chunk agreed. The allocation has a card copy now, and what this thread remembered of it must not let an access past the gate.
  {
    std::lock_guard<GateLock> g(gate.mu);   // (the runtime holds the gate's lock exclusively in note_simulated)
    VCHECK_EQ(keep_verified_chunks(res, mm, mm.live_ranges(), std::vector<std::vector<uint64_t>>{{0, 1, 2, 3}}, [&](Tracked& t) { publish(t, mm); }), uint64_t{4});
  }
  VCHECK_EQ(mm.card_resident_count().load(), 1);
  VCHECK_EQ(mm.load_scalar(a + 16, 4), uint64_t{16});
  VCHECK(gate.asked.load() >= 1);   // asked: the lookups were voided
  VCHECK_EQ(gate.pulled.load(), uint64_t{0});   // a read of a chunk both sides hold pulls nothing
  VCHECK(res.at(a).r.at(0) == Where::Both);
  // A host write to chunk 2 makes it the simulator's again: the next launch on the card pushes it, and only it.
  const uint32_t v = 0xdeadbeef;
  mm.write(a + 2 * kChunkSize + 8, &v, 4);
  VCHECK(res.at(a).r.host_chunks() == (std::vector<uint64_t>{2}));
  VCHECK_EQ(res.at(a).slot->load(), uint32_t{3});
  // A free drops the entry and the count.
  mm.free(a);
  VCHECK(res.count(a) == 0);
}

VTEST_MAIN
