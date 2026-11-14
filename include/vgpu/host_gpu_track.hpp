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
#include <cstring>
#include <map>
#include <mutex>
#include <shared_mutex>
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

// The gate's lock: a BigReaderLock (the readers count themselves in slots of their own, so the simulator's threads do not share a cache line the way they did on a
// std::shared_mutex, sp45) with a generation that makes a settled access answerable without the lock at all. The generation is odd while a writer holds the lock and
// changes on every exclusive section (once at its start and once at its end), so a number a thread read while no writer held the lock is current exactly as long as
// no exclusive section has started since.
class GateLock {
 public:
  void lock_shared() { l_.lock_shared(); }
  void unlock_shared() { l_.unlock_shared(); }
  void lock() {
    const bool outer = !l_.written_by_me();
    l_.lock();
    if (outer) {
      depth_ = 0;
      gen_.fetch_add(1, std::memory_order_acq_rel);   // odd: a section is open
    } else {
      ++depth_;
    }
  }
  void unlock() {
    if (depth_ == 0) gen_.fetch_add(1, std::memory_order_release);   // (outermost: even again, and another number)
    else --depth_;
    l_.unlock();
  }
  uint64_t generation() const { return gen_.load(std::memory_order_acquire); }
  bool cache_enabled = true;   // false: every access takes the lock (the reference the unit test compares with)

 private:
  BigReaderLock l_;
  std::atomic<uint64_t> gen_{next_base()};
  int depth_ = 0;   // nested exclusive sections inside the outermost (touched by the owner only)
  static uint64_t next_base() {   // locks never share a number: a thread's memory of one lock cannot match another that took its address
    static std::atomic<uint64_t> n{0};
    return (n.fetch_add(1, std::memory_order_relaxed) + 1) << 24;
  }
};

// What one thread remembers of chunks it has found settled (the gate would change nothing): up to four address ranges, each one chunk, and whether a write to
// it is settled too (the chunk is the simulator's alone), valid while the lock's generation is the one they were found under. A settled chunk stays settled for a
// read until an exclusive section changes it (a read changes a chunk from the card's alone to shared, a write to the simulator's alone: both only toward settled)
// and every other change is made in an exclusive section, which changes the generation before and after.
struct GateMemory {
  const void* owner;
  uint64_t gen;
  uint64_t lo[4], hi[4];
  uint8_t write_ok[4];
  unsigned next;
};
inline thread_local GateMemory t_gate_memory;   // (zero-initialised: no construction on the access path)

// The gate's reaction to one simulator access of [addr, addr + len) (Runtime::Impl::before_access, kept here so that a unit test runs the real function): the
// chunks the card alone holds that the access needs come back (`pull(entry, chunks)` copies them), a write makes its chunks the simulator's, and `update(entry)`
// publishes the entry's new count of card-valid chunks.
//
// Most accesses change nothing: while a card copy is valid the simulator's threads reach here for EVERY access of a launch they run (a verify run, or a
// launch that runs next to card-resident data), and all but the first access to a chunk find it already settled. Those take the shared lock and look at the
// accessed chunks only, so the threads do not serialise on one mutex and the cost does not grow with the allocation (before sp43 each access took an exclusive lock
// and counted every chunk of the allocation twice: PolyBench 2DCONV with VGPU_HOST_GPU_VERIFY=1 and 128 MiB live ran for more than 8 CPU-minutes, 35% of it in the
// count and 43% in the kernel's side of the mutex, against 0.97 s simulated). Only an access that does change something retakes the lock exclusively and
// decides again, for another thread may have changed the chunks in between.
//
// sp45: a thread that found a chunk settled remembers it (GateMemory) and answers the next access to that chunk from the lock's generation alone: no lock, no
// lookup of the allocation. 6 s of the 7 s a verify run of 2DCONV took beyond the plain simulation were the lock, the lookup and the chunk test, per access.
template <class Pull, class Update>
inline void gate_access(GateLock& mu, TrackedMap& res, uint64_t addr, uint64_t len, bool write, Pull&& pull, Update&& update) {
  GateMemory& m = t_gate_memory;
  const uint64_t g0 = mu.generation();
  if (mu.cache_enabled && m.owner == &mu && m.gen == g0 && !(g0 & 1)) {
    for (unsigned i = 0; i < 4; ++i)
      if (addr >= m.lo[i] && addr < m.hi[i] && len <= m.hi[i] - addr && (!write || m.write_ok[i])) return;
  }
  {
    std::shared_lock<GateLock> g(mu);
    auto it = res.upper_bound(addr);
    if (it == res.begin()) return;
    const Tracked& a = std::prev(it)->second;
    if (a.slot->load(std::memory_order_relaxed) == 0) return;
    const ChunkSpan sp = chunk_span(a.base, a.size, addr, len);
    if (!sp.any) return;
    if (!a.r.access_changes_state(sp.first, sp.last, write)) {
      const uint64_t lo = a.base + sp.first * kChunkSize, hi = std::min<uint64_t>(lo + kChunkSize, a.base + a.size);
      if (mu.cache_enabled && sp.first == sp.last && addr >= lo && len <= hi - addr) {
        const uint64_t g = mu.generation();   // (no writer can hold the lock while this thread holds it shared: even, and stable)
        if (m.owner != &mu || m.gen != g) {
          m.owner = &mu;
          m.gen = g;
          m.next = 0;
          for (unsigned i = 0; i < 4; ++i) m.lo[i] = m.hi[i] = 0;
        }
        const unsigned i = m.next++ & 3;
        m.lo[i] = lo;
        m.hi[i] = hi;
        m.write_ok[i] = a.r.at(sp.first) == Where::Host;
      }
      return;
    }
  }
  std::lock_guard<GateLock> g(mu);
  auto it = res.upper_bound(addr);
  if (it == res.begin()) return;
  Tracked& a = std::prev(it)->second;
  if (a.slot->load(std::memory_order_relaxed) == 0) return;
  const ChunkSpan sp = chunk_span(a.base, a.size, addr, len);
  if (!sp.any) return;
  if (!write) {
    pull(a, a.r.sim_read(sp.first, sp.last));
  } else {
    for (uint64_t c = sp.first; c <= sp.last; ++c) {
      const bool whole = c == sp.first ? sp.first_whole : c == sp.last ? sp.last_whole : true;
      pull(a, a.r.sim_write(c, c, whole));
    }
  }
  update(a);
}

// ---- verify mode: the compare, and the chunks the card may keep (sp67) ----
// A verify run (VGPU_HOST_GPU_VERIFY) runs a launch on the card, copies every live range back (`card[i]` holds the card's bytes of `ranges[i]`), gives the card
// copy up, runs the launch in the simulator and compares. `compare_with_card` reads the simulator's memory one 64 KiB chunk at a time into `scratch` (a buffer
// that stays in the cache: one pass over each side, where a copy of every range into one more buffer the size of the range was three, and a fresh such buffer per
// launch also a page fault per page) and calls `same(i, addr, n)` or `differ(i, addr, card, sim, n)` for each chunk; `equal[i]` gets the chunk indices of
// ranges[i] whose bytes agree.
template <class Same, class Differ>
inline void compare_with_card(const MemoryManager& mem, const std::vector<std::pair<uint64_t, uint64_t>>& ranges, const std::vector<std::vector<uint8_t>>& card,
                              std::vector<uint8_t>& scratch, std::vector<std::vector<uint64_t>>& equal, Same&& same, Differ&& differ) {
  equal.resize(ranges.size());
  scratch.resize(kChunkSize);
  for (size_t i = 0; i < ranges.size(); ++i) {
    equal[i].clear();
    const auto [base, size] = ranges[i];
    for (uint64_t off = 0; off < size; off += kChunkSize) {
      const uint64_t n = std::min<uint64_t>(kChunkSize, size - off);
      mem.read(base + off, scratch.data(), n);
      if (std::memcmp(card[i].data() + off, scratch.data(), n) == 0) {
        equal[i].push_back(off / kChunkSize);
        same(i, base + off, n);
      } else {
        differ(i, base + off, card[i].data() + off, scratch.data(), n);
      }
    }
  }
}

// After the compare, a chunk whose card bytes equal the simulator's is a valid card copy again: the card holds exactly what the simulator holds there (the card's
// memory has not changed since it was copied back: no launch ran on it in between). Each such chunk that is the simulator's alone (Host: the copy was given up
// before the simulator ran) becomes Both, so the next launch on the card does not push it. The others stay Host and are pushed: the next launch on the card still
// starts from the simulator's memory everywhere, the rule of Round sp59. Before sp67 every chunk was pushed again before every launch of a verify run (PolyBench
// 3DCONV, 254 launches on 128 MiB: 40 s against 0.38 s for the plain simulator, with a stand-in card).
//
// The gate then guards these chunks as it guards any card copy: a write of the host makes the chunk Host again (it is pushed), a read changes nothing (Both), a
// free drops the entry, and the simulator's next launch gives the copy up first (Runtime::before_simulated), so no simulator launch ever runs with it. An entry
// whose address or size is not the range's is left alone. `update(entry)` publishes an entry's new count of card-valid chunks (Runtime::Impl::update_slot); an
// allocation that had no card-valid chunk and has one now is dropped from every thread's memory of recent lookups (MemoryManager::card_void_lookups), as after
// a launch on the card: a remembered lookup would let the simulator's fast paths reach it without the gate. Returns the number of chunks made Both.
template <class Update>
inline uint64_t keep_verified_chunks(TrackedMap& res, MemoryManager& mem, const std::vector<std::pair<uint64_t, uint64_t>>& ranges,
                                     const std::vector<std::vector<uint64_t>>& equal, Update&& update) {
  uint64_t kept = 0;
  std::vector<uint64_t> newly;
  for (size_t i = 0; i < ranges.size() && i < equal.size(); ++i) {
    const auto it = res.find(ranges[i].first);
    if (it == res.end() || it->second.size != ranges[i].second || equal[i].empty()) continue;
    Tracked& a = it->second;
    const bool was = a.r.card_valid() > 0;
    uint64_t here = 0;
    for (uint64_t c : equal[i])
      if (c < a.r.chunks() && a.r.at(c) == Where::Host) {
        a.r.set(c, Where::Both);
        ++here;
      }
    if (!here) continue;
    kept += here;
    update(a);
    if (!was) newly.push_back(a.base);
  }
  if (!newly.empty()) mem.card_void_lookups(newly);
  return kept;
}

}  // namespace vgpu::hostgpu
