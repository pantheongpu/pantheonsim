// A reader-writer lock for a table read on every simulated memory access and
// written almost never (an allocation, a free, a mapping).
//
// A kernel's every load and store looks its address up in the device's
// allocation table, from as many host threads as are running work-groups, and
// with asynchronous streams a host thread may allocate while they do. A
// std::shared_mutex would put every one of those lookups on the same cache
// line. Here each reader counts itself in a slot of its own thread's (a
// "big-reader" lock): a lookup is one uncontended atomic add and a check of
// the writer's flag. A writer raises the flag and waits for every slot to
// empty, so writing is slow, which is the right way round for this table.
//
// The writer may take the lock again, and may read, from its own thread: the
// mutators call one another. Readers must not nest -- a reader waiting on a
// writer that is waiting on the reader's own outer count would never wake.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <pthread.h>

namespace vgpu {

class BigReaderLock {
 public:
  void lock_shared() {
    const uintptr_t me = self();
    if (owner_.load(std::memory_order_relaxed) == me) return;
    const unsigned s = slot();
    std::atomic<int>& n = slots_[s].n;
    // A slot is announced to writers the first time a thread uses it, so that a writer looks at the
    // slots in use and not at all 64 of them. Announced before the count: a writer that read the
    // mask without this bit started before this reader counted itself, and sees the flag below.
    const uint64_t bit = uint64_t{1} << s;
    if (!(used_.load(std::memory_order_relaxed) & bit)) used_.fetch_or(bit, std::memory_order_seq_cst);
    for (;;) {
      n.fetch_add(1, std::memory_order_seq_cst);
      if (!writer_.load(std::memory_order_seq_cst)) return;
      n.fetch_sub(1, std::memory_order_seq_cst);
      while (writer_.load(std::memory_order_relaxed)) std::this_thread::yield();
    }
  }
  void unlock_shared() {
    if (owner_.load(std::memory_order_relaxed) == self()) return;
    slots_[slot()].n.fetch_sub(1, std::memory_order_release);
  }
  // Takes the lock for writing and changes what epoch() returns, now and again when it is let go:
  // every answer a reader remembered is void. For a change to the table that can make one wrong.
  void lock() { acquire(true); }
  // Takes the lock for writing and leaves epoch() alone: for a change that cannot make an answer a
  // reader remembered wrong. An allocation is one (a new entry in the table touches no other); a
  // free is one only if it says so with note_free() before it takes the allocation apart.
  void lock_quiet() { acquire(false); }
  void unlock() {
    if (--depth_) return;
    if (bumped_) {
      stamp_.store(next_epoch(), std::memory_order_release);
      bumped_ = false;
    }
    writer_.store(false, std::memory_order_release);
    owner_.store(0, std::memory_order_relaxed);
    mu_.unlock();
  }

  // A number that changes whenever a writer takes the lock (not a quiet one) and again when it
  // lets go, and when an allocation is freed. A reader that looked something up under the shared
  // lock may remember the answer together with the number it saw, and trust it for as long as
  // epoch() still returns that number, or still_valid() says the answer survived the frees since:
  // no writer has changed the table in a way that touches it. Read it, then use the answer; a
  // writer that starts in between is no worse than one that starts just after the lookup handed back
  // its reference to a table entry, which is the case already.
  //
  // The numbers are drawn from a pseudo-random sequence rather than counted. A
  // count would repeat: a lock freed and another made at the same address (and
  // every library of the shim carries its own copy of this class, with a
  // counter of its own) would hand a reader's remembered answer a number it
  // still believes. A 64-bit random one is not equal by accident. The low 16 bits are the count of
  // frees since the random part was drawn (note_free), so a thread that remembered an allocation
  // can ask whether it is one of the last few that were freed instead of starting again.
  uint64_t epoch() const { return stamp_.load(std::memory_order_acquire); }

  // Records that the allocation at `base` is about to be freed; the writer holds the lock (quiet or
  // not) and calls this before the allocation's memory is touched. A remembered answer for any other
  // allocation stays good: still_valid().
  void note_free(uint64_t base) {
    const uint64_t s = stamp_.load(std::memory_order_relaxed);
    if ((s & kCountMask) == kCountMask) {   // the count is used up: a new random part voids everything
      stamp_.store(next_epoch(), std::memory_order_release);
      return;
    }
    ring_[(s + 1) % kRing].store(base, std::memory_order_relaxed);
    stamp_.store(s + 1, std::memory_order_release);
  }
  // Whether an answer remembered under epoch `seen` for the allocation at `base` is still right now:
  // nothing but frees of other allocations has happened since. On true, *now is the epoch to remember
  // it under from here on.
  bool still_valid(uint64_t seen, uint64_t base, uint64_t* now) const {
    const uint64_t cur = stamp_.load(std::memory_order_acquire);
    if (((seen ^ cur) >> 16) != 0 || cur < seen || cur - seen > kRing) return false;
    for (uint64_t q = seen + 1; q <= cur; ++q)
      if (ring_[q % kRing].load(std::memory_order_relaxed) == base) return false;
    // A writer that freed kRing more while this looked has reused the slots read.
    if (stamp_.load(std::memory_order_acquire) - seen > kRing) return false;
    *now = cur;
    return true;
  }
  // Whether this thread holds the lock for writing. What it reads is a table
  // part way through a change, and not worth remembering by epoch.
  bool written_by_me() const { return owner_.load(std::memory_order_relaxed) == self(); }

 private:
  static constexpr uint64_t kCountMask = 0xFFFF;
  static constexpr uint64_t kRing = 64;   // divides 65536, so a count's slot is its low bits' remainder
  void acquire(bool bump) {
    if (written_by_me()) {
      ++depth_;
      if (bump && !bumped_) {   // a full lock inside a quiet one: void the answers before the change
        bumped_ = true;
        stamp_.store(next_epoch(), std::memory_order_release);
      }
      return;
    }
    mu_.lock();
    owner_.store(self(), std::memory_order_relaxed);
    depth_ = 1;
    writer_.store(true, std::memory_order_seq_cst);
    uint64_t m = used_.load(std::memory_order_seq_cst);
    while (m) {
      Slot& s = slots_[__builtin_ctzll(m)];
      while (s.n.load(std::memory_order_seq_cst)) std::this_thread::yield();
      m &= m - 1;
    }
    bumped_ = bump;
    if (bump) stamp_.store(next_epoch(), std::memory_order_release);
  }
  static uint64_t next_epoch() {
    static std::atomic<uint64_t> state{
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count()) ^
        reinterpret_cast<uintptr_t>(&state)};
    uint64_t z = state.fetch_add(0x9E3779B97F4A7C15ull, std::memory_order_relaxed) + 0x9E3779B97F4A7C15ull;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    z ^= z >> 31;
    z &= ~kCountMask;
    return z ? z : (kCountMask + 1);   // zero is what a reader's empty memory holds
  }
  static constexpr unsigned kSlots = 64;
  struct alignas(64) Slot {
    std::atomic<int> n{0};
  };
  // This thread's identity and slot, fixed for its life: threads that share a slot only share a
  // count, which is still right. An ordinary (general-dynamic) thread-local: this class is linked
  // into every shim library, and an initial-exec one would take static TLS from each library a
  // process loads late (the surplus is about 1.6 KB for all of them, shared with ROCm and CUDA).
  struct ThreadTag {
    uintptr_t self;
    unsigned slot;
  };
  static inline thread_local ThreadTag t_tag = {0, 0};
  [[gnu::noinline]] static ThreadTag& make_tag() {
    ThreadTag& t = t_tag;
    t.self = reinterpret_cast<uintptr_t>(pthread_self());
    t.slot = static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id())) % kSlots;
    return t;
  }
  static uintptr_t self() {
    const uintptr_t s = t_tag.self;
    return __builtin_expect(s != 0, 1) ? s : make_tag().self;
  }
  static unsigned slot() {
    self();
    return t_tag.slot;
  }

  Slot slots_[kSlots];
  alignas(64) std::atomic<bool> writer_{false};
  std::atomic<uint64_t> used_{0};      // the slots some thread has counted in
  std::atomic<uintptr_t> owner_{0};    // the writing thread's tag, or 0
  alignas(64) std::atomic<uint64_t> stamp_{next_epoch()};
  std::atomic<uint64_t> ring_[kRing] = {};   // the allocations freed last, by count
  int depth_ = 0;       // touched only by the owner
  bool bumped_ = false;   // likewise: whether this hold voided the answers, and must again at its end
  std::mutex mu_;
};

// Scope guards over a lock that may be absent (null): the table's lock lives
// behind a pointer so the manager stays movable.
class SharedGuard {
 public:
  explicit SharedGuard(BigReaderLock* l) : l_(l) {
    if (l_) l_->lock_shared();
  }
  ~SharedGuard() {
    if (l_) l_->unlock_shared();
  }
  SharedGuard(const SharedGuard&) = delete;
  SharedGuard& operator=(const SharedGuard&) = delete;

 private:
  BigReaderLock* l_;
};
class ExclusiveGuard {
 public:
  explicit ExclusiveGuard(BigReaderLock* l, bool quiet = false) : l_(l) {
    if (l_) quiet ? l_->lock_quiet() : l_->lock();
  }
  ~ExclusiveGuard() {
    if (l_) l_->unlock();
  }
  ExclusiveGuard(const ExclusiveGuard&) = delete;
  ExclusiveGuard& operator=(const ExclusiveGuard&) = delete;

 private:
  BigReaderLock* l_;
};

}  // namespace vgpu
