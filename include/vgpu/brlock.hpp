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
#include <functional>
#include <mutex>
#include <thread>

namespace vgpu {

class BigReaderLock {
 public:
  void lock_shared() {
    if (owned_by_me()) return;
    std::atomic<int>& n = slots_[slot()].n;
    for (;;) {
      n.fetch_add(1, std::memory_order_seq_cst);
      if (!writer_.load(std::memory_order_seq_cst)) return;
      n.fetch_sub(1, std::memory_order_seq_cst);
      while (writer_.load(std::memory_order_relaxed)) std::this_thread::yield();
    }
  }
  void unlock_shared() {
    if (owned_by_me()) return;
    slots_[slot()].n.fetch_sub(1, std::memory_order_release);
  }
  void lock() {
    if (owned_by_me()) {
      ++depth_;
      return;
    }
    mu_.lock();
    owner_.store(std::this_thread::get_id(), std::memory_order_relaxed);
    depth_ = 1;
    writer_.store(true, std::memory_order_seq_cst);
    for (Slot& s : slots_)
      while (s.n.load(std::memory_order_seq_cst)) std::this_thread::yield();
  }
  void unlock() {
    if (--depth_) return;
    writer_.store(false, std::memory_order_release);
    owner_.store(std::thread::id(), std::memory_order_relaxed);
    mu_.unlock();
  }

 private:
  static constexpr unsigned kSlots = 64;
  struct alignas(64) Slot {
    std::atomic<int> n{0};
  };
  // Each thread's slot, fixed for its life: threads that share one only
  // share a count, which is still right.
  static unsigned slot() {
    static thread_local const unsigned mine =
        static_cast<unsigned>(std::hash<std::thread::id>{}(std::this_thread::get_id())) % kSlots;
    return mine;
  }
  bool owned_by_me() const { return owner_.load(std::memory_order_relaxed) == std::this_thread::get_id(); }

  Slot slots_[kSlots];
  std::atomic<bool> writer_{false};
  std::atomic<std::thread::id> owner_{};
  int depth_ = 0;   // touched only by the owner
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
  explicit ExclusiveGuard(BigReaderLock* l) : l_(l) {
    if (l_) l_->lock();
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
