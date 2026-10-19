// A HIP stream's work, run in order on a host thread of its own.
//
// Work on one stream runs one item after another; work on different streams
// runs at once, as it does on a card -- which is what lets a kernel on one
// stream wait on a flag another stream's kernel sets (RCCL's collectives are
// built on that), and what makes hipStreamQuery and hipEventQuery mean
// something. An item may also wait for an item on another queue to finish
// (a marker): that is how an event, hipStreamWaitEvent and the legacy null
// stream's ordering against the blocking streams are kept.
//
// A queue's thread starts with its first item. A failure is kept until the
// next synchronization asks for it, which is when a HIP program hears of an
// asynchronous error.
#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "vgpu/host_cpus.hpp"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace vgpu::amd {

// How long a thread that waits for the other side of a queue (the caller for a result, the queue's thread for work)
// spins before it sleeps: most hand-offs are answered in a few microseconds, and a sleep and a wake-up cost more
// than that on their own. In loop turns of a pause (some tens of nanoseconds); 0 -- VGPU_QUEUE_SPIN=0, or a process
// that may use one CPU only, where a spinning thread keeps the one it waits for off the processor -- never spins.
inline unsigned queue_spin_turns() {
  static const unsigned n = [] {
    if (vgpu::host_cpus() < 2) return 0u;
    const char* v = std::getenv("VGPU_QUEUE_SPIN");
    return v && *v ? static_cast<unsigned>(std::atoi(v)) : 400u;
  }();
  return n;
}
inline void queue_relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

template <typename Error>
class WorkQueue {
 public:
  // An item on some queue: done once the queue has finished its seq'th item.
  struct Marker {
    std::shared_ptr<WorkQueue> queue;
    uint64_t seq = 0;
  };
  using Work = std::function<Error()>;

  explicit WorkQueue(Error ok, Error failed) : ok_(ok), failed_(failed), error_(ok) {}
  ~WorkQueue() {
    stopping_.store(true, std::memory_order_release);   // (a thread that is spinning stops at once)
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    work_cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;

  // Queues `work` behind everything already here and behind each marker.
  // Returns its number, which a marker names it by.
  uint64_t submit(Work work, std::vector<Marker> after = {}) {
    uint64_t seq;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (!thread_.joinable()) thread_ = std::thread([this] { run(); });
      seq = ++submitted_;
      items_.push_back(Item{seq, std::move(work), std::move(after)});
      pushed_.store(++pushed_count_, std::memory_order_release);
    }
    work_cv_.notify_one();
    return seq;
  }
  // Runs `work` on the calling thread as the next item, if that changes nothing a thread could see but where it
  // runs: nothing is queued or running, and every marker in `after` is done. The result is the work's own (an
  // exception is the queue's failure, as it is for its thread). Says whether it ran; if not, nothing was done and
  // the work is still the caller's to submit. A caller that waits for the item anyway saves two thread hand-offs.
  bool run_inline(Work& work, const std::vector<Marker>& after, Error* result) {
    for (const Marker& m : after)
      if (m.queue.get() != this && !m.queue->done(m.seq)) return false;
    uint64_t seq;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (completed_ != submitted_ || inline_) return false;
      seq = ++submitted_;
      inline_ = true;
    }
    Error e = ok_;
    try {
      e = work();
    } catch (...) {
      e = failed_;
    }
    {
      std::lock_guard<std::mutex> lock(mu_);
      completed_ = seq;
      completed_hint_.store(seq, std::memory_order_release);
      inline_ = false;
      if (e != ok_ && error_ == ok_) error_ = e;
    }
    // (an item submitted meanwhile by another thread waits for this one, on its thread)
    work_cv_.notify_one();
    done_cv_.notify_all();
    *result = e;
    return true;
  }
  // Blocks until the seq'th item has run.
  void wait(uint64_t seq) {
    for (unsigned t = queue_spin_turns(); t && completed_hint_.load(std::memory_order_acquire) < seq; --t) queue_relax();
    std::unique_lock<std::mutex> lock(mu_);
    done_cv_.wait(lock, [&] { return completed_ >= seq; });
  }
  void drain() { wait(tail()); }
  // Waits for the seq'th item until `deadline`; says whether it ran.
  bool wait_until(uint64_t seq, std::chrono::steady_clock::time_point deadline) {
    std::unique_lock<std::mutex> lock(mu_);
    return done_cv_.wait_until(lock, deadline, [&] { return completed_ >= seq; });
  }
  // Whether the calling thread is this queue's own (a host function that
  // calls exit, say).
  bool on_worker() const {
    std::lock_guard<std::mutex> lock(mu_);
    return thread_.joinable() && thread_.get_id() == std::this_thread::get_id();
  }
  bool done(uint64_t seq) const {
    if (completed_hint_.load(std::memory_order_acquire) >= seq) return true;
    std::lock_guard<std::mutex> lock(mu_);
    return completed_ >= seq;
  }
  bool idle() const {
    std::lock_guard<std::mutex> lock(mu_);
    return completed_ == submitted_;
  }
  uint64_t tail() const {
    std::lock_guard<std::mutex> lock(mu_);
    return submitted_;
  }
  // The first failure since this was last asked, which it then forgets.
  Error take_error() {
    std::lock_guard<std::mutex> lock(mu_);
    return std::exchange(error_, ok_);
  }

 private:
  struct Item {
    uint64_t seq;
    Work work;
    std::vector<Marker> after;
  };
  void run() {
    uint64_t taken = 0;   // items this thread has taken so far (pushed_ counts the same way)
    for (;;) {
      Item item;
      for (unsigned t = queue_spin_turns();
           t && pushed_.load(std::memory_order_acquire) == taken && !stopping_.load(std::memory_order_relaxed); --t)
        queue_relax();
      {
        std::unique_lock<std::mutex> lock(mu_);
        work_cv_.wait(lock, [&] { return stop_ || (!items_.empty() && !inline_); });
        if (items_.empty()) return;   // stopping, and nothing left
        item = std::move(items_.front());
        items_.pop_front();
        ++taken;
      }
      for (const Marker& m : item.after)
        if (m.queue.get() != this) m.queue->wait(m.seq);
      Error e = ok_;
      try {
        e = item.work();
      } catch (...) {
        e = failed_;
      }
      {
        std::lock_guard<std::mutex> lock(mu_);
        completed_ = item.seq;
        completed_hint_.store(item.seq, std::memory_order_release);
        if (e != ok_ && error_ == ok_) error_ = e;
      }
      done_cv_.notify_all();
    }
  }

  const Error ok_, failed_;
  mutable std::mutex mu_;
  std::condition_variable work_cv_;   // for the thread: an item, or stop
  std::condition_variable done_cv_;   // for the waiters: an item finished
  std::deque<Item> items_;
  uint64_t submitted_ = 0, completed_ = 0;
  // Copies that a thread about to sleep can spin on without the lock, only ever a reason to look at the real ones:
  // how many items were queued for the thread (not the inline ones), and completed_.
  uint64_t pushed_count_ = 0;
  std::atomic<uint64_t> pushed_{0}, completed_hint_{0};
  std::atomic<bool> stopping_{false};
  bool inline_ = false;   // an item is running on a caller's thread (run_inline)
  Error error_;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace vgpu::amd

