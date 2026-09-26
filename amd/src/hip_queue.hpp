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

namespace vgpu::amd {

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
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    cv_.notify_all();
    if (thread_.joinable()) thread_.join();
  }
  WorkQueue(const WorkQueue&) = delete;
  WorkQueue& operator=(const WorkQueue&) = delete;

  // Queues `work` behind everything already here and behind each marker.
  // Returns its number, which a marker names it by.
  uint64_t submit(Work work, std::vector<Marker> after = {}) {
    std::lock_guard<std::mutex> lock(mu_);
    if (!thread_.joinable()) thread_ = std::thread([this] { run(); });
    items_.push_back(Item{++submitted_, std::move(work), std::move(after)});
    cv_.notify_all();
    return submitted_;
  }
  // Blocks until the seq'th item has run.
  void wait(uint64_t seq) {
    std::unique_lock<std::mutex> lock(mu_);
    cv_.wait(lock, [&] { return completed_ >= seq; });
  }
  void drain() { wait(tail()); }
  bool done(uint64_t seq) const {
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
    for (;;) {
      Item item;
      {
        std::unique_lock<std::mutex> lock(mu_);
        cv_.wait(lock, [&] { return stop_ || !items_.empty(); });
        if (items_.empty()) return;   // stopping, and nothing left
        item = std::move(items_.front());
        items_.pop_front();
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
        if (e != ok_ && error_ == ok_) error_ = e;
      }
      cv_.notify_all();
    }
  }

  const Error ok_, failed_;
  mutable std::mutex mu_;
  std::condition_variable cv_;
  std::deque<Item> items_;
  uint64_t submitted_ = 0, completed_ = 0;
  Error error_;
  bool stop_ = false;
  std::thread thread_;
};

}  // namespace vgpu::amd
