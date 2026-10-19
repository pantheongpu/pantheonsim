// A stream's queue (amd/src/hip_queue.hpp): work in order on a thread of its own, and a call that waits for its own
// work running it on the calling thread when the stream has nothing in it (run_inline), with the same order, the same
// numbering and the same failures as a queued item.
#include <atomic>
#include <chrono>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "../../src/hip_queue.hpp"
#include "vtest.hpp"

using Queue = vgpu::amd::WorkQueue<int>;
using namespace std::chrono_literals;

namespace {
constexpr int kOk = 0, kFailed = 99;

// Waits until `flag` is set, up to `limit`; says whether it was.
bool until(const std::atomic<bool>& flag, std::chrono::milliseconds limit = 20s) {
  const auto end = std::chrono::steady_clock::now() + limit;
  while (!flag.load()) {
    if (std::chrono::steady_clock::now() > end) return false;
    std::this_thread::sleep_for(100us);
  }
  return true;
}
}  // namespace

// Queued items run in the order they were queued, on one thread that is not the caller's, and wait() returns after
// its item.
VTEST(queued_items_run_in_order_on_the_queues_thread) {
  Queue q(kOk, kFailed);
  std::mutex mu;
  std::vector<int> order;
  std::vector<std::thread::id> threads;
  uint64_t last = 0;
  for (int i = 0; i < 200; ++i)
    last = q.submit([&, i] {
      std::lock_guard<std::mutex> lock(mu);
      order.push_back(i);
      threads.push_back(std::this_thread::get_id());
      return kOk;
    });
  q.wait(last);
  VCHECK_EQ(order.size(), size_t{200});
  for (int i = 0; i < 200; ++i) VCHECK_EQ(order[i], i);
  for (const auto& t : threads) VCHECK(t == threads[0] && t != std::this_thread::get_id());
}

// On an idle queue the item runs on the calling thread, and is numbered, finished and counted as a queued one is.
VTEST(an_idle_queue_runs_a_waited_for_item_on_the_calling_thread) {
  Queue q(kOk, kFailed);
  std::thread::id ran_on;
  int result = -1;
  Queue::Work w = [&] {
    ran_on = std::this_thread::get_id();
    return 7;
  };
  VCHECK(q.run_inline(w, {}, &result));
  VCHECK(ran_on == std::this_thread::get_id());
  VCHECK_EQ(result, 7);
  VCHECK(q.idle());
  VCHECK_EQ(q.tail(), uint64_t{1});
  VCHECK(q.done(1));
  // its failure is the queue's, kept until asked for, as a queued item's is
  VCHECK_EQ(q.take_error(), 7);
  VCHECK_EQ(q.take_error(), kOk);
  // the next item is the next number
  const uint64_t seq = q.submit([] { return kOk; });
  VCHECK_EQ(seq, uint64_t{2});
  q.wait(seq);
  VCHECK(q.done(2));
}

// An exception in inline work is the queue's failure, not the caller's.
VTEST(inline_work_that_throws_is_a_failure_of_the_queue) {
  Queue q(kOk, kFailed);
  int result = kOk;
  Queue::Work w = []() -> int { throw std::runtime_error("no"); };
  VCHECK(q.run_inline(w, {}, &result));
  VCHECK_EQ(result, kFailed);
  VCHECK_EQ(q.take_error(), kFailed);
  VCHECK(q.idle());
}

// Not on a queue that has an item in it (running or waiting), nor after a marker that is not done: then nothing is
// run and the work is still the caller's.
VTEST(a_busy_queue_or_a_pending_marker_refuses_inline_work) {
  Queue q(kOk, kFailed);
  std::atomic<bool> started{false}, release{false};
  const uint64_t seq = q.submit([&] {
    started = true;
    while (!release.load()) std::this_thread::sleep_for(100us);
    return kOk;
  });
  VCHECK(until(started));
  bool ran = false;
  Queue::Work w = [&] {
    ran = true;
    return kOk;
  };
  int result = -1;
  VCHECK(!q.run_inline(w, {}, &result));
  VCHECK(!ran);
  VCHECK(w != nullptr);   // (not consumed)
  release = true;
  q.wait(seq);

  // a marker on another queue that has not finished
  auto other = std::make_shared<Queue>(kOk, kFailed);
  std::atomic<bool> other_started{false}, other_release{false};
  const uint64_t other_seq = other->submit([&] {
    other_started = true;
    while (!other_release.load()) std::this_thread::sleep_for(100us);
    return kOk;
  });
  VCHECK(until(other_started));
  VCHECK(!q.run_inline(w, {Queue::Marker{other, other_seq}}, &result));
  VCHECK(!ran);
  other_release = true;
  other->wait(other_seq);
  VCHECK(q.run_inline(w, {Queue::Marker{other, other_seq}}, &result));   // done now
  VCHECK(ran);
}

// While inline work runs, an item another thread queues waits for it: the order is the order of the numbers.
VTEST(an_item_queued_during_inline_work_runs_after_it) {
  Queue q(kOk, kFailed);
  std::atomic<bool> in_work{false}, release{false}, inline_done{false};
  std::mutex mu;
  std::vector<std::string> events;
  Queue::Work w = [&] {
    in_work = true;
    while (!release.load()) std::this_thread::sleep_for(100us);
    std::lock_guard<std::mutex> lock(mu);
    events.push_back("inline");
    return kOk;
  };
  std::thread caller([&] {
    int result = -1;
    q.run_inline(w, {}, &result);
    inline_done = true;
  });
  VCHECK(until(in_work));
  const uint64_t seq = q.submit([&] {
    std::lock_guard<std::mutex> lock(mu);
    events.push_back("queued");
    return kOk;
  });
  std::this_thread::sleep_for(20ms);   // time for the queue's thread to (wrongly) start it
  {
    std::lock_guard<std::mutex> lock(mu);
    VCHECK(events.empty());
  }
  release = true;
  q.wait(seq);
  caller.join();
  VCHECK(inline_done.load());
  VCHECK_EQ(events.size(), size_t{2});
  VCHECK(events[0] == "inline" && events[1] == "queued");
}

// Callers that wait for their work, from many threads, with queued items between: each runs once, nothing is lost,
// and every queued item of one thread keeps its order.
VTEST(waited_for_and_queued_items_from_many_threads_all_run_once_in_order) {
  Queue q(kOk, kFailed);
  constexpr int kThreads = 6, kRounds = 400;
  std::vector<std::atomic<int>> counts(kThreads);
  for (auto& c : counts) c = 0;
  std::atomic<int> out_of_order{0};
  std::vector<std::thread> callers;
  for (int t = 0; t < kThreads; ++t)
    callers.emplace_back([&, t] {
      int expected = 0;
      for (int r = 0; r < kRounds; ++r) {
        // a queued item, then one waited for (inline when the queue happens to be idle)
        q.submit([&, t, r] {
          if (counts[t].load() != r * 2) ++out_of_order;
          ++counts[t];
          return kOk;
        });
        Queue::Work w = [&, t, r] {
          if (counts[t].load() != r * 2 + 1) ++out_of_order;
          ++counts[t];
          return kOk;
        };
        int result = -1;
        if (!q.run_inline(w, {}, &result)) {
          const uint64_t seq = q.submit(std::move(w));
          q.wait(seq);
        } else {
          // (run inline, so the queued item before it had finished)
        }
        ++expected;
      }
      (void)expected;
    });
  for (auto& c : callers) c.join();
  q.drain();
  for (int t = 0; t < kThreads; ++t) VCHECK_EQ(counts[t].load(), kRounds * 2);
  VCHECK_EQ(out_of_order.load(), 0);
  VCHECK(q.idle());
}

// A wait that outlasts the spin sleeps and is woken.
VTEST(a_long_item_is_waited_for_by_sleeping) {
  Queue q(kOk, kFailed);
  std::atomic<bool> finished{false};
  const uint64_t seq = q.submit([&] {
    std::this_thread::sleep_for(150ms);
    finished = true;
    return kOk;
  });
  q.wait(seq);
  VCHECK(finished.load());
  VCHECK(q.done(seq));
}

VTEST_MAIN
