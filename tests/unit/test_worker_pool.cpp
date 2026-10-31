// The host threads that run a grid's blocks (vgpu/exec/worker_pool.hpp), on
// their own: reuse, concurrency, nesting, several callers at once, what a body
// inherits from its caller, errors, and fork.
//
// The launch-level checks (the same kernels at VGPU_THREADS=1, 2 and N giving
// the same bytes and counters; launches from several host threads; child grids
// launched from a kernel) are in test_sched.cpp.
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cfenv>
#include <chrono>
#include <csignal>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vgpu/exec/worker_pool.hpp"
#include "vtest.hpp"

using vgpu::exec::run_parallel;
using vgpu::exec::worker_pool_stats;

namespace {

// ThreadSanitizer refuses to start a thread in a fork child of a process that
// had threads ("starting new threads after multi-threaded fork is not
// supported"), and the pool has threads by then. The fork test is for the
// other builds.
#if defined(__SANITIZE_THREAD__)
#define VGPU_TEST_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define VGPU_TEST_TSAN 1
#endif
#endif

// Each body waits until all n are running at once. If the pool ran them one
// after another, or on fewer threads than bodies, this would never finish
// (the deadline turns that into a failure instead of a hang).
bool all_run_at_once(unsigned n) {
  std::atomic<unsigned> arrived{0};
  std::atomic<unsigned> ok{0};
  run_parallel(n, [&](unsigned) {
    arrived.fetch_add(1);
    const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(20);
    while (arrived.load() < n) {
      if (std::chrono::steady_clock::now() > limit) return;
      std::this_thread::yield();
    }
    ok.fetch_add(1);
  });
  return ok.load() == n;
}

}  // namespace

VTEST(every_index_runs_exactly_once_each_on_a_thread_of_its_own) {
  for (unsigned n : {1u, 2u, 3u, 8u, 33u}) {
    std::vector<std::atomic<int>> hits(n);
    std::vector<std::thread::id> ids(n);
    run_parallel(n, [&](unsigned t) {
      hits[t].fetch_add(1);
      ids[t] = std::this_thread::get_id();
    });
    std::set<std::thread::id> distinct(ids.begin(), ids.end());
    for (unsigned t = 0; t < n; ++t) VCHECK_EQ(hits[t].load(), 1);
    VCHECK_EQ(distinct.size(), size_t{n});
    VCHECK(distinct.count(std::this_thread::get_id()) == 0);   // the caller only waits
  }
}

VTEST(the_bodies_are_alive_together_so_one_may_wait_for_another) {
  VCHECK(all_run_at_once(2));
  VCHECK(all_run_at_once(16));
}

VTEST(threads_are_made_once_and_reused_by_later_calls) {
  run_parallel(6, [](unsigned) {});
  const auto before = worker_pool_stats();
  VCHECK(before.created >= 6);
  std::set<std::thread::id> seen;
  std::mutex mu;
  for (int round = 0; round < 500; ++round)
    run_parallel(6, [&](unsigned) {
      std::lock_guard<std::mutex> g(mu);
      seen.insert(std::this_thread::get_id());
    });
  const auto after = worker_pool_stats();
  VCHECK_EQ(after.created, before.created);   // no thread was made for 3000 bodies
  VCHECK(seen.size() <= before.created);
  VCHECK_EQ(after.idle, after.created);       // and all are parked again
}

VTEST(a_body_may_call_run_parallel_itself) {
  std::atomic<int> inner{0};
  std::atomic<int> outer{0};
  run_parallel(4, [&](unsigned) {
    outer.fetch_add(1);
    run_parallel(4, [&](unsigned) { inner.fetch_add(1); });
  });
  VCHECK_EQ(outer.load(), 4);
  VCHECK_EQ(inner.load(), 16);
  // Nested bodies must also all be alive at once with the outer ones.
  std::atomic<unsigned> ok{0};
  run_parallel(3, [&](unsigned) { ok.fetch_add(all_run_at_once(3) ? 1 : 0); });
  VCHECK_EQ(ok.load(), 3u);
}

VTEST(several_host_threads_can_call_at_once) {
  constexpr int kCallers = 6;
  std::atomic<long> sum{0};
  std::atomic<int> failures{0};
  std::vector<std::thread> callers;
  for (int c = 0; c < kCallers; ++c)
    callers.emplace_back([&, c] {
      for (int i = 0; i < 200; ++i) {
        const unsigned n = 1 + static_cast<unsigned>((c + i) % 5);
        std::vector<int> mine(n, 0);
        run_parallel(n, [&](unsigned t) { mine[t] = static_cast<int>(t) + 1; });
        long local = 0;
        for (unsigned t = 0; t < n; ++t) {
          if (mine[t] != static_cast<int>(t) + 1) failures.fetch_add(1);
          local += mine[t];
        }
        sum.fetch_add(local);
      }
    });
  for (auto& c : callers) c.join();
  VCHECK_EQ(failures.load(), 0);
  long want = 0;
  for (int c = 0; c < kCallers; ++c)
    for (int i = 0; i < 200; ++i) {
      const long n = 1 + (c + i) % 5;
      want += n * (n + 1) / 2;
    }
  VCHECK_EQ(sum.load(), want);
}

VTEST(callers_waiting_on_each_other_get_threads_of_their_own) {
  // Two host threads each need 8 bodies alive together. Were the pool a fixed
  // set, one caller could hold half of it while the other held the rest.
  std::atomic<int> good{0};
  std::thread a([&] { good += all_run_at_once(8) ? 1 : 0; });
  std::thread b([&] { good += all_run_at_once(8) ? 1 : 0; });
  a.join();
  b.join();
  VCHECK_EQ(good.load(), 2);
}

VTEST(a_body_starts_with_the_callers_floating_point_environment) {
  const int saved = std::fegetround();
  const int modes[] = {FE_UPWARD, FE_DOWNWARD, FE_TOWARDZERO, FE_TONEAREST};
  for (int round = 0; round < 3; ++round)
    for (int m : modes) {
      std::fesetround(m);
      std::atomic<int> wrong{0};
      run_parallel(4, [&](unsigned) {
        if (std::fegetround() != m) wrong.fetch_add(1);
        // A body that leaves a directed mode behind (a fault between
        // fesetround and its restore does) must not leak it into the next call.
        std::fesetround(FE_DOWNWARD);
      });
      VCHECK_EQ(wrong.load(), 0);
    }
  std::fesetround(saved);
}

VTEST(a_body_starts_with_the_callers_signal_mask) {
  sigset_t block, old;
  sigemptyset(&block);
  sigaddset(&block, SIGUSR2);
  for (int round = 0; round < 3; ++round) {
    const bool blocked = round % 2 == 0;
    pthread_sigmask(blocked ? SIG_BLOCK : SIG_UNBLOCK, &block, &old);
    std::atomic<int> wrong{0};
    run_parallel(3, [&](unsigned) {
      sigset_t now;
      pthread_sigmask(SIG_BLOCK, nullptr, &now);
      if ((sigismember(&now, SIGUSR2) == 1) != blocked) wrong.fetch_add(1);
    });
    VCHECK_EQ(wrong.load(), 0);
    pthread_sigmask(SIG_SETMASK, &old, nullptr);
  }
}

VTEST(an_error_in_a_body_reaches_the_caller_after_every_body_has_returned) {
  std::atomic<int> finished{0};
  bool caught = false;
  try {
    run_parallel(5, [&](unsigned t) {
      if (t == 2) throw std::runtime_error("body 2");
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
      finished.fetch_add(1);
    });
  } catch (const std::runtime_error& e) {
    caught = std::string(e.what()) == "body 2";
  }
  VCHECK(caught);
  VCHECK_EQ(finished.load(), 4);   // the others were not abandoned
  VCHECK(all_run_at_once(5));      // and the pool still works
}

VTEST(zero_bodies_is_nothing) {
  const auto before = worker_pool_stats();
  run_parallel(0, [](unsigned) { throw std::logic_error("must not run"); });
  VCHECK_EQ(worker_pool_stats().created, before.created);
}

#ifndef VGPU_TEST_TSAN
// A program may fork after a launch. The child has none of the pool's threads
// and must get working ones of its own rather than wait for ghosts.
VTEST(a_fork_child_gets_a_working_pool_of_its_own) {
  run_parallel(4, [](unsigned) {});   // the parent's pool exists, with threads
  const pid_t pid = fork();
  VCHECK(pid >= 0);
  if (pid == 0) {
    // Only this thread exists here. Every path out is _exit: the child must
    // not run the parent's test-harness teardown.
    int bad = 0;
    for (int round = 0; round < 3; ++round) {
      std::atomic<int> hits{0};
      run_parallel(4, [&](unsigned) { hits.fetch_add(1); });
      bad += hits.load() != 4;
    }
    bad += !all_run_at_once(4);
    _exit(bad ? 1 : 0);
  }
  int status = 0;
  // A child that hangs on the pool's ghosts is killed rather than waited for.
  for (int waited = 0; waited < 600; ++waited) {
    const pid_t r = waitpid(pid, &status, WNOHANG);
    if (r == pid) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    if (waited == 599) {
      kill(pid, SIGKILL);
      waitpid(pid, &status, 0);
      status = -1;
    }
  }
  VCHECK(WIFEXITED(status));
  VCHECK_EQ(WEXITSTATUS(status), 0);
  VCHECK(all_run_at_once(4));   // and the parent's pool is untouched
}
#endif

VTEST(the_cap_keeps_small_launches_on_few_threads) {
  using vgpu::exec::cap_threads;
  const uint64_t kMin = 100'000;   // 100 us of work per thread
  // Nothing known, a request of one, or the cap off: the request stands.
  VCHECK_EQ(cap_threads(8, 64, 0, kMin), 8u);
  VCHECK_EQ(cap_threads(1, 64, 5000, kMin), 1u);
  VCHECK_EQ(cap_threads(8, 64, 5000, 0), 8u);
  // 64 blocks of 234 ns (a 15 us launch): one thread.
  VCHECK_EQ(cap_threads(8, 64, 234, kMin), 1u);
  VCHECK_EQ(cap_threads(16, 64, 234, kMin), 1u);
  // Just under, at and over two threads' worth, and four.
  VCHECK_EQ(cap_threads(8, 100, 1999, kMin), 1u);
  VCHECK_EQ(cap_threads(8, 100, 2000, kMin), 2u);
  VCHECK_EQ(cap_threads(8, 100, 4000, kMin), 4u);
  // Never above the request, however large the grid.
  VCHECK_EQ(cap_threads(4, 1u << 20, 100000, kMin), 4u);
  VCHECK_EQ(cap_threads(8, UINT64_MAX, UINT64_MAX, kMin), 8u);
  VCHECK_EQ(cap_threads(2, 1, 1000000, kMin), 2u);
}

VTEST(the_unit_cost_is_exact_on_one_thread_and_averaged_otherwise) {
  using vgpu::exec::blend_unit_cost;
  VCHECK_EQ(blend_unit_cost(0, 64, 1, 15000), uint64_t{234});
  VCHECK_EQ(blend_unit_cost(999999, 64, 1, 6400), uint64_t{100});      // one thread replaces
  VCHECK_EQ(blend_unit_cost(0, 64, 8, 53000), uint64_t{6625});         // nothing known: the sample
  VCHECK_EQ(blend_unit_cost(1000, 100, 4, 50000), uint64_t{1500});     // (1000 + 2000) / 2
  VCHECK_EQ(blend_unit_cost(77, 0, 4, 50000), uint64_t{77});           // no blocks: unchanged
  VCHECK(blend_unit_cost(0, 1u << 30, 1, 1) >= 1);                      // never zero (zero means unknown)
}

VTEST_MAIN
