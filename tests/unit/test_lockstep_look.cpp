// When lockstep_worker (vgpu/exec/lockstep.hpp) first looks at its blocks' registers to tell a block that waits from one that computes.
// A look folds every register file of every resident block, and a kernel with big register files makes that millisecond work. The worker
// used to take its first look 250 microseconds after the launch began whatever that cost, and only then learned how dear a look was, so a
// launch that ran for twenty milliseconds in all (a sort's thousands of short kernels) spent a third of its time on one look that found
// nothing. A launch now waits twelve times what the last look cost for each resident block, times the blocks it holds (at most half a
// second), before its first look; once it has looked, twelve times that look, as before. A look decides only whether the thread sleeps
// between passes, so none of this reaches a result; what is checked here is when the looks happen and that a block that stands still is
// still found out.
#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <thread>
#include <type_traits>
#include <utility>

#include "vgpu/exec/lockstep.hpp"
#include "vtest.hpp"

namespace {

using Clock = std::chrono::steady_clock;

struct Block {
  Clock::time_point start;
  bool started = false;
};

void busy(std::chrono::microseconds d) {
  const auto end = Clock::now() + d;
  while (Clock::now() < end) {}
}

// One launch of `units` blocks, all resident, each finishing `run_ms` after its first turn. A look costs `fold_us` a block. Returns the
// number of times a block was folded. (One call site, so one memory of what a look costs, shared by the calls below in order.)
template <int Tag>
unsigned launch(unsigned units, unsigned run_ms, unsigned fold_us) {
  std::atomic<uint64_t> ticket{0}, beat{0};
  unsigned folds = 0;
  vgpu::exec::lockstep_worker<Block>(
      ticket, beat, units, /*window=*/units, /*initial=*/units, /*chunk=*/units,
      [](Block& b, uint64_t, size_t) { b = Block{}; },
      [&](Block& b) {
        if (!b.started) {
          b.started = true;
          b.start = Clock::now();
        }
        return Clock::now() - b.start >= std::chrono::milliseconds(run_ms);
      },
      [](const Block&) -> uint64_t { return 0; },
      [&](const Block&) -> uint64_t {
        busy(std::chrono::microseconds(fold_us));
        return ++folds;   // (a different value each time: this block computes)
      },
      [](Block&) {}, [] { return false; });
  return folds;
}

// The same, for a block that stands still; its own call site. Returns the number of turns it was given in `run_ms`.
template <int Tag>
uint64_t spin(unsigned run_ms) {
  std::atomic<uint64_t> ticket{0}, beat{0};
  uint64_t turns = 0;
  vgpu::exec::lockstep_worker<Block>(
      ticket, beat, 1, 1, 1, 1, [](Block& b, uint64_t, size_t) { b = Block{}; },
      [&](Block& b) {
        if (!b.started) {
          b.started = true;
          b.start = Clock::now();
        }
        ++turns;
        return Clock::now() - b.start >= std::chrono::milliseconds(run_ms);
      },
      [](const Block&) -> uint64_t { return 0; }, [](const Block&) -> uint64_t { return 42; }, [](Block&) {}, [] { return false; });
  return turns;
}

// How many turns of spin()'s kind a thread takes in a microsecond on this machine and in this build (a turn reads the clock twice and yields): the
// answer differs by an order of magnitude between a plain build and a sanitizer build, and a fixed bound on the turns taken cannot be
// right for both.
double turns_per_us() {
  uint64_t n = 0;
  const auto start = Clock::now();
  while (Clock::now() - start < std::chrono::milliseconds(2)) {
    ++n;
    std::this_thread::yield();   // (a pass that waits gives the core up, as the worker does)
  }
  return static_cast<double>(n) / 2000.0;
}

// Its own call site again, for the cap on the wait.
template <int Tag>
unsigned launch_cap(unsigned run_us, unsigned fold_us) {
  std::atomic<uint64_t> ticket{0}, beat{0};
  unsigned folds = 0;
  vgpu::exec::lockstep_worker<Block>(
      ticket, beat, 1, 1, 1, 1, [](Block& b, uint64_t, size_t) { b = Block{}; },
      [&](Block& b) {
        if (!b.started) {
          b.started = true;
          b.start = Clock::now();
        }
        return Clock::now() - b.start >= std::chrono::microseconds(run_us);
      },
      [](const Block&) -> uint64_t { return 0; },
      [&](const Block&) -> uint64_t {
        busy(std::chrono::microseconds(fold_us));
        return ++folds;
      },
      [](Block&) {}, [] { return false; });
  return folds;
}

// These tests are about when things happen on the wall clock, so a thread that is descheduled for a few milliseconds (a loaded CI
// machine, and every sanitizer build) can change what a launch sees: a look that was preempted teaches the call site a cost of
// milliseconds, and the nap that follows never comes within the test's window. So a case may be tried again, up to kAttempts times, each
// time on call sites of its own (the Tag), which have learned nothing; it passes if one attempt gives the expected answers. A mistake in
// the worker gives the wrong answer in every attempt and still fails.
constexpr int kAttempts = 6;

template <class F, size_t... I>
bool some_attempt_holds(F f, std::index_sequence<I...>) {
  return (f(std::integral_constant<int, static_cast<int>(I)>{}) || ...);
}

template <class F>
bool eventually(F f) {
  return some_attempt_holds(f, std::make_index_sequence<kAttempts>{});
}

}  // namespace

VTEST(the_first_look_of_a_launch_waits_for_what_the_last_look_cost) {
  VCHECK(eventually([](auto tag) {
    constexpr int T = decltype(tag)::value;
    // Nothing is known yet: the look comes after 250 microseconds, and costs 2 ms a block for each of the two.
    if (launch<T>(2, 8, 2000) != 2u) return false;
    // One block, 20 ms: its look would cost 2 ms, so the first comes after 24 ms, and the launch is over before that.
    // (Not twelve times what both blocks' look cost, and not the 250 microseconds.)
    if (launch<T>(1, 20, 2000) != 0u) return false;
    // The same for 36 ms: the look comes at about 24 ms, and the next would be 24 ms after it.
    if (launch<T>(1, 36, 2000) != 1u) return false;
    // Two blocks resident: the cost is 2 ms for each, so the first look is after 48 ms, not 24.
    return launch<T>(2, 36, 2000) == 0u;
  }));
}

VTEST(a_block_that_stands_still_is_still_found_out_and_slept_on) {
  // Looks 250 microseconds apart (a look costs nothing), the third finding nothing changed: the thread then sleeps between passes. Without
  // that it would take a turn every microsecond or so for the 100 ms. With it, the turns are the ones before the third look (about
  // 750 microseconds' worth) and a few hundred more: about one hundredth of the others. The bound is a quarter, measured on this machine.
  VCHECK(eventually([](auto tag) {
    const double bound = turns_per_us() * 100000.0 / 4.0;
    return static_cast<double>(spin<decltype(tag)::value>(100)) < bound;
  }));
}

VTEST(the_wait_for_the_first_look_has_a_ceiling) {
  VCHECK(eventually([](auto tag) {
    constexpr int T = decltype(tag)::value;
    // Nothing is known: a launch over in 50 microseconds has not looked (the first look is 250 microseconds in, never at once).
    if (launch_cap<T>(50, 50000) != 0u) return false;
    // A look of 50 ms a block, taken at 250 microseconds, teaches the call site that.
    if (launch_cap<T>(5000, 50000) != 1u) return false;
    // Twelve times that would put the next launch's first look at 600 ms; the ceiling is 500 ms, and not 100.
    if (launch_cap<T>(300000, 50000) != 0u) return false;
    return launch_cap<T>(560000, 50000) == 1u;   // (the look at 500 ms, not the one at 600)
  }));
}

VTEST(once_it_has_looked_a_launch_looks_again_after_twelve_times_the_look) {
  VCHECK(eventually([](auto tag) {
    constexpr int T = decltype(tag)::value;
    // Taught 2 ms a block (first launch below), the first look is at 24 ms, the second at 50 and the third at 76: three in 80 ms.
    launch<T>(1, 30, 2000);
    const unsigned n = launch<T>(1, 80, 2000);
    return n >= 2 && n <= 3;
  }));
}

VTEST_MAIN
