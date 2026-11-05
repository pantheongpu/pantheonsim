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

// Its own call site again, for the cap on the wait.
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

}  // namespace

VTEST(the_first_look_of_a_launch_waits_for_what_the_last_look_cost) {
  // Nothing is known yet: the look comes after 250 microseconds, and costs 2 ms a block for each of the two.
  VCHECK_EQ(launch(2, 8, 2000), 2u);
  // One block, 20 ms: its look would cost 2 ms, so the first comes after 24 ms, and the launch is over before that.
  // (Not twelve times what both blocks' look cost, and not the 250 microseconds.)
  VCHECK_EQ(launch(1, 20, 2000), 0u);
  // The same for 36 ms: the look comes at about 24 ms, and the next would be 24 ms after it.
  VCHECK_EQ(launch(1, 36, 2000), 1u);
  // Two blocks resident: the cost is 2 ms for each, so the first look is after 48 ms, not 24.
  VCHECK_EQ(launch(2, 36, 2000), 0u);
}

VTEST(a_block_that_stands_still_is_still_found_out_and_slept_on) {
  // Looks 250 microseconds apart (a look costs nothing), the third finding nothing changed: the thread then sleeps between passes. Without
  // that it would take a turn every microsecond or so, tens of thousands in 100 ms.
  VCHECK(spin(100) < 5000);
}

VTEST(the_wait_for_the_first_look_has_a_ceiling) {
  // Nothing is known: a launch over in 50 microseconds has not looked (the first look is 250 microseconds in, never at once).
  VCHECK_EQ(launch_cap(50, 50000), 0u);
  // A look of 50 ms a block, taken at 250 microseconds, teaches the call site that.
  VCHECK_EQ(launch_cap(5000, 50000), 1u);
  // Twelve times that would put the next launch's first look at 600 ms; the ceiling is 500 ms, and not 100.
  VCHECK_EQ(launch_cap(300000, 50000), 0u);
  VCHECK_EQ(launch_cap(560000, 50000), 1u);   // (the look at 500 ms, not the one at 600)
}

VTEST(once_it_has_looked_a_launch_looks_again_after_twelve_times_the_look) {
  // Taught 2 ms a block (first launch below), the first look is at 24 ms, the second at 50 and the third at 76: three in 80 ms.
  launch(1, 30, 2000);
  const unsigned n = launch(1, 80, 2000);
  VCHECK(n >= 2 && n <= 3);
}

VTEST_MAIN
