// lockstep_worker (vgpu/exec/lockstep.hpp) when the host has fewer cores than threads. A block that waits for another gives its host
// thread's turn up, so the thread the awaited block runs on gets to run. Without that, the waiting thread spends its whole time slice on
// polls, and a program whose waits count their polls and give up after so many (tests/hipcc/lockstep.cpp: 2^18 of them) fails on a loaded
// machine (CI's lockstep.gfx942 "every wait is satisfied", intermittently), whatever the order the blocks started in.
//
// Here the whole process is pinned to ONE host core, so the threads can only take turns, and eight of them run a chain of eight blocks, each
// waiting on the one after it, in the worst order the OS can pick. Each wait is bounded by a poll count that a thread spending its time slice
// on polls runs through many times over; a thread that yields between passes needs a few hundred. The poll budget is a count, so a thread the
// OS has descheduled does not use it up: the test does not depend on how loaded the machine is.
#include <sched.h>

#include <atomic>
#include <cstdint>
#include <memory>
#include <thread>
#include <vector>

#include "vgpu/exec/lockstep.hpp"
#include "vtest.hpp"

namespace {

constexpr unsigned kUnits = 8, kThreads = 8, kPollsPerTurn = 64, kBudget = 20000, kRounds = 20;

struct Chain {
  std::atomic<unsigned> flag[kUnits + 1] = {};
  std::atomic<unsigned> timeouts{0}, finished{0};
};

struct Block {
  uint64_t unit = 0, polls = 0;
  bool done = false;
};

// Runs one chain; returns the number of waits that gave up.
unsigned run_chain() {
  Chain chain;
  std::atomic<uint64_t> ticket{0}, beat{0};
  std::atomic<unsigned> started{0};
  const auto worker = [&] {
    started.fetch_add(1);
    while (started.load() < kThreads) std::this_thread::yield();
    vgpu::exec::lockstep_worker<Block>(
        ticket, beat, kUnits, /*window=*/1, /*initial=*/1, /*chunk=*/1,
        [](Block& b, uint64_t unit, size_t) { b = Block{unit, 0, false}; },
        [&](Block& b) {
          // Unit u waits for unit u + 1 (the last waits for nobody), as a chain of flags run from the last block to the first.
          for (unsigned i = 0; i < kPollsPerTurn; ++i) {
            if (b.unit + 1 == kUnits || chain.flag[b.unit + 1].load(std::memory_order_acquire)) {
              chain.flag[b.unit].store(1, std::memory_order_release);
              chain.finished.fetch_add(1);
              return true;
            }
            if (++b.polls >= kBudget) {
              chain.timeouts.fetch_add(1);
              chain.flag[b.unit].store(1, std::memory_order_release);   // (let the others finish, as a timed-out wait does)
              return true;
            }
          }
          return false;
        },
        [](const Block&) -> uint64_t { return 0; },   // a poll stores nothing
        [](const Block& b) -> uint64_t { return b.polls; },   // (a wait that counts its polls never looks standing still)
        [](Block&) {}, [] { return false; });
  };
  std::vector<std::thread> threads;
  for (unsigned t = 0; t < kThreads; ++t) threads.emplace_back(worker);
  for (auto& t : threads) t.join();
  return chain.timeouts.load();
}

}  // namespace

VTEST(a_chain_of_waits_finishes_when_the_threads_share_one_core) {
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof set, &set) != 0 || CPU_COUNT(&set) == 0) return;
  int cpu = 0;
  while (!CPU_ISSET(cpu, &set)) ++cpu;
  cpu_set_t one;
  CPU_ZERO(&one);
  CPU_SET(cpu, &one);
  if (sched_setaffinity(0, sizeof one, &one) != 0) return;   // (threads made from here on share the core)
  unsigned timeouts = 0;
  for (unsigned r = 0; r < kRounds; ++r) timeouts += run_chain();
  sched_setaffinity(0, sizeof set, &set);
  VCHECK_EQ(timeouts, 0u);
}

VTEST_MAIN
