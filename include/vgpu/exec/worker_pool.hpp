// Host threads that run a grid's blocks, kept between launches.
//
// A launch that spreads its blocks over N host threads used to create N
// std::threads and join them before returning. For a program that issues
// thousands of small launches the creation and join cost more than the blocks
// did (README "Performance"; docs/perf-design/ has the measurement). The
// threads are now made once, on the first launch that needs them, and parked
// between launches.
//
// What does not change, and why:
//  * run_parallel(n, body) calls body(0) ... body(n-1), each on a thread of its
//    own, all alive at once -- exactly the set of threads the launch used to
//    create. A body may therefore wait for another body's progress. Nothing
//    here assumes the bodies are independent.
//  * The calling thread only waits, as it did when it joined. It never runs a
//    body, so the thread that holds the API lock and owns the thread_local
//    state of a CUDA call is never one of the workers.
//  * A body starts with the floating-point environment and the signal mask of
//    the thread that called run_parallel, which is what a freshly created
//    thread inherited. (CPU affinity and scheduling attributes are those of the
//    thread that created the worker; they change no result.)
//  * Nested and concurrent calls work: a call takes the idle workers it needs
//    and makes more when there are not enough, so a body may itself call
//    run_parallel, and so may several host threads at once, and none of them
//    can wait for a worker that is held by another.
//  * After fork() the child has none of the parent's threads. A pool made
//    before the fork is recognised by its process id and left alone; the child
//    makes a pool of its own on its first parallel launch.
//
// An exception that escapes a body is rethrown to the caller once every body
// has returned (the first one, if several threw). The bodies of the engines
// catch their own errors and never reach this.
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace vgpu::exec {

// The type-erased form. `fn(ctx, t)` runs for t in [0, n).
void run_parallel_raw(unsigned n, void (*fn)(void* ctx, unsigned t), void* ctx);

template <class Body>
void run_parallel(unsigned n, Body&& body) {
  run_parallel_raw(
      n, [](void* ctx, unsigned t) { (*static_cast<std::remove_reference_t<Body>*>(ctx))(t); },
      const_cast<void*>(static_cast<const void*>(&body)));
}

// What the pool holds, for tests: threads created since the process (or the
// fork) began, how many of them are parked right now, and how many calls were made.
struct WorkerPoolStats {
  size_t created = 0;
  size_t idle = 0;
  uint64_t calls = 0;   // run_parallel calls made (a launch kept to one thread makes none)
};
WorkerPoolStats worker_pool_stats();

// How many of the requested host threads a launch should use.
//
// A thread costs a hand-off per launch (a futex wake when it has gone cold,
// several microseconds even when it has not), so a launch whose blocks take
// 15 us in all is slower on 8 threads than on one: its period rises, the
// workers stop polling and sleep, and every later launch pays the wake again.
// The cap keeps small launches on few threads, from what the last launches of
// the same kernel cost. Which thread runs a block changes nothing it computes,
// so this moves time only, never a result or a counter.
//
// cap_threads is the policy: with `ns_per_unit` (the measured cost of one
// block, 0 when not known) a launch of `units` blocks gets at most
// units * ns_per_unit / min_work_ns threads, at least 1, and never more than
// `requested`. An unknown cost, min_work_ns == 0 and a request of 1 leave the
// request as it was.
unsigned cap_threads(unsigned requested, uint64_t units, uint64_t ns_per_unit, uint64_t min_work_ns);

// The cost of one block after a launch that took `wall_ns` on `threads` threads
// (their summed time is at most wall_ns * threads, an over-estimate when they
// waited, which errs towards more threads). One thread measures it exactly and
// replaces `old`; more are averaged into it.
uint64_t blend_unit_cost(uint64_t old_ns_per_unit, uint64_t units, unsigned threads, uint64_t wall_ns);

// The cost of one block from the time the threads spent running the launch's
// bodies (their summed `busy_ns`), which leaves out what a wall-clock figure
// holds: the wait for workers to wake, the wait for the slowest to finish.
// Those waits are what a launch on several threads costs on top of its work,
// and measuring them as work makes a launch look dear because it was slow,
// which keeps it on the pool, which keeps it slow. Never zero (zero means
// unknown).
uint64_t unit_cost_from_busy(uint64_t units, uint64_t busy_ns);

// run_parallel adds, to the calling thread's tally, the time its bodies ran
// (each body's own start to end, on whichever thread ran it). A LaunchWidth
// that was timed clears the tally when it is made and, when it ends, takes the
// cost of a block from it if the launch used more than one thread; with an
// empty tally (the bodies were not run through run_parallel) it uses the wall
// time as before. Exposed for tests, which stand in for the pool.
void note_busy_ns(uint64_t ns);

// Per-kernel use of the two above, for an engine's grid launch. Construct it
// after the thread count is known and before the blocks run; it lowers
// `threads` by the cap and, when it ends, records what the launch cost. `key`
// names the kernel (any address that is stable for it). A request of one
// thread, or one block, is left alone and not timed.
class LaunchWidth {
 public:
  // Out of line on purpose: a launch on one thread, which is most of them, runs the code it always did.
  [[gnu::noinline]] LaunchWidth(const void* key, uint64_t units, unsigned& threads);
  [[gnu::noinline]] ~LaunchWidth();
  LaunchWidth(const LaunchWidth&) = delete;
  LaunchWidth& operator=(const LaunchWidth&) = delete;

 private:
  const void* key_;
  uint64_t units_;
  unsigned used_;
  uint64_t begin_ns_;   // 0: not timed
};

}  // namespace vgpu::exec
