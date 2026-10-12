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
// fork) began, and how many of them are parked right now.
struct WorkerPoolStats {
  size_t created = 0;
  size_t idle = 0;
};
WorkerPoolStats worker_pool_stats();

}  // namespace vgpu::exec
