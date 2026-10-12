#pragma once
// How many host threads a library routine that computes on the host may use.
//
// The routines behind cuBLAS, cuDNN and the like each run on the calling
// thread, so a program with eight CPU threads each making a call would, if
// every call spread itself over the machine, ask for eight times the cores it
// has. Each call therefore takes a share: the CPUs this process may use (its
// affinity and cgroup quota, vgpu::host_cpus()), less the threads other such
// calls hold right now, and never more than VGPU_THREADS when that is set (1
// gives one thread and the strictly ordered sequence). The result of a call
// does not depend on the count: callers split their work only into pieces that
// are independent of each other.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <thread>
#include <vector>

#include "vgpu/host_cpus.hpp"

namespace vgpu::hostpar {

inline std::atomic<unsigned>& busy_threads() {
  static std::atomic<unsigned> n{0};
  return n;
}

inline unsigned thread_cap() {
  static const unsigned cap = [] {
    unsigned c = vgpu::host_cpus();
    if (const char* v = std::getenv("VGPU_THREADS"); v && *v) {
      const long t = std::atol(v);
      c = t >= 1 ? static_cast<unsigned>(t) : 1u;
    }
    return c ? c : 1u;
  }();
  return cap;
}

// Threads worth using for `work` elementary steps in `items` independent pieces.
// A step is one multiply-add or one element touched; below min_work a thread
// costs more than it saves.
inline unsigned plan(uint64_t work, int64_t items, uint64_t min_work = (1u << 21)) {
  if (items <= 1 || work < min_work) return 1;
  unsigned want = thread_cap();
  const unsigned cpus = vgpu::host_cpus();
  const unsigned busy = busy_threads().load(std::memory_order_relaxed);
  const unsigned spare = cpus > busy ? cpus - busy : 1u;
  if (spare < want) want = spare;
  if (static_cast<int64_t>(want) > items) want = static_cast<unsigned>(items);
  return want ? want : 1u;
}

// Runs fn(piece) for piece in [0, items) on n threads (the caller's included),
// handing pieces out one at a time.
template <class F>
void parallel_for(unsigned n, int64_t items, F&& fn) {
  std::atomic<int64_t> next{0};
  auto worker = [&] {
    for (;;) {
      const int64_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= items) break;
      fn(i);
    }
  };
  if (n <= 1) {
    worker();
    return;
  }
  busy_threads().fetch_add(n, std::memory_order_relaxed);
  std::vector<std::thread> pool;
  pool.reserve(n - 1);
  for (unsigned t = 1; t < n; ++t) pool.emplace_back(worker);
  worker();
  for (auto& t : pool) t.join();
  busy_threads().fetch_sub(n, std::memory_order_relaxed);
}

}  // namespace vgpu::hostpar
