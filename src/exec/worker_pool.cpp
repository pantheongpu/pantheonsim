#include "vgpu/exec/worker_pool.hpp"

#include <unistd.h>

#include <atomic>
#include <cfenv>
#include <chrono>
#include <condition_variable>
#include <csignal>
#include <cstring>
#include <exception>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <vector>

namespace vgpu::exec {
namespace {

#if defined(__x86_64__) || defined(__i386__)
inline void cpu_relax() { __builtin_ia32_pause(); }
#elif defined(__aarch64__)
inline void cpu_relax() { asm volatile("yield" ::: "memory"); }
#else
inline void cpu_relax() { std::this_thread::yield(); }
#endif

using Clock = std::chrono::steady_clock;

// How long a parked worker, and a caller waiting for its workers, poll before
// they sleep on a condition variable. A sleeping thread costs a futex wake and
// a trip through the scheduler to start again -- tens of microseconds, more on
// a loaded host -- which is more than a small launch's blocks take, so a loop
// of back-to-back launches would pay it on every launch for every worker.
// Polling for a little longer than the usual gap between two launches avoids
// that.
//
// Polling costs CPU, so a worker does it only while launches keep coming close
// together: it polls for kSpin when its last job came less than kSpin after it
// finished the one before (it is "hot"), and for kColdSpin otherwise, so a
// program that launches now and then leaves its workers asleep and costing
// nothing. One close pair of launches makes the workers hot again.
constexpr auto kSpin = std::chrono::microseconds(40);
constexpr auto kColdSpin = std::chrono::microseconds(2);

// One call of run_parallel. It lives on the caller's stack, which is safe
// because the caller does not return until every worker it posted it to has
// said it is done with it, and a worker touches nothing of it after that.
struct Job {
  void (*fn)(void*, unsigned) = nullptr;
  void* ctx = nullptr;
  std::fenv_t env;   // the caller's floating-point environment
  sigset_t mask;     // ...and signal mask, which a new thread would have inherited
  Clock::time_point posted;   // when the call began handing it out
};

struct Pool;

// One thread. Between launches it belongs to the pool's idle list; during one,
// to the call that took it. `posted` and `done` count the jobs it has been
// given and has finished: it has work when they differ.
struct Worker {
  std::thread thread;
  std::atomic<Job*> job{nullptr};
  std::atomic<unsigned> index{0};
  std::atomic<uint64_t> posted{0};
  std::atomic<uint64_t> done{0};
  std::atomic<bool> quit{false};
  std::exception_ptr error;   // written by the worker before `done`, read by the caller after it

  // The sleeping sides. A flag is raised under the mutex before sleeping and
  // the waker takes and drops the mutex before notifying, so a wake cannot fall
  // between the sleeper's last look at the state and its wait.
  std::mutex mu;
  std::condition_variable work_cv;   // the worker sleeps here, for work or for quit
  std::condition_variable done_cv;   // the caller sleeps here, for `done`
  std::atomic<bool> worker_asleep{false};
  std::atomic<bool> caller_asleep{false};

  bool has_work() const { return posted.load() != done.load(); }
};

struct Pool {
  explicit Pool(pid_t p) : pid(p) {}
  ~Pool();

  // The process that made the pool. A fork child has the memory but none of
  // the threads, and must not use it (pool() below).
  const pid_t pid;
  // The pool of the process this one was forked from, if that is how this one
  // came to be made. Kept only so that it stays reachable: its threads are not
  // ours to join, and freeing it would be a mistake for the same reason.
  Pool* previous = nullptr;

  std::mutex mu;
  std::vector<Worker*> idle;
  std::vector<std::unique_ptr<Worker>> all;
  size_t created = 0;

  bool quiescent() {
    std::lock_guard<std::mutex> g(mu);
    return idle.size() == all.size();
  }
  void give_back(const std::vector<Worker*>& ws) {
    std::lock_guard<std::mutex> g(mu);
    idle.insert(idle.end(), ws.begin(), ws.end());
  }
  Worker* spawn();
};

void worker_main(Worker* w) {
  sigset_t current;
  pthread_sigmask(SIG_BLOCK, nullptr, &current);   // what this thread was created with
  bool hot = false;
  Clock::time_point finished = Clock::now();
  for (;;) {
    // Wait for work: poll briefly, then sleep.
    if (!w->has_work() && !w->quit.load()) {
      const auto until = Clock::now() + (hot ? kSpin : kColdSpin);
      unsigned n = 0;
      while (!w->has_work() && !w->quit.load()) {
        cpu_relax();
        if ((++n & 15) == 0 && Clock::now() > until) break;
      }
    }
    if (!w->has_work()) {
      if (w->quit.load()) return;
      std::unique_lock<std::mutex> l(w->mu);
      w->worker_asleep.store(true);
      w->work_cv.wait(l, [&] { return w->has_work() || w->quit.load(); });
      w->worker_asleep.store(false);
      if (!w->has_work()) return;   // quit, and nothing left to run
    }
    Job* job = w->job.load();
    const unsigned index = w->index.load();
    hot = job->posted - finished < kSpin;
    std::fesetenv(&job->env);
    if (std::memcmp(&current, &job->mask, sizeof current) != 0) {
      pthread_sigmask(SIG_SETMASK, &job->mask, nullptr);
      current = job->mask;
    }
    try {
      job->fn(job->ctx, index);
    } catch (...) {
      w->error = std::current_exception();
    }
    finished = Clock::now();
    // The last thing done with the job; from here the caller may return.
    w->done.store(w->posted.load());
    if (w->caller_asleep.load()) {
      { std::lock_guard<std::mutex> g(w->mu); }
      w->done_cv.notify_one();
    }
  }
}

Worker* Pool::spawn() {
  auto w = std::make_unique<Worker>();
  try {
    w->thread = std::thread(worker_main, w.get());
  } catch (const std::system_error&) {
    return nullptr;   // out of threads: the caller runs the rest itself
  }
  Worker* raw = w.get();
  std::lock_guard<std::mutex> g(mu);
  all.push_back(std::move(w));
  ++created;
  return raw;
}

Pool::~Pool() {
  // Only called with every worker idle (Holder below), so each is told to
  // leave and joined.
  std::vector<std::unique_ptr<Worker>> mine;
  {
    std::lock_guard<std::mutex> g(mu);
    mine.swap(all);
    idle.clear();
  }
  for (auto& w : mine) {
    w->quit.store(true);
    if (w->worker_asleep.load()) {
      { std::lock_guard<std::mutex> g(w->mu); }
      w->work_cv.notify_one();
    }
    w->thread.join();
  }
}

// The pool of this process. Made on first use; never behind a mutex, so that
// a fork taken while another thread is inside it cannot leave the child with a
// lock nobody will release.
struct Holder {
  std::atomic<Pool*> pool{nullptr};
  ~Holder() {
    Pool* p = pool.load();
    // A pool some other host thread is still using as the process exits is
    // left as it is: its workers would otherwise find it gone.
    if (p && p->pid == getpid() && p->quiescent()) {
      keep = p->previous;
      p->previous = nullptr;
      delete p;
    }
  }
  Pool* keep = nullptr;   // the abandoned pool of a fork parent, still referenced at exit
};
Holder g_holder;

Pool& pool() {
  const pid_t me = getpid();
  Pool* p = g_holder.pool.load(std::memory_order_acquire);
  if (p && p->pid == me) return *p;
  Pool* fresh = new Pool(me);
  for (;;) {
    fresh->previous = p;
    if (g_holder.pool.compare_exchange_weak(p, fresh, std::memory_order_acq_rel, std::memory_order_acquire))
      return *fresh;
    if (p && p->pid == me) {   // another thread of this process got there first
      delete fresh;
      return *p;
    }
  }
}

}  // namespace

void run_parallel_raw(unsigned n, void (*fn)(void*, unsigned), void* ctx) {
  if (n == 0) return;
  Pool& p = pool();
  Job job;
  job.fn = fn;
  job.ctx = ctx;
  std::fegetenv(&job.env);
  pthread_sigmask(SIG_BLOCK, nullptr, &job.mask);
  job.posted = Clock::now();

  // The idle workers first (the most recently parked, whose caches are
  // warmest), then new ones.
  std::vector<Worker*> ws;
  ws.reserve(n);
  {
    std::lock_guard<std::mutex> g(p.mu);
    while (ws.size() < n && !p.idle.empty()) {
      ws.push_back(p.idle.back());
      p.idle.pop_back();
    }
  }
  while (ws.size() < n) {
    Worker* w = p.spawn();
    if (!w) break;
    ws.push_back(w);
  }
  const unsigned started = static_cast<unsigned>(ws.size());
  for (unsigned i = 0; i < started; ++i) {
    Worker* w = ws[i];
    w->error = nullptr;
    w->job.store(&job);
    w->index.store(i);
    w->posted.fetch_add(1);   // last: it is what the worker looks at
    if (w->worker_asleep.load()) {
      { std::lock_guard<std::mutex> g(w->mu); }
      w->work_cv.notify_one();
    }
  }
  // No thread to be had for the rest: they run here, one after another. The
  // engines' bodies take their work from a shared counter, so the first of
  // them finishes the grid and the others find nothing left.
  std::exception_ptr first_error;
  for (unsigned i = started; i < n; ++i) {
    try {
      fn(ctx, i);
    } catch (...) {
      if (!first_error) first_error = std::current_exception();
    }
  }
  for (Worker* w : ws) {
    if (!w->has_work()) continue;
    if (w->done.load() != w->posted.load()) {
      const auto until = Clock::now() + kSpin;
      unsigned spins = 0;
      while (w->done.load() != w->posted.load()) {
        cpu_relax();
        if ((++spins & 15) == 0 && Clock::now() > until) break;
      }
    }
    if (w->done.load() != w->posted.load()) {
      std::unique_lock<std::mutex> l(w->mu);
      w->caller_asleep.store(true);
      w->done_cv.wait(l, [&] { return w->done.load() == w->posted.load(); });
      w->caller_asleep.store(false);
    }
  }
  // Every body has returned; the workers are the pool's again.
  for (Worker* w : ws)
    if (w->error && !first_error) first_error = w->error;
  p.give_back(ws);
  if (first_error) std::rethrow_exception(first_error);
}

WorkerPoolStats worker_pool_stats() {
  Pool& p = pool();
  std::lock_guard<std::mutex> g(p.mu);
  return {p.created, p.idle.size()};
}

}  // namespace vgpu::exec
