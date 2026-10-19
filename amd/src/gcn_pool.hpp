// The host threads a dispatch's work-groups run on, kept between dispatches.
//
// A dispatch spread over N threads used to start N std::threads and join them, for every launch: tens of
// microseconds a launch, which a grid of a few hundred cheap work-groups does not earn back. run() hands the
// same job to threads that already exist (and are made as needed, so that N tasks that wait on one another are
// still N threads running at once, as they were), and returns when every one has finished.
//
// What a task sees is what it saw as a fresh thread: it runs fn(index) alone on its thread until it returns,
// with the floating-point environment and signal mask of the thread that launched (which is what a new thread
// inherits), and nothing but its own stack and thread-locals is shared between two tasks of different launches.
// Index 0 runs on the calling thread (which would otherwise only wait), the others on pool threads, so a launch
// that needs one task needs no hand-off at all, and a launch spread over every CPU is not one thread more than
// the CPUs there are (the worker pool of the NVIDIA engines, vgpu/exec/worker_pool.hpp, leaves the caller waiting:
// with the CPUs full, that thread's polling costs a work-group thread its CPU for a time slice, 35% of a launch of
// 1024 small work-groups on 8 CPUs, measured). fn must not throw (the callers catch what a work-group throws and
// report it themselves); one that does ends the process, as it did when it was a std::thread's body.
//
// A worker that has just finished a task spins for a few microseconds before it sleeps, so that launches
// following one another closely (a stream of many-group kernels) find it awake, and the caller does the same
// for the last task. Both give the processor back to the system at once when a task takes longer than that.
#pragma once

#include <atomic>
#include <cfenv>
#include <csignal>
#include <chrono>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <system_error>
#include <thread>
#include <type_traits>
#include <vector>

#include <pthread.h>

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace vgpu::amd::pool {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#endif
}

// How long a thread spins before it sleeps, in loop turns (a turn is a pause, some tens of nanoseconds on x86).
// VGPU_POOL_SPIN=0 turns spinning off.
inline unsigned spin_turns() {
  static const unsigned n = [] {
    const char* v = std::getenv("VGPU_POOL_SPIN");
    return v && *v ? static_cast<unsigned>(std::atoi(v)) : 400u;
  }();
  return n;
}

class Pool {
 public:
  using Fn = void (*)(void* ctx, unsigned index);

  // Runs fn(ctx, 0) .. fn(ctx, n - 1), each on a thread of its own, at once; returns when all are done.
  //
  // Where the host will not start a thread (std::system_error: EAGAIN at a process or user limit, or no address space
  // left) there are two ways to go on. Degrade (the default; for tasks that are independent of one another): the tasks
  // that got no thread run on the calling thread, one after another, once its own is done, so the launch finishes on
  // fewer threads with the bits it would have had. Strict (for tasks that wait on one another's progress, so one that
  // is not running at the same time as the rest could never finish): nothing is started, and the system_error is thrown
  // for the caller to report. (A task is never half-started: the threads are all found, or made, before any task begins.)
  enum class Refused { Degrade, Strict };
  void run(unsigned n, Fn fn, void* ctx, Refused refused = Refused::Degrade) {
    if (n == 0) return;
    if (n == 1) {
      fn(ctx, 0);
      return;
    }
    auto join = std::make_shared<Join>();
    join->pending.store(n - 1, std::memory_order_relaxed);
    std::fegetenv(&join->env);
    pthread_sigmask(SIG_BLOCK, nullptr, &join->mask);
    Worker* taken[kStack];
    std::vector<Worker*> more;
    Worker** w = taken;
    if (n - 1 > kStack) {
      more.resize(n - 1);
      w = more.data();
    }
    unsigned have = 0;
    {
      std::lock_guard<std::mutex> lock(mu_);
      while (have < n - 1 && !idle_.empty()) {
        w[have++] = idle_.back();
        idle_.pop_back();
      }
    }
    try {
      for (; have < n - 1; ++have) w[have] = make_worker();
    } catch (const std::system_error&) {
      if (refused == Refused::Strict) {
        std::lock_guard<std::mutex> lock(mu_);
        for (unsigned i = 0; i < have; ++i) idle_.push_back(w[i]);
        throw;
      }
    }
    // (`have` threads in all, the first n - 1 when none was refused.)
    join->pending.store(have, std::memory_order_relaxed);
    for (unsigned i = 0; i < have; ++i) w[i]->start(Task{fn, ctx, i + 1, join});
    fn(ctx, 0);
    for (unsigned i = have + 1; i < n; ++i) fn(ctx, i);   // (those the host would not give a thread)
    // The others: spin a little, then sleep on the counter.
    for (unsigned t = spin_turns(); t && join->pending.load(std::memory_order_acquire) != 0; --t) cpu_relax();
    for (uint32_t p; (p = join->pending.load(std::memory_order_acquire)) != 0;) join->pending.wait(p, std::memory_order_acquire);
  }

 private:
  static constexpr unsigned kStack = 64;
  struct Join {
    std::atomic<uint32_t> pending{0};
    std::fenv_t env;   // the launching thread's floating-point environment...
    sigset_t mask;     // ...and signal mask
  };
  struct Task {
    Fn fn = nullptr;
    void* ctx = nullptr;
    unsigned index = 0;
    std::shared_ptr<Join> join;
  };
  struct Worker {
    explicit Worker(Pool* p) : pool(p) {}
    Pool* pool;
    std::atomic<uint32_t> go{0};   // 1: a task is in `task`; 2: stop
    Task task;
    std::thread thread;
    void start(Task t) {
      task = std::move(t);
      go.store(1, std::memory_order_release);
      go.notify_one();
    }
    void loop() {
      sigset_t current;
      pthread_sigmask(SIG_BLOCK, nullptr, &current);   // what this thread was created with
      for (;;) {
        for (unsigned t = spin_turns(); t && go.load(std::memory_order_acquire) == 0; --t) cpu_relax();
        for (uint32_t g; (g = go.load(std::memory_order_acquire)) == 0;) go.wait(0, std::memory_order_acquire);
        if (go.load(std::memory_order_acquire) == 2) return;
        Task t = std::move(task);
        go.store(0, std::memory_order_relaxed);
        std::fesetenv(&t.join->env);
        if (std::memcmp(&current, &t.join->mask, sizeof current) != 0) {
          pthread_sigmask(SIG_SETMASK, &t.join->mask, nullptr);
          current = t.join->mask;
        }
        t.fn(t.ctx, t.index);
        // Back to the idle list before the caller is told, so that the next launch finds this thread there.
        pool->give_back(this);
        std::shared_ptr<Join> join = std::move(t.join);
        if (join->pending.fetch_sub(1, std::memory_order_acq_rel) == 1) join->pending.notify_all();
      }
    }
  };

  Worker* make_worker() {
    auto w = std::make_unique<Worker>(this);
    Worker* raw = w.get();
    raw->thread = std::thread([raw] { raw->loop(); });
    raw->thread.detach();   // never joined: the pool lives to the end of the process
    std::lock_guard<std::mutex> lock(mu_);
    all_.push_back(std::move(w));
    return raw;
  }
  void give_back(Worker* w) {
    std::lock_guard<std::mutex> lock(mu_);
    idle_.push_back(w);
  }

  std::mutex mu_;
  std::vector<Worker*> idle_;
  std::vector<std::unique_ptr<Worker>> all_;
};

// The process's pool. Never destroyed (its threads are still running at exit), and made afresh in a forked child,
// which has none of its parent's threads (the old one is left as it is: a lock in it may be held by a thread that
// does not exist there).
inline std::atomic<Pool*>& instance_slot() {
  static std::atomic<Pool*> p{nullptr};
  return p;
}
inline Pool& instance() {
  static const bool atfork = [] {
    pthread_atfork(nullptr, nullptr, [] { instance_slot().store(nullptr); });
    return true;
  }();
  (void)atfork;
  std::atomic<Pool*>& slot = instance_slot();
  Pool* p = slot.load(std::memory_order_acquire);
  if (!p) {
    Pool* fresh = new Pool;
    if (slot.compare_exchange_strong(p, fresh)) p = fresh;
    else delete fresh;
  }
  return *p;
}

// fn(0) .. fn(n - 1), each on a thread of its own at once, the first on the calling thread. fn is any callable of
// an unsigned index that does not throw.
// Where the host will not start all the threads, the tasks without one run here in turn (see Pool::run).
template <class F>
void run(unsigned n, F&& fn) {
  struct Ctx {
    std::remove_reference_t<F>* f;
  } ctx{&fn};
  instance().run(
      n, [](void* c, unsigned i) { (*static_cast<Ctx*>(c)->f)(i); }, &ctx);
}

// The same for tasks that wait on one another: throws std::system_error, with no task started, where the host would not
// start a thread for each.
template <class F>
void run_together(unsigned n, F&& fn) {
  struct Ctx {
    std::remove_reference_t<F>* f;
  } ctx{&fn};
  instance().run(
      n, [](void* c, unsigned i) { (*static_cast<Ctx*>(c)->f)(i); }, &ctx, Pool::Refused::Strict);
}

}  // namespace vgpu::amd::pool
