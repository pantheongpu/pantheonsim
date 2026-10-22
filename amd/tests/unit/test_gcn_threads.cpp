// The host threads a dispatch's work-groups run on (amd/src/gcn_pool.hpp): kept between dispatches, not started for each.
// What a dispatch computes, writes and counts must not depend on how many threads it was spread over, how many dispatches
// came before it, or how many callers launch at once; and the tasks of one launch still run at the same time, each on a
// thread of its own, since work-groups may wait on one another.
//
// The kernels are amd/tests/data/vector_add.c (one element a work-item) and its LDS reduction (a barrier, one result
// a work-group), compiled for the target in VGPU_TEST_TARGET.
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cfenv>
#include <csignal>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

#include "../../src/gcn_pool.hpp"
#include "vgpu/amd_codeobject.hpp"
#include "vgpu/amd_exec.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {

amd::CodeObject object() {
  const std::string path = std::string(VGPU_SOURCE_DIR) + "/amd/tests/data/vector_add." + vtest::amd_target() + ".o";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw vtest::Failure("no code object at " + path);
  return amd::load_code_object(std::string((std::istreambuf_iterator<char>(in)), {}), path);
}

void set_threads(unsigned n) { setenv("VGPU_THREADS", std::to_string(n).c_str(), 1); }

struct Result {
  std::vector<uint8_t> memory;   // every output byte
  amd::DispatchStats stats;
};

uint64_t put(MemoryManager& mem, const void* p, size_t bytes) {
  const uint64_t a = mem.alloc(bytes ? bytes : 1);
  if (bytes) mem.write(a, p, bytes);
  return a;
}

// One dispatch of `kernel` over `groups` work-groups of 256 work-items, on as many threads as VGPU_THREADS says (set
// by the caller: setenv is not for threads to race); its outputs and counts.
Result dispatch(const amd::CodeObject& o, const char* kernel, uint32_t groups, uint32_t seed) {
  const amd::Kernel* k = amd::find_kernel(o, kernel);
  if (!k) throw vtest::Failure(std::string("no kernel named ") + kernel);
  MemoryManager mem(256ull << 20);
  const uint32_t n = groups * 256;
  std::vector<float> a(n), b(n), in(n);
  uint32_t x = seed * 2654435761u + 1;
  for (uint32_t i = 0; i < n; ++i) {
    x = x * 1664525u + 1013904223u;
    // whole numbers below 2^10: every sum in the reduction is exact in any order, so a bad split cannot hide in rounding
    a[i] = static_cast<float>((x >> 8) & 1023);
    b[i] = static_cast<float>((x >> 18) & 1023);
    in[i] = a[i];
  }
  const bool reduce = std::strcmp(kernel, "reduce_sum") == 0;
  const uint64_t pa = put(mem, reduce ? in.data() : a.data(), n * 4ull), pb = put(mem, b.data(), n * 4ull);
  const uint64_t out = mem.alloc(n * 4ull);
  std::vector<uint8_t> zeros(n * 4ull, 0);
  mem.write(out, zeros.data(), zeros.size());
  std::vector<uint8_t> args(k->kernarg_size, 0);
  const auto set64 = [&](size_t at, uint64_t v) { std::memcpy(&args[at], &v, 8); };
  const auto set32 = [&](size_t at, uint32_t v) { std::memcpy(&args[at], &v, 4); };
  if (reduce) {   // (in, out, n, scale)
    set64(k->args[0].offset, pa);
    set64(k->args[1].offset, out);
    set32(k->args[2].offset, n);
    const float one = 1.0f;
    std::memcpy(&args[k->args[3].offset], &one, 4);
  } else {   // (a, b, out, n)
    set64(k->args[0].offset, pa);
    set64(k->args[1].offset, pb);
    set64(k->args[2].offset, out);
    set32(k->args[3].offset, n);
  }
  amd::Dispatch d;
  d.object = &o;
  d.kernel = k;
  d.kernarg = put(mem, args.data(), args.size());
  d.groups[0] = groups;
  d.group_size[0] = 256;
  Result r;
  r.stats = amd::execute(d, mem);
  r.memory.resize(n * 4ull);
  mem.read(out, r.memory.data(), r.memory.size());
  return r;
}

bool same(const Result& x, const Result& y) {
  return x.memory == y.memory && x.stats.waves == y.stats.waves && x.stats.instructions == y.stats.instructions &&
         x.stats.barriers == y.stats.barriers && x.stats.counts.valu == y.stats.counts.valu &&
         x.stats.counts.salu == y.stats.counts.salu && x.stats.counts.vmem == y.stats.counts.vmem &&
         x.stats.counts.lds == y.stats.counts.lds && x.stats.counts.branch == y.stats.counts.branch &&
         x.stats.counts.vmem_rd == y.stats.counts.vmem_rd && x.stats.counts.vmem_wr == y.stats.counts.vmem_wr &&
         std::memcmp(x.stats.counts.mix, y.stats.counts.mix, sizeof x.stats.counts.mix) == 0;
}

}  // namespace

// The tasks of one launch run at once, one to a thread, and each exactly once: n tasks that wait for each other
// (every one spins until all have started) finish only if they really are n threads.
VTEST(a_launch_runs_every_task_once_and_all_at_the_same_time) {
  for (unsigned round = 0; round < 300; ++round) {
    const unsigned n = 1 + round % 40;
    std::atomic<unsigned> started{0};
    std::vector<std::atomic<unsigned>> ran(n);
    for (auto& r : ran) r = 0;
    std::atomic<bool> timed_out{false};
    amd::pool::run(n, [&](unsigned i) {
      ++ran[i];
      ++started;
      const auto limit = std::chrono::steady_clock::now() + std::chrono::seconds(20);
      while (started.load() < n)
        if (std::chrono::steady_clock::now() > limit) {
          timed_out = true;
          break;
        }
    });
    VCHECK(!timed_out.load());
    for (unsigned i = 0; i < n; ++i) VCHECK_EQ(ran[i].load(), 1u);
  }
}

// The first task runs on the calling thread; the rest do not.
VTEST(the_first_task_runs_on_the_calling_thread) {
  const auto self = std::this_thread::get_id();
  std::vector<std::thread::id> ids(6);
  amd::pool::run(6, [&](unsigned i) { ids[i] = std::this_thread::get_id(); });
  VCHECK(ids[0] == self);
  for (unsigned i = 1; i < 6; ++i) VCHECK(ids[i] != self);
}

// Several callers at once, each with launches of its own, as streams have: every task of every launch runs once.
VTEST(launches_from_several_threads_at_once_do_not_mix_their_tasks) {
  std::atomic<unsigned> bad{0};
  std::vector<std::thread> callers;
  for (unsigned c = 0; c < 6; ++c)
    callers.emplace_back([&, c] {
      for (unsigned round = 0; round < 400; ++round) {
        const unsigned n = 1 + (round * 7 + c) % 12;
        std::vector<unsigned> mine(n, 0);
        amd::pool::run(n, [&](unsigned i) { mine[i] += 1 + c; });
        for (unsigned i = 0; i < n; ++i)
          if (mine[i] != 1 + c) ++bad;
      }
    });
  for (auto& t : callers) t.join();
  VCHECK_EQ(bad.load(), 0u);
}

// A task starts with the floating-point environment and signal mask of the thread that launched, whatever the worker
// had before: what a thread made for the launch would have had.
VTEST(a_task_starts_with_the_launching_threads_float_environment_and_signal_mask) {
  for (int round = 0; round < 6; ++round) {
    const int mode = round % 2 ? FE_UPWARD : FE_TONEAREST;
    std::fesetround(mode);
    sigset_t mine;
    sigemptyset(&mine);
    if (round % 3 == 1) sigaddset(&mine, SIGUSR2);
    pthread_sigmask(SIG_SETMASK, &mine, nullptr);
    std::atomic<unsigned> wrong{0};
    amd::pool::run(5, [&](unsigned) {
      if (std::fegetround() != mode) ++wrong;
      sigset_t now;
      pthread_sigmask(SIG_BLOCK, nullptr, &now);
      if (sigismember(&now, SIGUSR2) != sigismember(&mine, SIGUSR2)) ++wrong;
    });
    VCHECK_EQ(wrong.load(), 0u);
  }
  std::fesetround(FE_TONEAREST);
  sigset_t none;
  sigemptyset(&none);
  pthread_sigmask(SIG_SETMASK, &none, nullptr);
}

// A forked child has none of its parent's threads; its first launch makes its own.
VTEST(a_forked_child_launches_on_threads_of_its_own) {
#if defined(__SANITIZE_THREAD__)
  // ThreadSanitizer does not support a child that starts threads after a fork of a process that has them ("starting new threads after
  // multi-threaded fork is not supported. Dying"): that is the case this test makes. Under it, the case is left to the other builds.
  std::printf("SKIP: ThreadSanitizer does not support starting threads in a forked child\n");
  return;
#endif
  std::atomic<unsigned> sum{0};
  amd::pool::run(4, [&](unsigned i) { sum += i; });   // (the parent's pool exists)
  VCHECK_EQ(sum.load(), 6u);
  const pid_t pid = fork();
  if (pid == 0) {
    std::atomic<unsigned> s{0};
    for (int r = 0; r < 20; ++r) amd::pool::run(5, [&](unsigned i) { s += i + 1; });
    _exit(s.load() == 20u * 15u ? 0 : 1);
  }
  int status = 0;
  VCHECK(waitpid(pid, &status, 0) == pid);
  VCHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

// What a dispatch writes and counts does not depend on its thread count.
VTEST(a_dispatch_gives_the_same_memory_and_counts_on_any_number_of_threads) {
  const amd::CodeObject o = object();
  for (const char* kernel : {"vector_add", "reduce_sum"})
    for (uint32_t groups : {1u, 2u, 7u, 64u, 513u, 3000u}) {
      set_threads(1);
      const Result serial = dispatch(o, kernel, groups, groups);
      for (unsigned threads : {2u, 3u, 5u, 8u, 17u, 32u}) {
        set_threads(threads);
        const Result r = dispatch(o, kernel, groups, groups);
        if (!same(serial, r))
          throw vtest::Failure(std::string(kernel) + " over " + std::to_string(groups) + " groups on " +
                               std::to_string(threads) + " threads differs from one thread");
      }
    }
}

// The same, a long run of dispatches one after another (the pool's threads reused each time, with grids of every
// size), and from several threads at once.
VTEST(many_dispatches_in_a_row_and_side_by_side_match_the_serial_run) {
  const amd::CodeObject o = object();
  std::vector<Result> serial;
  std::vector<uint32_t> sizes;
  for (uint32_t i = 0; i < 60; ++i) sizes.push_back(1 + (i * 37) % 900);
  set_threads(1);
  for (uint32_t i = 0; i < sizes.size(); ++i) serial.push_back(dispatch(o, i % 2 ? "reduce_sum" : "vector_add", sizes[i], i));
  set_threads(6);   // (the callers below share it: setenv is the process's)
  std::atomic<unsigned> bad{0};
  std::vector<std::thread> callers;
  for (unsigned c = 0; c < 4; ++c)
    callers.emplace_back([&] {
      for (uint32_t i = 0; i < sizes.size(); ++i) {
        const Result r = dispatch(o, i % 2 ? "reduce_sum" : "vector_add", sizes[i], i);
        if (!same(serial[i], r)) ++bad;
      }
    });
  for (auto& t : callers) t.join();
  VCHECK_EQ(bad.load(), 0u);
}

// A dispatch that fails -- here a store past the end of the output -- throws on any number of threads, the same kind of
// error, and the threads are good for the next dispatch.
VTEST(a_failing_dispatch_throws_on_any_number_of_threads_and_the_next_one_runs) {
  const amd::CodeObject o = object();
  const amd::Kernel* k = amd::find_kernel(o, "vector_add");
  VCHECK(k != nullptr);
  for (unsigned threads : {1u, 2u, 8u}) {
    set_threads(threads);
    MemoryManager mem(64ull << 20);
    const uint32_t groups = 64, n = groups * 256;
    std::vector<float> a(n, 1.0f), b(n, 2.0f);
    const uint64_t pa = put(mem, a.data(), n * 4ull), pb = put(mem, b.data(), n * 4ull);
    const uint64_t out = mem.alloc(n * 2ull);   // half of what the kernel writes
    std::vector<uint8_t> args(k->kernarg_size, 0);
    const auto set64 = [&](size_t at, uint64_t v) { std::memcpy(&args[at], &v, 8); };
    set64(k->args[0].offset, pa);
    set64(k->args[1].offset, pb);
    set64(k->args[2].offset, out);
    std::memcpy(&args[k->args[3].offset], &n, 4);
    amd::Dispatch d;
    d.object = &o;
    d.kernel = k;
    d.kernarg = put(mem, args.data(), args.size());
    d.groups[0] = groups;
    d.group_size[0] = 256;
    bool threw = false;
    try {
      amd::execute(d, mem);
    } catch (const std::exception&) {
      threw = true;
    }
    VCHECK(threw);
    // and the next, a good one, on the same threads
    set_threads(threads);
    const Result good = dispatch(o, "vector_add", 33, 7);
    set_threads(1);
    const Result serial = dispatch(o, "vector_add", 33, 7);
    VCHECK(same(good, serial));
  }
}

VTEST_MAIN
