// How a launch spreads its blocks over host threads, checked at the launch
// level: the same kernel at VGPU_THREADS=1, 2 and many gives the same bytes and
// the same counters; many small launches reuse the pool's threads; launches
// from several host threads at once, a kernel that launches a child grid, and
// a cooperative launch each come out right. (The pool itself is in
// test_worker_pool.cpp.)
#include <sched.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/worker_pool.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;
using vgpu::exec::LaunchStats;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
std::vector<uint8_t> arg_u32(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

const char* kHeader = ".version 8.3\n.target sm_90\n.address_size 64\n";

// out[gid] = 3*gid + 1 for gid < n; sum += gid; and the block's shared memory
// is used (a barrier, a reverse read) so blocks of several warps are real work.
const char* kFill = R"(
.visible .entry fill(.param .u64 out, .param .u64 sum, .param .u32 n)
{
    .reg .pred %p<3>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<10>;
    .shared .align 4 .b8 tile[1024];
    ld.param.u64 %rd1, [out];
    ld.param.u64 %rd2, [sum];
    ld.param.u32 %r1, [n];
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.u32 %r5, %r2, %r3, %r4;
    mad.lo.u32 %r6, %r5, 3, 1;
    mul.wide.u32 %rd3, %r4, 4;
    mov.u64 %rd4, tile;
    add.u64 %rd5, %rd4, %rd3;
    st.shared.u32 [%rd5], %r6;
    bar.sync 0;
    sub.u32 %r7, %r3, %r4;
    sub.u32 %r7, %r7, 1;
    mul.wide.u32 %rd6, %r7, 4;
    add.u64 %rd7, %rd4, %rd6;
    ld.shared.u32 %r8, [%rd7];
    setp.ge.u32 %p1, %r5, %r1;
    @%p1 bra DONE;
    mul.wide.u32 %rd8, %r5, 4;
    add.u64 %rd9, %rd1, %rd8;
    st.global.u32 [%rd9], %r6;
    atom.global.add.u32 %r9, [%rd2], %r5;
    atom.global.add.u32 %r10, [%rd2+4], %r8;
DONE:
    ret;
}
)";

struct Result {
  std::vector<uint32_t> out;
  uint32_t sum = 0, sum2 = 0;
  LaunchStats st;
};

// One launch of `fill` over `blocks` blocks of `threads` at the given
// VGPU_THREADS ("" leaves it unset: the default).
Result run_fill(const ptx::Module& m, const DeviceProfile& prof, const char* host_threads, uint32_t blocks,
                uint32_t threads) {
  if (*host_threads) ::setenv("VGPU_THREADS", host_threads, 1);
  else ::unsetenv("VGPU_THREADS");
  const uint32_t n = blocks * threads - 5;   // a ragged tail
  MemoryManager mem{1 << 24};
  const uint64_t out = mem.alloc(size_t{n} * 4);
  const uint64_t sum = mem.alloc(8);
  mem.store_scalar(sum, 4, 0);
  mem.store_scalar(sum + 4, 4, 0);
  LaunchConfig cfg;
  cfg.grid = {blocks, 1, 1};
  cfg.block = {threads, 1, 1};
  Result r;
  r.st = exec::launch(m.entries[0], cfg, {arg_u64(out), arg_u64(sum), arg_u32(n)}, mem, prof);
  r.out.resize(n);
  mem.read(out, r.out.data(), size_t{n} * 4);
  r.sum = static_cast<uint32_t>(mem.load_scalar(sum, 4));
  r.sum2 = static_cast<uint32_t>(mem.load_scalar(sum + 4, 4));
  ::unsetenv("VGPU_THREADS");
  return r;
}

bool same_counters(const LaunchStats& a, const LaunchStats& b) {
  if (a.inst_by_opcode != b.inst_by_opcode) return false;
  return std::memcmp(&a, &b, exec::kCounterWords * sizeof(uint64_t)) == 0;
}

}  // namespace

VTEST(a_grid_gives_the_same_bytes_and_counters_at_every_host_thread_count) {
  const auto m = ptx::parse(std::string(kHeader) + kFill);
  const DeviceProfile prof = load_gpu("nvidia/h100");
  for (uint32_t threads : {32u, 96u, 256u}) {
    const Result one = run_fill(m, prof, "1", 53, threads);
    VCHECK(one.st.instructions > 0);
    VCHECK_EQ(one.st.blocks, uint64_t{53});
    for (uint32_t i = 0; i < one.out.size(); ++i) VCHECK_EQ(one.out[i], 3 * i + 1);
    for (const char* t : {"2", "3", "7", "64", ""}) {
      const Result r = run_fill(m, prof, t, 53, threads);
      VCHECK(r.out == one.out);
      VCHECK_EQ(r.sum, one.sum);
      VCHECK_EQ(r.sum2, one.sum2);
      VCHECK(same_counters(r.st, one.st));
    }
  }
}

VTEST(many_small_launches_reuse_the_pools_threads) {
  const auto m = ptx::parse(std::string(kHeader) + kFill);
  const DeviceProfile prof = load_gpu("nvidia/h100");
  run_fill(m, prof, "4", 16, 32);   // makes the threads
  const size_t created = exec::worker_pool_stats().created;
  VCHECK(created >= 4);
  const Result first = run_fill(m, prof, "4", 16, 32);
  for (int i = 0; i < 300; ++i) {
    const Result r = run_fill(m, prof, "4", 16, 32);
    VCHECK(r.out == first.out);
    VCHECK(same_counters(r.st, first.st));
  }
  VCHECK_EQ(exec::worker_pool_stats().created, created);
  // A launch asking for more threads grows the pool by the difference and
  // then reuses that too.
  run_fill(m, prof, "6", 16, 32);
  const size_t grown = exec::worker_pool_stats().created;
  VCHECK(grown >= created);
  for (int i = 0; i < 20; ++i) run_fill(m, prof, "6", 16, 32);
  VCHECK_EQ(exec::worker_pool_stats().created, grown);
}

// A loop of small launches (the launch64 shape: few blocks, a few microseconds of
// work each) must give the same bytes and the same counters at every thread
// count, and once the cost of the kernel is known must stay off the pool: a
// worker's hand-off costs more than the blocks do (docs/performance.md, Round sp12).
VTEST(small_launches_in_a_loop_stay_off_the_pool_and_give_the_same_results) {
  const auto m = ptx::parse(std::string(kHeader) + kFill);
  const DeviceProfile prof = load_gpu("nvidia/h100");
  // Any launch this small is below the 1 s of work a thread would need.
  ::setenv("VGPU_POOL_MIN_WORK_US", "1000000", 1);
  const Result ref = run_fill(m, prof, "1", 8, 32);
  for (const char* t : {"2", "8", "16"}) {
    run_fill(m, prof, t, 8, 32);   // the first launch of a kernel has no cost to go by
    run_fill(m, prof, t, 8, 32);
    const uint64_t calls = exec::worker_pool_stats().calls;
    LaunchStats total;
    for (int i = 0; i < 200; ++i) {
      const Result r = run_fill(m, prof, t, 8, 32);
      VCHECK(r.out == ref.out);
      VCHECK_EQ(r.sum, ref.sum);
      VCHECK_EQ(r.sum2, ref.sum2);
      VCHECK(same_counters(r.st, ref.st));
    }
    VCHECK_EQ(exec::worker_pool_stats().calls, calls);   // none went to the pool
  }
  // With the cap off the same loop does use the pool, so the check above is not vacuous.
  ::setenv("VGPU_POOL_MIN_WORK_US", "0", 1);
  const uint64_t calls = exec::worker_pool_stats().calls;
  for (int i = 0; i < 5; ++i) {
    const Result r = run_fill(m, prof, "4", 8, 32);
    VCHECK(r.out == ref.out);
    VCHECK(same_counters(r.st, ref.st));
  }
  VCHECK(exec::worker_pool_stats().calls >= calls + 5);
  // A grid with plenty of work is not capped: a 1 us threshold lets 8 blocks have 4 threads.
  ::setenv("VGPU_POOL_MIN_WORK_US", "1", 1);
  run_fill(m, prof, "4", 8, 32);
  const uint64_t before = exec::worker_pool_stats().calls;
  for (int i = 0; i < 5; ++i) run_fill(m, prof, "4", 8, 32);
  VCHECK(exec::worker_pool_stats().calls >= before + 5);
  ::unsetenv("VGPU_POOL_MIN_WORK_US");
}

// ThreadSanitizer refuses to start threads in a fork child of a process that had them (the pool has some by now).
// AddressSanitizer makes a block cost more than the pool's break-even (100 us), so the launches below are genuinely worth the pool there.
#if defined(__SANITIZE_THREAD__) || defined(__SANITIZE_ADDRESS__)
#define VGPU_SCHED_TSAN 1
#elif defined(__has_feature)
#if __has_feature(thread_sanitizer) || __has_feature(address_sanitizer)
#define VGPU_SCHED_TSAN 1
#endif
#endif

#ifndef VGPU_SCHED_TSAN
// The pool can be slow for a reason that is not the kernel: here four threads on one CPU, so each hand-off waits for a
// time slice. A launch that took long because of that must not be taken for one with much work, or it stays on the pool
// and stays slow for good (the launch64 loop at VGPU_THREADS=4 took 10 times as long as at 1, every launch on the pool).
// The cost is taken from the time threads spent on blocks. A child process, so the pool's threads are born on the one CPU.
VTEST(a_pool_slowed_by_a_crowded_cpu_does_not_keep_small_launches_on_it) {
  const pid_t pid = fork();
  VCHECK(pid >= 0);
  if (pid == 0) {
    alarm(120);
    int bad = 0;
    cpu_set_t all;
    CPU_ZERO(&all);
    if (sched_getaffinity(0, sizeof all, &all) != 0) _exit(2);
    cpu_set_t one;
    CPU_ZERO(&one);
    for (int c = 0; c < CPU_SETSIZE; ++c)
      if (CPU_ISSET(c, &all)) { CPU_SET(c, &one); break; }
    if (sched_setaffinity(0, sizeof one, &one) != 0) _exit(2);
    ::unsetenv("VGPU_POOL_MIN_WORK_US");
    const auto m = ptx::parse(std::string(kHeader) + kFill);
    const DeviceProfile prof = load_gpu("nvidia/h100");
    const Result ref = run_fill(m, prof, "1", 8, 32);
    run_fill(m, prof, "4", 8, 32);   // the first launch of a kernel has no cost to go by
    run_fill(m, prof, "4", 8, 32);
    const uint64_t calls = exec::worker_pool_stats().calls;
    for (int i = 0; i < 400; ++i) {
      const Result r = run_fill(m, prof, "4", 8, 32);
      bad += r.out != ref.out || !same_counters(r.st, ref.st);
    }
    // Some launches may still go to the pool (a cost read while the process was preempted); not most of them.
    bad += exec::worker_pool_stats().calls - calls > 40;
    _exit(bad ? 1 : 0);
  }
  int status = 0;
  VCHECK_EQ(waitpid(pid, &status, 0), pid);
  VCHECK(WIFEXITED(status));
  VCHECK_EQ(WEXITSTATUS(status), 0);
}
#endif

VTEST(one_thread_runs_inline_and_makes_no_pool_thread) {
  const auto m = ptx::parse(std::string(kHeader) + kFill);
  const DeviceProfile prof = load_gpu("nvidia/h100");
  const size_t created = exec::worker_pool_stats().created;
  const Result r = run_fill(m, prof, "1", 40, 64);
  VCHECK_EQ(r.st.blocks, uint64_t{40});
  VCHECK_EQ(exec::worker_pool_stats().created, created);
}

VTEST(launches_from_several_host_threads_at_once) {
  const auto m = ptx::parse(std::string(kHeader) + kFill);
  const DeviceProfile prof = load_gpu("nvidia/h100");
  ::setenv("VGPU_THREADS", "3", 1);
  const Result want = run_fill(m, prof, "3", 21, 64);
  ::setenv("VGPU_THREADS", "3", 1);
  std::atomic<int> bad{0};
  std::vector<std::thread> hosts;
  for (int h = 0; h < 5; ++h)
    hosts.emplace_back([&] {
      for (int i = 0; i < 40; ++i) {
        // Each host thread has a device of its own, as separate processes'
        // would; they share the pool, which is what is under test.
        MemoryManager mem{1 << 22};
        const uint32_t n = 21 * 64 - 5;
        const uint64_t out = mem.alloc(size_t{n} * 4);
        const uint64_t sum = mem.alloc(8);
        mem.store_scalar(sum, 4, 0);
        mem.store_scalar(sum + 4, 4, 0);
        LaunchConfig cfg;
        cfg.grid = {21, 1, 1};
        cfg.block = {64, 1, 1};
        const LaunchStats st = exec::launch(m.entries[0], cfg, {arg_u64(out), arg_u64(sum), arg_u32(n)}, mem, prof);
        std::vector<uint32_t> got(n);
        mem.read(out, got.data(), size_t{n} * 4);
        if (got != want.out || mem.load_scalar(sum, 4) != want.sum || !same_counters(st, want.st)) bad.fetch_add(1);
      }
    });
  for (auto& t : hosts) t.join();
  ::unsetenv("VGPU_THREADS");
  VCHECK_EQ(bad.load(), 0);
}

// A kernel that launches a child grid of many blocks; the child grid runs after
// the parent's, on the pool, when the parent's launch is completing.
VTEST(a_child_grid_launched_from_a_kernel_runs_on_the_pool) {
  const std::string decls = R"(
.extern .func (.param .b64 func_retval0) __cudaCDP2GetParameterBufferV2
(.param .b64 f, .param .align 4 .b8 g[12], .param .align 4 .b8 b[12], .param .b32 s);
.extern .func (.param .b32 func_retval0) __cudaCDP2LaunchDeviceV2
(.param .b64 buf, .param .b64 stream);
)";
  const std::string src = std::string(kHeader) + decls + R"(
.visible .entry child(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mov.u32 %r3, %tid.x;
    mad.lo.u32 %r4, %r1, %r2, %r3;
    mul.wide.u32 %rd2, %r4, 4;
    add.u64 %rd3, %rd1, %rd2;
    add.u32 %r5, %r4, 7;
    st.global.u32 [%rd3], %r5;
    ret;
}
.visible .entry parent(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    setp.ne.u32 %p1, %r1, 0;
    @%p1 bra DONE;
    mov.u64 %rd19, child;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd19;
    .param .align 4 .b8 param1[12];
    st.param.b32 [param1+0], 48;
    st.param.b32 [param1+4], 1;
    st.param.b32 [param1+8], 1;
    .param .align 4 .b8 param2[12];
    st.param.b32 [param2+0], 32;
    st.param.b32 [param2+4], 1;
    st.param.b32 [param2+8], 1;
    .param .b32 param3;
    st.param.b32 [param3+0], 0;
    .param .b64 retval0;
    call.uni (retval0), __cudaCDP2GetParameterBufferV2, (param0, param1, param2, param3);
    ld.param.b64 %rd20, [retval0+0];
    }
    st.u64 [%rd20], %rd1;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd20;
    .param .b64 param1;
    st.param.b64 [param1+0], 0;
    .param .b32 retval0;
    call.uni (retval0), __cudaCDP2LaunchDeviceV2, (param0, param1);
    ld.param.b32 %r22, [retval0+0];
    }
DONE:
    ret;
}
)";
  const auto m = ptx::parse(src);
  std::map<std::string, uint64_t> symbols;
  exec::KernelTable kernels;
  uint64_t va = kKernelVaBase;
  for (const auto& e : m.entries) {
    symbols[e.name] = va;
    va += kKernelVaStride;
  }
  for (const auto& e : m.entries) kernels[symbols[e.name]] = exec::KernelRef{&e, &symbols};
  const ptx::EntryFn* parent = nullptr;
  for (const auto& e : m.entries)
    if (e.name == "parent") parent = &e;
  VCHECK(parent != nullptr);
  const DeviceProfile prof = load_gpu("nvidia/h100");
  std::vector<std::vector<uint32_t>> results;
  for (const char* t : {"1", "4", "9"}) {
    ::setenv("VGPU_THREADS", t, 1);
    MemoryManager mem{1 << 22};
    const uint64_t out = mem.alloc(48 * 32 * 4);
    LaunchConfig c;
    c.block = {32, 1, 1};
    c.kernels = &kernels;
    for (int rep = 0; rep < 5; ++rep) {
      exec::launch(*parent, c, {arg_u64(out)}, mem, prof, &symbols);
      std::vector<uint32_t> got(48 * 32);
      mem.read(out, got.data(), got.size() * 4);
      for (uint32_t i = 0; i < got.size(); ++i) VCHECK_EQ(got[i], i + 7);
      results.push_back(got);
    }
  }
  ::unsetenv("VGPU_THREADS");
  for (const auto& r : results) VCHECK(r == results[0]);
}

// A cooperative launch ignores VGPU_THREADS and runs on the calling thread
// (docs/cooperative.md): the blocks wait on one another, and the order is fixed.
VTEST(a_cooperative_launch_still_runs_on_one_thread) {
  const auto m = ptx::parse(std::string(kHeader) + R"(
.visible .entry k(.param .u64 counter, .param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [counter];
    ld.param.u64 %rd2, [out];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %nctaid.x;
    mul.wide.u32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    add.s32 %r3, %r1, 1;
    st.global.u32 [%rd6], %r3;
    mov.u32 %r4, 1;
    atom.add.release.gpu.global.u32 %r5, [%rd3], %r4;
SPIN:
    ld.acquire.gpu.global.u32 %r6, [%rd3];
    setp.lt.u32 %p1, %r6, %r2;
    @%p1 bra SPIN;
    add.s32 %r7, %r1, 1;
    rem.u32 %r8, %r7, %r2;
    mul.wide.u32 %rd7, %r8, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.u32 %r9, [%rd8];
    add.s32 %r10, %r1, %r2;
    mul.wide.u32 %rd9, %r10, 4;
    add.s64 %rd10, %rd4, %rd9;
    st.global.u32 [%rd10], %r9;
    ret;
}
)");
  const DeviceProfile prof = load_gpu("nvidia/h100");
  ::setenv("VGPU_THREADS", "8", 1);
  const size_t created = exec::worker_pool_stats().created;
  MemoryManager mem{1 << 20};
  const uint64_t counter = mem.alloc(4);
  const uint64_t out = mem.alloc(128);
  mem.store_scalar(counter, 4, 0);
  LaunchConfig cfg;
  cfg.grid = {6, 1, 1};
  cfg.block = {1, 1, 1};
  cfg.cooperative = true;
  exec::launch(m.entries[0], cfg, {arg_u64(counter), arg_u64(out)}, mem, prof);
  ::unsetenv("VGPU_THREADS");
  VCHECK_EQ(mem.load_scalar(counter, 4), uint64_t{6});
  for (uint32_t b = 0; b < 6; ++b) VCHECK_EQ(mem.load_scalar(out + (6 + b) * 4, 4), uint64_t{(b + 1) % 6 + 1});
  VCHECK_EQ(exec::worker_pool_stats().created, created);
}

// A launch that fails reports it, and the pool is fine for the next one.
VTEST(a_failing_grid_reports_its_error_and_the_pool_survives) {
  const auto m = ptx::parse(std::string(kHeader) + R"(
.visible .entry bad(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %ctaid.x;
    setp.eq.u32 %p1, %r1, 5;
    @%p1 st.global.u32 [%rd1+4000000000], %r1;
    ret;
}
)");
  const DeviceProfile prof = load_gpu("nvidia/h100");
  ::setenv("VGPU_THREADS", "4", 1);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(64);
  LaunchConfig cfg;
  cfg.grid = {32, 1, 1};
  cfg.block = {32, 1, 1};
  for (int i = 0; i < 3; ++i) {
    bool threw = false;
    try {
      exec::launch(m.entries[0], cfg, {arg_u64(out)}, mem, prof);
    } catch (const Error&) {
      threw = true;
    }
    VCHECK(threw);
  }
  ::unsetenv("VGPU_THREADS");
  const auto fill = ptx::parse(std::string(kHeader) + kFill);
  const Result r = run_fill(fill, prof, "4", 12, 32);
  for (uint32_t i = 0; i < r.out.size(); ++i) VCHECK_EQ(r.out[i], 3 * i + 1);
}

VTEST_MAIN
