// Tests for the SIMT warp interpreter: arithmetic, divergence, barriers,
// determinism, and diagnostics. This is also where vectorAdd first runs
// end-to-end at the engine level.
#include "vgpu/exec/launch.hpp"

#include <cstring>
#include <fstream>
#include <sstream>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include <cstdlib>
#include <string>
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

std::string read_file(const std::string& path) {
  std::ifstream f(path);
  VCHECK(f.good());
  std::ostringstream os;
  os << f.rdbuf();
  return os.str();
}

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

const char* kDivergePtx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry diverge(.param .u64 in_ptr, .param .u64 out_ptr)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [in_ptr];
    ld.param.u64 %rd2, [out_ptr];
    mov.u32 %r1, %tid.x;
    mul.wide.s32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3;
    add.s64 %rd5, %rd2, %rd3;
    ld.global.u32 %r2, [%rd4];
    setp.lt.s32 %p1, %r1, 16;
    @%p1 bra THEN;
    add.s32 %r3, %r2, 1;
    st.global.u32 [%rd5], %r3;
    bra END;
THEN:
    mul.lo.s32 %r3, %r2, 2;
    st.global.u32 [%rd5], %r3;
END:
    ret;
}
)";

const char* kBarrierPtx = R"(
.version 8.3
.target sm_90
.address_size 64
// out[tid] = buf[(tid+1) % ntid] after all threads wrote buf[tid] = tid.
// Warps run to the barrier in scheduler order, so a correct cross-warp
// bar.sync is required for the neighbor reads to observe the writes.
.visible .entry neighbor(.param .u64 buf_ptr, .param .u64 out_ptr)
{
    .reg .pred %p<2>;
    .reg .b32 %r<6>;
    .reg .b64 %rd<9>;
    ld.param.u64 %rd1, [buf_ptr];
    ld.param.u64 %rd2, [out_ptr];
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ntid.x;
    mul.wide.s32 %rd3, %r1, 4;
    add.s64 %rd4, %rd1, %rd3;
    st.global.u32 [%rd4], %r1;
    bar.sync 0;
    add.s32 %r3, %r1, 1;
    setp.eq.s32 %p1, %r3, %r2;
    @%p1 mov.u32 %r3, 0;
    mul.wide.s32 %rd5, %r3, 4;
    add.s64 %rd6, %rd1, %rd5;
    ld.global.u32 %r4, [%rd6];
    add.s64 %rd7, %rd2, %rd3;
    st.global.u32 [%rd7], %r4;
    ret;
}
)";

struct VecAddRun {
  std::vector<float> result;
  MemoryManager mem{1ull << 32};
};

VecAddRun run_vector_add(uint32_t n, uint32_t block = 256) {
  ptx::Module m = ptx::parse(read_file(VGPU_KERNEL_DIR "/vector_add.ptx"));
  const ptx::EntryFn* fn = m.find_entry("vecAdd");
  VCHECK(fn != nullptr);

  VecAddRun run;
  DeviceProfile prof = load_gpu("nvidia/h100");
  std::vector<float> a(n), b(n);
  for (uint32_t i = 0; i < n; ++i) {
    a[i] = 0.5f * static_cast<float>(i);
    b[i] = 0.25f * static_cast<float>(i);
  }
  uint64_t da = run.mem.alloc(n * 4), db = run.mem.alloc(n * 4), dc = run.mem.alloc(n * 4);
  run.mem.write(da, a.data(), n * 4);
  run.mem.write(db, b.data(), n * 4);

  LaunchConfig cfg;
  cfg.grid = {(n + block - 1) / block, 1, 1};
  cfg.block = {block, 1, 1};
  exec::launch(*fn, cfg, {arg_u64(da), arg_u64(db), arg_u64(dc), arg_u32(n)}, run.mem, prof);

  run.result.resize(n);
  run.mem.read(dc, run.result.data(), n * 4);
  return run;
}

}  // namespace

VTEST(vector_add_end_to_end) {
  const uint32_t n = 1000;  // deliberately not a multiple of 256: exercises divergence + inactive lanes
  auto run = run_vector_add(n);
  for (uint32_t i = 0; i < n; ++i) {
    float expect = 0.5f * static_cast<float>(i) + 0.25f * static_cast<float>(i);
    VCHECK_EQ(run.result[i], expect);  // exact float equality — IEEE add, deterministic
  }
}

VTEST(vector_add_is_deterministic) {
  auto r1 = run_vector_add(777);
  auto r2 = run_vector_add(777);
  VCHECK(std::memcmp(r1.result.data(), r2.result.data(), 777 * 4) == 0);
}

VTEST(vector_add_odd_block_sizes) {
  for (uint32_t block : {1u, 32u, 33u, 100u, 1024u}) {
    auto run = run_vector_add(257, block);
    for (uint32_t i = 0; i < 257; ++i) VCHECK_EQ(run.result[i], 0.75f * static_cast<float>(i));
  }
}

VTEST(divergent_branches_take_both_paths) {
  ptx::Module m = ptx::parse(kDivergePtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  std::vector<uint32_t> in(32);
  for (uint32_t i = 0; i < 32; ++i) in[i] = i + 100;
  uint64_t din = mem.alloc(32 * 4), dout = mem.alloc(32 * 4);
  mem.write(din, in.data(), 32 * 4);

  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(din), arg_u64(dout)}, mem, prof);

  std::vector<uint32_t> out(32);
  mem.read(dout, out.data(), 32 * 4);
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t expect = i < 16 ? (i + 100) * 2 : (i + 100) + 1;
    VCHECK_EQ(out[i], expect);
  }
}

VTEST(barrier_synchronizes_warps) {
  ptx::Module m = ptx::parse(kBarrierPtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  const uint32_t nthreads = 96;  // three warps
  uint64_t dbuf = mem.alloc(nthreads * 4), dout = mem.alloc(nthreads * 4);

  LaunchConfig cfg;
  cfg.block = {nthreads, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(dbuf), arg_u64(dout)}, mem, prof);

  std::vector<uint32_t> out(nthreads);
  mem.read(dout, out.data(), nthreads * 4);
  for (uint32_t i = 0; i < nthreads; ++i) VCHECK_EQ(out[i], (i + 1) % nthreads);
}

VTEST(multi_dimensional_grid) {
  // c[i] = a[i] + b[i] with a 2x2 grid and 2D blocks still covers every index
  // exactly once via ctaid/ntid math? vectorAdd is 1D; instead verify grid.y/z
  // and tid.y/z sregs directly with a small custom kernel.
  const char* ptx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry writeids(.param .u64 out_ptr)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out_ptr];
    mov.u32 %r1, %ctaid.y;
    mov.u32 %r2, %ntid.y;
    mov.u32 %r3, %tid.y;
    mad.lo.s32 %r4, %r1, %r2, %r3;    // global y index
    mov.u32 %r5, %nctaid.y;           // (unused, exercises sreg)
    mul.wide.s32 %rd2, %r4, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r4;
    ret;
}
)";
  ptx::Module m = ptx::parse(ptx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  uint64_t dout = mem.alloc(64 * 4);
  LaunchConfig cfg;
  cfg.grid = {1, 8, 1};
  cfg.block = {1, 8, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(dout)}, mem, prof);
  std::vector<uint32_t> out(64);
  mem.read(dout, out.data(), 64 * 4);
  for (uint32_t y = 0; y < 64; ++y) VCHECK_EQ(out[y], y);
}

VTEST(kernel_oob_fault_names_kernel_lane_and_profile) {
  // Run vecAdd with n larger than the allocations: lane faults with context.
  ptx::Module m = ptx::parse(read_file(VGPU_KERNEL_DIR "/vector_add.ptx"));
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  uint64_t da = mem.alloc(64), db = mem.alloc(64), dc = mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg,
                                          {arg_u64(da), arg_u64(db), arg_u64(dc), arg_u32(64)}, mem, prof));
  VCHECK(err.code() == Err::OutOfBounds);
  VCHECK_CONTAINS(err.what(), "vecAdd");
  VCHECK_CONTAINS(err.what(), "lane");
  VCHECK_CONTAINS(err.what(), "nvidia/h100");
}

VTEST(uninitialized_register_reaching_memory_is_diagnosed_in_strict_mode) {
  // A value nothing has written, on its way to memory, looks like a bug -- but
  // CUB's radix sort stores exactly that into the unused part of a shared tile
  // and never reads it back, so this cannot be the default. VGPU_STRICT=1 is
  // for looking for a bug rather than running a workload.
  const char* ptx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    st.global.u32 [%rd2], %r2;
    ret;
}
)";
  ptx::Module m = ptx::parse(ptx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  uint64_t out = mem.alloc(4);
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  setenv("VGPU_STRICT", "1", 1);
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {arg}, mem, prof));
  unsetenv("VGPU_STRICT");
  VCHECK(err.code() == Err::UninitializedRegister);
  VCHECK_CONTAINS(err.what(), "%r2");
}

VTEST(uninitialized_address_register_is_diagnosed) {
  const char* ptx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry k()
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    ld.global.u32 %r1, [%rd2];
    ret;
}
)";
  ptx::Module m = ptx::parse(ptx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {}, mem, prof));
  VCHECK(err.code() == Err::UninitializedRegister);
  VCHECK_CONTAINS(err.what(), "%rd2");
}

VTEST(uninitialized_register_in_arithmetic_reads_as_zero) {
  // Not every undefined read is a bug: compilers emit them deliberately, and
  // faulting here made ggml's flash-attention kernels -- which seed an unrolled
  // chain of selp with a register nothing has written, then overwrite every
  // lane's copy of it -- impossible to run. NVIDIA's own compute-sanitizer
  // checks uninitialized memory rather than registers for the same reason. So
  // arithmetic reads zero, deterministically, and the diagnosis moves to the
  // point where such a value would become a result.
  const char* ptx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    add.s32 %r1, %r2, 7;
    st.global.u32 [%rd2], %r1;
    ret;
}
)";
  ptx::Module m = ptx::parse(ptx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  uint64_t out = mem.alloc(4);
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  exec::launch(m.entries[0], LaunchConfig{}, {arg}, mem, prof);
  VCHECK_EQ(mem.load_scalar(out, 4), uint64_t{7});  // 0 + 7, not a fault
}

VTEST(infinite_loop_hits_step_budget) {
  const char* ptx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry spin()
{
LOOP:
    bra LOOP;
}
)";
  ptx::Module m = ptx::parse(ptx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  cfg.max_steps = 10000;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, mem, prof));
  VCHECK(err.code() == Err::ExecLimit);
  VCHECK_CONTAINS(err.what(), "spin");
}

VTEST(launch_limits_enforced) {
  ptx::Module m = ptx::parse(read_file(VGPU_KERNEL_DIR "/vector_add.ptx"));
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  cfg.block = {2048, 1, 1};  // over max_threads_per_block AND max_block_dim.x
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg,
                                          {arg_u64(0), arg_u64(0), arg_u64(0), arg_u32(0)}, mem, prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "1024");
}

VTEST(wrong_arg_count_is_diagnosed) {
  ptx::Module m = ptx::parse(read_file(VGPU_KERNEL_DIR "/vector_add.ptx"));
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(0)}, mem, prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "expects 4 parameters");
}

VTEST_MAIN

// --- VGPU_MAX_STEPS -------------------------------------------------------
// The step budget catches infinite loops, but a kernel that is legitimately
// long trips it too. These pin the override that raises or removes it.

namespace {
// A kernel that runs a bounded number of iterations: long enough to exceed a
// deliberately tiny budget, short enough to finish when the budget allows it.
const char* kCountdownPtx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry countdown()
{
    .reg .s32 %r<3>;
    mov.s32 %r1, 500;
LOOP:
    sub.s32 %r1, %r1, 1;
    setp.gt.s32 %p1, %r1, 0;
    @%p1 bra LOOP;
    ret;
}
)";

struct EnvGuard {
  const char* name;
  std::string saved;
  bool had = false;
  explicit EnvGuard(const char* n) : name(n) {
    if (const char* v = std::getenv(n)) { saved = v; had = true; }
  }
  void set(const char* v) { ::setenv(name, v, 1); }
  ~EnvGuard() { had ? (void)::setenv(name, saved.c_str(), 1) : (void)::unsetenv(name); }
};
}  // namespace

VTEST(max_steps_env_raises_the_budget) {
  ptx::Module m = ptx::parse(kCountdownPtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  cfg.max_steps = 100;  // far too small for 500 iterations

  EnvGuard g("VGPU_MAX_STEPS");
  ::unsetenv("VGPU_MAX_STEPS");
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, mem, prof));
  VCHECK(err.code() == Err::ExecLimit);

  g.set("100000");  // now it fits
  exec::launch(m.entries[0], cfg, {}, mem, prof);
}

VTEST(max_steps_env_zero_disables_the_budget) {
  ptx::Module m = ptx::parse(kCountdownPtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  cfg.max_steps = 1;

  EnvGuard g("VGPU_MAX_STEPS");
  g.set("0");
  exec::launch(m.entries[0], cfg, {}, mem, prof);  // no guard at all
}

VTEST(max_steps_env_ignores_junk) {
  // A value that is not a whole number must leave the guard alone rather than
  // silently removing the protection against a runaway kernel.
  ptx::Module m = ptx::parse(kCountdownPtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  cfg.max_steps = 100;

  EnvGuard g("VGPU_MAX_STEPS");
  for (const char* junk : {"abc", "12x", "", "-5"}) {
    g.set(junk);
    auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, mem, prof));
    VCHECK(err.code() == Err::ExecLimit);
  }
}

VTEST(step_budget_counts_each_warp_not_the_whole_launch) {
  // The budget is there to catch one thread looping forever. A big grid of
  // threads that each finish adds up to far more instructions than any one of
  // them runs, and must not be mistaken for a runaway -- nor may the verdict
  // depend on how many host threads the grid is spread over.
  ptx::Module m = ptx::parse(kCountdownPtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  cfg.grid = {16, 1, 1};
  cfg.block = {256, 1, 1};
  cfg.max_steps = 10000;  // ~1,500 per warp; at least 16 warps, so over 24,000 in all

  EnvGuard steps("VGPU_MAX_STEPS");
  ::unsetenv("VGPU_MAX_STEPS");
  EnvGuard threads("VGPU_THREADS");
  for (const char* n : {"1", "4"}) {
    threads.set(n);
    exec::LaunchStats stats = exec::launch(m.entries[0], cfg, {}, mem, prof);
    VCHECK(stats.instructions > cfg.max_steps);
  }
}

VTEST(launch_bounds_limit_the_total_not_each_dimension) {
  // __launch_bounds__(128) emits ".maxntid 128, 1, 1". A 32x4x1 block is 128
  // threads and hardware accepts it; checking dimension by dimension rejected
  // any block with a y extent, which is most real kernels.
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry bounded()
.maxntid 128, 1, 1
{
    ret;
}
)";
  ptx::Module m = ptx::parse(kPtx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 4, 1};  // 128 threads, within the bound
  exec::launch(m.entries[0], cfg, {}, mem, prof);
  // Over the total is still refused.
  cfg.block = {32, 8, 1};  // 256
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, mem, prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "128");
}

// Launch counters. These are exact counts of what executed, so a test can
// predict them precisely -- which is the point: hardware counters sample and
// multiplex, these do not.
VTEST(launch_counters_are_exact) {
  // 2 blocks x 32 threads, each thread doing one global load and one store.
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry cnt(.param .u64 p)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<6>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %ctaid.x;
  mov.u32 %r2, %ntid.x;
  mov.u32 %r3, %tid.x;
  mad.lo.s32 %r1, %r1, %r2, %r3;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd4, %rd2, %rd3;
  ld.global.u32 %r2, [%rd4];
  st.global.u32 [%rd4], %r2;
  ret;
}
)";
  ptx::Module m = ptx::parse(kPtx);
  MemoryManager mem(1 << 20);
  uint64_t buf = mem.alloc(64 * 4);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {2, 1, 1};
  cfg.block = {32, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &buf, 8);
  auto st = exec::launch(m.entries[0], cfg, {arg}, mem, prof);

  VCHECK_EQ(st.blocks, 2ull);
  VCHECK_EQ(st.warps, 2ull);          // 32 threads is exactly one warp per block
  // One load and one store per thread: 64 threads.
  VCHECK_EQ(st.global_loads, 64ull);
  VCHECK_EQ(st.global_stores, 64ull);
  VCHECK_EQ(st.global_bytes_read, 64ull * 4);
  VCHECK_EQ(st.global_bytes_written, 64ull * 4);
  VCHECK_EQ(st.shared_loads, 0ull);
  VCHECK_EQ(st.atomics, 0ull);
  VCHECK_EQ(st.barriers, 0ull);
  // No branch in this kernel, so nothing diverges.
  VCHECK_EQ(st.divergent_branches, 0ull);
  // Every lane is active throughout, so the per-lane total is exactly 32x the
  // warp-level count.
  VCHECK_EQ(st.thread_instructions, st.instructions * 32);
  mem.free(buf);
}

VTEST(divergence_is_counted_only_when_lanes_disagree) {
  // Half the warp takes the branch, so it diverges exactly once per warp.
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry dv(.param .u64 p)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  .reg .pred %p<2>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  setp.lt.u32 %p1, %r1, 16;
  @%p1 bra SKIP;
  st.global.u32 [%rd2], %r1;
SKIP:
  ret;
}
)";
  ptx::Module m = ptx::parse(kPtx);
  MemoryManager mem(1 << 20);
  uint64_t buf = mem.alloc(4);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &buf, 8);
  auto st = exec::launch(m.entries[0], cfg, {arg}, mem, prof);
  VCHECK_EQ(st.divergent_branches, 1ull);
  // Only the 16 lanes that fell through reach the store.
  VCHECK_EQ(st.global_stores, 16ull);
  // And with lanes idle on the diverged paths, utilisation is below full.
  VCHECK(st.thread_instructions < st.instructions * 32);
  mem.free(buf);
}
