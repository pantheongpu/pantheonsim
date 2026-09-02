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

VTEST(uninitialized_register_read_is_diagnosed) {
  const char* ptx = R"(
.version 8.3
.target sm_90
.address_size 64
.visible .entry k()
{
    .reg .b32 %r<3>;
    add.s32 %r1, %r2, 1;
    ret;
}
)";
  ptx::Module m = ptx::parse(ptx);
  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig cfg;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, mem, prof));
  VCHECK(err.code() == Err::UninitializedRegister);
  VCHECK_CONTAINS(err.what(), "%r2");
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
