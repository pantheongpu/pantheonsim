// Dynamic parallelism at the PTX level: kernels launching kernels through the
// device runtime's entry points, with the kernel table the runtime would
// supply built by hand. nvidia/tests/e2e/dynamic_parallelism.cu covers the
// same from CUDA C++ built with -rdc.
#include <cstring>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// The declarations nvcc emits for CDP2, and a helper that launches `kernel`
// with a <<<grid, block>>> of (%r20, 1, 1) x (%r21, 1, 1) and returns the
// parameter buffer in %rd20 (0 if none was given).
const char* kDecls = R"(
.extern .func (.param .b64 func_retval0) __cudaCDP2GetParameterBufferV2
(.param .b64 f, .param .align 4 .b8 g[12], .param .align 4 .b8 b[12], .param .b32 s);
.extern .func (.param .b32 func_retval0) __cudaCDP2LaunchDeviceV2
(.param .b64 buf, .param .b64 stream);
)";

std::string get_buffer(const std::string& kernel) {
  return R"(
    mov.u64 %rd19, )" + kernel + R"(;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd19;
    .param .align 4 .b8 param1[12];
    st.param.b32 [param1+0], %r20;
    st.param.b32 [param1+4], 1;
    st.param.b32 [param1+8], 1;
    .param .align 4 .b8 param2[12];
    st.param.b32 [param2+0], %r21;
    st.param.b32 [param2+4], 1;
    st.param.b32 [param2+8], 1;
    .param .b32 param3;
    st.param.b32 [param3+0], 0;
    .param .b64 retval0;
    call.uni (retval0), __cudaCDP2GetParameterBufferV2, (param0, param1, param2, param3);
    ld.param.b64 %rd20, [retval0+0];
    }
)";
}
const char* kLaunch = R"(
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd20;
    .param .b64 param1;
    st.param.b64 [param1+0], 0;
    .param .b32 retval0;
    call.uni (retval0), __cudaCDP2LaunchDeviceV2, (param0, param1);
    ld.param.b32 %r22, [retval0+0];
    }
)";

struct Loaded {
  ptx::Module mod;
  std::map<std::string, uint64_t> symbols;
  exec::KernelTable kernels;
};
// Kernel addresses the way the runtime hands them out.
Loaded load(const std::string& body) {
  Loaded l;
  l.mod = ptx::parse(".version 8.3\n.target sm_90\n.address_size 64\n" + std::string(kDecls) + body);
  uint64_t va = kKernelVaBase;
  for (const auto& e : l.mod.entries) {
    l.symbols[e.name] = va;
    va += kKernelVaStride;
  }
  for (const auto& e : l.mod.entries) l.kernels[l.symbols[e.name]] = exec::KernelRef{&e, &l.symbols};
  return l;
}
const ptx::EntryFn& entry(const Loaded& l, const std::string& name) {
  for (const auto& e : l.mod.entries)
    if (e.name == name) return e;
  throw Error::make(Err::NotFound, name);
}

}  // namespace

// Each of four threads launches an 8-thread child that fills its slice; the
// child's second parameter sits at offset 8 after a pointer, as nvcc lays it.
VTEST(a_parent_launches_children_that_run_before_the_launch_returns) {
  Loaded l = load(R"(
.visible .entry child(.param .u64 out, .param .u32 base)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    ld.param.u32 %r1, [base];
    mov.u32 %r2, %tid.x;
    add.u32 %r3, %r1, %r2;
    mul.wide.u32 %rd2, %r3, 4;
    add.u64 %rd3, %rd1, %rd2;
    st.global.u32 [%rd3], %r3;
    ret;
}
.visible .entry parent(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    setp.ge.u32 %p1, %r1, 4;
    @%p1 bra DONE;
    mov.u32 %r20, 1;
    mov.u32 %r21, 8;
)" + get_buffer("child") + R"(
    st.u64 [%rd20], %rd1;
    shl.b32 %r2, %r1, 3;
    st.u32 [%rd20+8], %r2;
)" + kLaunch + R"(
DONE:
    ret;
}
)");
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(32 * 4);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig c;
  c.block = {32, 1, 1};
  c.kernels = &l.kernels;
  exec::launch(entry(l, "parent"), c, {arg_u64(out)}, mem, prof, &l.symbols);
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(mem.load_scalar(out + i * 4, 4), uint64_t{i});
}

// A kernel that launches itself one level deeper each time stops at CUDA's
// nesting limit of 24 with an error that says so.
VTEST(nesting_past_24_levels_is_refused) {
  Loaded l = load(R"(
.visible .entry deeper(.param .u64 out)
{
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    ld.param.u64 %rd1, [out];
    atom.global.add.u32 %r1, [%rd1], 1;
    mov.u32 %r20, 1;
    mov.u32 %r21, 1;
)" + get_buffer("deeper") + R"(
    st.u64 [%rd20], %rd1;
)" + kLaunch + R"(
    ret;
}
)");
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(4);
  mem.store_scalar(out, 4, 0);
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig c;
  c.kernels = &l.kernels;
  auto err = VCAPTURE(Error, exec::launch(entry(l, "deeper"), c, {arg_u64(out)}, mem, prof, &l.symbols));
  VCHECK_CONTAINS(err.message(), "nesting depth 25");
  VCHECK_EQ(mem.load_scalar(out, 4), uint64_t{25});   // levels 0..24 ran
}

// A launch naming an address that is not a kernel is reported, not run.
VTEST(a_launch_of_something_that_is_not_a_kernel_is_reported) {
  Loaded l = load(R"(
.visible .entry bad()
{
    .reg .b32 %r<24>;
    .reg .b64 %rd<24>;
    mov.u32 %r20, 1;
    mov.u32 %r21, 1;
    mov.u64 %rd19, 0x1234;
    {
    .param .b64 param0;
    st.param.b64 [param0+0], %rd19;
    .param .align 4 .b8 param1[12];
    st.param.b32 [param1+0], 1;
    st.param.b32 [param1+4], 1;
    st.param.b32 [param1+8], 1;
    .param .align 4 .b8 param2[12];
    st.param.b32 [param2+0], 1;
    st.param.b32 [param2+4], 1;
    st.param.b32 [param2+8], 1;
    .param .b32 param3;
    st.param.b32 [param3+0], 0;
    .param .b64 retval0;
    call.uni (retval0), __cudaCDP2GetParameterBufferV2, (param0, param1, param2, param3);
    ld.param.b64 %rd20, [retval0+0];
    }
    ret;
}
)");
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/h100");
  LaunchConfig c;
  c.kernels = &l.kernels;
  auto err = VCAPTURE(Error, exec::launch(entry(l, "bad"), c, {}, mem, prof, &l.symbols));
  VCHECK_CONTAINS(err.message(), "not the address of a kernel");
}

VTEST_MAIN
