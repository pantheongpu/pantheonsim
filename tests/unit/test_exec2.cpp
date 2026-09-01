// Tests for the pantheon-era PTX additions: cvt, atomics, vector ld/st,
// funnel shifts, predicate logic, aggregate params, module globals, .local
// memory, and device printf.
#include <algorithm>
#include <cstring>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {

const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct Env {
  MemoryManager mem{1 << 24};
  DeviceProfile prof = load_gpu("nvidia/a10");  // sm_86, like the kernels
};

}  // namespace

VTEST(cvt_sign_and_zero_extension) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, -1;
    cvt.s64.s32 %rd3, %r1;          // sign-extend -> 0xFFFFFFFFFFFFFFFF
    st.global.u64 [%rd2], %rd3;
    cvt.u64.u32 %rd4, %r1;          // zero-extend -> 0x00000000FFFFFFFF
    st.global.u64 [%rd2+8], %rd4;
    mov.u64 %rd5, 0x100000005;
    cvt.u32.u64 %r2, %rd5;          // truncate -> 5
    st.global.u32 [%rd2+16], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(24);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 8), 0xFFFFFFFFFFFFFFFFull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 8), 0x00000000FFFFFFFFull);
  VCHECK_EQ(e.mem.load_scalar(out + 16, 4), 5ull);
}

VTEST(atomic_add_returns_unique_old_values) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 ctr, .param .u64 olds)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [ctr];
    ld.param.u64 %rd2, [olds];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    atom.global.add.u32 %r1, [%rd3], 1;
    mov.u32 %r2, %ctaid.x;
    mov.u32 %r3, %ntid.x;
    mov.u32 %r4, %tid.x;
    mad.lo.s32 %r5, %r2, %r3, %r4;
    mul.wide.u32 %rd5, %r5, 4;
    add.s64 %rd6, %rd4, %rd5;
    st.global.u32 [%rd6], %r1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint32_t n = 256;
  uint64_t ctr = e.mem.alloc(4), olds = e.mem.alloc(n * 4);
  LaunchConfig cfg;
  cfg.grid = {4, 1, 1};
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(ctr), arg_u64(olds)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(ctr, 4), uint64_t{n});
  std::vector<uint32_t> got(n);
  e.mem.read(olds, got.data(), n * 4);
  std::sort(got.begin(), got.end());
  for (uint32_t i = 0; i < n; ++i) VCHECK_EQ(got[i], i);  // every old value seen exactly once
}

VTEST(vector_v4_load_store_and_alignment) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 in, .param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [in];
    ld.param.u64 %rd2, [out];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r5, %tid.x;
    mul.wide.u32 %rd5, %r5, 16;
    add.s64 %rd6, %rd3, %rd5;
    add.s64 %rd7, %rd4, %rd5;
    ld.global.cs.v4.u32 {%r1, %r2, %r3, %r4}, [%rd6];
    st.global.cs.v4.u32 [%rd7], {%r4, %r3, %r2, %r1};
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint32_t lanes = 8;
  std::vector<uint32_t> in(lanes * 4);
  for (uint32_t i = 0; i < in.size(); ++i) in[i] = i * 17;
  uint64_t din = e.mem.alloc(in.size() * 4), dout = e.mem.alloc(in.size() * 4);
  e.mem.write(din, in.data(), in.size() * 4);
  LaunchConfig cfg;
  cfg.block = {lanes, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(din), arg_u64(dout)}, e.mem, e.prof);
  std::vector<uint32_t> out(in.size());
  e.mem.read(dout, out.data(), out.size() * 4);
  for (uint32_t t = 0; t < lanes; ++t)
    for (int j = 0; j < 4; ++j) VCHECK_EQ(out[t * 4 + j], in[t * 4 + (3 - j)]);

  // Misaligned v4 access is a fault, matching hardware.
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(din + 4), arg_u64(dout)}, e.mem, e.prof));
  VCHECK(err.code() == Err::MisalignedAccess);
  VCHECK_CONTAINS(err.what(), "16-byte alignment");
}

VTEST(funnel_shift_wrap) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0x9ABCDEF0;
    mov.u32 %r2, 0x12345678;
    mov.u32 %r3, 8;
    shf.l.wrap.b32 %r4, %r1, %r2, %r3;   // high 32 of (b:a) << 8
    st.global.u32 [%rd2], %r4;
    shf.r.wrap.b32 %r5, %r1, %r2, %r3;   // low 32 of (b:a) >> 8
    st.global.u32 [%rd2+4], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(8);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), 0x3456789Aull);
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 0x789ABCDEull);
}

VTEST(predicate_logic_ops) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<5>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    setp.lt.u32 %p1, %r1, 2;          // tid 0,1
    setp.eq.s32 %p2, %r1, 1;          // tid 1
    and.pred %p3, %p1, %p2;           // tid 1
    or.pred %p4, %p1, %p2;            // tid 0,1
    selp.b32 %r2, 100, 200, %p3;
    selp.b32 %r3, 10, 20, %p4;
    add.s32 %r4, %r2, %r3;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {4, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out + 0, 4), 210ull);   // tid0: 200+10
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 110ull);   // tid1: 100+10
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), 220ull);   // tid2: 200+20
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), 220ull);  // tid3: 200+20
}

VTEST(aggregate_byte_array_param) {
  // Models PantheonFaultLog passed by value: { u32* count; u32 capacity; pad; }
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .align 8 .b8 k_param_0[16])
{
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [k_param_0];
    ld.param.u32 %r1, [k_param_0+8];
    cvta.to.global.u64 %rd2, %rd1;
    st.global.u32 [%rd2], %r1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  VCHECK_EQ(m.entries[0].params[0].size, 16u);
  uint64_t buf = e.mem.alloc(4);
  std::vector<uint8_t> arg(16, 0);
  std::memcpy(arg.data(), &buf, 8);
  uint32_t cap = 0xC0FFEE;
  std::memcpy(arg.data() + 8, &cap, 4);
  exec::launch(m.entries[0], LaunchConfig{}, {arg}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(buf, 4), uint64_t{0xC0FFEE});
}

VTEST(module_global_string_and_symbols) {
  // "AB" + zero padding, read through a symbol address.
  std::string ptx = std::string(kHeader) + R"(
.global .align 1 .b8 $str[4] = {65, 66};
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<3>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, $str;
    cvta.global.u64 %rd4, %rd3;
    ld.global.u32 %r1, [%rd4];
    st.global.u32 [%rd2], %r1;
    ret;
}
)";
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  uint64_t out = dev.memory().alloc(4);
  dev.launch(*fn, LaunchConfig{}, {arg_u64(out)}, rt.device(0).symbols(mod));
  VCHECK_EQ(dev.memory().load_scalar(out, 4), 0x00004241ull);  // 'A','B',0,0 little-endian
}

VTEST(local_memory_and_device_printf) {
  // The full nvcc printf pattern: local depot, %SP/%SPL, callseq slots,
  // vprintf, retval. Checks the returned character count; the text itself
  // lands on stdout (verified by the pantheon workloads end-to-end).
  std::string ptx = std::string(kHeader) + R"(
.extern .func (.param .b32 func_retval0) vprintf (.param .b64 p0, .param .b64 p1);
.global .align 1 .b8 $str[15] = {118, 97, 108, 61, 37, 100, 32, 104, 101, 120, 61, 37, 120, 10};
.visible .entry printer(.param .u64 out)
{
    .local .align 8 .b8 __local_depot0[8];
    .reg .b64 %SP, %SPL;
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    mov.u64 %SPL, __local_depot0;
    cvta.local.u64 %SP, %SPL;
    ld.param.u64 %rd1, [out];
    mov.u32 %r1, %tid.x;
    st.local.u32 [%SPL+0], %r1;
    st.local.u32 [%SPL+4], %r1;
    mov.u64 %rd2, $str;
    cvta.global.u64 %rd3, %rd2;
    add.u64 %rd4, %SP, 0;
    { // callseq 0, 0
    .reg .b32 temp_param_reg;
    .param .b64 param0;
    st.param.b64 [param0+0], %rd3;
    .param .b64 param1;
    st.param.b64 [param1+0], %rd4;
    .param .b32 retval0;
    call.uni (retval0), vprintf, (param0, param1);
    ld.param.b32 %r2, [retval0+0];
    } // callseq 0
    cvta.to.global.u64 %rd5, %rd1;
    mul.wide.u32 %rd6, %r1, 4;
    add.s64 %rd7, %rd5, %rd6;
    st.global.u32 [%rd7], %r2;
    ret;
}
)";
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "printer");
  VCHECK_EQ(fn->local_frame_size, 8u);
  uint64_t out = dev.memory().alloc(8);
  LaunchConfig cfg;
  cfg.block = {2, 1, 1};
  std::printf("--- expected device printf output: ---\n");
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  std::printf("--------------------------------------\n");
  VCHECK_EQ(dev.memory().load_scalar(out, 4), 12ull);      // "val=0 hex=0\n"
  VCHECK_EQ(dev.memory().load_scalar(out + 4, 4), 12ull);  // "val=1 hex=1\n"
}

VTEST_MAIN
