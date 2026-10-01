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

VTEST(signed_narrow_params_sign_extend) {
  // A kernel taking signed char and short, as nvcc compiles it: ld.param.s8
  // into a 16-bit register and ld.param.s16 straight into a 32-bit one. Both
  // deliver the value (-128, -32768), not the bit pattern (128, 32768), by
  // name and through a register holding the parameter's address. The
  // unsigned load of the same byte is the control.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out, .param .u8 a, .param .u8 b, .param .u16 c)
{
    .reg .b16 %rs<2>;
    .reg .b32 %r<7>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.s8 %rs1, [a];
    cvt.s32.s16 %r1, %rs1;
    st.global.u32 [%rd2], %r1;
    ld.param.s16 %r2, [c];
    st.global.u32 [%rd2+4], %r2;
    ld.param.s8 %r3, [b];
    st.global.u32 [%rd2+8], %r3;
    ld.param.u8 %r4, [a];
    st.global.u32 [%rd2+12], %r4;
    mov.u64 %rd3, c;
    ld.param.s16 %r5, [%rd3];
    st.global.u32 [%rd2+16], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(20);
  std::vector<uint8_t> o(8);
  std::memcpy(o.data(), &out, 8);
  exec::launch(m.entries[0], LaunchConfig{}, {o, {0x80}, {0x05}, {0x00, 0x80}}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out + 0, 4), uint64_t{0xFFFFFF80});   // -128
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0xFFFF8000});   // -32768
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), 5ull);
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), 128ull);
  VCHECK_EQ(e.mem.load_scalar(out + 16, 4), uint64_t{0xFFFF8000});  // through a register
}

VTEST(vector_param_accesses_through_call_slots) {
  // A struct passed to and returned from a device function, as nvcc writes
  // it: st.param.v4.b8 into the argument, ld.param.v4.u8 and a signed byte
  // (ld.param.s8) out of it in the callee, and the result back through a
  // .v2 return slot. Boost.Math's quantile finders pass their arguments so.
  // An RTX 3060 runs this PTX to ffffff80 and 1 + 2*10 + 3*100 + 200*1000.
  std::string ptx = std::string(kHeader) + R"(
.func (.param .align 8 .b8 r[8]) pack(.param .align 4 .b8 p[8])
{
    .reg .b16 %rs<5>;
    .reg .b32 %r<4>;
    ld.param.v4.u8 {%rs1, %rs2, %rs3, %rs4}, [p];
    ld.param.s8 %r1, [p+4];
    cvt.u32.u16 %r2, %rs1;
    cvt.u32.u16 %r3, %rs2;
    mad.lo.u32 %r2, %r3, 10, %r2;
    cvt.u32.u16 %r3, %rs3;
    mad.lo.u32 %r2, %r3, 100, %r2;
    cvt.u32.u16 %r3, %rs4;
    mad.lo.u32 %r2, %r3, 1000, %r2;
    st.param.v2.b32 [r], {%r1, %r2};
    ret;
}
.visible .entry k(.param .u64 out)
{
    .reg .b16 %rs<6>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u16 %rs1, 1;
    mov.u16 %rs2, 2;
    mov.u16 %rs3, 3;
    mov.u16 %rs4, 200;
    mov.u16 %rs5, 128;
    {
    .param .align 4 .b8 param0[8];
    st.param.v4.b8 [param0], {%rs1, %rs2, %rs3, %rs4};
    st.param.b8 [param0+4], %rs5;
    .param .align 8 .b8 retval0[8];
    call.uni (retval0), pack, (param0);
    ld.param.v2.b32 {%r1, %r2}, [retval0];
    }
    st.global.v2.u32 [%rd2], {%r1, %r2};
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(8);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0xFFFFFF80});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 200321ull);
}

VTEST(running_off_the_end_of_a_function_returns) {
  // A kernel and a .func whose last block ends without ret. Lanes that run
  // off the end of the .func return to the caller with the value they stored
  // (lane 3), and lanes that run off the end of the kernel exit. An RTX 3060
  // runs this PTX to 0 2 104 106. nvcc emits kernels like this after a call
  // whose result nothing uses (Boost.Math's inverse Gaussian quantile).
  std::string ptx = std::string(kHeader) + R"(
.func (.param .b32 r) f(.param .b32 v)
{
    .reg .pred %q;
    .reg .b32 %t<3>;
    ld.param.b32 %t1, [v];
    shl.b32 %t2, %t1, 1;
    st.param.b32 [r], %t2;
    setp.eq.u32 %q, %t1, 3;
    @%q bra FEND;
    ret;
FEND:
}
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd2, %rd2, %rd3;
    {
    .param .b32 param0;
    st.param.b32 [param0], %r1;
    .param .b32 retval0;
    call.uni (retval0), f, (param0);
    ld.param.b32 %r2, [retval0];
    }
    st.global.u32 [%rd2], %r2;
    setp.lt.u32 %p1, %r1, 2;
    @%p1 bra DONE;
    add.u32 %r3, %r2, 100;
    st.global.u32 [%rd2], %r3;
    bra.uni END;
DONE:
    ret;
END:
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  std::vector<uint8_t> zero(16, 0);
  e.mem.write(out, zero.data(), 16);
  LaunchConfig cfg;
  cfg.block = {4, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), 0ull);
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 2ull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), 104ull);
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), 106ull);
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

VTEST(vector_param_return_and_argument) {
  // A device function returning a 16-byte struct and taking a vector
  // argument: CUDA 13 writes both with st.param.v4 / ld.param.v4 (sm_100+).
  std::string ptx = std::string(kHeader) + R"(
.func (.param .align 16 .b8 func_retval0[16]) make(.param .align 8 .b8 p0[8])
{
    .reg .f32 %f<7>;
    ld.param.v2.f32 {%f1, %f5}, [p0];
    add.f32 %f2, %f1, 0f3F800000;
    add.f32 %f3, %f1, 0f40000000;
    add.f32 %f4, %f5, 0f40400000;
    st.param.v4.f32 [func_retval0], {%f1, %f2, %f3, %f4};
    ret;
}
.visible .entry k(.param .u64 out)
{
    .reg .f32 %f<6>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    {
    .param .align 8 .b8 param0[8];
    st.param.v2.f32 [param0], {0f41200000, 0f42C80000};
    .param .align 16 .b8 retval0[16];
    call.uni (retval0), make, (param0);
    ld.param.v4.f32 {%f1, %f2, %f3, %f4}, [retval0];
    }
    st.global.v4.f32 [%rd2], {%f1, %f2, %f3, %f4};
    ret;
}
)";
  runtime::Runtime rt(load_gpu("nvidia/b200"));
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  uint64_t out = dev.memory().alloc(16);
  dev.launch(*fn, LaunchConfig{}, {arg_u64(out)}, rt.device(0).symbols(mod));
  float got[4];
  dev.memory().read(out, got, 16);
  VCHECK_EQ(got[0], 10.0f);
  VCHECK_EQ(got[1], 11.0f);
  VCHECK_EQ(got[2], 12.0f);
  VCHECK_EQ(got[3], 103.0f);
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
