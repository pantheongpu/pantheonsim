// A .func whose parameters or result are registers: ".func f(.reg .u64 data)", called through
// ".callprototype _(.reg .u64 data)" (Dr.Jit compiles virtual calls to these). The argument register is
// copied into the callee's register of that name at the call, a register result is copied back into the
// caller's after it, and a call that only some lanes make moves only those lanes' values.
#include <cstring>

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
const char* kHeader = ".version 8.0\n.target sm_80\n.address_size 64\n";
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
// Loads `ptx` through the runtime (which links the .func into the module) and launches `k(out)`.
uint64_t run(const std::string& ptx, uint32_t threads, size_t out_bytes, runtime::Runtime& rt) {
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  const uint64_t out = dev.memory().alloc(out_bytes);
  LaunchConfig cfg;
  cfg.block = {threads, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  return out;
}
}  // namespace

// The module of the report: a register parameter that is an address, an indirect call, a prototype with .reg.
VTEST(reg_parameter_called_through_a_register_prototype) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .func f(.reg .u64 data)
{
    .reg .b32 %r<2>;
    mov.b32 %r0, 1;
    st.global.b32 [data], %r0;
    ret;
}

.visible .entry k(.param .u64 p)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd0, [p];
    mov.u64 %rd1, f;
    {
        proto: .callprototype _(.reg .u64 data);
        call %rd1, (%rd0), proto;
    }
    ret;
}
)";
  runtime::Runtime rt(load_gpu("nvidia/a100"));
  const uint64_t out = run(ptx, 1, 8, rt);
  VCHECK_EQ(rt.device(0).memory().load_scalar(out, 4), 1ull);
}

// Several register parameters of different widths and kinds, a register result, and lanes of one warp
// taking different callees through a table.
VTEST(reg_parameters_and_result_with_divergent_indirect_calls) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .func (.reg .b32 res) scale(.reg .u32 x, .reg .f32 s, .reg .u64 base)
{
    .reg .b32 %r<4>; .reg .f32 %f<3>; .reg .b64 %rd<3>;
    cvt.rn.f32.u32 %f1, x;
    mul.f32 %f2, %f1, s;
    cvt.rzi.u32.f32 %r1, %f2;
    ld.global.u32 %r2, [base];
    add.u32 res, %r1, %r2;
    ret;
}
.visible .func (.reg .b32 res) twice(.reg .u32 x, .reg .f32 s, .reg .u64 base)
{
    shl.b32 res, x, 1;
    ret;
}
.global .align 8 .u64 table[2] = {scale, twice};
.visible .entry k(.param .u64 po)
{
    .reg .b32 %r<10>; .reg .f32 %f<3>; .reg .b64 %rd<10>; .reg .pred %p<3>;
    ld.param.u64 %rd0, [po];
    cvta.to.global.u64 %rd0, %rd0;
    st.global.u32 [%rd0+1024], 1000;
    add.u64 %rd9, %rd0, 1024;
    mov.u32 %r1, %tid.x;
    mov.f32 %f1, 0f40000000;
    and.b32 %r2, %r1, 1;
    mov.u64 %rd3, table;
    mul.wide.u32 %rd4, %r2, 8;
    add.u64 %rd5, %rd3, %rd4;
    ld.global.u64 %rd6, [%rd5];
    {
        proto: .callprototype (.reg .b32 _) _ (.reg .u32 _, .reg .f32 _, .reg .u64 _);
        call (%r3), %rd6, (%r1, %f1, %rd9), proto;
    }
    mul.wide.u32 %rd7, %r1, 4;
    add.u64 %rd8, %rd0, %rd7;
    st.global.u32 [%rd8], %r3;
    ret;
}
)";
  runtime::Runtime rt(load_gpu("nvidia/a100"));
  const uint64_t out = run(ptx, 32, 2048, rt);
  for (uint32_t t = 0; t < 32; ++t) {
    const uint64_t want = (t & 1) ? uint64_t{2 * t} : uint64_t{2 * t + 1000};
    VCHECK_EQ(rt.device(0).memory().load_scalar(out + 4 * t, 4), want);
  }
}

// A direct call under a guard, mixed with a .param argument; the result stays untouched in the lanes that
// did not call.
VTEST(guarded_direct_call_with_mixed_param_and_reg_arguments) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .func (.reg .b32 res) addp(.param .b32 a, .reg .u32 b)
{
    .reg .b32 %r<3>;
    ld.param.b32 %r1, [a];
    add.u32 res, %r1, b;
    ret;
}
.visible .entry k(.param .u64 po)
{
    .reg .b32 %r<6>; .reg .b64 %rd<5>; .reg .pred %p<2>;
    ld.param.u64 %rd0, [po];
    cvta.to.global.u64 %rd0, %rd0;
    mov.u32 %r1, %tid.x;
    mov.u32 %r3, 7777;
    setp.lt.u32 %p1, %r1, 4;
    {
        .param .b32 a0;
        st.param.b32 [a0], %r1;
        @%p1 call (%r3), addp, (a0, %r1);
    }
    mul.wide.u32 %rd1, %r1, 4;
    add.u64 %rd2, %rd0, %rd1;
    st.global.u32 [%rd2], %r3;
    ret;
}
)";
  runtime::Runtime rt(load_gpu("nvidia/a100"));
  const uint64_t out = run(ptx, 8, 64, rt);
  for (uint32_t t = 0; t < 8; ++t)
    VCHECK_EQ(rt.device(0).memory().load_scalar(out + 4 * t, 4), t < 4 ? uint64_t{2 * t} : uint64_t{7777});
}

// A constant argument has no width to pass it at: refused by name, not guessed.
VTEST(constant_call_argument_is_refused_by_name) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .func g(.reg .u32 x) { ret; }
.visible .entry k(.param .u64 po)
{
    {
        call g, (5);
    }
    ret;
}
)";
  const auto err = VCAPTURE(vgpu::Error, ptx::parse(ptx));
  VCHECK_CONTAINS(err.what(), "constant as a call argument");
}

VTEST_MAIN
