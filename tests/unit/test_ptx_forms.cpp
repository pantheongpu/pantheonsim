// PTX forms that real producers emit (drjit-core, JIT compilers) and that the frontend used to refuse:
//   * a negated predicate source operand (`and.pred %p2, !%p0, %p1;`)    issue 474
//   * `ldu` (uniform load), which reads what `ld` reads    issue 472
//   * the array-element address `var[imm]` (without or with the outer brackets)    issue 476
//   * `.reg` parameters and return values of a .func, and the call through them    issue 475
// Each result is compared with what the PTX ISA says the instruction computes, worked out here on the host.
#include <cstdint>
#include <cstring>
#include <string>
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

// Loads `ptx` through the runtime (which places the module's globals and its function addresses), launches kernel `k` with
// `threads` threads and one pointer argument to `words` zeroed 32-bit words, and returns the words.
std::vector<uint32_t> run(const std::string& ptx, uint32_t threads, size_t words) {
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  const uint64_t out = dev.memory().alloc(words * 4);
  for (size_t i = 0; i < words; ++i) dev.memory().store_scalar(out + 4 * i, 4, 0);
  LaunchConfig cfg;
  cfg.block = {threads, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  std::vector<uint32_t> r(words);
  for (size_t i = 0; i < words; ++i) r[i] = static_cast<uint32_t>(dev.memory().load_scalar(out + 4 * i, 4));
  return r;
}
}  // namespace

// ---- 474: a negated predicate source ----

// Every thread holds a different pair of predicate values (lane bit 0 and bit 1), so all four combinations occur in a warp.
// Each operation writes 1 or 0 through selp. The expected value is the boolean formula, written on the host.
VTEST(a_negated_predicate_source_in_and_or_xor_not_and_selp) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .pred %p<16>;
  .reg .b32 %r<16>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  and.b32 %r1, %r0, 1;
  and.b32 %r2, %r0, 2;
  setp.ne.u32 %p0, %r1, 0;      // a = lane bit 0
  setp.ne.u32 %p1, %r2, 0;      // b = lane bit 1
  and.pred %p2, !%p0, %p1;
  xor.pred %p3, %p0, !%p1;
  or.pred  %p4, !%p0, !%p1;
  and.pred %p5, %p0, !%p1;
  xor.pred %p6, !%p0, !%p1;
  not.pred %p7, !%p0;
  selp.u32 %r3, 1, 0, !%p0;
  and.pred %p8, !%p0, 1;
  setp.eq.and.u32 %p9, %r1, 0, !%p1;
  selp.u32 %r4, 1, 0, %p2;
  selp.u32 %r5, 1, 0, %p3;
  selp.u32 %r6, 1, 0, %p4;
  selp.u32 %r7, 1, 0, %p5;
  selp.u32 %r8, 1, 0, %p6;
  selp.u32 %r9, 1, 0, %p7;
  selp.u32 %r10, 1, 0, %p8;
  selp.u32 %r11, 1, 0, %p9;
  mul.wide.u32 %rd1, %r0, 36;
  add.u64 %rd2, %rd0, %rd1;
  st.global.u32 [%rd2+0], %r4;
  st.global.u32 [%rd2+4], %r5;
  st.global.u32 [%rd2+8], %r6;
  st.global.u32 [%rd2+12], %r7;
  st.global.u32 [%rd2+16], %r8;
  st.global.u32 [%rd2+20], %r9;
  st.global.u32 [%rd2+24], %r3;
  st.global.u32 [%rd2+28], %r10;
  st.global.u32 [%rd2+32], %r11;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 32, 32 * 9);
  for (uint32_t t = 0; t < 32; ++t) {
    const bool a = t & 1, b = (t >> 1) & 1;
    const uint32_t want[9] = {!a && b, a != !b, !a || !b, a && !b, !a != !b, a, !a, !a, !a && !b};
    for (int i = 0; i < 9; ++i) VCHECK_EQ(uint64_t{r[t * 9 + i]}, uint64_t{want[i]});
  }
}

// The guard form `@!%p` keeps working next to the new operand form.
VTEST(a_negated_guard_still_works_beside_a_negated_source) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .pred %p<4>;
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  and.b32 %r1, %r0, 1;
  setp.ne.u32 %p0, %r1, 0;
  mov.u32 %r2, 5;
  @!%p0 mov.u32 %r2, 9;
  and.pred %p1, !%p0, !%p0;
  @!%p1 add.u32 %r2, %r2, 100;
  mul.wide.u32 %rd1, %r0, 4;
  add.u64 %rd2, %rd0, %rd1;
  st.global.u32 [%rd2], %r2;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 32, 32);
  for (uint32_t t = 0; t < 32; ++t) VCHECK_EQ(uint64_t{r[t]}, uint64_t{(t & 1) ? 105u : 9u});
}

// ---- 472: ldu ----

VTEST(ldu_loads_what_ld_loads_scalar_and_vector) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .b64 %rd<3>;
  .reg .b32 %r<8>;
  ld.param.u64 %rd1, [po];
  cvta.to.global.u64 %rd1, %rd1;
  mov.u32 %r0, 41;
  st.global.u32 [%rd1+16], %r0;
  mov.u32 %r0, 1;
  st.global.u32 [%rd1+32], %r0;
  mov.u32 %r0, 2;
  st.global.u32 [%rd1+36], %r0;
  mov.u32 %r0, 3;
  st.global.u32 [%rd1+40], %r0;
  mov.u32 %r0, 4;
  st.global.u32 [%rd1+44], %r0;
  ldu.global.u32 %r1, [%rd1+16];
  add.u32 %r1, %r1, 1;
  st.global.u32 [%rd1], %r1;
  ldu.global.v4.u32 {%r2, %r3, %r4, %r5}, [%rd1+32];
  add.u32 %r6, %r2, %r3;
  add.u32 %r6, %r6, %r4;
  add.u32 %r6, %r6, %r5;
  st.global.u32 [%rd1+4], %r6;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 1, 12);
  VCHECK_EQ(uint64_t{r[0]}, uint64_t{42});
  VCHECK_EQ(uint64_t{r[1]}, uint64_t{1 + 2 + 3 + 4});
}

VTEST(ldu_outside_global_is_refused) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .b32 %r<2>;
  .reg .b64 %rd<2>;
  ldu.shared.u32 %r1, [%rd1];
  ret;
}
)";
  auto e = VCAPTURE(Error, ptx::parse(ptx));
  VCHECK_CONTAINS(std::string(e.what()), "ldu outside .global");
}

// ---- 476: var[imm] ----

VTEST(array_element_address_on_local_scales_by_the_element_size) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .f32 %f<4>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<8>;
  .local .align 4 .b32 arr[4];
  .local .align 8 .b64 big[3];
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  mov.b32 %f0, 0x3f800000;
  st.local.b32 arr[0], %f0;
  add.u32 %r1, %r0, 10;
  st.local.b32 arr[1], %r1;
  add.u32 %r2, %r0, 20;
  st.local.b32 [arr[2]], %r2;
  add.u32 %r3, %r0, 30;
  st.local.b32 [arr+12], %r3;                 // the same place as arr[3]
  cvt.u64.u32 %rd1, %r0;
  add.u64 %rd2, %rd1, 1000;
  st.local.b64 big[2], %rd2;
  ld.local.b32 %f1, arr[0];
  ld.local.b32 %r4, arr[1];
  ld.local.b32 %r5, [arr[2]];
  ld.local.b32 %r6, arr[3];
  ld.local.b64 %rd3, big[2];
  mov.b32 %r7, %f1;
  mul.wide.u32 %rd4, %r0, 24;
  add.u64 %rd5, %rd0, %rd4;
  st.global.u32 [%rd5+0], %r7;
  st.global.u32 [%rd5+4], %r4;
  st.global.u32 [%rd5+8], %r5;
  st.global.u32 [%rd5+12], %r6;
  st.global.u64 [%rd5+16], %rd3;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 32, 32 * 6);
  for (uint32_t t = 0; t < 32; ++t) {
    VCHECK_EQ(uint64_t{r[t * 6 + 0]}, uint64_t{0x3f800000u});
    VCHECK_EQ(uint64_t{r[t * 6 + 1]}, uint64_t{t + 10});
    VCHECK_EQ(uint64_t{r[t * 6 + 2]}, uint64_t{t + 20});
    VCHECK_EQ(uint64_t{r[t * 6 + 3]}, uint64_t{t + 30});
    VCHECK_EQ(uint64_t{r[t * 6 + 4]}, uint64_t{t + 1000});
    VCHECK_EQ(uint64_t{r[t * 6 + 5]}, uint64_t{0});
  }
}

VTEST(array_element_address_on_shared_global_and_const) {
  const std::string ptx = std::string(kHeader) + R"(
.global .align 4 .u32 gl[4];
.const .align 4 .u32 cn[4] = {11, 22, 33, 44};
.visible .entry k(.param .u64 po)
{
  .reg .b32 %r<8>;
  .reg .b64 %rd<4>;
  .shared .align 4 .u32 sh[4];
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, 5;
  st.shared.u32 sh[2], %r0;
  mov.u32 %r1, 6;
  st.global.u32 gl[3], %r1;
  ld.shared.u32 %r2, sh[2];
  ld.global.u32 %r3, gl[3];
  ld.const.u32 %r4, cn[2];
  ld.global.u32 %r5, [cn[3]];
  st.global.u32 [%rd0+0], %r2;
  st.global.u32 [%rd0+4], %r3;
  st.global.u32 [%rd0+8], %r4;
  st.global.u32 [%rd0+12], %r5;
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 1, 4);
  VCHECK_EQ(uint64_t{r[0]}, uint64_t{5});
  VCHECK_EQ(uint64_t{r[1]}, uint64_t{6});
  VCHECK_EQ(uint64_t{r[2]}, uint64_t{33});
  VCHECK_EQ(uint64_t{r[3]}, uint64_t{44});
}

VTEST(array_element_index_outside_the_array_is_refused) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
  .reg .b32 %r<2>;
  .local .align 4 .b32 arr[4];
  st.local.b32 arr[4], %r0;
  ret;
}
)";
  auto e = VCAPTURE(Error, ptx::parse(ptx));
  VCHECK_CONTAINS(std::string(e.what()), "outside 'arr'");
}

// ---- 475: .reg parameters and return values ----

// The issue's PTX: a .func with a .reg parameter, called through a .callprototype with the .reg form.
VTEST(an_indirect_call_passes_a_reg_parameter) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .func f(.reg .u64 data)
{
  .reg .b32 %r<2>;
  mov.b32 %r0, 77;
  st.global.b32 [data], %r0;
  ret;
}
.visible .entry k(.param .u64 p)
{
  .reg .b64 %rd<4>;
  ld.param.u64 %rd0, [p];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u64 %rd1, f;
  {
    proto: .callprototype _(.reg .u64 data);
    call %rd1, (%rd0), proto;
  }
  ret;
}
)";
  const std::vector<uint32_t> r = run(ptx, 1, 1);
  VCHECK_EQ(uint64_t{r[0]}, uint64_t{77});
}

// Per-lane arguments, a .reg return value, an immediate argument, and a guarded call: the lanes that do not call keep their
// result register. The same function written with .param must give the same words (the two forms are one call).
VTEST(direct_call_with_reg_parameters_and_a_reg_return_value) {
  const std::string reg_form = std::string(kHeader) + R"(
.func (.reg .b32 res) scale(.reg .u64 data, .reg .b32 x, .reg .b32 k)
{
  .reg .b32 %r<4>;
  mad.lo.u32 %r0, x, k, 1;
  st.global.u32 [data], %r0;
  add.u32 res, x, 7;
  ret;
}
.visible .entry k(.param .u64 po)
{
  .reg .pred %p<2>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<6>;
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  mul.wide.u32 %rd1, %r0, 8;
  add.u64 %rd2, %rd0, %rd1;
  mov.u32 %r2, 1000;
  and.b32 %r3, %r0, 1;
  setp.ne.u32 %p0, %r3, 0;
  @%p0 call (%r2), scale, (%rd2, %r0, 3);
  st.global.u32 [%rd2+4], %r2;
  ret;
}
)";
  const std::string param_form = std::string(kHeader) + R"(
.func (.param .b32 res) scale(.param .u64 data, .param .b32 x, .param .b32 k)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<2>;
  ld.param.u64 %rd0, [data];
  ld.param.b32 %r1, [x];
  ld.param.b32 %r2, [k];
  mad.lo.u32 %r0, %r1, %r2, 1;
  st.global.u32 [%rd0], %r0;
  add.u32 %r3, %r1, 7;
  st.param.b32 [res], %r3;
  ret;
}
.visible .entry k(.param .u64 po)
{
  .reg .pred %p<2>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<6>;
  ld.param.u64 %rd0, [po];
  cvta.to.global.u64 %rd0, %rd0;
  mov.u32 %r0, %tid.x;
  mul.wide.u32 %rd1, %r0, 8;
  add.u64 %rd2, %rd0, %rd1;
  mov.u32 %r2, 1000;
  and.b32 %r3, %r0, 1;
  setp.ne.u32 %p0, %r3, 0;
  @%p0 bra DO;
  bra DONE;
DO:
  {
    .param .u64 a0;
    .param .b32 a1;
    .param .b32 a2;
    .param .b32 r0;
    st.param.u64 [a0], %rd2;
    st.param.b32 [a1], %r0;
    mov.u32 %r4, 3;
    st.param.b32 [a2], %r4;
    call (r0), scale, (a0, a1, a2);
    ld.param.b32 %r2, [r0];
  }
DONE:
  st.global.u32 [%rd2+4], %r2;
  ret;
}
)";
  const std::vector<uint32_t> a = run(reg_form, 32, 64);
  const std::vector<uint32_t> b = run(param_form, 32, 64);
  for (uint32_t t = 0; t < 32; ++t) {
    const bool odd = t & 1;
    VCHECK_EQ(uint64_t{a[2 * t]}, uint64_t{odd ? t * 3 + 1 : 0u});
    VCHECK_EQ(uint64_t{a[2 * t + 1]}, uint64_t{odd ? t + 7 : 1000u});
  }
  VCHECK(a == b);
}

VTEST_MAIN
