// The array-element operand `var[imm]` (no brackets round the whole operand): element imm of a declared
// array, at var + imm * the element's size, as ptxas lays it out. Dr.Jit declares per-thread arrays as
// `.local .align 4 .b32 arr_4[10];` and accesses them as `st.local.b32 arr_4[0], %f15;`.
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
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
struct Env {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a100");
};
}  // namespace

// The element index is scaled by the element size, so arr[1] is the second 32-bit word and agrees
// with [arr+4]; each thread has its own array.
VTEST(local_array_element_form_matches_the_byte_offset_form) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
    .reg .b32 %r<8>; .reg .b64 %rd<4>;
    .local .align 4 .b32 arr[4];
    ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
    mov.u32 %r1,%tid.x;
    st.local.b32 arr[0], %r1;
    add.u32 %r2,%r1,100;
    st.local.b32 arr[1], %r2;
    add.u32 %r3,%r1,200;
    st.local.b32 [arr+8], %r3;
    ld.local.b32 %r4, arr[0];
    ld.local.b32 %r5, [arr+4];
    ld.local.b32 %r6, arr[2];
    mul.wide.u32 %rd2,%r1,12; add.u64 %rd3,%rd1,%rd2;
    st.global.u32 [%rd3],%r4;
    st.global.u32 [%rd3+4],%r5;
    st.global.u32 [%rd3+8],%r6;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t out = e.mem.alloc(8 * 12);
  LaunchConfig cfg;
  cfg.block = {8, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t t = 0; t < 8; ++t) {
    VCHECK_EQ(e.mem.load_scalar(out + t * 12, 4), uint64_t{t});
    VCHECK_EQ(e.mem.load_scalar(out + t * 12 + 4, 4), uint64_t{t + 100});
    VCHECK_EQ(e.mem.load_scalar(out + t * 12 + 8, 4), uint64_t{t + 200});
  }
}

// A wider element scales by its own size; .shared arrays take the same operand.
VTEST(element_size_scales_the_index_in_local_and_shared) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 po)
{
    .reg .b32 %r<4>; .reg .b64 %rd<8>;
    .local .align 8 .b64 w[3];
    .shared .align 2 .b16 h[8];
    ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
    mov.u64 %rd2, 77;
    st.local.b64 w[2], %rd2;
    ld.local.b64 %rd3, [w+16];
    mov.u32 %r1, 9;
    st.shared.b16 h[3], %r1;
    ld.shared.u16 %r2, [h+6];
    st.global.u64 [%rd1], %rd3;
    st.global.u32 [%rd1+8], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t out = e.mem.alloc(16);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 8), 77ull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), 9ull);
}

// The report's module loads.
VTEST(module_of_the_report_loads) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p)
{
    .reg .f32 %f<3>;
    .local .align 4 .b32 arr[4];
    mov.b32 %f0, 0x3f800000;
    st.local.b32 arr[0], %f0;
    st.local.b32 arr[1], %f0;
    ld.local.b32 %f1, arr[0];
    ret;
}
)";
  auto m = ptx::parse(ptx);
  VCHECK(m.entries.size() == 1);
}

// A name that is not a declared array is refused, not read as a byte offset.
VTEST(index_on_an_undeclared_name_is_refused_by_name) {
  const std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p)
{
    .reg .b32 %r<2>;
    ld.local.b32 %r1, nothing[1];
    ret;
}
)";
  const auto err = VCAPTURE(vgpu::Error, ptx::parse(ptx));
  VCHECK_CONTAINS(err.what(), "nothing[1]");
}

VTEST_MAIN
