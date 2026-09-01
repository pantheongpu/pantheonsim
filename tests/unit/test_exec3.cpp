// Tests for the arithmetic/conversion ops added to run real pantheon kernels:
// neg, prmt.b32 (byte permute), and the full cvt family (int<->float, rounding).
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
const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
struct Env {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
};
float as_f32(uint64_t v) {
  float f;
  uint32_t u = static_cast<uint32_t>(v);
  std::memcpy(&f, &u, 4);
  return f;
}
}  // namespace

VTEST(neg_float_and_int) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .f32 %f<3>;
    .reg .f64 %fd<3>;
    .reg .b32 %r<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f40490FDB;      // 3.14159
    neg.f32 %f2, %f1;
    st.global.f32 [%rd2], %f2;
    mov.u32 %r1, 7;
    neg.s32 %r2, %r1;
    st.global.u32 [%rd2+4], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(8);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK(as_f32(e.mem.load_scalar(out, 4)) < -3.14f);
  VCHECK_EQ(static_cast<int32_t>(e.mem.load_scalar(out + 4, 4)), -7);
}

VTEST(prmt_default_byte_select) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0x03020100;   // bytes: a = [00,01,02,03]
    mov.u32 %r2, 0x07060504;   // bytes: b = [04,05,06,07]
    mov.u32 %r3, 0x00007654;   // select bytes 4,5,6,7 into result -> 0x07060504
    prmt.b32 %r4, %r1, %r2, %r3;
    st.global.u32 [%rd2], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), 0x07060504ull);
}

VTEST(cvt_int_to_float_and_back) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .f32 %f<3>;
    .reg .f64 %fd<3>;
    .reg .b32 %r<4>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 10;
    cvt.rn.f32.s32 %f1, %r1;       // 10 -> 10.0f
    st.global.f32 [%rd2], %f1;
    cvt.f64.f32 %fd1, %f1;         // widen to 10.0
    st.global.f64 [%rd2+8], %fd1;
    mov.f32 %f2, 0f42F6E979;       // 123.456
    cvt.rzi.s32.f32 %r2, %f2;      // truncate -> 123
    st.global.u32 [%rd2+16], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(24);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out, 4)), 10.0f);
  double d;
  uint64_t db = e.mem.load_scalar(out + 8, 8);
  std::memcpy(&d, &db, 8);
  VCHECK_EQ(d, 10.0);
  VCHECK_EQ(static_cast<int32_t>(e.mem.load_scalar(out + 16, 4)), 123);
}

VTEST(cvt_float_to_int_clamps) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .f32 %f<2>;
    .reg .b32 %r<2>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f5F000000;       // 9.22e18, well over INT32_MAX
    cvt.rzi.s32.f32 %r1, %f1;      // clamps to INT32_MAX
    st.global.u32 [%rd2], %r1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(static_cast<int32_t>(e.mem.load_scalar(out, 4)), 2147483647);
}

VTEST_MAIN
