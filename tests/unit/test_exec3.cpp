// Tests for the arithmetic/conversion ops added to run real pantheon kernels:
// neg, prmt.b32 (byte permute), and the full cvt family (int<->float, rounding).
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include <array>
#include <cmath>
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

VTEST(shared_memory_reduction_across_warps) {
  // Classic block reduction: every thread writes its tid to shared memory,
  // barrier, then thread 0 sums the whole block. Exercises .shared + bar.sync
  // across multiple warps.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry reduce(.param .u64 out)
{
    .shared .align 4 .b8 buf[512];
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u64 %rd3, buf;
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd3, %rd4;
    st.shared.u32 [%rd5], %r1;
    bar.sync 0;
    setp.ne.s32 %p1, %r1, 0;
    @%p1 bra DONE;
    mov.u32 %r2, 0;          // accumulator
    mov.u32 %r3, 0;          // index
    mov.u32 %r4, %ntid.x;
LOOP:
    setp.ge.u32 %p2, %r3, %r4;
    @%p2 bra STORE;
    mul.wide.u32 %rd6, %r3, 4;
    add.s64 %rd7, %rd3, %rd6;
    ld.shared.u32 %r5, [%rd7];
    add.s32 %r2, %r2, %r5;
    add.s32 %r3, %r3, 1;
    bra LOOP;
STORE:
    st.global.u32 [%rd2], %r2;
DONE:
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4);
  LaunchConfig cfg;
  cfg.block = {96, 1, 1};  // three warps
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{95 * 96 / 2});  // sum 0..95
}

VTEST(shared_memory_is_zeroed_and_per_block) {
  // Each block writes only its own slot; untouched shared reads as zero and
  // nothing leaks between blocks.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .shared .align 4 .b8 buf[64];
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, buf;
    ld.shared.u32 %r1, [%rd3+4];      // never written: must read 0
    mov.u32 %r2, %ctaid.x;
    add.s32 %r3, %r2, 100;
    st.shared.u32 [%rd3], %r3;
    ld.shared.u32 %r4, [%rd3];
    add.s32 %r5, %r4, %r1;
    mul.wide.u32 %rd4, %r2, 4;
    add.s64 %rd5, %rd2, %rd4;
    st.global.u32 [%rd5], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.grid = {4, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t b = 0; b < 4; ++b) VCHECK_EQ(e.mem.load_scalar(out + b * 4, 4), uint64_t{100 + b});
}

VTEST(shared_overflow_is_diagnosed) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k()
{
    .shared .align 4 .b8 buf[16];
    .reg .b32 %r<3>;
    .reg .b64 %rd<4>;
    mov.u64 %rd1, buf;
    mov.u32 %r1, 5;
    st.shared.u32 [%rd1+64], %r1;    // past the 16-byte allocation
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], LaunchConfig{}, {}, e.mem, e.prof));
  VCHECK(err.code() == Err::OutOfBounds);
  VCHECK_CONTAINS(err.what(), "shared memory");
}

VTEST(warp_shuffle_butterfly_reduction) {
  // The standard warp-level sum: shfl.sync.bfly halving reduction.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry wsum(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<20>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;              // value = tid
    mov.u32 %r10, 31;
    mov.u32 %r11, -1;
    mov.u32 %r2, 16;
    shfl.sync.bfly.b32 %r3, %r1, %r2, %r10, %r11;
    add.s32 %r1, %r1, %r3;
    mov.u32 %r2, 8;
    shfl.sync.bfly.b32 %r3, %r1, %r2, %r10, %r11;
    add.s32 %r1, %r1, %r3;
    mov.u32 %r2, 4;
    shfl.sync.bfly.b32 %r3, %r1, %r2, %r10, %r11;
    add.s32 %r1, %r1, %r3;
    mov.u32 %r2, 2;
    shfl.sync.bfly.b32 %r3, %r1, %r2, %r10, %r11;
    add.s32 %r1, %r1, %r3;
    mov.u32 %r2, 1;
    shfl.sync.bfly.b32 %r3, %r1, %r2, %r10, %r11;
    add.s32 %r1, %r1, %r3;
    mov.u32 %r4, %laneid;
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  // A butterfly reduction leaves the full warp sum (0..31 = 496) in EVERY lane.
  for (uint32_t l = 0; l < 32; ++l) VCHECK_EQ(e.mem.load_scalar(out + l * 4, 4), uint64_t{496});
}

VTEST(shuffle_idx_broadcast_and_predicate) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<10>;
    .reg .b64 %rd<5>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    add.s32 %r2, %r1, 1000;
    mov.u32 %r3, 7;               // broadcast lane 7
    mov.u32 %r4, 31;
    mov.u32 %r5, -1;
    shfl.sync.idx.b32 %r6|%p1, %r2, %r3, %r4, %r5;
    selp.b32 %r7, 1, 0, %p1;
    add.s32 %r8, %r6, %r7;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r8;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  // Every lane gets lane 7's value (1007), +1 because the predicate is true.
  for (uint32_t l = 0; l < 32; ++l) VCHECK_EQ(e.mem.load_scalar(out + l * 4, 4), uint64_t{1008});
}

VTEST(bitfield_and_transcendental_ops) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .f32 %f<6>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0xFFFFFFFF;
    mov.u32 %r2, 0x0000FF00;
    bfi.b32 %r3, %r1, %r2, 16, 8;     // insert 8 bits of 0xFF at pos 16
    st.global.u32 [%rd2], %r3;
    mov.u32 %r4, 0x12345678;
    bfe.u32 %r5, %r4, 8, 8;           // extract byte 1 -> 0x56
    st.global.u32 [%rd2+4], %r5;
    mov.f32 %f1, 0f41000000;          // 8.0
    lg2.approx.f32 %f2, %f1;          // 3.0
    st.global.f32 [%rd2+8], %f2;
    mov.f32 %f3, 0f40800000;          // 4.0
    rsqrt.approx.f32 %f4, %f3;        // 0.5
    st.global.f32 [%rd2+12], %f4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), 0x00FFFF00ull);
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 0x56ull);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 8, 4)), 3.0f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 12, 4)), 0.5f);
}

VTEST(nan_aware_setp_variants) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<4>;
    .reg .f32 %f<4>;
    .reg .b32 %r<5>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f7FC00000;          // NaN
    mov.f32 %f2, 0f3F800000;          // 1.0
    setp.neu.f32 %p1, %f1, %f2;       // unordered -> true
    selp.b32 %r1, 1, 0, %p1;
    st.global.u32 [%rd2], %r1;
    setp.ne.f32 %p2, %f1, %f2;        // ordered ne with NaN -> false
    selp.b32 %r2, 1, 0, %p2;
    st.global.u32 [%rd2+4], %r2;
    setp.nan.f32 %p3, %f1, %f2;       // is-NaN test -> true
    selp.b32 %r3, 1, 0, %p3;
    st.global.u32 [%rd2+8], %r3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(12);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), 1ull);
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), 0ull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), 1ull);
}

VTEST(f16_conversion_roundtrip) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .f32 %f<4>;
    .reg .b16 %rs<3>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f40490FDB;          // 3.14159
    cvt.rn.f16.f32 %rs1, %f1;
    cvt.f32.f16 %f2, %rs1;            // back to f32, now half-rounded
    st.global.f32 [%rd2], %f2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  float got = as_f32(e.mem.load_scalar(out, 4));
  // f16 has ~3 decimal digits: 3.14159 -> 3.140625 exactly.
  VCHECK_EQ(got, 3.140625f);
}

VTEST(f16x2_packed_arithmetic) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b16 %rs<5>;
    .reg .f32 %f<4>;
    .reg .b32 %r<8>;
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f40000000;          // 2.0
    cvt.rn.f16.f32 %rs1, %f1;
    mov.f32 %f2, 0f40400000;          // 3.0
    cvt.rn.f16.f32 %rs2, %f2;
    mov.b32 %r1, {%rs1, %rs2};        // packed (2.0, 3.0)
    mov.b32 %r2, {%rs2, %rs1};        // packed (3.0, 2.0)
    mul.rn.f16x2 %r3, %r1, %r2;       // -> (6.0, 6.0)
    st.global.u32 [%rd2], %r3;
    add.rn.f16x2 %r4, %r1, %r2;       // -> (5.0, 5.0)
    st.global.u32 [%rd2+4], %r4;
    neg.f16x2 %r5, %r1;               // -> (-2.0, -3.0)
    st.global.u32 [%rd2+8], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(12);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  auto half = [&](uint64_t packed, int h) { return (packed >> (16 * h)) & 0xFFFF; };
  uint64_t mul = e.mem.load_scalar(out, 4);
  VCHECK_EQ(half(mul, 0), 0x4600ull);  // 6.0 in binary16
  VCHECK_EQ(half(mul, 1), 0x4600ull);
  uint64_t add = e.mem.load_scalar(out + 4, 4);
  VCHECK_EQ(half(add, 0), 0x4500ull);  // 5.0
  VCHECK_EQ(half(add, 1), 0x4500ull);
  uint64_t neg = e.mem.load_scalar(out + 8, 4);
  VCHECK_EQ(half(neg, 0), 0xC000ull);  // -2.0
  VCHECK_EQ(half(neg, 1), 0xC200ull);  // -3.0
}

VTEST(wmma_m16n16k16_matmul) {
  // Fill A with 2.0 and B with 3.0 (every element), C with 0. Then every
  // element of D must be sum over k=0..15 of 2*3 = 96. This is the shape
  // real tensor-core kernels use when hand-filling fragments.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry mma(.param .u64 out)
{
    .reg .b16 %rs<4>;
    .reg .f32 %f<40>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f40000000;          // 2.0
    cvt.rn.f16.f32 %rs1, %f1;
    mov.b32 %r1, {%rs1, %rs1};        // A fragment word
    mov.f32 %f2, 0f40400000;          // 3.0
    cvt.rn.f16.f32 %rs2, %f2;
    mov.b32 %r2, {%rs2, %rs2};        // B fragment word
    mov.f32 %f3, 0f00000000;          // C = 0
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
        {%f10, %f11, %f12, %f13, %f14, %f15, %f16, %f17},
        {%r1, %r1, %r1, %r1, %r1, %r1, %r1, %r1},
        {%r2, %r2, %r2, %r2, %r2, %r2, %r2, %r2},
        {%f3, %f3, %f3, %f3, %f3, %f3, %f3, %f3};
    mov.u32 %r5, %laneid;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f10;       // each lane stores its first D element
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t l = 0; l < 32; ++l)
    VCHECK_EQ(as_f32(e.mem.load_scalar(out + l * 4, 4)), 96.0f);  // 16 * 2 * 3
}

VTEST(wmma_accumulator_is_added) {
  // Same as above but with a non-zero C: D = A*B + C = 96 + 10 = 106.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry mma(.param .u64 out)
{
    .reg .b16 %rs<4>;
    .reg .f32 %f<40>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f40000000;
    cvt.rn.f16.f32 %rs1, %f1;
    mov.b32 %r1, {%rs1, %rs1};
    mov.f32 %f2, 0f40400000;
    cvt.rn.f16.f32 %rs2, %f2;
    mov.b32 %r2, {%rs2, %rs2};
    mov.f32 %f3, 0f41200000;          // 10.0
    wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32
        {%f10, %f11, %f12, %f13, %f14, %f15, %f16, %f17},
        {%r1, %r1, %r1, %r1, %r1, %r1, %r1, %r1},
        {%r2, %r2, %r2, %r2, %r2, %r2, %r2, %r2},
        {%f3, %f3, %f3, %f3, %f3, %f3, %f3, %f3};
    mov.u32 %r5, %laneid;
    mul.wide.u32 %rd3, %r5, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.f32 [%rd4], %f17;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t l = 0; l < 32; ++l)
    VCHECK_EQ(as_f32(e.mem.load_scalar(out + l * 4, 4)), 106.0f);
}

// Blocks run on several host threads by default, so an atomic that is only
// atomic within a block silently loses updates. This is the test that says so:
// 256 blocks of 64 threads each add one to the same counter.
VTEST(atomics_are_atomic_across_blocks) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry bump(.param .u64 p)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  atom.global.add.u32 %r1, [%rd2], 1;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t counter = mem.alloc(4);
  uint32_t zero = 0;
  mem.write(counter, &zero, 4);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {256, 1, 1};
  cfg.block = {64, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &counter, 8);
  exec::launch(m.entries[0], cfg, {arg}, mem, prof);
  uint32_t total = 0;
  mem.read(counter, &total, 4);
  VCHECK_EQ(total, 256u * 64u);
}


// Float atomics are what reductions, gradient accumulation, and embedding
// backward passes are built out of, so a simulator without them cannot run ML
// code. atom.add.f32 must add the values, not their bit patterns.
VTEST(float_atomic_add_accumulates_across_blocks) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry acc(.param .u64 p)
{
  .reg .f32 %f<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.f32 %f1, 0f3F000000;          // 0.5
  atom.global.add.f32 %f2, [%rd2], %f1;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t acc = mem.alloc(4);
  float zero = 0.0f;
  mem.write(acc, &zero, 4);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {128, 1, 1};
  cfg.block = {32, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &acc, 8);
  exec::launch(m.entries[0], cfg, {arg}, mem, prof);
  float total = 0.0f;
  mem.read(acc, &total, 4);
  VCHECK_EQ(total, 128.0f * 32.0f * 0.5f);
}

VTEST(float_atomic_min_max_follow_fmin_ordering) {
  // CUDA's float min/max take the non-NaN operand, which is fmin/fmax
  // ordering rather than a plain comparison.
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry mm(.param .u64 p, .param .f32 v)
{
  .reg .f32 %f<4>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.f32 %f1, [v];
  cvta.to.global.u64 %rd2, %rd1;
  atom.global.min.f32 %f2, [%rd2], %f1;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t cell = mem.alloc(4);
  float start = 5.0f;
  mem.write(cell, &start, 4);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  std::vector<uint8_t> pa(8), va(4);
  std::memcpy(pa.data(), &cell, 8);
  float smaller = 2.0f;
  std::memcpy(va.data(), &smaller, 4);
  exec::launch(m.entries[0], cfg, {pa, va}, mem, prof);
  float got = 0.0f;
  mem.read(cell, &got, 4);
  VCHECK_EQ(got, 2.0f);
  // A NaN operand leaves the stored value, per fmin.
  float nan_v = std::nanf("");
  std::memcpy(va.data(), &nan_v, 4);
  exec::launch(m.entries[0], cfg, {pa, va}, mem, prof);
  mem.read(cell, &got, 4);
  VCHECK_EQ(got, 2.0f);
}

// mov.pred sets a predicate from a constant or copies another. Predicates are a
// separate register file, so this cannot go through the value mov path.
VTEST(mov_pred_from_immediate_and_register) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry mp(.param .u64 p)
{
  .reg .b32 %r<4>;
  .reg .b64 %rd<4>;
  .reg .pred %p<4>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.pred %p1, 1;
  mov.pred %p2, %p1;          // copy
  mov.pred %p3, 0;
  mov.u32 %r1, 0;
  @%p2 mov.u32 %r1, 7;        // taken:      r1 = 7
  @%p3 mov.u32 %r1, 99;       // not taken:  stays 7
  st.global.u32 [%rd2], %r1;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(4);
  uint32_t zero = 0;
  mem.write(out, &zero, 4);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  exec::launch(m.entries[0], cfg, {arg}, mem, prof);
  uint32_t got = 0;
  mem.read(out, &got, 4);
  VCHECK_EQ(got, 7u);
}

// dp4a is the four-way byte dot product quantized inference is built on, so a
// wrong answer here corrupts every quantized matmul while still producing
// plausible-looking output. Concrete values, checked by hand.
VTEST(dp4a_signed_and_unsigned) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry dp(.param .u64 p, .param .u32 av, .param .u32 bv, .param .u32 cv)
{
  .reg .b32 %r<8>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r1, [av];
  ld.param.u32 %r2, [bv];
  ld.param.u32 %r3, [cv];
  cvta.to.global.u64 %rd2, %rd1;
  dp4a.u32.u32 %r4, %r1, %r2, %r3;
  st.global.u32 [%rd2], %r4;
  dp4a.s32.s32 %r5, %r1, %r2, %r3;
  st.global.u32 [%rd2+4], %r5;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(8);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  auto run = [&](uint32_t a, uint32_t b, uint32_t c) {
    std::vector<uint8_t> pa(8), aa(4), ba(4), ca(4);
    std::memcpy(pa.data(), &out, 8);
    std::memcpy(aa.data(), &a, 4);
    std::memcpy(ba.data(), &b, 4);
    std::memcpy(ca.data(), &c, 4);
    exec::launch(m.entries[0], cfg, {pa, aa, ba, ca}, mem, prof);
    uint32_t got[2] = {0, 0};
    mem.read(out, got, 8);
    return std::pair<uint32_t, int32_t>{got[0], static_cast<int32_t>(got[1])};
  };

  // bytes of a = 4,3,2,1 ; bytes of b = 1,1,1,1 -> 4+3+2+1 = 10
  auto r1 = run(0x01020304u, 0x01010101u, 0);
  VCHECK_EQ(r1.first, 10u);
  VCHECK_EQ(r1.second, 10);

  // a = all 0xFF. Unsigned that is 255 each: 255*4 = 1020.
  // Signed it is -1 each: -1*4 = -4.
  auto r2 = run(0xFFFFFFFFu, 0x01010101u, 0);
  VCHECK_EQ(r2.first, 1020u);
  VCHECK_EQ(r2.second, -4);

  // The accumulator is added in.
  auto r3 = run(0x01020304u, 0x01010101u, 7);
  VCHECK_EQ(r3.first, 17u);
  VCHECK_EQ(r3.second, 17);

  // Larger products: bytes 2,2,2,2 against 3,3,3,3 -> 4 * 6 = 24
  auto r4 = run(0x02020202u, 0x03030303u, 0);
  VCHECK_EQ(r4.first, 24u);
  mem.free(out);
}

// A signed narrow load sign-extends into the destination register. Masking to
// the type width instead turns -1 into 255, and the cvt that follows reads the
// positive number: that is how a quantized weight of -1 became +255 and
// corrupted every dequantized tensor while still looking like a plain copy.
VTEST(signed_narrow_loads_sign_extend) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry ldsext(.param .u64 src, .param .u64 dst)
{
  .reg .b16 %rs<4>;
  .reg .b32 %r<6>;
  .reg .f32 %f<4>;
  .reg .b64 %rd<6>;
  ld.param.u64 %rd1, [src];
  ld.param.u64 %rd2, [dst];
  cvta.to.global.u64 %rd3, %rd1;
  cvta.to.global.u64 %rd4, %rd2;
  ld.global.s8 %rs1, [%rd3];        // signed byte
  cvt.rn.f32.s16 %f1, %rs1;         // must see the negative value
  st.global.f32 [%rd4], %f1;
  ld.global.s16 %rs2, [%rd3+2];     // signed halfword
  cvt.s32.s16 %r1, %rs2;
  st.global.u32 [%rd4+4], %r1;
  ld.global.u8 %rs3, [%rd3];        // unsigned stays zero-extended
  cvt.u32.u16 %r2, %rs3;
  st.global.u32 [%rd4+8], %r2;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t src = mem.alloc(16), dst = mem.alloc(16);
  // byte 0 = -1 (0xFF); halfword at +2 = -1000
  uint8_t in[8] = {0xFF, 0x00, 0x00, 0x00, 0, 0, 0, 0};
  int16_t neg = -1000;
  std::memcpy(in + 2, &neg, 2);
  mem.write(src, in, 8);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  std::vector<uint8_t> a0(8), a1(8);
  std::memcpy(a0.data(), &src, 8);
  std::memcpy(a1.data(), &dst, 8);
  exec::launch(m.entries[0], cfg, {a0, a1}, mem, prof);

  float as_float = 0.0f;
  int32_t as_int = 0;
  uint32_t unsigned_byte = 0;
  mem.read(dst, &as_float, 4);
  mem.read(dst + 4, &as_int, 4);
  mem.read(dst + 8, &unsigned_byte, 4);
  VCHECK_EQ(as_float, -1.0f);       // not 255.0
  VCHECK_EQ(as_int, -1000);
  VCHECK_EQ(unsigned_byte, 255u);   // ld.u8 must NOT sign-extend
  mem.free(src);
  mem.free(dst);
}

// Registers are 32 or 64 bits wide, but an operand type can be narrower. Bits
// above the operand's width belong to whatever the register held before and
// must take no part: shr.u16 of 0xFFFFFFFF shifts 0xFFFF, and shr.s16 takes its
// sign from bit 15. Treating every non-64-bit type as 32-bit got both wrong.
VTEST(narrow_integer_ops_use_their_own_width) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry nw(.param .u64 p)
{
  .reg .b16 %rs<8>;
  .reg .b32 %r<8>;
  .reg .b64 %rd<4>;
  .reg .pred %p<4>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, -1;              // register holds 0xFFFFFFFF
  cvt.u16.u32 %rs1, %r1;        // as a 16-bit value that is 0xFFFF
  shr.u16 %rs2, %rs1, 4;        // must be 0x0FFF, not 0xFFFF
  cvt.u32.u16 %r2, %rs2;
  st.global.u32 [%rd2], %r2;
  shr.s16 %rs3, %rs1, 4;        // sign from bit 15: -1 >> 4 = -1
  cvt.s32.s16 %r3, %rs3;
  st.global.u32 [%rd2+4], %r3;
  mov.u32 %r4, 32768;           // 0x8000: negative as s16, positive as s32
  cvt.u16.u32 %rs4, %r4;
  setp.lt.s16 %p1, %rs4, 0;
  selp.b32 %r5, 1, 0, %p1;
  st.global.u32 [%rd2+8], %r5;  // must be 1
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(16);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  exec::launch(m.entries[0], cfg, {arg}, mem, prof);
  uint32_t shr_u = 0, shr_s = 0, is_neg = 0;
  mem.read(out, &shr_u, 4);
  mem.read(out + 4, &shr_s, 4);
  mem.read(out + 8, &is_neg, 4);
  VCHECK_EQ(shr_u, 0x0FFFu);
  VCHECK_EQ(static_cast<int32_t>(shr_s), -1);
  VCHECK_EQ(is_neg, 1u);
  mem.free(out);
}

// The "i" rounding modes round to an integral value while keeping the float
// type: cvt.rpi.f32.f32 is ceilf. Treating them as the bare .rm/.rp float
// rounding modes made ceilf, floorf and truncf return their argument, and
// roundf return x + 0.5 -- the rounding step was simply absent.
VTEST(integral_cvt_rounding_modes_round) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry rnd(.param .u64 p, .param .f32 v)
{
  .reg .f32 %f<8>;
  .reg .b64 %rd<4>;
  ld.param.u64 %rd1, [p];
  ld.param.f32 %f1, [v];
  cvta.to.global.u64 %rd2, %rd1;
  cvt.rpi.f32.f32 %f2, %f1;      // ceil
  st.global.f32 [%rd2], %f2;
  cvt.rmi.f32.f32 %f3, %f1;      // floor
  st.global.f32 [%rd2+4], %f3;
  cvt.rzi.f32.f32 %f4, %f1;      // trunc
  st.global.f32 [%rd2+8], %f4;
  cvt.rni.f32.f32 %f5, %f1;      // nearest, ties to even
  st.global.f32 [%rd2+12], %f5;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(16);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {1, 1, 1};
  auto run = [&](float v) {
    std::vector<uint8_t> pa(8), va(4);
    std::memcpy(pa.data(), &out, 8);
    std::memcpy(va.data(), &v, 4);
    exec::launch(m.entries[0], cfg, {pa, va}, mem, prof);
    float got[4] = {0, 0, 0, 0};
    mem.read(out, got, 16);
    return std::array<float, 4>{got[0], got[1], got[2], got[3]};
  };

  auto a = run(100.75f);
  VCHECK_EQ(a[0], 101.0f);   // ceil
  VCHECK_EQ(a[1], 100.0f);   // floor
  VCHECK_EQ(a[2], 100.0f);   // trunc
  VCHECK_EQ(a[3], 101.0f);   // nearest

  auto b = run(-100.75f);
  VCHECK_EQ(b[0], -100.0f);
  VCHECK_EQ(b[1], -101.0f);
  VCHECK_EQ(b[2], -100.0f);
  VCHECK_EQ(b[3], -101.0f);

  // Ties go to even, not away from zero.
  auto c = run(2.5f);
  VCHECK_EQ(c[3], 2.0f);
  auto d = run(3.5f);
  VCHECK_EQ(d[3], 4.0f);
  mem.free(out);
}

// redux.sync reduces a value across the participating lanes of a warp and
// gives every one of them the result.
VTEST(redux_sync_reduces_across_the_warp) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry rdx(.param .u64 p)
{
  .reg .b32 %r<8>;
  .reg .b64 %rd<6>;
  ld.param.u64 %rd1, [p];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  redux.sync.add.u32 %r2, %r1, -1;
  redux.sync.max.u32 %r3, %r1, -1;
  redux.sync.min.u32 %r4, %r1, -1;
  mul.wide.u32 %rd3, %r1, 4;
  add.s64 %rd4, %rd2, %rd3;
  st.global.u32 [%rd4], %r2;
  setp.eq.u32 %p1, %r1, 0;
  @%p1 st.global.u32 [%rd2+128], %r3;
  @%p1 st.global.u32 [%rd2+132], %r4;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(256);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  std::vector<uint8_t> arg(8);
  std::memcpy(arg.data(), &out, 8);
  exec::launch(m.entries[0], cfg, {arg}, mem, prof);
  // 0 + 1 + ... + 31 = 496, and every lane must see it.
  std::vector<uint32_t> sums(32, 0);
  mem.read(out, sums.data(), 32 * 4);
  for (uint32_t v : sums) VCHECK_EQ(v, 496u);
  uint32_t mx = 0, mn = 0;
  mem.read(out + 128, &mx, 4);
  mem.read(out + 132, &mn, 4);
  VCHECK_EQ(mx, 31u);
  VCHECK_EQ(mn, 0u);
  mem.free(out);
}

// bar.red is a barrier that also produces a value: a predicate reduced across
// every thread in the block. It cannot complete until every warp has arrived,
// so it contributes, waits, and collects on release.
VTEST(bar_red_reduces_across_the_block) {
  const char* kPtx = R"(
.version 8.3
.target sm_86
.address_size 64
.visible .entry br(.param .u64 p, .param .u32 thresh)
{
  .reg .b32 %r<8>;
  .reg .b64 %rd<6>;
  .reg .pred %p<6>;
  ld.param.u64 %rd1, [p];
  ld.param.u32 %r7, [thresh];
  cvta.to.global.u64 %rd2, %rd1;
  mov.u32 %r1, %tid.x;
  setp.lt.u32 %p1, %r1, %r7;      // true for the first `thresh` threads
  bar.red.or.pred %p2, 0, %p1;    // any thread in the block?
  bar.red.and.pred %p3, 0, %p1;   // all threads in the block?
  selp.b32 %r2, 1, 0, %p2;
  selp.b32 %r3, 1, 0, %p3;
  setp.eq.u32 %p4, %r1, 0;
  @%p4 st.global.u32 [%rd2], %r2;
  @%p4 st.global.u32 [%rd2+4], %r3;
  ret;
}
)";
  auto m = ptx::parse(kPtx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(16);
  DeviceProfile prof = load_gpu("nvidia/a10");
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {128, 1, 1};  // four warps, so the reduction spans warps
  auto run = [&](uint32_t thresh) {
    std::vector<uint8_t> pa(8), ta(4);
    std::memcpy(pa.data(), &out, 8);
    std::memcpy(ta.data(), &thresh, 4);
    exec::launch(m.entries[0], cfg, {pa, ta}, mem, prof);
    uint32_t any = 0, all = 0;
    mem.read(out, &any, 4);
    mem.read(out + 4, &all, 4);
    return std::pair<uint32_t, uint32_t>{any, all};
  };

  // Nobody: neither any nor all.
  auto none = run(0);
  VCHECK_EQ(none.first, 0u);
  VCHECK_EQ(none.second, 0u);
  // One thread, and it is in the first warp: any but not all. This is the case
  // that only works if warps beyond the first also contribute.
  auto one = run(1);
  VCHECK_EQ(one.first, 1u);
  VCHECK_EQ(one.second, 0u);
  // A whole warp's worth, still not the whole block.
  auto warp = run(32);
  VCHECK_EQ(warp.first, 1u);
  VCHECK_EQ(warp.second, 0u);
  // Everyone: both.
  auto every = run(128);
  VCHECK_EQ(every.first, 1u);
  VCHECK_EQ(every.second, 1u);
  mem.free(out);
}

// ---- cp.async ----
//
// The instruction's whole meaning is that the copy is *not* finished when it
// issues. A test that only checked the data arrives would pass just as well
// against a plain synchronous copy and prove nothing, so these pin the timing.

VTEST(cp_async_lands_only_after_the_wait) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 src, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .shared .align 16 .b8 tile[16];
    ld.param.u64 %rd1, [src];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [out];
    cvta.to.global.u64 %rd4, %rd3;
    mov.u32 %r1, tile;
    mov.u32 %r6, 0xAAAAAAAA;
    st.shared.u32 [%r1], %r6;                 // poison the destination
    cp.async.cg.shared.global [%r1], [%rd2], 16;
    cp.async.commit_group;
    ld.shared.u32 %r2, [%r1];                 // before the wait: still poison
    st.global.u32 [%rd4], %r2;
    cp.async.wait_group 0;
    ld.shared.u32 %r3, [%r1];                 // after the wait: the copy
    st.global.u32 [%rd4+4], %r3;
    ld.shared.u32 %r4, [%r1+12];
    st.global.u32 [%rd4+8], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t src = e.mem.alloc(16), out = e.mem.alloc(16);
  for (int i = 0; i < 4; ++i) e.mem.store_scalar(src + 4 * i, 4, 0x11111111ull * (i + 1));
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(src), arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0xAAAAAAAA});      // not yet copied
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0x11111111});  // after the wait
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), uint64_t{0x44444444});  // all 16 bytes
}

VTEST(cp_async_wait_group_keeps_later_groups_pending) {
  // Two groups, waiting with one still allowed outstanding: the first must have
  // landed and the second must not. This is the double-buffering pattern every
  // tiled kernel is built on.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 src, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .shared .align 16 .b8 tile[32];
    ld.param.u64 %rd1, [src];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [out];
    cvta.to.global.u64 %rd4, %rd3;
    mov.u32 %r1, tile;
    mov.u32 %r6, 0xAAAAAAAA;
    mov.u32 %r7, 0xBBBBBBBB;
    st.shared.u32 [%r1], %r6;
    st.shared.u32 [%r1+16], %r7;
    cp.async.ca.shared.global [%r1], [%rd2], 4;
    cp.async.commit_group;
    cp.async.ca.shared.global [%r1+16], [%rd2+4], 4;
    cp.async.commit_group;
    cp.async.wait_group 1;
    ld.shared.u32 %r2, [%r1];
    st.global.u32 [%rd4], %r2;
    ld.shared.u32 %r3, [%r1+16];
    st.global.u32 [%rd4+4], %r3;
    cp.async.wait_all;
    ld.shared.u32 %r4, [%r1+16];
    st.global.u32 [%rd4+8], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t src = e.mem.alloc(16), out = e.mem.alloc(16);
  e.mem.store_scalar(src, 4, 0x11111111u);
  e.mem.store_scalar(src + 4, 4, 0x22222222u);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(src), arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0x11111111});      // group 0 landed
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0xBBBBBBBB});  // group 1 pending
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), uint64_t{0x22222222});  // wait_all drains it
}

VTEST(cp_async_zero_fills_past_the_source_size) {
  // The src-size operand is how a kernel reads a tile that runs off the end of
  // a tensor without branching: the bytes past it read as zero, not as
  // whatever happened to follow it in memory.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 src, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .shared .align 16 .b8 tile[16];
    ld.param.u64 %rd1, [src];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [out];
    cvta.to.global.u64 %rd4, %rd3;
    mov.u32 %r1, tile;
    mov.u32 %r5, 4;
    cp.async.cg.shared.global [%r1], [%rd2], 16, %r5;
    cp.async.wait_all;
    ld.shared.u32 %r2, [%r1];
    st.global.u32 [%rd4], %r2;
    ld.shared.u32 %r3, [%r1+4];
    st.global.u32 [%rd4+4], %r3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t src = e.mem.alloc(16), out = e.mem.alloc(16);
  for (int i = 0; i < 4; ++i) e.mem.store_scalar(src + 4 * i, 4, 0x11111111ull * (i + 1));
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(src), arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0x11111111});  // within src-size
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0});       // past it: zero-filled
}

VTEST(movmatrix_transposes_the_warps_8x8_tile) {
  // Element (r,c) = r*8+c, so the transpose is unmistakable: after it, the
  // lane holding row r must hold what column r held.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 2;          // row = lane / 4
    and.b32 %r3, %r1, 3;          // colpair = lane % 4
    shl.b32 %r4, %r2, 3;          // row * 8
    shl.b32 %r5, %r3, 1;          // colpair * 2
    add.s32 %r6, %r4, %r5;        // low half  = row*8 + colpair*2
    add.s32 %r7, %r6, 1;          // high half = that + 1
    shl.b32 %r7, %r7, 16;
    or.b32 %r6, %r6, %r7;
    movmatrix.sync.aligned.m8n8.trans.b16 %r5, %r6;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd3, %rd2, %rd3;
    st.global.u32 [%rd3], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t row = lane / 4, colpair = lane % 4;
    const uint64_t got = e.mem.load_scalar(out + lane * 4, 4);
    // d[row][c] = a[c][row] = c*8 + row, for c = colpair*2 and colpair*2+1.
    const uint32_t lo = (colpair * 2) * 8 + row;
    const uint32_t hi = (colpair * 2 + 1) * 8 + row;
    VCHECK_EQ(got & 0xFFFFu, uint64_t{lo});
    VCHECK_EQ((got >> 16) & 0xFFFFu, uint64_t{hi});
  }
}

VTEST(integer_division_by_zero_follows_the_hardware) {
  // PTX leaves this undefined and real GPUs do not trap. VirtualGPU used to,
  // on the reasoning that a div-by-zero is almost always a bug -- but ggml's
  // flash-attention passes zero for a stride its configuration does not use,
  // takes a remainder from it, and throws the answer away. Trapping made those
  // kernels unrunnable over arithmetic that never mattered.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 7;
    mov.u32 %r2, 0;
    div.s32 %r3, %r1, %r2;
    st.global.u32 [%rd2], %r3;
    rem.s32 %r4, %r1, %r2;
    st.global.u32 [%rd2+4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(8);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  // Deterministic, which is more than hardware promises: all-ones for the
  // quotient, the dividend for the remainder.
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0xFFFFFFFF});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{7});
}

VTEST_MAIN
