// Tests for the arithmetic/conversion ops added to run real pantheon kernels:
// neg, prmt.b32 (byte permute), and the full cvt family (int<->float, rounding).
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
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

VTEST_MAIN

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
