// Tests for the arithmetic/conversion ops added to run real pantheon kernels:
// neg, prmt.b32 (byte permute), and the full cvt family (int<->float, rounding).
#include <cstring>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/exec/texture.hpp"
#include "vgpu/exec/scheduler.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include <array>
#include <cmath>
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;
using vgpu::exec::TextureTable;
using vgpu::exec::TextureDesc;
using vgpu::exec::ChannelKind;
using vgpu::exec::TexAddress;
using vgpu::exec::TexKind;

namespace {
const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";
// Hopper and later, for the features that only exist there.
const char* kHeader90 = ".version 8.3\n.target sm_90a\n.address_size 64\n";
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
uint64_t f32_bits(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  return u;
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

VTEST(lane_masks_have_the_values_warp_algorithms_depend_on) {
  // popc(%lanemask_le & mask) is how CUB ranks a lane among its peers. Getting
  // these wrong is not an off-by-one in a number, it is an address.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<10>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 20;
    add.s64 %rd4, %rd2, %rd3;
    mov.u32 %r2, %lanemask_eq;
    st.global.u32 [%rd4], %r2;
    mov.u32 %r3, %lanemask_lt;
    st.global.u32 [%rd4+4], %r3;
    mov.u32 %r4, %lanemask_le;
    st.global.u32 [%rd4+8], %r4;
    mov.u32 %r5, %lanemask_gt;
    st.global.u32 [%rd4+12], %r5;
    mov.u32 %r6, %lanemask_ge;
    st.global.u32 [%rd4+16], %r6;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 20);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t l = 0; l < 32; ++l) {
    const uint64_t base = out + l * 20;
    const uint32_t eq = 1u << l;
    const uint32_t lt = l == 0 ? 0u : (~0u >> (32 - l));
    const uint32_t le = lt | eq;
    const uint32_t ge = ~0u << l;
    const uint32_t gt = ge & ~eq;
    VCHECK_EQ(e.mem.load_scalar(base, 4), uint64_t{eq});
    VCHECK_EQ(e.mem.load_scalar(base + 4, 4), uint64_t{lt});
    VCHECK_EQ(e.mem.load_scalar(base + 8, 4), uint64_t{le});
    VCHECK_EQ(e.mem.load_scalar(base + 12, 4), uint64_t{gt});
    VCHECK_EQ(e.mem.load_scalar(base + 16, 4), uint64_t{ge});
  }
}

VTEST(ballot_returns_the_lanes_that_voted_not_their_complement) {
  // vote.ballot's negate_src flag had no default initializer, so a ballot with
  // no '!' read whatever was on the stack: usually true, which returned the
  // complement of the mask. Almost nothing noticed, because most uses feed
  // popc or compare against zero and both survive a complement. CUB's radix
  // sort does not -- it ranks a lane with popc(%lanemask_le & ballot), and a
  // complemented ballot ranked every lane zero and stored below its array.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    .reg .pred %p<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %laneid;
    mul.wide.u32 %rd3, %r1, 12;
    add.s64 %rd4, %rd2, %rd3;
    setp.lt.u32 %p1, %r1, 16;
    vote.sync.ballot.b32 %r2, %p1, -1;
    st.global.u32 [%rd4], %r2;
    setp.eq.s32 %p2, %r1, 0;
    vote.sync.ballot.b32 %r3, %p2, -1;
    st.global.u32 [%rd4+4], %r3;
    vote.sync.ballot.b32 %r4, !%p2, -1;
    st.global.u32 [%rd4+8], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 12);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t l = 0; l < 32; ++l) {
    VCHECK_EQ(e.mem.load_scalar(out + l * 12, 4), uint64_t{0x0000FFFF});
    VCHECK_EQ(e.mem.load_scalar(out + l * 12 + 4, 4), uint64_t{0x00000001});
    VCHECK_EQ(e.mem.load_scalar(out + l * 12 + 8, 4), uint64_t{0xFFFFFFFE});
  }
}

VTEST(activemask_is_the_lanes_still_running) {
  // Half the warp returns early; the rest must see only themselves.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;
    .reg .pred %p<3>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %laneid;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    setp.lt.u32 %p1, %r1, 16;
    @%p1 bra DONE;
    activemask.b32 %r2;
    st.global.u32 [%rd4], %r2;
DONE:
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t l = 16; l < 32; ++l)
    VCHECK_EQ(e.mem.load_scalar(out + l * 4, 4), uint64_t{0xFFFF0000});
}

// ---- counters derived from the addresses themselves ----
//
// Every expected number below is worked out by hand from the access pattern,
// not read back from the engine. A counter checked against itself measures
// nothing.

VTEST(coalesced_and_strided_loads_touch_the_sectors_they_should) {
  // 32 lanes x 4 bytes, consecutive: 128 bytes, which is exactly 4 sectors.
  // The same 32 lanes at a 32-byte stride touch a sector each: 32.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p, .param .u32 stride)
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u32 %r2, [stride];
    mov.u32 %r1, %tid.x;
    mul.lo.s32 %r3, %r1, %r2;
    mul.wide.u32 %rd3, %r3, 1;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r4, [%rd4];
    st.global.u32 [%rd4], %r4;
    ret;
}
)";
  auto m = ptx::parse(ptx);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};

  for (uint32_t stride : {4u, 32u}) {
    Env e;
    uint64_t buf = e.mem.alloc(32 * 64);
    std::vector<uint8_t> sarg(4);
    std::memcpy(sarg.data(), &stride, 4);
    auto st = exec::launch(m.entries[0], cfg, {arg_u64(buf), sarg}, e.mem, e.prof);
    // One load and one store, so two requests from one warp.
    VCHECK_EQ(st.global_requests, 2ull);
    const uint64_t want = stride == 4 ? 4ull : 32ull;   // per request
    VCHECK_EQ(st.global_sectors, want * 2);
  }
}

VTEST(shared_bank_conflicts_are_counted_the_way_hardware_serializes) {
  // Shared memory is 32 banks of 4 bytes, so bank = (address/4) % 32.
  //   stride 1 word : lane i -> bank i          -> no conflict
  //   stride 32 words: every lane -> bank 0, all different words -> 32-way,
  //                    which costs 31 extra passes
  //   every lane the same address -> broadcast, no conflict
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u32 stride)
{
    .reg .b32 %r<8>;
    .shared .align 4 .b8 tile[16384];
    ld.param.u32 %r2, [stride];
    mov.u32 %r1, %tid.x;
    mul.lo.s32 %r3, %r1, %r2;
    shl.b32 %r4, %r3, 2;
    mov.u32 %r5, tile;
    add.s32 %r6, %r5, %r4;
    ld.shared.u32 %r7, [%r6];
    ret;
}
)";
  auto m = ptx::parse(ptx);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  struct Case { uint32_t stride; uint64_t conflicts; };
  for (const Case& c : {Case{1, 0}, Case{32, 31}, Case{0, 0}}) {
    Env e;
    std::vector<uint8_t> sarg(4);
    std::memcpy(sarg.data(), &c.stride, 4);
    auto st = exec::launch(m.entries[0], cfg, {sarg}, e.mem, e.prof);
    VCHECK_EQ(st.shared_requests, 1ull);
    VCHECK_EQ(st.shared_bank_conflicts, c.conflicts);
  }
}

VTEST(a_vector_load_is_one_request_over_the_sectors_it_spans) {
  // 32 lanes x 16 bytes consecutive = 512 bytes = 16 sectors, in one
  // instruction. Counting per lane instead would say 32 requests and miss that
  // this is the efficient shape.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 16;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.v4.u32 {%r2,%r3,%r4,%r5}, [%rd4];
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t buf = e.mem.alloc(32 * 16);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(buf)}, e.mem, e.prof);
  VCHECK_EQ(st.global_requests, 1ull);
  VCHECK_EQ(st.global_sectors, 16ull);
}

VTEST(counters_survive_the_merge_across_host_threads) {
  // A launch of one block returns its statistics directly; more than one block
  // splits across host threads and folds the totals together at the end. The
  // fold used to name each field by hand, so a counter added without touching
  // it read zero -- every single-block test passed while a real kernel
  // reported nothing. This runs enough blocks to take the threaded path.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [p];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, %ctaid.x;
    shl.b32 %r3, %r2, 5;
    add.s32 %r4, %r3, %r1;
    mul.wide.u32 %rd3, %r4, 4;
    add.s64 %rd4, %rd2, %rd3;
    ld.global.u32 %r5, [%rd4];
    mov.u32 %r6, tile;
    shl.b32 %r7, %r1, 2;
    add.s32 %r8, %r6, %r7;
    st.shared.u32 [%r8], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint32_t nblocks = 64;
  uint64_t buf = e.mem.alloc(nblocks * 32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  cfg.grid = {nblocks, 1, 1};
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(buf)}, e.mem, e.prof);
  VCHECK_EQ(st.blocks, uint64_t{nblocks});
  // One coalesced global load per block: 32 lanes x 4 bytes = 4 sectors each.
  VCHECK_EQ(st.global_requests, uint64_t{nblocks});
  VCHECK_EQ(st.global_sectors, uint64_t{nblocks} * 4);
  // One conflict-free shared store per block.
  VCHECK_EQ(st.shared_requests, uint64_t{nblocks});
  VCHECK_EQ(st.shared_bank_conflicts, 0ull);
}

VTEST(the_instruction_classes_partition_the_instructions) {
  // The interesting property is not what any one class counts but that they
  // partition: a classifier that silently drops a case looks exactly like one
  // that works, until the totals are compared. This kernel deliberately
  // touches every class.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 p)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .reg .f32 %f<8>;
    .reg .f64 %fd<4>;
    .reg .pred %pr<4>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [p];              // memory
    cvta.to.global.u64 %rd2, %rd1;       // bit convert
    mov.u32 %r1, %tid.x;                 // misc
    mul.wide.u32 %rd3, %r1, 4;           // integer
    add.s64 %rd4, %rd2, %rd3;            // integer
    ld.global.f32 %f1, [%rd4];           // memory
    add.f32 %f2, %f1, 0f3F800000;        // fp32
    fma.rn.f32 %f3, %f2, %f2, %f1;       // fp32
    cvt.f64.f32 %fd1, %f3;               // bit convert
    add.f64 %fd2, %fd1, %fd1;            // fp64
    cvt.rn.f32.f64 %f4, %fd2;            // bit convert
    setp.gt.f32 %pr1, %f4, 0f00000000;   // misc
    selp.f32 %f5, %f4, %f3, %pr1;        // misc
    st.global.f32 [%rd4], %f5;           // memory
    ret;                                 // control
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t buf = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(buf)}, e.mem, e.prof);

  uint64_t sum = 0;
  for (size_t i = 0; i < static_cast<size_t>(exec::InstClass::Count); ++i)
    sum += st.inst_by_class[i];
  VCHECK_EQ(sum, st.thread_instructions);

  // And the classes that must be non-zero for this kernel actually are, so a
  // classifier that put everything in Misc would not pass by summing right.
  const auto at = [&](exec::InstClass c) { return st.inst_by_class[static_cast<size_t>(c)]; };
  VCHECK(at(exec::InstClass::Fp32) > 0);
  VCHECK(at(exec::InstClass::Fp64) > 0);
  VCHECK(at(exec::InstClass::Integer) > 0);
  VCHECK(at(exec::InstClass::BitConvert) > 0);
  VCHECK(at(exec::InstClass::Memory) > 0);
  VCHECK(at(exec::InstClass::Control) > 0);
  VCHECK(at(exec::InstClass::Misc) > 0);
  VCHECK_EQ(at(exec::InstClass::Tensor), 0ull);   // this kernel has none
}

VTEST(tensor_instructions_are_counted_once_per_warp) {
  // An mma is one instruction the warp issues together. Counting per lane
  // would report 32 for something that happened once, which is the number
  // people would then divide by to get tensor-core throughput.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0;
    mov.f32 %f1, 0f00000000;
    mov.f32 %f2, 0f00000000;
    mov.f32 %f3, 0f00000000;
    mov.f32 %f4, 0f00000000;
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r1,%r1,%r1}, {%r1,%r1}, {%f1,%f2,%f3,%f4};
    mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%f1,%f2,%f3,%f4}, {%r1,%r1,%r1,%r1}, {%r1,%r1}, {%f1,%f2,%f3,%f4};
    st.global.f32 [%rd2], %f1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};   // exactly one warp
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(st.tensor_instructions, 2ull);                                   // per warp
  VCHECK_EQ(st.inst_by_class[static_cast<size_t>(exec::InstClass::Tensor)],
            2ull * 32);                                                      // per lane
}

// ---- shared-memory race detection ----

static const char* kRaceKernel = R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .reg .pred %p<3>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, tile;
    // Warp 0 writes word 0; warp 1 reads it. NO bar.sync between them, so
    // which one goes first is not something this program decided.
    setp.gt.u32 %p1, %r1, 31;
    @%p1 bra READER;
    mov.u32 %r3, 7;
    st.shared.u32 [%r2], %r3;
    bra DONE;
READER:
    ld.shared.u32 %r4, [%r2];
    st.global.u32 [%rd2], %r4;
DONE:
    ret;
}
)";

VTEST(the_per_opcode_histogram_counts_every_mnemonic) {
  // Nine classes say a kernel is memory-heavy; this says which instruction
  // made it so. The first attempt at this interned the opcode at two of the
  // parser's dozens of construction sites and counted almost nothing, so the
  // test checks specific mnemonics rather than a non-empty total.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 1;
    add.s32 %r2, %r1, 1;
    add.s32 %r3, %r2, 1;
    add.s32 %r4, %r3, 1;
    st.global.u32 [%rd2], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{4});

  const auto& names = vgpu::ptx::opcode_names();
  auto count_of = [&](const char* mnemonic) -> uint64_t {
    for (size_t i = 1; i < names.size() && i < st.inst_by_opcode.size(); ++i)
      if (names[i] == mnemonic) return st.inst_by_opcode[i];
    return 0;
  };
  VCHECK_EQ(count_of("add"), uint64_t{3});
  VCHECK_EQ(count_of("st"), uint64_t{1});
  VCHECK_EQ(count_of("ret"), uint64_t{1});
  VCHECK_EQ(count_of("cvta"), uint64_t{1});
  // And the histogram sums to the warp-level issue count, which is what makes
  // it comparable with `instructions` rather than a separate accounting.
  uint64_t total = 0;
  for (uint64_t n : st.inst_by_opcode) total += n;
  VCHECK_EQ(total, st.instructions);
}

VTEST(merging_per_thread_counters_folds_the_opcode_histogram_too) {
  // inst_by_opcode is a vector, not a uint64_t, so it sits outside the block
  // that add() walks by reinterpretation. Folding it by hand is the part that
  // is easy to forget -- and forgetting it leaves the histogram reading zero
  // on any multi-threaded launch however carefully it was collected.
  vgpu::exec::LaunchStats a, b;
  a.instructions = 10;
  a.inst_by_opcode = {0, 5, 3};
  b.instructions = 4;
  b.inst_by_opcode = {0, 1, 0, 7};  // longer than a's, so it must grow
  a.add(b);
  VCHECK_EQ(a.instructions, uint64_t{14});
  VCHECK_EQ(a.inst_by_opcode.size(), size_t{4});
  VCHECK_EQ(a.inst_by_opcode[1], uint64_t{6});
  VCHECK_EQ(a.inst_by_opcode[2], uint64_t{3});
  VCHECK_EQ(a.inst_by_opcode[3], uint64_t{7});
}

// ---- scheduler modes ----
//
// A kernel with a read-modify-write race between two warps through *global*
// memory. The shared-memory detector cannot see this one -- it is not shared
// memory -- so the only thing that exposes it is an execution order in which
// the two warps interleave.
//
// Under the deterministic scheduler they never do: a warp runs from one
// barrier to the next without interruption, so warp 0 completes its whole
// read-modify-write before warp 1 starts, and the answer comes out "right"
// every time. That is the failure mode these modes exist for.
static const char* kGlobalRaceKernel = R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    // Every thread does a non-atomic increment of the same word, 8 times.
    // With any interleaving at all, updates are lost.
    mov.u32 %r5, 0;
LOOP:
    ld.global.u32 %r1, [%rd2];
    add.s32 %r2, %r1, 1;
    st.global.u32 [%rd2], %r2;
    add.s32 %r5, %r5, 1;
    setp.lt.u32 %p1, %r5, 8;
    @%p1 bra LOOP;
    ret;
}
)";

static uint64_t run_global_race(Env& e, const ptx::Module& m, vgpu::exec::SchedulerKind kind,
                                uint64_t seed) {
  uint64_t out = e.mem.alloc(16);
  e.mem.store_scalar(out, 4, 0);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};  // two warps
  cfg.scheduler = kind;
  cfg.scheduler_seed = seed;
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  return e.mem.load_scalar(out, 4);
}

VTEST(the_same_seed_replays_the_same_execution) {
  // The property the whole mode rests on: a race you cannot re-run is a race
  // you cannot fix.
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kGlobalRaceKernel);
  const uint64_t a = run_global_race(e, m, vgpu::exec::SchedulerKind::Random, 12345);
  const uint64_t b = run_global_race(e, m, vgpu::exec::SchedulerKind::Random, 12345);
  VCHECK_EQ(a, b);
  const uint64_t c = run_global_race(e, m, vgpu::exec::SchedulerKind::Adversarial, 999);
  const uint64_t d = run_global_race(e, m, vgpu::exec::SchedulerKind::Adversarial, 999);
  VCHECK_EQ(c, d);
}

VTEST(an_adversarial_order_loses_updates_the_deterministic_one_does_not) {
  // 64 threads x 8 increments = 512 if every update landed. The deterministic
  // order runs each warp to completion, so within a warp the lanes are
  // lockstep and the two warps do not interleave: it reports one fixed answer.
  // The adversarial order preempts every instruction, so updates are lost --
  // which is what the hardware would also do, and what the fixed order hides.
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kGlobalRaceKernel);
  const uint64_t det = run_global_race(e, m, vgpu::exec::SchedulerKind::Deterministic, 0);
  const uint64_t adv = run_global_race(e, m, vgpu::exec::SchedulerKind::Adversarial, 7);
  VCHECK(adv < det);
}

VTEST(different_seeds_explore_different_orders) {
  // If every seed produced the same answer the mode would be a search in name
  // only. At least one pair out of several must differ.
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kGlobalRaceKernel);
  std::vector<uint64_t> seen;
  for (uint64_t seed = 1; seed <= 12; ++seed)
    seen.push_back(run_global_race(e, m, vgpu::exec::SchedulerKind::Random, seed));
  bool any_differ = false;
  for (size_t i = 1; i < seen.size(); ++i)
    if (seen[i] != seen[0]) any_differ = true;
  VCHECK(any_differ);
}

VTEST(a_race_free_kernel_gives_the_same_answer_under_every_order) {
  // The other half, and the one that keeps the modes honest: reordering must
  // not change a correct program. A scheduler that broke race-free kernels
  // would make every report suspect.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    mul.lo.s32 %r2, %r1, 3;
    st.global.u32 [%rd4], %r2;
    ret;
}
)";
  auto check = [&](vgpu::exec::SchedulerKind kind, uint64_t seed) {
    Env e;
    auto m = ptx::parse(ptx);
    uint64_t out = e.mem.alloc(512);
    LaunchConfig cfg;
    cfg.block = {64, 1, 1};
    cfg.scheduler = kind;
    cfg.scheduler_seed = seed;
    exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
    for (uint32_t i = 0; i < 64; ++i)
      VCHECK_EQ(e.mem.load_scalar(out + i * 4, 4), uint64_t{i * 3});
  };
  check(vgpu::exec::SchedulerKind::Deterministic, 0);
  check(vgpu::exec::SchedulerKind::Random, 4242);
  check(vgpu::exec::SchedulerKind::Adversarial, 4242);
}

VTEST(an_unknown_scheduler_name_is_refused_rather_than_defaulted) {
  // Running the default under a name nobody recognises would mean a result
  // that cannot be attributed to an execution order, which is the one thing
  // these modes exist to provide.
  setenv("VGPU_SCHEDULER", "aggressive", 1);
  vgpu::exec::SchedulerKind k = vgpu::exec::SchedulerKind::Deterministic;
  uint64_t seed = 0;
  auto err = VCAPTURE(Error, vgpu::exec::scheduler_from_env(&k, &seed));
  unsetenv("VGPU_SCHEDULER");
  VCHECK(err.code() == Err::InvalidValue);
}

// ---- special registers added for kernels that read them ----

VTEST(clock64_advances_and_never_goes_backwards) {
  // The property kernels actually depend on. A spin-with-a-deadline loop needs
  // the value to move; it does not need it to be a duration, and here it is
  // not one -- see the note at sreg_value().
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, %clock64;
    // Some work between the two reads.
    mov.u32 %r1, 0;
    add.s32 %r1, %r1, 1;
    add.s32 %r1, %r1, 1;
    add.s32 %r1, %r1, 1;
    mov.u64 %rd4, %clock64;
    sub.s64 %rd5, %rd4, %rd3;
    st.global.u64 [%rd2], %rd5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  const uint64_t elapsed = e.mem.load_scalar(out, 8);
  VCHECK(elapsed > 0);                 // it advanced
  VCHECK(elapsed < (1ull << 32));      // and did not wrap or go negative
}

VTEST(smid_is_within_the_devices_multiprocessor_count) {
  // A persistent kernel partitions work by %smid, so the values have to be
  // distinct and in range. Round robin over the profile's SM count gives both.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %smid;
    mov.u32 %r3, %nsmid;
    mul.wide.u32 %rd3, %r1, 8;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r2;
    st.global.u32 [%rd4+4], %r3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(128);
  LaunchConfig cfg;
  cfg.grid = {8, 1, 1};
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  const uint64_t sms = e.prof.limits.multiprocessors;
  VCHECK(sms > 0);
  for (uint32_t b = 0; b < 8; ++b) {
    VCHECK_EQ(e.mem.load_scalar(out + b * 8, 4), uint64_t{b % sms});
    VCHECK_EQ(e.mem.load_scalar(out + b * 8 + 4, 4), sms);
  }
}

VTEST(the_shared_memory_size_registers_report_what_the_launch_gave) {
  // %dynamic_smem_size is the launch's dynamic bytes; %total_smem_size adds the
  // module's static declarations. Both are known exactly, so there is no reason
  // for a kernel to be refused over them.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %dynamic_smem_size;
    mov.u32 %r2, %total_smem_size;
    st.global.u32 [%rd2], %r1;
    st.global.u32 [%rd2+4], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  cfg.shared_bytes = 512;
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{512});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{512 + 256});
}

VTEST(fp8_e4m3_and_e5m2_are_different_formats) {
  // The two FP8 formats are not one shape with a different bias. e4m3 spends
  // its top exponent on ordinary numbers -- it has NO infinity -- so its
  // largest finite value is 448 and 1000 saturates to it. e5m2 is IEEE-shaped
  // with a wider range, and represents 1000 as 1024 (two mantissa bits).
  //
  // Treating e4m3's top exponent as reserved would cost half its range, and
  // the values that vanish are exactly the large activations FP8 inference is
  // scaled to keep.
  std::string ptx = std::string(kHeader90) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<20>;
    .reg .f32 %f<20>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f3F800000;   // 1.0
    mov.f32 %f2, 0f40000000;   // 2.0
    cvt.rn.satfinite.e4m3x2.f32 %r1, %f2, %f1;   // a is the high byte
    st.global.u32 [%rd2], %r1;
    cvt.rn.f16x2.e4m3x2 %r2, %r1;                // and back again
    st.global.u32 [%rd2+4], %r2;
    mov.f32 %f3, 0f447A0000;   // 1000.0
    mov.f32 %f4, 0f00000000;
    cvt.rn.satfinite.e4m3x2.f32 %r3, %f3, %f4;
    cvt.rn.f16x2.e4m3x2 %r4, %r3;
    st.global.u32 [%rd2+8], %r4;
    cvt.rn.satfinite.e5m2x2.f32 %r5, %f3, %f4;
    cvt.rn.f16x2.e5m2x2 %r6, %r5;
    st.global.u32 [%rd2+12], %r6;
    ret;
}
)";
  Env e;
  e.prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  auto half = [](uint16_t h) {
    const uint32_t s = (h >> 15) & 1, ex = (h >> 10) & 0x1F, mn = h & 0x3FF;
    if (ex == 0) return static_cast<float>((s ? -1 : 1) * std::ldexp(static_cast<double>(mn), -24));
    const double v = std::ldexp(static_cast<double>(mn | 0x400), static_cast<int>(ex) - 25);
    return static_cast<float>(s ? -v : v);
  };
  // The packed bytes themselves: e4m3 1.0 is 0x38, 2.0 is 0x40.
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0x4038});
  const uint32_t rt = static_cast<uint32_t>(e.mem.load_scalar(out + 4, 4));
  VCHECK_EQ(half(static_cast<uint16_t>(rt & 0xFFFF)), 1.0f);
  VCHECK_EQ(half(static_cast<uint16_t>(rt >> 16)), 2.0f);
  // 1000 saturates in e4m3 and does not in e5m2 -- the whole difference.
  const uint32_t big4 = static_cast<uint32_t>(e.mem.load_scalar(out + 8, 4));
  const uint32_t big5 = static_cast<uint32_t>(e.mem.load_scalar(out + 12, 4));
  VCHECK_EQ(half(static_cast<uint16_t>(big4 >> 16)), 448.0f);
  VCHECK_EQ(half(static_cast<uint16_t>(big5 >> 16)), 1024.0f);
}

VTEST(nan_propagating_min_and_half_atomics) {
  // min.NaN/max.NaN return NaN when either operand is NaN; plain min/max
  // return the other operand, which is fmin/fmax's rule. A clamp written as
  // max.NaN to keep NaNs visible would quietly launder them away if the
  // modifier were dropped, so the two forms are checked against each other.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f7FC00000;      // NaN
    mov.f32 %f2, 0f40400000;      // 3.0
    min.f32 %f3, %f1, %f2;        // plain: returns 3.0
    min.NaN.f32 %f4, %f1, %f2;    // .NaN: returns NaN
    st.global.f32 [%rd2], %f3;
    st.global.f32 [%rd2+4], %f4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out, 4)), 3.0f);
  VCHECK(std::isnan(as_f32(e.mem.load_scalar(out + 4, 4))));
}

VTEST(packed_half_atomics_update_both_channels) {
  // atom.add.f16x2 is one atomic over two independent halves, which is the
  // point of it: a gradient accumulation touches both channels of a half2
  // without racing on two separate atomics. 64 threads each add {1.0, 2.0}.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f3F800000;      // 1.0
    mov.f32 %f2, 0f40000000;      // 2.0
    cvt.rn.f16x2.f32 %r1, %f2, %f1;   // hi = 2.0, lo = 1.0
    red.global.add.noftz.f16x2 [%rd2], %r1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(8);
  e.mem.store_scalar(out, 4, 0);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  const uint32_t packed = static_cast<uint32_t>(e.mem.load_scalar(out, 4));
  // f16 has an 11-bit mantissa, so 64 and 128 are both exact.
  auto half_to_float = [](uint16_t h) {
    const uint32_t sign = (h >> 15) & 1, exp = (h >> 10) & 0x1F, man = h & 0x3FF;
    if (exp == 0) return std::ldexp(static_cast<float>(man), -24) * (sign ? -1 : 1);
    const float v = std::ldexp(static_cast<float>(man | 0x400), static_cast<int>(exp) - 25);
    return sign ? -v : v;
  };
  VCHECK_EQ(half_to_float(static_cast<uint16_t>(packed & 0xFFFF)), 64.0f);
  VCHECK_EQ(half_to_float(static_cast<uint16_t>(packed >> 16)), 128.0f);
}

VTEST(bf16_arithmetic_is_bf16_not_f16) {
  // bf16 is not an f16 with a different bias: it has f32's exponent range and
  // a 7-bit mantissa. The distinction is visible at 300.0, which bf16 rounds
  // to 300 exactly but which is well inside f16's range too -- so the value
  // that separates them is one f16 cannot hold at all. 70000 overflows f16 to
  // infinity and is an ordinary bf16.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f477A3000;      // 64099 -- finite in bf16, infinite in f16
    cvt.rn.bf16.f32 %r1, %f1;
    mov.f32 %f2, 0f40000000;      // 2.0
    cvt.rn.bf16.f32 %r2, %f2;
    mul.rn.bf16 %r3, %r1, %r2;    // ~128k, still finite in bf16
    cvt.f32.bf16 %f3, %r3;
    st.global.f32 [%rd2], %f3;
    // packed bf16x2: two independent halves
    mov.f32 %f4, 0f3F800000;      // 1.0
    cvt.rn.bf16.f32 %r4, %f4;
    shl.b32 %r5, %r2, 16;
    or.b32 %r6, %r4, %r5;         // {lo=1.0, hi=2.0}
    add.rn.bf16x2 %r7, %r6, %r6;  // {2.0, 4.0}
    and.b32 %r8, %r7, 65535;
    cvt.f32.bf16 %f5, %r8;
    shr.u32 %r9, %r7, 16;
    cvt.f32.bf16 %f6, %r9;
    st.global.f32 [%rd2+4], %f5;
    st.global.f32 [%rd2+8], %f6;
    // max propagates the non-NaN operand, which fmin/fmax do and < does not
    max.bf16 %r10, %r4, %r2;
    cvt.f32.bf16 %f7, %r10;
    st.global.f32 [%rd2+12], %f7;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  // 64099 rounds to bf16 as 64256; doubled that is 128512, and it must be
  // finite -- an f16 decode would have made it infinity long before here.
  const float doubled = as_f32(e.mem.load_scalar(out, 4));
  VCHECK(std::isfinite(doubled));
  VCHECK(doubled > 100000.0f && doubled < 160000.0f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 4, 4)), 2.0f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 8, 4)), 4.0f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 12, 4)), 2.0f);
}

VTEST(bfind_elect_and_isspacep) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<8>;
    .reg .pred %p<4>;
    .shared .align 4 .b8 tile[64];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 1024;            // bit 10
    bfind.u32 %r2, %r1;
    bfind.shiftamt.u32 %r3, %r1;  // 31 - 10
    mov.u32 %r4, 0;
    bfind.u32 %r5, %r4;           // no set bit -> 0xFFFFFFFF
    mov.u32 %r6, 4294967291;      // -5 as s32; ~(-5) = 4, top differing bit is 2
    bfind.s32 %r7, %r6;
    st.global.u32 [%rd2], %r2;
    st.global.u32 [%rd2+4], %r3;
    st.global.u32 [%rd2+8], %r5;
    st.global.u32 [%rd2+12], %r7;
    // one leader for the whole warp
    elect.sync %r8|%p0, -1;
    selp.b32 %r9, 1, 0, %p0;
    st.global.u32 [%rd2+16], %r8;
    // generic pointers: a shared address is shared, a global one is global
    mov.u64 %rd3, tile;
    cvta.shared.u64 %rd4, %rd3;
    isspacep.shared %p1, %rd4;
    selp.b32 %r10, 1, 0, %p1;
    isspacep.global %p2, %rd4;
    selp.b32 %r11, 1, 0, %p2;
    isspacep.global %p3, %rd1;
    selp.b32 %r12, 1, 0, %p3;
    st.global.u32 [%rd2+20], %r10;
    st.global.u32 [%rd2+24], %r11;
    st.global.u32 [%rd2+28], %r12;
    griddepcontrol.wait;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{10});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{21});          // 31 - 10
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), uint64_t{0xFFFFFFFFu}); // no bit set
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), uint64_t{2});          // bfind.s32 of -5
  VCHECK_EQ(e.mem.load_scalar(out + 16, 4), uint64_t{0});          // lane 0 elected
  VCHECK_EQ(e.mem.load_scalar(out + 20, 4), uint64_t{1});          // shared is shared
  VCHECK_EQ(e.mem.load_scalar(out + 24, 4), uint64_t{0});          // and is not global
  VCHECK_EQ(e.mem.load_scalar(out + 28, 4), uint64_t{1});          // the buffer is global
}

VTEST(cp_async_mbarrier_arrive_lands_the_copy_before_the_arrival) {
  // The ordering an Ampere pipeline depends on. Warp 0 issues a cp.async into
  // shared memory and signals the barrier with cp.async.mbarrier.arrive; every
  // thread waits and then reads. If the arrival were signalled before the copy
  // landed, the consumers would read the zero that cp.async deliberately
  // leaves visible until its wait -- so this reads 7 or it reads 0.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 src, .param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<12>;
    .reg .pred %p<4>;
    .shared .align 8 .b8 bar[8];
    .shared .align 16 .b8 stage[16];
    ld.param.u64 %rd1, [src];
    cvta.to.global.u64 %rd2, %rd1;
    ld.param.u64 %rd3, [out];
    cvta.to.global.u64 %rd4, %rd3;
    mov.u32 %r1, %tid.x;
    mov.u64 %rd5, bar;
    mov.u64 %rd6, stage;
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra INITDONE;
    mov.u32 %r2, 64;
    mbarrier.init.shared.b64 [%rd5], %r2;
    mov.u32 %r3, 0;
    st.shared.u32 [%rd6], %r3;
INITDONE:
    bar.sync 0;
    @%p0 bra ARRIVE;
    cp.async.ca.shared.global [%rd6], [%rd2], 4;
    cp.async.mbarrier.arrive.shared.b64 [%rd5];
    bra WAIT;
ARRIVE:
    mbarrier.arrive.shared.b64 %rd7, [%rd5];
WAIT:
    mbarrier.arrive.shared.b64 %rd8, [%rd5];
SPIN:
    mbarrier.test_wait.shared.b64 %p1, [%rd5], %rd8;
    @!%p1 bra SPIN;
    ld.shared.u32 %r5, [%rd6];
    mul.wide.u32 %rd9, %r1, 4;
    add.s64 %rd10, %rd4, %rd9;
    st.global.u32 [%rd10], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint32_t kThreads = 64;
  uint64_t src = e.mem.alloc(16);
  e.mem.store_scalar(src, 4, 7);
  uint64_t out = e.mem.alloc(kThreads * 4);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {kThreads, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(src), arg_u64(out)}, e.mem, e.prof);
  for (uint32_t t = 0; t < kThreads; ++t)
    VCHECK_EQ(e.mem.load_scalar(out + t * 4, 4), uint64_t{7});
}

VTEST(mbarrier_orders_a_producer_against_a_consumer) {
  // The test that matters for a split barrier: warp 0 writes a value, all
  // threads arrive, and every thread spins on the barrier before reading. If
  // the wait returned true early -- the easy way to get the phase comparison
  // backwards -- the consumers would read the zero that was there before, and
  // this would fail with 0 rather than 41.
  //
  // Nothing blocks in the engine here. The wait is a predicate and the kernel
  // spins; the scheduler preempts the spinning warp, the others arrive, and
  // the spinner then sees the phase flip.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .reg .pred %p<4>;
    .shared .align 8 .b8 bar[8];
    .shared .align 4 .b8 buf[4];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u64 %rd3, bar;
    mov.u64 %rd4, buf;
    // thread 0 initializes the barrier for all 64 threads
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra INITDONE;
    mov.u32 %r2, 64;
    mbarrier.init.shared.b64 [%rd3], %r2;
    mov.u32 %r3, 0;
    st.shared.u32 [%rd4], %r3;
INITDONE:
    bar.sync 0;
    // the producer publishes before arriving
    @%p0 bra ARRIVE;
    mov.u32 %r4, 41;
    st.shared.u32 [%rd4], %r4;
ARRIVE:
    mbarrier.arrive.shared.b64 %rd5, [%rd3];
SPIN:
    mbarrier.test_wait.shared.b64 %p1, [%rd3], %rd5;
    @!%p1 bra SPIN;
    // every thread reads only after the barrier released
    ld.shared.u32 %r5, [%rd4];
    mul.wide.u32 %rd6, %r1, 4;
    add.s64 %rd7, %rd2, %rd6;
    st.global.u32 [%rd7], %r5;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const uint32_t kThreads = 64;
  uint64_t out = e.mem.alloc(kThreads * 4);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {kThreads, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t t = 0; t < kThreads; ++t)
    VCHECK_EQ(e.mem.load_scalar(out + t * 4, 4), uint64_t{41});
}

VTEST(mbarrier_counts_threads_not_warps) {
  // A barrier initialized to blockDim.x completes only if every *thread*
  // counts as an arrival. Counting one per warp is the mistake that makes it
  // never complete -- and it would show up as a hang, not a wrong number, so
  // pending_count is checked directly instead.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    .reg .pred %p<4>;
    .shared .align 8 .b8 bar[8];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u64 %rd3, bar;
    setp.ne.u32 %p0, %r1, 0;
    @%p0 bra INITDONE;
    mov.u32 %r2, 96;              // more than the 64 threads that will arrive
    mbarrier.init.shared.b64 [%rd3], %r2;
INITDONE:
    bar.sync 0;
    mbarrier.arrive.shared.b64 %rd5, [%rd3];
    bar.sync 0;
    @%p0 bra DONE;
    mbarrier.pending_count.shared.b64 %r6, [%rd3];
    st.global.u32 [%rd2], %r6;
DONE:
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(8);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  // 96 expected, 64 threads arrived: 32 outstanding. Per warp it would be 94.
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{32});
}

VTEST(an_uninitialized_mbarrier_is_refused) {
  // Waiting on a barrier nobody initialized is a real bug with a silent
  // failure mode: treated as "expected 0", it would complete immediately and
  // the pipeline would read a buffer nobody filled.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k()
{
    .reg .b64 %rd<4>;
    .shared .align 8 .b8 bar[8];
    mov.u64 %rd1, bar;
    mbarrier.arrive.shared.b64 %rd2, [%rd1];
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, e.mem, e.prof));
  VCHECK(err.code() == Err::UnsupportedPtx);
  VCHECK_CONTAINS(err.what(), "has not been initialized");
}

VTEST(red_is_an_atomic_that_keeps_no_answer) {
  // nvcc emits `red` whenever an atomicAdd()'s result is unused, which in a
  // reduction or a histogram is every call -- so a kernel full of atomics can
  // contain no `atom` at all. The memory side must be identical to atom's;
  // only the write-back is skipped.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    add.s32 %r2, %r1, 1;
    red.global.add.u32 [%rd2], %r2;
    red.global.max.u32 [%rd2+4], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  e.mem.store_scalar(out, 4, 0);
  e.mem.store_scalar(out + 4, 4, 0);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{64 * 65 / 2});  // 1..64 summed
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{64});
}

VTEST(match_any_groups_lanes_by_value) {
  // CUB's and cooperative_groups' value-keyed partitions are this instruction.
  // Lanes are given tid/8, so each group of 8 consecutive lanes shares a value
  // and must see exactly its own 8 bits set.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    shr.u32 %r2, %r1, 3;
    match.any.sync.b32 %r3, %r2, -1;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32 * 4);
  LaunchConfig cfg;
  cfg.grid = {1, 1, 1};
  cfg.block = {32, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t lane = 0; lane < 32; ++lane) {
    const uint32_t group = lane / 8;
    const uint64_t want = uint64_t{0xFFu} << (group * 8);
    VCHECK_EQ(e.mem.load_scalar(out + lane * 4, 4), want);
  }
}

VTEST(mul24_szext_and_fns) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 16777215;        // 0xFFFFFF, the widest 24-bit value
    mul24.lo.u32 %r2, %r1, %r1;
    mul24.hi.u32 %r3, %r1, %r1;
    st.global.u32 [%rd2], %r2;
    st.global.u32 [%rd2+4], %r3;
    mov.u32 %r4, 255;             // 0xFF
    mov.u32 %r5, 8;
    szext.clamp.s32 %r6, %r4, %r5;   // sign-extend 0xFF from 8 bits -> -1
    szext.clamp.u32 %r7, %r4, %r5;   // zero-extend -> 255
    st.global.u32 [%rd2+8], %r6;
    st.global.u32 [%rd2+12], %r7;
    mov.u32 %r8, 164;             // 0b10100100: bits 2, 5, 7
    mov.u32 %r9, 0;
    mov.u32 %r10, 2;
    fns.b32 %r11, %r8, %r9, %r10;    // 2nd set bit at or above 0 -> 5
    st.global.u32 [%rd2+16], %r11;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  const uint64_t prod = uint64_t{0xFFFFFFu} * 0xFFFFFFu;   // 48 bits wide
  VCHECK_EQ(e.mem.load_scalar(out, 4), prod & 0xFFFFFFFFull);
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), (prod >> 24) & 0xFFFFFFFFull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), uint64_t{0xFFFFFFFFu});  // -1
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), uint64_t{255});
  VCHECK_EQ(e.mem.load_scalar(out + 16, 4), uint64_t{5});
}

VTEST(lop3_computes_the_truth_table_it_is_given) {
  // ptxas fuses bitwise chains into lop3, so optimized PTX is full of these
  // and a wrong truth table is a wrong mask rather than a crash. The immLut
  // values here are the canonical ones: evaluate the expression on
  // a=0xF0, b=0xCC, c=0xAA and the result is the table.
  //   0xF8 = a | (b & c)      0x96 = a ^ b ^ c
  //   0xFE = a | b | c        0x80 = a & b & c
  //   0x01 = ~(a | b | c)
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 4042322160;      // 0xF0F0F0F0
    mov.u32 %r2, 3435973836;      // 0xCCCCCCCC
    mov.u32 %r3, 2863311530;      // 0xAAAAAAAA
    lop3.b32 %r4, %r1, %r2, %r3, 0xf8;
    lop3.b32 %r5, %r1, %r2, %r3, 0x96;
    lop3.b32 %r6, %r1, %r2, %r3, 0xfe;
    lop3.b32 %r7, %r1, %r2, %r3, 0x80;
    lop3.b32 %r8, %r1, %r2, %r3, 0x01;
    st.global.u32 [%rd2], %r4;
    st.global.u32 [%rd2+4], %r5;
    st.global.u32 [%rd2+8], %r6;
    st.global.u32 [%rd2+12], %r7;
    st.global.u32 [%rd2+16], %r8;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  const uint32_t a = 0xF0F0F0F0u, b = 0xCCCCCCCCu, c = 0xAAAAAAAAu;
  // This one is also the defining property of the encoding: feeding the
  // canonical constants back in returns the table itself, 0xF8 repeated. The
  // first draft of this test asserted a & (b | c) here, which is 0xE0's table,
  // not 0xF8's -- the check caught the comment, not the code.
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{a | (b & c)});
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0xF8F8F8F8u});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{a ^ b ^ c});
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), uint64_t{a | b | c});
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), uint64_t{a & b & c});
  VCHECK_EQ(e.mem.load_scalar(out + 16, 4), uint64_t{~(a | b | c) & 0xFFFFFFFFu});
}

VTEST(slct_testp_and_sad) {
  // slct's selector is compared as a float when the source type says so, which
  // is the case a bit comparison gets wrong: -0.0 has its sign bit set but is
  // >= 0, so it must select a. NaN is not >= 0 and must select b.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<16>;
    .reg .f32 %f<8>;
    .reg .pred %p<4>;
    .reg .b64 %rd<4>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.f32 %f1, 0f42C80000;      // 100.0 -> "a"
    mov.f32 %f2, 0fC2C80000;      // -100.0 -> "b"
    mov.f32 %f3, 0f80000000;      // -0.0 : >= 0, so picks a
    mov.f32 %f4, 0f7FC00000;      // NaN  : not >= 0, so picks b
    slct.f32.f32 %f5, %f1, %f2, %f3;
    slct.f32.f32 %f6, %f1, %f2, %f4;
    st.global.f32 [%rd2], %f5;
    st.global.f32 [%rd2+4], %f6;
    testp.finite.f32 %p0, %f4;
    selp.b32 %r1, 1, 0, %p0;
    testp.notanumber.f32 %p1, %f4;
    selp.b32 %r2, 1, 0, %p1;
    st.global.u32 [%rd2+8], %r1;
    st.global.u32 [%rd2+12], %r2;
    mov.u32 %r3, 7;
    mov.u32 %r4, 20;
    mov.u32 %r5, 5;
    sad.u32 %r6, %r3, %r4, %r5;   // |7-20| + 5 = 18
    st.global.u32 [%rd2+16], %r6;
    mov.u32 %r7, 4294967286;      // -10 as s32
    mov.u32 %r8, 5;
    sad.s32 %r9, %r7, %r8, %r5;   // |-10-5| + 5 = 20
    st.global.u32 [%rd2+20], %r9;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(32);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out, 4)), 100.0f);      // -0.0 selects a
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 4, 4)), -100.0f); // NaN selects b
  VCHECK_EQ(e.mem.load_scalar(out + 8, 4), uint64_t{0});     // NaN is not finite
  VCHECK_EQ(e.mem.load_scalar(out + 12, 4), uint64_t{1});    // NaN is notanumber
  VCHECK_EQ(e.mem.load_scalar(out + 16, 4), uint64_t{18});
  VCHECK_EQ(e.mem.load_scalar(out + 20, 4), uint64_t{20});
}

VTEST(cache_hints_are_accepted_and_do_nothing) {
  // prefetch and createpolicy say where data should be kept, never what a load
  // returns, so with no cache model here honouring them and ignoring them are
  // the same result -- and refusing the kernel would fail it over a
  // performance note. createpolicy still has to write its destination, or the
  // handle would be read later and diagnosed as read-before-write.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    prefetch.global.L2 [%rd2];
    createpolicy.fractional.L2::evict_last.b64 %rd3, 1.0;
    mov.u32 %r1, 42;
    st.global.u32 [%rd2], %r1;
    cvt.u32.u64 %r2, %rd3;
    st.global.u32 [%rd2+4], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{42});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0});  // a policy nothing consults
}

VTEST(cluster_registers_tile_the_grid) {
  // A 2x2 cluster over a 4x2 grid is two clusters side by side. Every block
  // writes where it thinks it is, and the check is against the tiling worked
  // out by hand rather than against the same arithmetic the engine used.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ctaid.y;
    mov.u32 %r3, %nctaid.x;
    mad.lo.s32 %r4, %r2, %r3, %r1;        // linear block id
    mul.lo.s32 %r5, %r4, 16;              // 4 words per block
    cvt.u64.u32 %rd3, %r5;
    add.s64 %rd4, %rd2, %rd3;
    mov.u32 %r6, %clusterid.x;
    mov.u32 %r7, %cluster_ctaid.x;
    mov.u32 %r8, %cluster_ctarank;
    mov.u32 %r9, %cluster_nctarank;
    st.global.u32 [%rd4], %r6;
    st.global.u32 [%rd4+4], %r7;
    st.global.u32 [%rd4+8], %r8;
    st.global.u32 [%rd4+12], %r9;
    ret;
}
)";
  Env e;
  e.prof = load_gpu("nvidia/h100");   // clusters are sm_90 and later
  auto m = ptx::parse(ptx);
  const uint32_t kBlocks = 8;
  uint64_t out = e.mem.alloc(kBlocks * 16);
  LaunchConfig cfg;
  cfg.grid = {4, 2, 1};
  cfg.block = {1, 1, 1};
  cfg.cluster = {2, 2, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t by = 0; by < 2; ++by) {
    for (uint32_t bx = 0; bx < 4; ++bx) {
      const uint32_t b = by * 4 + bx;
      const uint64_t base = out + b * 16;
      VCHECK_EQ(e.mem.load_scalar(base, 4), uint64_t{bx / 2});          // %clusterid.x
      VCHECK_EQ(e.mem.load_scalar(base + 4, 4), uint64_t{bx % 2});      // %cluster_ctaid.x
      // rank is x-fastest within the 2x2 cluster
      VCHECK_EQ(e.mem.load_scalar(base + 8, 4), uint64_t{(bx % 2) + (by % 2) * 2});
      VCHECK_EQ(e.mem.load_scalar(base + 12, 4), uint64_t{4});          // %cluster_nctarank
    }
  }
}

VTEST(without_a_cluster_every_block_is_its_own) {
  // PTX defines a launch with no cluster dimension as behaving like a 1x1x1
  // cluster, so these registers answer on any launch rather than being a
  // cluster-only feature. %clusterid then equals %ctaid, the rank is 0, and
  // %is_explicit_cluster is false -- which is what an H100 reports, and the
  // reason a kernel reading them need not be refused.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %ctaid.x;
    mul.lo.s32 %r2, %r1, 16;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd2, %rd3;
    mov.u32 %r3, %clusterid.x;
    mov.u32 %r4, %cluster_ctarank;
    mov.u32 %r5, %cluster_nctarank;
    mov.u32 %r6, %is_explicit_cluster;
    st.global.u32 [%rd4], %r3;
    st.global.u32 [%rd4+4], %r4;
    st.global.u32 [%rd4+8], %r5;
    st.global.u32 [%rd4+12], %r6;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(3 * 16);
  LaunchConfig cfg;
  cfg.grid = {3, 1, 1};
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t b = 0; b < 3; ++b) {
    const uint64_t base = out + b * 16;
    VCHECK_EQ(e.mem.load_scalar(base, 4), uint64_t{b});    // %clusterid.x == %ctaid.x
    VCHECK_EQ(e.mem.load_scalar(base + 4, 4), uint64_t{0});
    VCHECK_EQ(e.mem.load_scalar(base + 8, 4), uint64_t{1});
    VCHECK_EQ(e.mem.load_scalar(base + 12, 4), uint64_t{0});
  }
}

VTEST(a_cluster_that_does_not_tile_the_grid_is_refused) {
  // Hardware rejects a grid that is not a whole number of clusters, because
  // the leftover blocks belong to no cluster. Inventing a partial one would
  // give %cluster_nctarank a value no block in it agrees with.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k() { ret; }
)";
  Env e;
  e.prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  LaunchConfig cfg;
  cfg.grid = {5, 1, 1};      // 5 is not a multiple of 2
  cfg.block = {1, 1, 1};
  cfg.cluster = {2, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, e.mem, e.prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "not a multiple of the cluster");
}

VTEST(clusters_need_hopper) {
  // Reporting a scheduling level a part does not have is the failure mode this
  // engine exists to avoid, so an A10 refuses rather than pretending.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k() { ret; }
)";
  Env e;   // nvidia/a10, sm_86
  auto m = ptx::parse(ptx);
  LaunchConfig cfg;
  cfg.grid = {4, 1, 1};
  cfg.block = {1, 1, 1};
  cfg.cluster = {2, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, e.mem, e.prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "compute capability 9.0");
}

VTEST(cluster_dims_compiled_into_the_kernel_apply_without_a_launch_attribute) {
  // __cluster_dims__(2,1,1) becomes .reqnctapercluster in the PTX. It is a
  // property of the kernel, so a plain launch still runs 2-block clusters --
  // and %is_explicit_cluster is true, because one was.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out) .reqnctapercluster 2, 1, 1
{
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %ctaid.x;
    mul.lo.s32 %r2, %r1, 8;
    cvt.u64.u32 %rd3, %r2;
    add.s64 %rd4, %rd2, %rd3;
    mov.u32 %r3, %cluster_ctarank;
    mov.u32 %r4, %is_explicit_cluster;
    st.global.u32 [%rd4], %r3;
    st.global.u32 [%rd4+4], %r4;
    ret;
}
)";
  Env e;
  e.prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(4 * 8);
  LaunchConfig cfg;
  cfg.grid = {4, 1, 1};
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  for (uint32_t b = 0; b < 4; ++b) {
    VCHECK_EQ(e.mem.load_scalar(out + b * 8, 4), uint64_t{b % 2});
    VCHECK_EQ(e.mem.load_scalar(out + b * 8 + 4, 4), uint64_t{1});
  }
}

VTEST(gridid_differs_between_launches) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, %gridid;
    st.global.u64 [%rd2], %rd3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t a = e.mem.alloc(8), b = e.mem.alloc(8);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(a)}, e.mem, e.prof);
  exec::launch(m.entries[0], cfg, {arg_u64(b)}, e.mem, e.prof);
  const uint64_t first = e.mem.load_scalar(a, 8), second = e.mem.load_scalar(b, 8);
  VCHECK(first != 0);
  VCHECK(second == first + 1);
}

VTEST(local_traffic_is_counted_in_bytes_as_well_as_operations) {
  // Global and shared had byte totals and local did not, so a spill-heavy
  // kernel could not be compared against a global-memory-heavy one in the same
  // units.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    .local .align 4 .b8 depot[64];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, depot;
    mov.u32 %r1, 42;
    st.local.u32 [%rd3], %r1;
    ld.local.u32 %r2, [%rd3];
    st.global.u32 [%rd2], %r2;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  LaunchConfig cfg;
  cfg.block = {4, 1, 1};
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{42});
  VCHECK_EQ(st.local_stores, uint64_t{4});
  VCHECK_EQ(st.local_loads, uint64_t{4});
  VCHECK_EQ(st.local_bytes_written, uint64_t{16});  // 4 lanes x 4 bytes
  VCHECK_EQ(st.local_bytes_read, uint64_t{16});
}

VTEST(atomics_report_the_bytes_they_moved) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 1;
    atom.global.add.u32 %r2, [%rd2], %r1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  e.mem.store_scalar(out, 4, 0);
  LaunchConfig cfg;
  cfg.block = {8, 1, 1};
  auto st = exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{8});
  VCHECK_EQ(st.atomics, uint64_t{8});
  VCHECK_EQ(st.atomic_bytes, uint64_t{32});  // 8 lanes x 4 bytes
}

// ---- texture and surface objects ----
//
// The handle a kernel receives is only a number; what it means comes from the
// launch's texture table. These pin the addressing and the format conversion,
// which are the parts of a fetch that change results.

VTEST(tex_1d_reads_a_texel_by_integer_index) {
  // tex1Dfetch over linear memory: no filtering, no normalisation, just an
  // indexed read. This is the form ML code uses, because it is a cached load.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 t, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [t];
    ld.param.u64 %rd2, [out];
    cvta.to.global.u64 %rd3, %rd2;
    mov.u32 %r1, %tid.x;
    tex.1d.v4.f32.s32 {%f1, %f2, %f3, %f4}, [%rd1, {%r1}];
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t data = e.mem.alloc(64);
  uint64_t out = e.mem.alloc(64);
  for (uint32_t i = 0; i < 8; ++i) e.mem.store_scalar(data + i * 4, 4, f32_bits(i * 1.5f));

  TextureTable tex;
  TextureDesc d;
  d.base = data;
  d.width = 8;
  d.channels = 1;
  d.channel_bits[0] = 32;
  d.texel_bytes = 4;
  d.kind = ChannelKind::Float;
  tex[0x1234] = d;

  LaunchConfig cfg;
  cfg.block = {8, 1, 1};
  cfg.textures = &tex;
  exec::launch(m.entries[0], cfg, {arg_u64(0x1234), arg_u64(out)}, e.mem, e.prof);
  for (uint32_t i = 0; i < 8; ++i)
    VCHECK_EQ(as_f32(e.mem.load_scalar(out + i * 4, 4)), i * 1.5f);
}

VTEST(an_unknown_texture_handle_is_named_rather_than_read) {
  // The failure mode this prevents: a handle that was never created is a
  // number like any other, and reading through it would produce plausible
  // garbage from wherever it pointed.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 t, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [t];
    mov.u32 %r1, %tid.x;
    tex.1d.v4.f32.s32 {%f1, %f2, %f3, %f4}, [%rd1, {%r1}];
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  TextureTable tex;
  TextureDesc d;
  d.base = out;
  d.width = 4;
  tex[0x1] = d;
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  cfg.textures = &tex;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(0xBEEF), arg_u64(out)},
                                          e.mem, e.prof));
  VCHECK(err.code() == Err::InvalidValue);
}

VTEST(tex_clamps_out_of_range_coordinates_to_the_edge) {
  // cudaAddressModeClamp: the default, and the one that hides bugs if it is
  // wrong, because an off-by-one only shows at the boundary.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 t, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [t];
    ld.param.u64 %rd2, [out];
    cvta.to.global.u64 %rd3, %rd2;
    mov.u32 %r1, %tid.x;
    // index = tid - 2, so lanes 0 and 1 fall off the low edge and 6, 7 off
    // the high one.
    sub.s32 %r2, %r1, 2;
    tex.1d.v4.f32.s32 {%f1, %f2, %f3, %f4}, [%rd1, {%r2}];
    mul.wide.u32 %rd4, %r1, 4;
    add.s64 %rd5, %rd3, %rd4;
    st.global.f32 [%rd5], %f1;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t data = e.mem.alloc(64);
  uint64_t out = e.mem.alloc(64);
  for (uint32_t i = 0; i < 4; ++i) e.mem.store_scalar(data + i * 4, 4, f32_bits(10.0f + i));
  TextureTable tex;
  TextureDesc d;
  d.base = data;
  d.width = 4;
  d.channel_bits[0] = 32;
  d.texel_bytes = 4;
  d.kind = ChannelKind::Float;
  d.address[0] = TexAddress::Clamp;
  tex[7] = d;
  LaunchConfig cfg;
  cfg.block = {8, 1, 1};
  cfg.textures = &tex;
  exec::launch(m.entries[0], cfg, {arg_u64(7), arg_u64(out)}, e.mem, e.prof);
  const float want[8] = {10, 10, 10, 11, 12, 13, 13, 13};
  for (uint32_t i = 0; i < 8; ++i) VCHECK_EQ(as_f32(e.mem.load_scalar(out + i * 4, 4)), want[i]);
}

VTEST(a_missing_channel_reads_as_zero_and_alpha_as_one) {
  // Hardware returns 0 for absent x/y/z and 1 for absent w. A kernel reading
  // .w of a one-channel texture expects 1, and getting 0 is the kind of wrong
  // that looks like a black image rather than an error.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 t, .param .u64 out)
{
    .reg .b32 %r<8>;
    .reg .f32 %f<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [t];
    ld.param.u64 %rd2, [out];
    cvta.to.global.u64 %rd3, %rd2;
    mov.u32 %r1, 0;
    tex.1d.v4.f32.s32 {%f1, %f2, %f3, %f4}, [%rd1, {%r1}];
    st.global.f32 [%rd3], %f1;
    st.global.f32 [%rd3+4], %f2;
    st.global.f32 [%rd3+8], %f3;
    st.global.f32 [%rd3+12], %f4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t data = e.mem.alloc(16);
  uint64_t out = e.mem.alloc(32);
  e.mem.store_scalar(data, 4, f32_bits(2.5f));
  TextureTable tex;
  TextureDesc d;
  d.base = data;
  d.width = 1;
  d.channels = 1;
  d.channel_bits[0] = 32;
  d.texel_bytes = 4;
  d.kind = ChannelKind::Float;
  tex[3] = d;
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  cfg.textures = &tex;
  exec::launch(m.entries[0], cfg, {arg_u64(3), arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out, 4)), 2.5f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 4, 4)), 0.0f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 8, 4)), 0.0f);
  VCHECK_EQ(as_f32(e.mem.load_scalar(out + 12, 4)), 1.0f);
}

VTEST(a_surface_write_then_read_round_trips) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 s)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [s];
    mov.u32 %r1, %tid.x;
    shl.b32 %r2, %r1, 2;
    mov.u32 %r3, 0;
    suld.b.2d.b32.trap {%r4}, [%rd1, {%r2, %r3}];
    add.s32 %r5, %r4, 100;
    sust.b.2d.b32.trap [%rd1, {%r2, %r3}], {%r5};
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t data = e.mem.alloc(64);
  for (uint32_t i = 0; i < 4; ++i) e.mem.store_scalar(data + i * 4, 4, i);
  TextureTable tex;
  TextureDesc d;
  d.base = data;
  d.width = 4;
  d.height = 1;
  d.channel_bits[0] = 32;
  d.texel_bytes = 4;
  d.object = TexKind::Surface;
  tex[9] = d;
  LaunchConfig cfg;
  cfg.block = {4, 1, 1};
  cfg.textures = &tex;
  exec::launch(m.entries[0], cfg, {arg_u64(9)}, e.mem, e.prof);
  for (uint32_t i = 0; i < 4; ++i) VCHECK_EQ(e.mem.load_scalar(data + i * 4, 4), uint64_t{i + 100});
}

VTEST(a_surface_access_past_the_edge_faults_rather_than_wrapping) {
  // suld/sust carry a ".trap" out-of-range policy, and it means what it says.
  // Clamping instead would turn an indexing bug into a plausible picture.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 s)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [s];
    mov.u32 %r2, 64;
    mov.u32 %r3, 0;
    suld.b.2d.b32.trap {%r4}, [%rd1, {%r2, %r3}];
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t data = e.mem.alloc(64);
  TextureTable tex;
  TextureDesc d;
  d.base = data;
  d.width = 4;
  d.height = 1;
  d.channel_bits[0] = 32;
  d.texel_bytes = 4;
  d.object = TexKind::Surface;
  tex[9] = d;
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  cfg.textures = &tex;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(9)}, e.mem, e.prof));
  VCHECK(err.code() == Err::OutOfBounds);
}

VTEST(a_texture_handle_used_as_a_surface_is_refused) {
  // The two have the same shape of handle and are not interchangeable.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 s)
{
    .reg .b32 %r<8>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [s];
    mov.u32 %r2, 0;
    mov.u32 %r3, 0;
    suld.b.2d.b32.trap {%r4}, [%rd1, {%r2, %r3}];
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t data = e.mem.alloc(64);
  TextureTable tex;
  TextureDesc d;
  d.base = data;
  d.width = 4;
  d.height = 1;
  d.texel_bytes = 4;
  d.object = TexKind::Texture;  // a texture, used by a surface instruction
  tex[9] = d;
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  cfg.textures = &tex;
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(9)}, e.mem, e.prof));
  VCHECK(err.code() == Err::InvalidValue);
}

// ---- cooperative launch: a grid-wide barrier ----
//
// cg::this_grid().sync() is not an instruction. It compiles to an atomic
// increment of a counter in global memory followed by a spin on that counter,
// so it only terminates if the blocks that have not arrived yet can still run.
// This is that pattern written directly: each block adds one to a counter, then
// spins until the counter reaches the block count, then reads a word another
// block wrote.
static const char* kGridBarrierKernel = R"(
.visible .entry k(.param .u64 counter, .param .u64 out)
{
    .reg .pred %p<4>;
    .reg .b32 %r<16>;
    .reg .b64 %rd<12>;
    ld.param.u64 %rd1, [counter];
    ld.param.u64 %rd2, [out];
    cvta.to.global.u64 %rd3, %rd1;
    cvta.to.global.u64 %rd4, %rd2;
    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %nctaid.x;
    // Each block writes its own id + 1 into out[blockIdx].
    mul.wide.u32 %rd5, %r1, 4;
    add.s64 %rd6, %rd4, %rd5;
    add.s32 %r3, %r1, 1;
    st.global.u32 [%rd6], %r3;
    // Arrive.
    // One thread per block, so the arrival is one increment per block with no
    // intra-block divergence to reason about -- this is a test of the grid
    // barrier, not of how a block gathers its own threads first.
    mov.u32 %r4, 1;
    atom.add.release.gpu.global.u32 %r5, [%rd3], %r4;
    // Spin until every block has arrived.
SPIN:
    ld.acquire.gpu.global.u32 %r6, [%rd3];
    setp.lt.u32 %p1, %r6, %r2;
    @%p1 bra SPIN;
    // Now read the word the *next* block wrote. Only correct if the barrier
    // held: without it this block may arrive before that one has written.
    add.s32 %r7, %r1, 1;
    rem.u32 %r8, %r7, %r2;
    mul.wide.u32 %rd7, %r8, 4;
    add.s64 %rd8, %rd4, %rd7;
    ld.global.u32 %r9, [%rd8];
    // Stash it at out[nctaid + blockIdx] so the check can see both halves.
    add.s32 %r10, %r1, %r2;
    mul.wide.u32 %rd9, %r10, 4;
    add.s64 %rd10, %rd4, %rd9;
    st.global.u32 [%rd10], %r9;
    ret;
}
)";

VTEST(a_cooperative_launch_lets_blocks_wait_for_each_other) {
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kGridBarrierKernel);
  uint64_t counter = e.mem.alloc(4);
  uint64_t out = e.mem.alloc(64);
  e.mem.store_scalar(counter, 4, 0);
  LaunchConfig cfg;
  cfg.grid = {4, 1, 1};
  cfg.block = {1, 1, 1};
  cfg.cooperative = true;
  exec::launch(m.entries[0], cfg, {arg_u64(counter), arg_u64(out)}, e.mem, e.prof);
  // Every block arrived.
  VCHECK_EQ(e.mem.load_scalar(counter, 4), uint64_t{4});
  // ...and each read the next block's value, which only the barrier makes safe.
  for (uint32_t b = 0; b < 4; ++b) {
    VCHECK_EQ(e.mem.load_scalar(out + b * 4, 4), uint64_t{b + 1});
    VCHECK_EQ(e.mem.load_scalar(out + (4 + b) * 4, 4), uint64_t{(b + 1) % 4 + 1});
  }
}

// The same kernel without the cooperative flag must not quietly work: blocks
// run one at a time, so the first to spin waits for a block that has not
// started. That is a hang, and the step budget is what turns it into a
// diagnosable error rather than a wedged process.
//
// VGPU_THREADS=1 is the point of the test, not a workaround for it. An ordinary
// launch is free to run blocks on parallel workers, and when it does, a grid
// barrier can be satisfied by luck -- which is precisely why grid.sync()
// outside a cooperative launch is undefined rather than merely slow. Pinning to
// one worker asks the question the test means to ask: with blocks run in
// sequence, does the spin get diagnosed?
VTEST(the_same_kernel_without_a_cooperative_launch_does_not_hang_forever) {
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kGridBarrierKernel);
  uint64_t counter = e.mem.alloc(4);
  uint64_t out = e.mem.alloc(64);
  e.mem.store_scalar(counter, 4, 0);
  LaunchConfig cfg;
  cfg.grid = {4, 1, 1};
  cfg.block = {1, 1, 1};
  cfg.cooperative = false;
  cfg.max_steps = 100000;  // small, so the spin is caught quickly
  setenv("VGPU_THREADS", "1", 1);
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(counter), arg_u64(out)},
                                          e.mem, e.prof));
  unsetenv("VGPU_THREADS");
  VCHECK(err.code() == Err::ExecLimit);
}

VTEST(a_shared_race_between_warps_is_reported) {
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kRaceKernel);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};   // two warps, so they can race with each other
  setenv("VGPU_RACE", "1", 1);
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof));
  unsetenv("VGPU_RACE");
  VCHECK(err.code() == Err::DataRace);
  VCHECK_CONTAINS(err.what(), "shared memory");
  VCHECK_CONTAINS(err.what(), "bar.sync");
}

VTEST(the_same_kernel_is_silent_when_a_barrier_orders_it) {
  // The identical accesses, with a bar.sync between the write and the read.
  // If the detector fired here it would be useless: every real kernel does
  // this, and a checker that cannot tell ordered from unordered is noise.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .reg .pred %p<3>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, tile;
    setp.gt.u32 %p1, %r1, 31;
    @%p1 bra AFTER;
    mov.u32 %r3, 7;
    st.shared.u32 [%r2], %r3;
AFTER:
    bar.sync 0;
    ld.shared.u32 %r4, [%r2];
    st.global.u32 [%rd2], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  setenv("VGPU_RACE", "1", 1);
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  unsetenv("VGPU_RACE");
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{7});
}

VTEST(the_racy_kernel_runs_without_complaint_when_detection_is_off) {
  // Detection is opt-in, and its cost is a shadow word per shared word. A
  // kernel nobody is checking must not pay for it or be stopped by it.
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kRaceKernel);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
}

// Two warps storing the *same* value to the same shared word.
//
// This is llama.cpp's mul_mat_q, reduced: with `need_check` on, the tile loader
// clamps out-of-range rows with `i = min(i, i_max)`, so several warps recompute
// the same source pointer and write the same bytes to the same word. It is a
// race by the strict definition and cannot affect the result -- no reader and
// no other writer can tell which store won, because the bytes are identical
// either way.
//
// Reporting it would make the detector useless on the code it exists to check,
// so by default it is silent. VGPU_RACE=2 still reports it, because the strict
// definition has its uses.
static const char* kSameValueRaceKernel = R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r2, tile;
    // Every thread in both warps stores 7 to word 0, with no bar.sync anywhere.
    mov.u32 %r3, 7;
    st.shared.u32 [%r2], %r3;
    ld.shared.u32 %r4, [%r2];
    st.global.u32 [%rd2], %r4;
    ret;
}
)";

VTEST(two_warps_writing_the_same_value_is_not_reported) {
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kSameValueRaceKernel);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};  // two warps
  setenv("VGPU_RACE", "1", 1);
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  unsetenv("VGPU_RACE");
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{7});
}

VTEST(the_same_value_write_is_still_a_race_under_strict_mode) {
  Env e;
  auto m = ptx::parse(std::string(kHeader) + kSameValueRaceKernel);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  setenv("VGPU_RACE", "2", 1);
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof));
  unsetenv("VGPU_RACE");
  VCHECK(err.code() == Err::DataRace);
}

// The converse, and the reason the value check is not simply "ignore the second
// write": a store that leaves the bytes alone is still recorded as a write, so
// a later store of a *different* value is caught against it.
VTEST(a_differing_write_after_a_redundant_one_is_still_reported) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<3>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mov.u32 %r2, tile;
    // Word 0 already holds 7 when warp 0 stores 7 into it -- a store that
    // changes nothing. Warp 1 then stores 9, which does change it.
    mov.u32 %r3, 7;
    st.shared.u32 [%r2], %r3;
    bar.sync 0;
    setp.gt.u32 %p1, %r1, 31;
    @%p1 bra SECOND;
    st.shared.u32 [%r2], %r3;
    bra DONE;
SECOND:
    mov.u32 %r5, 9;
    st.shared.u32 [%r2], %r5;
DONE:
    st.global.u32 [%rd2], %r3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {64, 1, 1};
  setenv("VGPU_RACE", "1", 1);
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof));
  unsetenv("VGPU_RACE");
  VCHECK(err.code() == Err::DataRace);
}

VTEST(one_warp_reusing_its_own_shared_words_is_not_a_race) {
  // A warp racing with itself is impossible: its own accesses are ordered by
  // the program. A detector that keyed on the word alone would say otherwise.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    .shared .align 4 .b8 tile[256];
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r2, tile;
    mov.u32 %r3, 11;
    st.shared.u32 [%r2], %r3;
    ld.shared.u32 %r4, [%r2];
    add.s32 %r5, %r4, 1;
    st.shared.u32 [%r2], %r5;
    ld.shared.u32 %r6, [%r2];
    st.global.u32 [%rd2], %r6;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};   // one warp
  setenv("VGPU_RACE", "1", 1);
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  unsetenv("VGPU_RACE");
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{12});
}

// ---- extended-precision arithmetic (the condition-code carry bit) ----
//
// These are what a compiler emits when it synthesises arithmetic wider than the
// native register: Numba builds every 64-bit array index this way, so a wrong
// carry shows up as a wrong address rather than a wrong number.

VTEST(add_cc_then_addc_carries_between_halves) {
  // 0xFFFFFFFF + 1 overflows the low half and must set the carry, which addc
  // then folds into the high half: {lo=0xFFFFFFFF, hi=0} + 1 == {0, 1}.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 4294967295;
    mov.u32 %r2, 0;
    add.cc.u32 %r3, %r1, 1;
    addc.u32 %r4, %r2, 0;
    st.global.u32 [%rd2], %r3;
    st.global.u32 [%rd2+4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{1});
}

VTEST(add_cc_leaves_the_carry_clear_when_nothing_overflows) {
  // The complement of the test above: addc must not invent a carry. Getting
  // this backwards is invisible in the overflow case and wrong everywhere else.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 5;
    add.cc.u32 %r3, %r1, 6;
    addc.u32 %r4, 100, 0;
    st.global.u32 [%rd2], %r3;
    st.global.u32 [%rd2+4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{11});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{100});
}

VTEST(sub_cc_then_subc_borrows_between_halves) {
  // {lo=0, hi=1} - 1 == {0xFFFFFFFF, 0}. PTX defines the subtract's carry as
  // the carry-out of (a + ~b + 1), so a borrow *clears* the bit and subc
  // subtracts the extra one. Treating a borrow as a set bit gives hi=1 here.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 0;
    mov.u32 %r2, 1;
    sub.cc.u32 %r3, %r1, 1;
    subc.u32 %r4, %r2, 0;
    st.global.u32 [%rd2], %r3;
    st.global.u32 [%rd2+4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0xFFFFFFFFull});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0});
}

VTEST(mad_lo_cc_and_madc_hi_build_a_64_bit_product) {
  // The full 64x64 pattern ptxas emits: 0xFFFFFFFF * 3 == 0x2FFFFFFFD, so the
  // low half wraps and the high half must receive the carry on top of the
  // product's own high half.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<12>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, 4294967295;
    mov.u32 %r2, 3;
    mad.lo.cc.u32 %r3, %r1, %r2, 0;
    madc.hi.u32 %r4, %r1, %r2, 0;
    st.global.u32 [%rd2], %r3;
    st.global.u32 [%rd2+4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  // 0xFFFFFFFF * 3 = 0x2_FFFFFFFD
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{0xFFFFFFFDull});
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{2});
}

VTEST(the_carry_bit_is_per_lane_not_per_warp) {
  // Lane 0 overflows and lane 1 does not. A carry bit shared across the warp
  // would give both lanes the same high half; each lane owns its own.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .pred %p<2>;
    .reg .b32 %r<12>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    // lane 0 adds 0xFFFFFFFF + 1 (overflows), lane 1 adds 1 + 1 (does not)
    setp.eq.u32 %p1, %r1, 0;
    selp.b32 %r2, 4294967295, 1, %p1;
    add.cc.u32 %r3, %r2, 1;
    addc.u32 %r4, 0, 0;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {2, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 4), uint64_t{1});      // lane 0 carried
  VCHECK_EQ(e.mem.load_scalar(out + 4, 4), uint64_t{0});  // lane 1 did not
}

VTEST(decimal_literals_above_int64_max_keep_their_bit_pattern) {
  // ptxas prints the float sign-bit mask as decimal 9223372036854775808, which
  // overflows a signed parse. Rejecting it made every kernel that negates a
  // double unloadable.
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b32 %r<4>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u64 %rd3, 9223372036854775808;
    st.global.u64 [%rd2], %rd3;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(64);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 8), uint64_t{0x8000000000000000ull});
}

// INT64_MIN / -1 is the one signed quotient that does not fit. It was a SIGFPE
// that took the process down; it wraps, and the remainder is zero.
VTEST(signed_64bit_division_of_min_by_minus_one_wraps) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [out];
    mov.s64 %rd2, 9223372036854775808;
    div.s64 %rd3, %rd2, -1;
    rem.s64 %rd4, %rd2, -1;
    st.global.u64 [%rd1], %rd3;
    st.global.u64 [%rd1+8], %rd4;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(16);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 8), 0x8000000000000000ull);
  VCHECK_EQ(e.mem.load_scalar(out + 8, 8), 0ull);
}

// 64-bit float -> int saturates at the integer limits. Clamping to INT64_MAX as
// a double rounded up to 2^63, and +inf came out as INT64_MIN (0 for u64).
VTEST(cvt_float_to_64bit_int_saturates_at_the_limits) {
  std::string ptx = std::string(kHeader) + R"(
.visible .entry k(.param .u64 out)
{
    .reg .b64 %rd<16>;
    ld.param.u64 %rd1, [out];
    mov.f64 %rd2, 0d7FF0000000000000;
    cvt.s64.f64 %rd3, %rd2;
    st.global.u64 [%rd1], %rd3;
    mov.f64 %rd4, 0d43E0000000000000;
    cvt.s64.f64 %rd5, %rd4;
    st.global.u64 [%rd1+8], %rd5;
    cvt.u64.f64 %rd6, %rd2;
    st.global.u64 [%rd1+16], %rd6;
    mov.f64 %rd7, 0d43F0000000000000;
    cvt.u64.f64 %rd8, %rd7;
    st.global.u64 [%rd1+24], %rd8;
    mov.f64 %rd9, 0dFFF0000000000000;
    cvt.s64.f64 %rd10, %rd9;
    st.global.u64 [%rd1+32], %rd10;
    mov.f64 %rd11, 0d7FF8000000000000;
    cvt.s64.f64 %rd12, %rd11;
    st.global.u64 [%rd1+40], %rd12;
    ret;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  uint64_t out = e.mem.alloc(48);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 8), 0x7fffffffffffffffull);        // +inf -> s64 max
  VCHECK_EQ(e.mem.load_scalar(out + 8, 8), 0x7fffffffffffffffull);    // 2^63 -> s64 max
  VCHECK_EQ(e.mem.load_scalar(out + 16, 8), 0xffffffffffffffffull);   // +inf -> u64 max
  VCHECK_EQ(e.mem.load_scalar(out + 24, 8), 0xffffffffffffffffull);   // 2^64 -> u64 max
  VCHECK_EQ(e.mem.load_scalar(out + 32, 8), 0x8000000000000000ull);   // -inf -> s64 min
  VCHECK_EQ(e.mem.load_scalar(out + 40, 8), 0ull);                    // NaN -> 0
}

// Malformed PTX gets a precise parse error, never a crash, a wrong register or
// a silently truncated number.
VTEST(malformed_ptx_is_refused_with_a_reason) {
  auto refuses = [](const std::string& body, const char* says) {
    const std::string ptx = std::string(kHeader) + ".visible .entry k(.param .u64 out)\n{\n" + body + "\n    ret;\n}\n";
    auto err = VCAPTURE(Error, ptx::parse(ptx));
    VCHECK_CONTAINS(err.what(), says);
  };
  // A register redeclared at another width kept its narrow id: a heap overflow.
  refuses("    .reg .b32 %r<4>;\n    .reg .b64 %r2;", "already used or declared as a 32-bit register");
  // Used first as 32-bit, then declared 64-bit: the value written was lost.
  refuses("    mov.u64 %x9, 7;\n    .reg .b64 %x9;", "already used or declared as a 32-bit register");
  refuses("    .reg .b32 %r<100000000>;", "register count must be 1 to");
  refuses("    .reg .b32 %r<abc>;", "bad integer literal");
  refuses("    .reg .b64 %rd<2>;\n    mov.u64 %rd1, 12abc;", "bad integer literal '12abc'");
  refuses("    .reg .b64 %rd<2>;\n    mov.u64 %rd1, 12uu;", "bad integer literal '12uu'");
  // C integer suffixes are part of a number: nvcc writes "0x3fb8aa3bU".
  ptx::parse(std::string(kHeader) + ".visible .entry k(.param .u64 out)\n{\n    .reg .b64 %rd<3>;\n"
             "    mov.u64 %rd1, 0x3fb8aa3bU;\n    mov.u64 %rd2, 12ULL;\n    ret;\n}\n");
  refuses("    .reg .b32 %r<2>;\n    mov.f32 %r1, 0fZZZZZZZZ;", "bad float literal");
  refuses("    .shared .align 4 .b32 a[1073741824];", "is too large");
  refuses("    .shared .align 3 .b32 a[4];", "alignment must be a power of two");
  refuses("    .local .b32 a[-1];", "array size cannot be negative");
  refuses("    .reg .b64 %rd<2>;\n    .reg .pred %p<2>;\n    ld.global.v2.pred {%p1, %p0}, [%rd1];", "no storage size");
  auto err = VCAPTURE(Error, ptx::parse(std::string(kHeader) + ".visible .entry k(.param .pred out)\n{\n    ret;\n}\n"));
  VCHECK_CONTAINS(err.what(), "no storage size");
  // The most negative literal is still accepted, and still exact.
  const auto m = ptx::parse(std::string(kHeader) +
      ".visible .entry k(.param .u64 out)\n{\n    .reg .b64 %rd<3>;\n    ld.param.u64 %rd1, [out];\n"
      "    mov.s64 %rd2, -9223372036854775808;\n    st.global.u64 [%rd1], %rd2;\n    ret;\n}\n");
  Env e;
  uint64_t out = e.mem.alloc(8);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  VCHECK_EQ(e.mem.load_scalar(out, 8), 0x8000000000000000ull);
}

// __assertfail's line is an `unsigned int`, a 4-byte argument. Reading it as 8
// bytes took the next lane's copy for the high half, and an assert on line 55
// in a full warp reported line 236223201335 (55 * 2^32 + 55).
VTEST(device_assert_reports_its_line_from_a_full_warp) {
  std::string ptx = std::string(kHeader) + R"(
.extern .func __assertfail
(
    .param .b64 __assertfail_param_0,
    .param .b64 __assertfail_param_1,
    .param .b32 __assertfail_param_2,
    .param .b64 __assertfail_param_3,
    .param .b64 __assertfail_param_4
)
;
.visible .entry k(.param .u64 text)
{
    .reg .b64 %rd<3>;
    ld.param.u64 %rd1, [text];
    {
    .param .b64 param0;
    st.param.b64 [param0], %rd1;
    .param .b64 param1;
    st.param.b64 [param1], %rd1;
    .param .b32 param2;
    st.param.b32 [param2], 55;
    .param .b64 param3;
    st.param.b64 [param3], %rd1;
    .param .b64 param4;
    st.param.b64 [param4], 1;
    call.uni __assertfail, (param0, param1, param2, param3, param4);
    }
    trap;
}
)";
  Env e;
  auto m = ptx::parse(ptx);
  const char text[] = "x.cu";
  uint64_t t = e.mem.alloc(sizeof text);
  e.mem.write(t, text, sizeof text);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {arg_u64(t)}, e.mem, e.prof));
  VCHECK(err.code() == Err::DeviceAssert);
  VCHECK_CONTAINS(err.message(), "x.cu:55 in x.cu");
}

VTEST_MAIN
