// Four PTX-interpreter fixes reported against 0.2.0:
//  #522 "mov.u32%r1, %laneid;" (no space after the type, which nvcc emits for
//       inline asm written "mov.u32%0, ...") was rejected as "mov missing type".
//  #503 "elect.sync _|%p, mask;" (sink destination) was rejected.
//  #502 setp/set on .bf16 compared raw bits: sign ignored, NaN never seen.
//  #488 shfl.sync did not wait for a member lane in a long loop when the
//       first lane's member mask did not name it (two half-warp tiles).
// The PTX forms of #522, #503 and #502 are what ptxas accepts (checked by
// assembling them with ptxas 13.0); the expected values of #502 follow the
// PTX ISA (the same comparisons on .f16 already passed here).
#include <cstring>
#include <string>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/ptx/parser.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using namespace vgpu;
using vgpu::exec::LaunchConfig;

namespace {
const char* kHeader = ".version 8.5\n.target sm_90\n.address_size 64\n";
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// Runs one entry "k(po)" with `threads` threads, returns the first `words` u32 of the buffer.
std::vector<uint32_t> run(const std::string& body, uint32_t threads, size_t words) {
  const std::string ptx = std::string(kHeader) + ".visible .entry k(.param .u64 po){\n" + body + "}\n";
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/h100");
  auto m = ptx::parse(ptx);
  const uint64_t buf = mem.alloc(words * 4 + 16);
  for (size_t i = 0; i < words; ++i) mem.store_scalar(buf + 4 * i, 4, 0);
  LaunchConfig cfg;
  cfg.block = {threads, 1, 1};
  exec::launch(m.entries[0], cfg, {arg_u64(buf)}, mem, prof);
  std::vector<uint32_t> out(words);
  for (size_t i = 0; i < words; ++i) out[i] = static_cast<uint32_t>(mem.load_scalar(buf + 4 * i, 4));
  return out;
}
}  // namespace

VTEST(mov_type_directly_followed_by_a_register) {
  const auto r = run(R"(
 .reg .b64 %rd<3>; .reg .b32 %r<4>;
 ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
 mov.u32%r1, %laneid;
 mov.u32%r2, %tid.x;
 mul.wide.u32 %rd2, %r2, 4; add.s64 %rd2, %rd1, %rd2;
 st.global.u32 [%rd2], %r1;
 ret;
)", 32, 32);
  for (uint32_t l = 0; l < 32; ++l) VCHECK_EQ(r[l], l);
}

VTEST(elect_sync_accepts_the_sink_destination) {
  const auto r = run(R"(
 .reg .b64 %rd<3>; .reg .b32 %r<6>; .reg .pred %p<3>;
 ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
 mov.u32 %r1, %tid.x;
 elect.sync _|%p1, 0xffffffff;
 selp.u32 %r2, 1, 0, %p1;
 mul.wide.u32 %rd2, %r1, 4; add.s64 %rd2, %rd1, %rd2;
 st.global.u32 [%rd2], %r2;
 ret;
)", 32, 32);
  uint32_t leaders = 0;
  for (uint32_t l = 0; l < 32; ++l) leaders += r[l];
  VCHECK_EQ(leaders, 1u);
  VCHECK_EQ(r[0], 1u);   // the lowest active lane wins
}

// The kernel of #502 plus the NaN cases of its comments.
VTEST(bf16_compares_look_at_values_not_bits) {
  const auto r = run(R"(
 .reg .b64 %rd<4>; .reg .b32 %r<8>; .reg .b16 %h<8>; .reg .pred %p<12>;
 ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
 mov.b16 %h1, 0x8000;
 mov.b16 %h2, 0x0000;
 mov.b16 %h3, 0xBF80;
 mov.b16 %h4, 0x3F80;
 mov.b16 %h5, 0x7FC0;
 setp.eq.bf16 %p1, %h1, %h2;
 setp.lt.bf16 %p2, %h3, %h4;
 setp.lt.bf16 %p3, %h4, %h3;
 setp.nan.bf16 %p4, %h4, %h4;
 setp.ne.bf16 %p5, %h5, %h4;
 setp.neu.bf16 %p6, %h5, %h4;
 setp.nan.bf16 %p7, %h5, %h5;
 setp.eq.bf16 %p8, %h5, %h5;
 set.nan.u16.bf16 %r7, %h5, %h5;
 selp.u32 %r1,1,0,%p1; st.global.u32 [%rd1+0],%r1;
 selp.u32 %r1,1,0,%p2; st.global.u32 [%rd1+4],%r1;
 selp.u32 %r1,1,0,%p3; st.global.u32 [%rd1+8],%r1;
 selp.u32 %r1,1,0,%p4; st.global.u32 [%rd1+12],%r1;
 selp.u32 %r1,1,0,%p5; st.global.u32 [%rd1+16],%r1;
 selp.u32 %r1,1,0,%p6; st.global.u32 [%rd1+20],%r1;
 selp.u32 %r1,1,0,%p7; st.global.u32 [%rd1+24],%r1;
 selp.u32 %r1,1,0,%p8; st.global.u32 [%rd1+28],%r1;
 and.b32 %r7, %r7, 0xffff;
 st.global.u32 [%rd1+32], %r7;
 ret;
)", 1, 9);
  VCHECK_EQ(r[0], 1u);   // -0 == +0
  VCHECK_EQ(r[1], 1u);   // -1 < 1
  VCHECK_EQ(r[2], 0u);   // 1 < -1
  VCHECK_EQ(r[3], 0u);   // 1 is not NaN
  VCHECK_EQ(r[4], 0u);   // ne is ordered: NaN != 1 is false
  VCHECK_EQ(r[5], 1u);   // neu is unordered: true
  VCHECK_EQ(r[6], 1u);   // nan.bf16 x,x
  VCHECK_EQ(r[7], 0u);   // eq.bf16 x,x
  VCHECK(r[8] != 0u);    // set.nan.u16.bf16 x,x is true (all ones for an integer destination)
}

// A member lane in a long loop: both half-warp tiles shuffle with their own mask.
VTEST(shfl_sync_waits_for_a_member_still_in_a_loop) {
  const auto r = run(R"(
 .reg .b64 %rd<4>; .reg .b32 %r<12>; .reg .pred %p<4>;
 ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;
 mov.u32 %r1, %laneid;
 setp.eq.u32 %p1, %r1, 22;
 @%p1 bra $L_slow;
$L_shfl:
 and.b32 %r3, %r1, 16;
 mov.b32 %r4, 0xffff;
 shl.b32 %r4, %r4, %r3;
 add.u32 %r5, %r1, 1000;
 shfl.sync.idx.b32 %r6, %r5, 6, 0x100f, %r4;
 mul.wide.u32 %rd2, %r1, 4; add.s64 %rd2, %rd1, %rd2;
 st.global.u32 [%rd2], %r6;
 ret;
$L_slow:
 mov.u32 %r2, 0;
$L_spin:
 add.u32 %r2, %r2, 1;
 setp.lt.u32 %p2, %r2, 30000;
 @%p2 bra $L_spin;
 bra $L_shfl;
)", 32, 32);
  for (uint32_t l = 0; l < 32; ++l) VCHECK_EQ(r[l], 1006u + (l & 16));
}

VTEST_MAIN
