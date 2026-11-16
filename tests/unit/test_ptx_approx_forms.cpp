// The approximate math instructions as ptxas makes them, so that a kernel's PTX gives the bits its SASS gives.
//
// ptxas does not turn every .approx instruction into one MUFU. Read from the SASS nvcc 13.0 writes for sm_75 to sm_120
// (tools/numerics/mathfn.cu, cuobjdump -sass) and reproduced by the SASS executor on the same kernels:
//
//   sin.approx.f32, cos.approx.f32      FMUL.RZ x, 0.15915494 (1/2pi, rounded toward zero), then MUFU.SIN / MUFU.COS, whose
//                                       argument is in turns
//   ex2.approx.f32 (no .ftz)            below -126: MUFU.EX2(x * 0.5), squared (MUFU.EX2 cannot give a subnormal)
//   lg2.approx.f32 (no .ftz)            not >= 2^-126: MUFU.LG2(x * 2^24) + -24 (MUFU.LG2 flushes a subnormal)
//   rcp.approx.ftz.f64                  MUFU.RCP64H: the upper word of the reciprocal of the operand's upper word; the lower
//   rsqrt.approx.ftz.f64                word of the result is zero (MUFU.RSQ64H), as the PTX ISA documents both
//   rsqrt.approx.f64                    MUFU.RSQ64H and one correction step; a subnormal scaled by 2^54 first
//
// The interpreter computed each of these as the function at host precision, and a program run from its PTX differed from
// the same program run from its SASS: rsqrt(double) on 13 % of arguments, hypot(double) on 10 %, exp2f on subnormal results,
// cbrtf on subnormal arguments, and sinf, cosf and tanf under -use_fast_math on most (tools/numerics, docs/numerics/
// math-functions.md). The expected values were the SASS executor's on those kernels while MUFU was the function at host
// precision. Since the card's unit is modelled (include/vgpu/exec/mufu_model.hpp) the single-precision cases whose value
// goes through MUFU.EX2, LG2, SIN or COS are an RTX 3060's results for the same instruction and operand (run on the card
// 2026-11-15 with tools/numerics/mufu_forms.cu's kernels; nine values changed in their last bits, one -- the cosine of
// 3e18 -- altogether; each line says what it was). The double-precision cases did not change: on these operands the card's
// MUFU.RCP64H and MUFU.RSQ64H are the upper word of the correctly rounded value. tests/unit/test_mufu_model.cpp holds every
// form to the card on more operands.
#include <cstring>
#include <string>
#include <vector>

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

// Runs "<op> %b, %a" on an f32 or f64 register loaded with `a`'s bits and returns %b's bits.
uint64_t run(const std::string& op, int width, uint64_t a) {
  const std::string t = width == 64 ? "f64" : "f32";
  const std::string ptx = std::string(kHeader) + ".visible .entry k(.param .u64 po){\n .reg .b64 %rd1; .reg ." + t +
                          " %a, %b;\n ld.param.u64 %rd1,[po]; cvta.to.global.u64 %rd1,%rd1;\n ld.global." + t +
                          " %a,[%rd1];\n " + op + " %b, %a;\n st.global." + t + " [%rd1+8],%b; ret; }\n";
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a100");
  auto m = ptx::parse(ptx);
  const uint64_t buf = mem.alloc(16);
  mem.store_scalar(buf, width / 8, a);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(buf)}, mem, prof);
  return mem.load_scalar(buf + 8, width / 8);
}
}  // namespace

// rcp.approx.ftz.f64 and rsqrt.approx.ftz.f64 read the upper 32 bits of the operand and leave the lower 32 of the result zero.
VTEST(the_ftz_double_approximations_are_the_upper_word_only) {
  VCHECK_EQ(run("rcp.approx.ftz.f64", 64, 0x4008000000000000ull), 0x3FD5555500000000ull);   // 1/3 is 0x3FD5555555555555
  VCHECK_EQ(run("rcp.approx.ftz.f64", 64, 0x40080000FFFFFFFFull), 0x3FD5555500000000ull);   // the lower word is not read
  VCHECK_EQ(run("rcp.approx.ftz.f64", 64, 0x3FF0000000000000ull), 0x3FF0000000000000ull);
  VCHECK_EQ(run("rcp.approx.ftz.f64", 64, 0xC000000012345678ull), 0xBFE0000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.ftz.f64", 64, 0x4000000000000000ull), 0x3FE6A09E00000000ull);   // 2^-1/2 is 0x3FE6A09E667F3BCD
  VCHECK_EQ(run("rsqrt.approx.ftz.f64", 64, 0x40000000DEADBEEFull), 0x3FE6A09E00000000ull);
  VCHECK_EQ(run("rsqrt.approx.ftz.f64", 64, 0x4010000000000000ull), 0x3FE0000000000000ull);
}

// rsqrt.approx.f64: MUFU.RSQ64H and one step. Within an ulp of the root's reciprocal, and not always the nearest double.
VTEST(rsqrt_approx_f64_is_the_expansion_ptxas_makes) {
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x3FE0000000000000ull), 0x3FF6A09E667F3BCDull);   // nearest: ...3BCC
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x6E73E372E2338ACAull), 0x28ACB3A9B9D8209Full);   // nearest: ...209E
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x40416E2C8B133996ull), 0x3FC5ADDAFCABDDB4ull);   // nearest: ...DDB3
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x5E30000000000002ull), 0x30CFFFFFFFFFFFFEull);   // (a published worst case of ROCm's)
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x3FF0000000000000ull), 0x3FF0000000000000ull);
  // The slow path: zeros, a negative number, the infinities, a NaN, subnormals (scaled by 2^54, the result by 2^27).
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x0000000000000000ull), 0x7FF0000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x8000000000000000ull), 0xFFF0000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0xBFF0000000000000ull), 0xFFF8000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0xFFF0000000000000ull), 0xFFF8000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x7FF0000000000000ull), 0x0000000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x7FF8000000001234ull), 0x7FF8000000001234ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x0000000000000001ull), 0x6180000000000000ull);
  VCHECK_EQ(run("rsqrt.approx.f64", 64, 0x000FFFFFFFFFFFFFull), 0x5FE0000000000001ull);
}

// ex2.approx.f32 below -126 is a product of two rounded halves; with .ftz it is MUFU.EX2 alone, which flushes.
VTEST(ex2_approx_f32_gives_subnormals_by_squaring) {
  VCHECK_EQ(run("ex2.approx.f32", 32, 0xC2FE686Eu), 0x00378FF6ull);   // the card's MUFU.EX2 of the half, squared (at host precision: 0x00378FF7)
  VCHECK_EQ(run("ex2.approx.f32", 32, 0xC2FF2F30u), 0x002A7449ull);   // (at host precision: 0x002A744A)
  VCHECK_EQ(run("ex2.approx.f32", 32, 0xC2FC0000u), 0x00800000ull);   // -126: the smallest normal, MUFU.EX2 itself
  VCHECK_EQ(run("ex2.approx.f32", 32, 0xC3150000u), 0x00000001ull);   // -149
  VCHECK_EQ(run("ex2.approx.f32", 32, 0xC3160000u), 0x00000000ull);   // -150: 2^-75 squared rounds to zero
  VCHECK_EQ(run("ex2.approx.f32", 32, 0xFF800000u), 0x00000000ull);
  VCHECK_EQ(run("ex2.approx.f32", 32, 0x7FC00000u), 0x7FFFFFFFull);
  VCHECK_EQ(run("ex2.approx.f32", 32, 0x40400000u), 0x41000000ull);
  VCHECK_EQ(run("ex2.approx.ftz.f32", 32, 0xC2FE686Eu), 0x00000000ull);
}

// lg2.approx.f32 of a subnormal is the logarithm of the scaled number rounded to float, less 24, rounded again.
VTEST(lg2_approx_f32_scales_a_subnormal) {
  VCHECK_EQ(run("lg2.approx.f32", 32, 0x000AE398u), 0xC3018E21ull);   // (at host precision: 0xC3018E22)
  VCHECK_EQ(run("lg2.approx.f32", 32, 0x00000001u), 0xC3150000ull);   // -149
  VCHECK_EQ(run("lg2.approx.f32", 32, 0x007FFFFFu), 0xC2FC0000ull);
  VCHECK_EQ(run("lg2.approx.f32", 32, 0x00000000u), 0xFF800000ull);
  VCHECK_EQ(run("lg2.approx.f32", 32, 0x80000000u), 0xFF800000ull);
  VCHECK_EQ(run("lg2.approx.f32", 32, 0xBF800000u), 0x7FFFFFFFull);
  VCHECK_EQ(run("lg2.approx.f32", 32, 0x41000000u), 0x40400000ull);
  VCHECK_EQ(run("lg2.approx.ftz.f32", 32, 0x000AE398u), 0xFF800000ull);   // flushed: the logarithm of zero
}

// sin.approx and cos.approx: the argument times 1/2pi, rounded toward zero, is what the sine is taken of.
VTEST(sin_and_cos_approx_take_the_scaled_argument) {
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0x3F800000u), 0x3F576AA3ull);   // sin(1) rounded once: 0x3F576AA5; the scaled argument at host precision: ...A4
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0xBF800000u), 0xBF576AA3ull);
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0x4A47AE3Bu), 0xBE47C5C0ull);   // 3271566.75: sin of it is 0x3DFFA15D; at host precision 0xBE47C5C2
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0x472B5407u), 0xBE7BDBA7ull);   // (at host precision: 0xBE7BDBA4)
  VCHECK_EQ(run("sin.approx.f32", 32, 0x4A47AE3Bu), 0xBE47C5C0ull);
  VCHECK_EQ(run("cos.approx.ftz.f32", 32, 0x3F800000u), 0x3F0A5141ull);   // cos(1) rounded once: 0x3F0A5140
  VCHECK_EQ(run("cos.approx.ftz.f32", 32, 0x5E29E84Eu), 0x3F800000ull);   // 3.06e18 turns x 2pi: the unit keeps 25 fraction bits of the turns, none are left (host precision: 0xBEE8BEC3)
  VCHECK_EQ(run("cos.approx.ftz.f32", 32, 0x478B9A09u), 0x3DEE3866ull);   // (at host precision: 0x3DEE3876)
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0x00000000u), 0x00000000ull);
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0x80000000u), 0x80000000ull);
  VCHECK_EQ(run("cos.approx.ftz.f32", 32, 0x80000000u), 0x3F800000ull);
  VCHECK_EQ(run("sin.approx.ftz.f32", 32, 0x7F800000u), 0x7FFFFFFFull);
  VCHECK_EQ(run("cos.approx.ftz.f32", 32, 0x7FC00000u), 0x7FFFFFFFull);
}

VTEST_MAIN
