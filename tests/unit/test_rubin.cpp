// sm_107 (Rubin) PTX forms, from PTX ISA 9.4 and its figures. None of this is
// checked against a card: ptxas of CUDA 13.2 has no sm_107 target, and there is
// no such GPU to run on. Each test is the ISA's own example or a case the text
// states, written independently of the interpreter's code.
#include <cstdlib>
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

const char* kHeader = ".version 9.4\n.target sm_107a\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

struct Env {
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/vr200");
};

// Runs one thread of `body` (with %rd2 the output pointer) and returns `n` words.
std::vector<uint32_t> run(const std::string& decls, const std::string& body, size_t n) {
  const std::string ptx = std::string(kHeader) + ".visible .entry k(.param .u64 out)\n{\n" + decls +
                          "  .reg .b64 %rd<3>;\n  ld.param.u64 %rd1, [out];\n  cvta.to.global.u64 %rd2, %rd1;\n" + body + "  ret;\n}\n";
  Env e;
  auto m = ptx::parse(ptx);
  const uint64_t out = e.mem.alloc(4 * n);
  exec::launch(m.entries[0], LaunchConfig{}, {arg_u64(out)}, e.mem, e.prof);
  std::vector<uint32_t> r(n);
  for (size_t i = 0; i < n; ++i) r[i] = static_cast<uint32_t>(e.mem.load_scalar(out + 4 * i, 4));
  return r;
}

}  // namespace

// Figure 43: MAXABS over s8, two groups of four in two registers.
VTEST(spcompress_figure_43) {
  const auto r = run(".reg .b32 m<1>, c<1>, d<2>, spdesc;\n",
                     "  mov.b32 d0, 0x0003F900;\n  mov.b32 d1, 0x06030500;\n  mov.b32 spdesc, 0x5;\n"
                     "  spcompress.b8.b4.sp::2:4.x1 {m0}, {c0}, {d0, d1}, spdesc;\n"
                     "  st.global.b32 [%rd2], c0;\n  st.global.b32 [%rd2+4], m0;\n",
                     2);
  VCHECK_EQ(r[0], 0x060503F9u);   // -7, 3, 5, 6
  VCHECK_EQ(r[1], 0x3121u);       // the indices, a nibble each, group 0 first
}

// Figure 44: the same back again, with the dropped elements zero.
VTEST(spdecompress_figure_44) {
  const auto r = run(".reg .b32 m<1>, c<1>, d<2>;\n",
                     "  mov.b32 c0, 0x060503F9;\n  mov.b32 m0, 0x3121;\n"
                     "  spdecompress.b8.b4.sp::2:4.x2 {d0, d1}, {m0}, {c0};\n"
                     "  st.global.b32 [%rd2], d0;\n  st.global.b32 [%rd2+4], d1;\n",
                     2);
  VCHECK_EQ(r[0], 0x0003F900u);
  VCHECK_EQ(r[1], 0x06000500u);
}

// The selections over f16: MAX takes the two greatest, MIN the two smallest, the
// -abs forms compare magnitudes; a NaN is always taken; -0.0 is below +0.0.
VTEST(spcompress_selections_f16) {
  // One group: 1.0, -3.0, 2.0, 0.5 (f16 0x3C00, 0xC200, 0x4000, 0x3800).
  const std::string setup =
      "  mov.b32 d0, 0xC2003C00;\n  mov.b32 d1, 0x38004000;\n";
  const auto run_sel = [&](unsigned desc) {
    return run(".reg .b32 m<1>, c<1>, d<2>, spdesc;\n",
               setup + "  mov.b32 spdesc, " + std::to_string(desc) + ";\n"
               "  spcompress.b16.b4.sp::2:4.x1 {m0}, {c0}, {d0, d1}, spdesc;\n"
               "  st.global.b32 [%rd2], c0;\n  st.global.b32 [%rd2+4], m0;\n",
               2);
  };
  auto r = run_sel(0);          // MAX: 2.0 (index 2) and 1.0 (index 0): in index order 1.0, 2.0
  VCHECK_EQ(r[0], 0x40003C00u);
  VCHECK_EQ(r[1], 0x20u);
  r = run_sel(1);               // MAXABS: -3.0 (1) and 2.0 (2)
  VCHECK_EQ(r[0], 0x4000C200u);
  VCHECK_EQ(r[1], 0x21u);
  r = run_sel(2);               // MIN: -3.0 (1) and 0.5 (3)
  VCHECK_EQ(r[0], 0x3800C200u);
  VCHECK_EQ(r[1], 0x31u);
  r = run_sel(3);               // MINABS: 0.5 (3) and 1.0 (0)
  VCHECK_EQ(r[0], 0x38003C00u);
  VCHECK_EQ(r[1], 0x30u);
  // A NaN is taken whatever the selection: 1.0, NaN, 2.0, 0.5 with MIN takes the NaN and 0.5.
  const auto nan = run(".reg .b32 m<1>, c<1>, d<2>, spdesc;\n",
                       "  mov.b32 d0, 0x7E003C00;\n  mov.b32 d1, 0x38004000;\n  mov.b32 spdesc, 2;\n"
                       "  spcompress.b16.b4.sp::2:4.x1 {m0}, {c0}, {d0, d1}, spdesc;\n"
                       "  st.global.b32 [%rd2], c0;\n  st.global.b32 [%rd2+4], m0;\n",
                       2);
  VCHECK_EQ(nan[0], 0x38007E00u);
  VCHECK_EQ(nan[1], 0x31u);
  // -0.0 is below +0.0: MAX of 0.0, -0.0, -1.0, -2.0 takes +0.0 and -0.0.
  const auto zero = run(".reg .b32 m<1>, c<1>, d<2>, spdesc;\n",
                        "  mov.b32 d0, 0x80000000;\n  mov.b32 d1, 0xC000BC00;\n  mov.b32 spdesc, 0;\n"
                        "  spcompress.b16.b4.sp::2:4.x1 {m0}, {c0}, {d0, d1}, spdesc;\n"
                        "  st.global.b32 [%rd2], c0;\n  st.global.b32 [%rd2+4], m0;\n",
                        2);
  VCHECK_EQ(zero[0], 0x80000000u);
  VCHECK_EQ(zero[1], 0x10u);
}

// The factors other than 2:4 of spdecompress, and several iterations per register.
VTEST(spdecompress_other_factors) {
  // 1:4 over b8, x4: each source byte goes to the position its b2 index names among four.
  // cdata = 0x44332211 (bytes 11, 22, 33, 44); indices 3, 0, 2, 1 as b2 fields -> 0b01_10_00_11 = 0x63.
  const auto r = run(".reg .b32 m<1>, c<1>, d<4>;\n",
                     "  mov.b32 c0, 0x44332211;\n  mov.b32 m0, 0x63;\n"
                     "  spdecompress.b8.b2.sp::1:4.x4 {d0, d1, d2, d3}, {m0}, {c0};\n"
                     "  st.global.b32 [%rd2], d0;\n  st.global.b32 [%rd2+4], d1;\n"
                     "  st.global.b32 [%rd2+8], d2;\n  st.global.b32 [%rd2+12], d3;\n",
                     4);
  VCHECK_EQ(r[0], 0x11000000u);   // iteration 0: 0x11 at index 3
  VCHECK_EQ(r[1], 0x00000022u);   // iteration 1: 0x22 at index 0
  VCHECK_EQ(r[2], 0x00330000u);   // iteration 2: 0x33 at index 2
  VCHECK_EQ(r[3], 0x00004400u);   // iteration 3: 0x44 at index 1
}

// Only on sm_107a.
VTEST(spcompress_needs_sm_107a) {
  const std::string ptx =
      ".version 9.4\n.target sm_100a\n.address_size 64\n.visible .entry k()\n{\n .reg .b32 m<1>, c<1>, d<2>, s;\n"
      "  spcompress.b8.b4.sp::2:4.x1 {m0}, {c0}, {d0, d1}, s;\n  ret;\n}\n";
  bool threw = false;
  try {
    ptx::parse(ptx);
  } catch (const Error&) {
    threw = true;
  }
  VCHECK(threw);
}

// cvt.rz to the narrow floating-point types (PTX ISA 9.4): the value truncated, where .rn rounds.
VTEST(cvt_rz_to_fp8_and_fp4) {
  const auto r = run(".reg .b32 a, b, d, e;\n",
                     // e4m3 near 1.2: spacing 0.125, so .rz gives 1.125 (0x39) and .rn 1.25 (0x3a).
                     "  mov.b32 a, 0x3F99999A;\n  mov.b32 b, 0xBF99999A;\n"
                     "  cvt.rz.satfinite.e4m3x2.f32 d, a, b;\n  cvt.rn.satfinite.e4m3x2.f32 e, a, b;\n"
                     "  st.global.b32 [%rd2], d;\n  st.global.b32 [%rd2+4], e;\n"
                     // e2m1 holds 0, 0.5, 1, 1.5, 2, 3, 4, 6: 2.9 -> 2 (code 4) .rz, 3 (code 5) .rn; 5.5 -> 4 (6) and 6 (7).
                     "  mov.b32 a, 0x40399999;\n  mov.b32 b, 0x40B00000;\n"
                     "  cvt.rz.satfinite.e2m1x2.f32 d, a, b;\n  cvt.rn.satfinite.e2m1x2.f32 e, a, b;\n"
                     "  st.global.b32 [%rd2+8], d;\n  st.global.b32 [%rd2+12], e;\n",
                     4);
  VCHECK_EQ(r[0], 0x39B9u);
  VCHECK_EQ(r[1], 0x3ABAu);
  VCHECK_EQ(r[2], 0x46u);
  VCHECK_EQ(r[3], 0x57u);
}

// .pzo: a -0.0 result is +0.0, on the results of the format conversion only.
VTEST(cvt_pzo) {
  const auto r = run(".reg .b32 a, b, d, e;\n",
                     // -0.0 and a negative value too small for f16 (-1.08e-19): both -0.0 in f16.
                     "  mov.b32 a, 0x80000000;\n  mov.b32 b, 0xA0000000;\n"
                     "  cvt.rn.pzo.f16x2.f32 d, a, b;\n  cvt.rn.f16x2.f32 e, a, b;\n"
                     "  st.global.b32 [%rd2], d;\n  st.global.b32 [%rd2+4], e;\n"
                     // The same to e4m3 (0x80 is -0.0).
                     "  cvt.rn.satfinite.pzo.e4m3x2.f32 d, a, b;\n  cvt.rn.satfinite.e4m3x2.f32 e, a, b;\n"
                     "  st.global.b32 [%rd2+8], d;\n  st.global.b32 [%rd2+12], e;\n",
                     4);
  VCHECK_EQ(r[0], 0x00000000u);
  VCHECK_EQ(r[1], 0x80008000u);
  VCHECK_EQ(r[2], 0x0000u);
  VCHECK_EQ(r[3], 0x8080u);
}

// .scaled::n1::ue8m0: one scale factor, the low byte, divides both inputs.
VTEST(cvt_scaled_n1) {
  const auto r = run(".reg .b32 a, b, d;\n.reg .b16 sf;\n",
                     // 8.0 and 4.0 divided by 2^3 (ue8m0 130): 1.0 (0x38) and 0.5 (0x30).
                     "  mov.b32 a, 0x41000000;\n  mov.b32 b, 0x40800000;\n  mov.b16 sf, 0x82;\n"
                     "  cvt.rn.satfinite.scaled::n1::ue8m0.e4m3x2.f32 d, a, b, sf;\n"
                     "  st.global.b32 [%rd2], d;\n",
                     1);
  VCHECK_EQ(r[0], 0x3830u);
}

// The packed integer forms of PTX ISA 9.2 and 9.4: each lane of a 4 x 8-bit or 2 x 16-bit register
// on its own. Lane 0 is the low byte.
VTEST(packed_integer_add_sub_neg) {
  const auto r = run(".reg .b32 a, b, d;\n",
                     "  mov.b32 a, 0xF07F8080;\n  mov.b32 b, 0x20010180;\n"   // bytes (high to low): F0 7F 80 80 + 20 01 01 80
                     "  add.u8x4 d, a, b;\n  st.global.b32 [%rd2], d;\n"
                     "  add.sat.u8x4 d, a, b;\n  st.global.b32 [%rd2+4], d;\n"
                     "  add.sat.s8x4 d, a, b;\n  st.global.b32 [%rd2+8], d;\n"
                     "  sub.u8x4 d, a, b;\n  st.global.b32 [%rd2+12], d;\n"
                     "  sub.sat.u8x4 d, a, b;\n  st.global.b32 [%rd2+16], d;\n"
                     "  sub.sat.s8x4 d, a, b;\n  st.global.b32 [%rd2+20], d;\n"
                     "  mov.b32 a, 0x8001FF00;\n  neg.s8x4 d, a;\n  st.global.b32 [%rd2+24], d;\n"
                     "  mov.b32 a, 0xFFF00001;\n  mov.b32 b, 0x0020FFFF;\n"
                     "  add.u16x2 d, a, b;\n  st.global.b32 [%rd2+28], d;\n"
                     "  add.sat.u16x2 d, a, b;\n  st.global.b32 [%rd2+32], d;\n",
                     9);
  // u8 lanes: 0x80+0x80 = 0x100, 0x80+0x01 = 0x81, 0x7F+0x01 = 0x80, 0xF0+0x20 = 0x110.
  VCHECK_EQ(r[0], 0x10808100u);
  VCHECK_EQ(r[1], 0xFF8081FFu);
  // s8 lanes: -128 + -128 -> -128, -128 + 1 = -127 (0x81), 127 + 1 -> 127, -16 + 32 = 16.
  VCHECK_EQ(r[2], 0x107F8180u);
  // u8 lanes: 0x80-0x80 = 0, 0x80-0x01 = 0x7F, 0x7F-0x01 = 0x7E, 0xF0-0x20 = 0xD0.
  VCHECK_EQ(r[3], 0xD07E7F00u);
  VCHECK_EQ(r[4], 0xD07E7F00u);
  // s8 lanes: -128 - -128 = 0, -128 - 1 -> -128, 127 - 1 = 126, -16 - 32 = -48 (0xD0).
  VCHECK_EQ(r[5], 0xD07E8000u);
  VCHECK_EQ(r[6], 0x80FF0100u);   // -(0x00, 0xFF, 0x01, 0x80): 0, 1, -1, -128 (wraps)
  VCHECK_EQ(r[7], 0x00100000u);   // 0xFFF0 + 0x0020 wraps to 0x0010; 0x0001 + 0xFFFF to 0
  VCHECK_EQ(r[8], 0xFFFFFFFFu);
}

VTEST(packed_integer_min_max_relu_set) {
  const auto r = run(".reg .b32 a, b, d;\n",
                     // a's bytes (low to high) 127, -128, -5, 5; b's -2, -3, 4, -2.
                     "  mov.b32 a, 0x05FB807F;\n  mov.b32 b, 0xFE04FDFE;\n"
                     "  max.s8x4 d, a, b;\n  st.global.b32 [%rd2], d;\n"
                     "  max.relu.s8x4 d, a, b;\n  st.global.b32 [%rd2+4], d;\n"
                     "  min.s8x4 d, a, b;\n  st.global.b32 [%rd2+8], d;\n"
                     "  min.u8x4 d, a, b;\n  st.global.b32 [%rd2+12], d;\n"
                     "  set.lt.s8x4 d, a, b;\n  st.global.b32 [%rd2+16], d;\n"
                     "  set.lo.u8x4 d, a, b;\n  st.global.b32 [%rd2+20], d;\n"
                     "  set.ge.s16x2 d, a, b;\n  st.global.b32 [%rd2+24], d;\n"
                     "  set.ne.u16x2 d, a, a;\n  st.global.b32 [%rd2+28], d;\n",
                     8);
  VCHECK_EQ(r[0], 0x0504FD7Fu);   // 127, -3, 4, 5
  VCHECK_EQ(r[1], 0x0504007Fu);   // .relu: -3 becomes 0
  VCHECK_EQ(r[2], 0xFEFB80FEu);   // -2, -128, -5, -2
  VCHECK_EQ(r[3], 0x0504807Fu);   // min.u8x4: 0x7F, 0x80, 0x04, 0x05
  VCHECK_EQ(r[4], 0x00FFFF00u);   // -128 < -3 and -5 < 4
  VCHECK_EQ(r[5], 0xFF00FFFFu);   // as unsigned: 0x7F < 0xFE, 0x80 < 0xFD, 0xFB > 0x04, 0x05 < 0xFE
  VCHECK_EQ(r[6], 0xFFFF0000u);   // s16 halves: 0x807F = -32641 >= 0xFDFE = -514? no; 0x05FB >= 0xFE04 = -508 yes
  VCHECK_EQ(r[7], 0x00000000u);
}

VTEST(packed_integer_needs_its_target) {
  const auto parse_on = [](const char* header, const char* ins) {
    return VCAPTURE(Error, ptx::parse(std::string(header) + ".visible .entry k()\n{\n .reg .b32 a, b, d;\n " + ins + "\n ret;\n}\n")).message();
  };
  VCHECK_CONTAINS(parse_on(".version 9.4\n.target sm_100a\n.address_size 64\n", "add.sat.u8x4 d, a, b;"), "sm_107f or sm_120f");
  VCHECK_CONTAINS(parse_on(".version 8.0\n.target sm_86\n.address_size 64\n", "add.u16x2 d, a, b;"), "sm_90");
}

VTEST_MAIN
