// 16-bit signed integer ALU forms on a register whose upper bits are zero (what mov.b16 of a constant and cvt.u16.u32 leave).
//   * abs.s16 read the 32-bit register as s32, so 0xff81 came back as 0xff81 instead of 0x007f    issue 493
// The siblings (neg, min, max, add, mad, sad, mul.wide, div, rem, shr, cvt, setp, abs.s32/s64/f16/bf16) are checked the same way.
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
std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// Runs one thread of `body` (PTX that leaves its answer in %r9) and returns %r9. `a` and `b` are put in 16-bit registers
// %h0 and %h1 with mov.b16, so the upper register bits are zero.
uint32_t run(const std::string& body, uint16_t a, uint16_t b) {
  const std::string ptx = std::string(".version 8.3\n.target sm_86\n.address_size 64\n") +
                          ".visible .entry k(.param .u64 po)\n{\n  .reg .b16 %h<8>;\n  .reg .b32 %r<16>;\n  .reg .b64 %rd<4>;\n"
                          "  .reg .pred %p<4>;\n  .reg .s16 %s<4>;\n  ld.param.u64 %rd0, [po];\n  cvta.to.global.u64 %rd0, %rd0;\n"
                          "  mov.b16 %h0, " + std::to_string(a) + ";\n  mov.b16 %h1, " + std::to_string(b) + ";\n" + body +
                          "  st.global.u32 [%rd0], %r9;\n  ret;\n}\n";
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  const uint64_t out = dev.memory().alloc(4);
  dev.memory().store_scalar(out, 4, 0xdeadbeef);
  LaunchConfig cfg;
  cfg.block = {1, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  return static_cast<uint32_t>(dev.memory().load_scalar(out, 4));
}

uint32_t abs16(uint16_t a) { return run("  abs.s16 %h2, %h0;\n  cvt.u32.u16 %r9, %h2;\n", a, 0); }
}  // namespace

VTEST(abs_s16_of_a_negative_value_with_zero_upper_bits) {
  VCHECK_EQ(uint64_t{abs16(0xff81)}, uint64_t{0x007f});
  VCHECK_EQ(uint64_t{abs16(0xfffd)}, uint64_t{3});
  VCHECK_EQ(uint64_t{abs16(0xffff)}, uint64_t{1});
  VCHECK_EQ(uint64_t{abs16(0x8001)}, uint64_t{0x7fff});
}

VTEST(abs_s16_of_a_negative_value_with_sign_extended_upper_bits) {
  // cvt.s32.s16 sign-extends into the 32-bit register; abs.s16 then reads the same 16 bits.
  const uint32_t r = run("  mov.b32 %r1, 0xffffff81;\n  cvt.u16.u32 %h2, %r1;\n  abs.s16 %h3, %h2;\n  cvt.s32.s16 %r9, %h3;\n", 0, 0);
  VCHECK_EQ(uint64_t{r}, uint64_t{0x7f});
  // The same value held in a .s16 register that came from cvt.s16.s8.
  const uint32_t r2 = run("  mov.b32 %r1, 0xffffff81;\n  cvt.s16.s8 %s0, %r1;\n  abs.s16 %s1, %s0;\n  cvt.u32.u16 %r9, %s1;\n", 0, 0);
  VCHECK_EQ(uint64_t{r2}, uint64_t{0x7f});
}

VTEST(abs_s16_of_zero_positive_and_the_most_negative_value) {
  VCHECK_EQ(uint64_t{abs16(0)}, uint64_t{0});
  VCHECK_EQ(uint64_t{abs16(5)}, uint64_t{5});
  VCHECK_EQ(uint64_t{abs16(0x7fff)}, uint64_t{0x7fff});
  // abs.s16 of -32768 is -32768 (no overflow trap): the 16-bit result is 0x8000.
  VCHECK_EQ(uint64_t{abs16(0x8000)}, uint64_t{0x8000});
}

// The signed 16-bit siblings, on negative inputs with zero upper bits, against host arithmetic.
VTEST(the_other_signed_16_bit_forms_agree_with_the_isa) {
  const int16_t as[] = {0, 5, -3, -127, -32768, 32767, -1};
  const int16_t bs[] = {0, 7, -5, 3, -2, 32767, -32768};
  for (int16_t a : as)
    for (int16_t b : bs) {
      const uint16_t ua = static_cast<uint16_t>(a), ub = static_cast<uint16_t>(b);
      const int32_t wa = a, wb = b;
      VCHECK_EQ(uint64_t{run("  neg.s16 %h2, %h0;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(-wa)});
      VCHECK_EQ(uint64_t{run("  max.s16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(wa > wb ? wa : wb)});
      VCHECK_EQ(uint64_t{run("  min.s16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(wa < wb ? wa : wb)});
      VCHECK_EQ(uint64_t{run("  add.s16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(wa + wb)});
      VCHECK_EQ(uint64_t{run("  mad.lo.s16 %h2, %h0, %h1, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)},
                uint64_t{uint16_t(wa * wb + wb)});
      VCHECK_EQ(uint64_t{run("  cvt.s32.s16 %r9, %h0;\n", ua, ub)}, uint64_t{uint32_t(wa)});
      VCHECK_EQ(uint64_t{run("  mul.wide.s16 %r9, %h0, %h1;\n", ua, ub)}, uint64_t{uint32_t(wa * wb)});
      VCHECK_EQ(uint64_t{run("  setp.lt.s16 %p0, %h0, %h1;\n  selp.u32 %r9, 1, 0, %p0;\n", ua, ub)}, uint64_t{wa < wb});
      VCHECK_EQ(uint64_t{run("  shr.s16 %h2, %h0, 3;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(wa >> 3)});
      const int32_t sad = (wa > wb ? wa - wb : wb - wa);
      VCHECK_EQ(uint64_t{run("  sad.s16 %h2, %h0, %h1, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)},
                uint64_t{uint16_t(sad + wb)});
      if (b != 0 && !(a == -32768 && b == -1)) {
        VCHECK_EQ(uint64_t{run("  div.s16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(wa / wb)});
        VCHECK_EQ(uint64_t{run("  rem.s16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", ua, ub)}, uint64_t{uint16_t(wa % wb)});
      }
    }
}

// abs at the other widths, so the sign extension added for .s16 is not applied to the wrong ones.
VTEST(abs_s32_s64_f16_and_bf16) {
  VCHECK_EQ(uint64_t{run("  mov.b32 %r1, 0xffffff81;\n  abs.s32 %r9, %r1;\n", 0, 0)}, uint64_t{0x7f});
  VCHECK_EQ(uint64_t{run("  mov.b32 %r1, 0x80000000;\n  abs.s32 %r9, %r1;\n", 0, 0)}, uint64_t{0x80000000});
  VCHECK_EQ(uint64_t{run("  mov.b64 %rd1, 0xffffffff00000081;\n  abs.s64 %rd2, %rd1;\n  shr.u64 %rd3, %rd2, 32;\n"
                         "  cvt.u32.u64 %r9, %rd3;\n", 0, 0)},
            uint64_t{0});  // -(-0xffffff7f) = 0xffffff7f
  VCHECK_EQ(uint64_t{run("  mov.b64 %rd1, 0xffffffff00000081;\n  abs.s64 %rd2, %rd1;\n  cvt.u32.u64 %r9, %rd2;\n", 0, 0)},
            uint64_t{0xffffff7f});
  VCHECK_EQ(uint64_t{run("  abs.f16 %h2, %h0;\n  cvt.u32.u16 %r9, %h2;\n", 0xbc00, 0)}, uint64_t{0x3c00});  // -1.0 -> 1.0
  VCHECK_EQ(uint64_t{run("  abs.bf16 %h2, %h0;\n  cvt.u32.u16 %r9, %h2;\n", 0xbf80, 0)}, uint64_t{0x3f80});  // -1.0 -> 1.0
}

VTEST_MAIN
