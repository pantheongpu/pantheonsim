// setp and set on .bf16 compare the values, not the bits (PTX ISA 9.7.9.8 and 9.7.9.9, sm_90).
//   * Type::is_float() is false for .bf16, so the comparator fell through to the integer compare of the raw 16 bits: the sign was
//     ignored (-1 < 1 was false) and a NaN was never unordered                                                    issue 502
// Each result is compared with what the ISA says, worked out on the host from the bfloat16 value (the top half of an f32).
// (An RTX 3060 is sm_86 and ptxas rejects setp.bf16 for it, so the reference is the ISA and the f32 comparison of the same values.)
#include <cmath>
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

// Runs one thread of `body` (PTX that leaves its answer in %r9) with the 16-bit registers %h0 and %h1 set to `a` and `b`, and the
// 32-bit register %r0 to a<<0 | b<<16 (a bf16x2 pair).
uint32_t run(const std::string& body, uint16_t a, uint16_t b, uint16_t a_hi = 0, uint16_t b_hi = 0) {
  const std::string ptx = std::string(".version 8.5\n.target sm_90\n.address_size 64\n") +
                          ".visible .entry k(.param .u64 po)\n{\n  .reg .b16 %h<8>;\n  .reg .b32 %r<16>;\n  .reg .b64 %rd<4>;\n"
                          "  .reg .pred %p<8>;\n  ld.param.u64 %rd0, [po];\n  cvta.to.global.u64 %rd0, %rd0;\n"
                          "  mov.b16 %h0, " + std::to_string(a) + ";\n  mov.b16 %h1, " + std::to_string(b) + ";\n"
                          "  mov.b32 %r0, " + std::to_string(uint32_t{a} | (uint32_t{a_hi} << 16)) + ";\n"
                          "  mov.b32 %r1, " + std::to_string(uint32_t{b} | (uint32_t{b_hi} << 16)) + ";\n" + body +
                          "  st.global.u32 [%rd0], %r9;\n  ret;\n}\n";
  runtime::Runtime rt(load_gpu("nvidia/h100"));
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

constexpr uint16_t kNeg0 = 0x8000, kPos0 = 0x0000, kNeg1 = 0xBF80, kPos1 = 0x3F80, kNeg2 = 0xC000, kPos2 = 0x4000, kNaN = 0x7FC0,
                   kNegNaN = 0xFFC0, kInf = 0x7F80, kNegInf = 0xFF80, kSub = 0x0001, kNegSub = 0x8001;

float val(uint16_t h) {
  const uint32_t b = uint32_t{h} << 16;
  float f;
  std::memcpy(&f, &b, 4);
  return f;
}

struct Cmp { const char* name; bool ordered; bool (*f)(float, float); };
// The ISA's meaning: the ordered forms are false when either operand is NaN, the u forms true; num and nan test for NaN.
const Cmp kCmps[] = {
    {"eq", true, [](float x, float y) { return x == y; }},   {"ne", true, [](float x, float y) { return x != y; }},
    {"lt", true, [](float x, float y) { return x < y; }},    {"le", true, [](float x, float y) { return x <= y; }},
    {"gt", true, [](float x, float y) { return x > y; }},    {"ge", true, [](float x, float y) { return x >= y; }},
    {"equ", false, [](float x, float y) { return x == y; }}, {"neu", false, [](float x, float y) { return x != y; }},
    {"ltu", false, [](float x, float y) { return x < y; }},  {"leu", false, [](float x, float y) { return x <= y; }},
    {"gtu", false, [](float x, float y) { return x > y; }},  {"geu", false, [](float x, float y) { return x >= y; }},
};

bool expect(const Cmp& c, uint16_t a, uint16_t b) {
  const bool unordered = std::isnan(val(a)) || std::isnan(val(b));
  if (unordered) return !c.ordered;
  return c.f(val(a), val(b));
}

const uint16_t kValues[] = {kNeg0, kPos0, kNeg1, kPos1, kNeg2, kPos2, kNaN, kNegNaN, kInf, kNegInf, kSub, kNegSub, 0x3F81, 0xBF81};
}  // namespace

VTEST(the_issue_cases) {
  const auto cmp = [](const char* c, uint16_t a, uint16_t b) {
    return run(std::string("  setp.") + c + ".bf16 %p1, %h0, %h1;\n  selp.u32 %r9, 1, 0, %p1;\n", a, b);
  };
  VCHECK_EQ(uint64_t{cmp("eq", kNeg0, kPos0)}, uint64_t{1});
  VCHECK_EQ(uint64_t{cmp("lt", kNeg1, kPos1)}, uint64_t{1});
  VCHECK_EQ(uint64_t{cmp("lt", kPos1, kNeg1)}, uint64_t{0});
  VCHECK_EQ(uint64_t{cmp("nan", kPos1, kPos1)}, uint64_t{0});
  // A bf16 NaN against 1.0.
  VCHECK_EQ(uint64_t{cmp("ne", kNaN, kPos1)}, uint64_t{0});
  VCHECK_EQ(uint64_t{cmp("neu", kNaN, kPos1)}, uint64_t{1});
  VCHECK_EQ(uint64_t{cmp("nan", kNaN, kNaN)}, uint64_t{1});
  VCHECK_EQ(uint64_t{cmp("eq", kNaN, kNaN)}, uint64_t{0});
  VCHECK_EQ(uint64_t{cmp("num", kNaN, kPos1)}, uint64_t{0});
  VCHECK_EQ(uint64_t{cmp("num", kPos1, kNeg1)}, uint64_t{1});
}

VTEST(setp_bf16_all_comparisons_on_all_value_pairs) {
  for (const Cmp& c : kCmps)
    for (uint16_t a : kValues)
      for (uint16_t b : kValues) {
        const uint32_t got = run(std::string("  setp.") + c.name + ".bf16 %p1, %h0, %h1;\n  selp.u32 %r9, 1, 0, %p1;\n", a, b);
        if (got != (expect(c, a, b) ? 1u : 0u)) {
          std::fprintf(stderr, "setp.%s.bf16 0x%04x, 0x%04x: got %u\n", c.name, a, b, got);
          VCHECK(false);
        }
      }
}

VTEST(set_bf16_to_integer_and_float_destinations) {
  for (const Cmp& c : kCmps)
    for (uint16_t a : {kNeg0, kNeg2, kPos1, kNaN})
      for (uint16_t b : {kPos0, kNeg1, kPos1, kNaN}) {
        const bool t = expect(c, a, b);
        const uint32_t u = run(std::string("  set.") + c.name + ".u32.bf16 %r9, %h0, %h1;\n", a, b);
        const uint32_t f = run(std::string("  set.") + c.name + ".f32.bf16 %r9, %h0, %h1;\n", a, b);
        if (u != (t ? 0xFFFFFFFFu : 0u) || f != (t ? 0x3F800000u : 0u)) {
          std::fprintf(stderr, "set.%s.{u32,f32}.bf16 0x%04x, 0x%04x: got 0x%x, 0x%x\n", c.name, a, b, u, f);
          VCHECK(false);
        }
      }
  // The issue's set.nan.u16.bf16.
  VCHECK_EQ(uint64_t{run("  set.nan.u16.bf16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", kNaN, kPos1)}, uint64_t{0xFFFF});
  VCHECK_EQ(uint64_t{run("  set.nan.u16.bf16 %h2, %h0, %h1;\n  cvt.u32.u16 %r9, %h2;\n", kPos1, kPos1)}, uint64_t{0});
}

VTEST(setp_bf16x2_compares_each_half_by_value) {
  // Low halves go to p, high halves to q: lt of (-1, -2) against (1, -1): low -1 < 1 true, high -2 < -1 true.
  const auto r = [](uint16_t a, uint16_t b, uint16_t ah, uint16_t bh) {
    return run("  setp.lt.bf16x2 %p1|%p2, %r0, %r1;\n  selp.u32 %r2, 1, 0, %p1;\n  selp.u32 %r3, 2, 0, %p2;\n  or.b32 %r9, %r2, %r3;\n", a, b,
               ah, bh);
  };
  VCHECK_EQ(uint64_t{r(kNeg1, kPos1, kNeg2, kNeg1)}, uint64_t{3});
  VCHECK_EQ(uint64_t{r(kPos1, kNeg1, kNeg1, kNeg2)}, uint64_t{0});
  VCHECK_EQ(uint64_t{r(kNaN, kPos1, kPos1, kNaN)}, uint64_t{0});
}

VTEST_MAIN
