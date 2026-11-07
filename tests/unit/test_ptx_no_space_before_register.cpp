// A register operand written straight after the type suffix, with no white space ("mov.u32%r1, %laneid;"), is accepted.
//   * the lexer kept '%' inside a word, so the opcode read "mov.u32%r1" and the parser stopped with "mov missing type"   issue 522
// nvcc writes this for inline asm such as "mov.u32%0, %%laneid;", and ptxas accepts it.
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

// Runs 32 threads of `body` (it leaves its answer in %r9) and returns the 32 values.
std::vector<uint32_t> run(const std::string& body) {
  const std::string ptx = std::string(".version 8.3\n.target sm_86\n.address_size 64\n") +
                          ".visible .entry k(.param .u64 po)\n{\n  .reg .b32 %r<16>;\n  .reg .b64 %rd<4>;\n  .reg .pred %p<4>;\n"
                          "  ld.param.u64%rd0, [po];\n  cvta.to.global.u64%rd0,%rd0;\n" + body +
                          "  mov.u32%r10, %laneid;\n  mul.wide.u32%rd1,%r10, 4;\n  add.u64%rd0,%rd0,%rd1;\n"
                          "  st.global.u32[%rd0],%r9;\n  ret;\n}\n";
  runtime::Runtime rt(load_gpu("nvidia/a10"));
  auto& dev = rt.device(0);
  const uint64_t mod = dev.load_module(ptx);
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  const uint64_t out = dev.memory().alloc(32 * 4);
  LaunchConfig cfg;
  cfg.block = {32, 1, 1};
  dev.launch(*fn, cfg, {arg_u64(out)}, dev.symbols(mod));
  std::vector<uint32_t> r(32);
  for (int i = 0; i < 32; ++i) r[i] = static_cast<uint32_t>(dev.memory().load_scalar(out + 4 * i, 4));
  return r;
}
}  // namespace

VTEST(mov_u32_straight_into_a_register_reads_the_lane_id) {
  const auto r = run("  mov.u32%r9, %laneid;\n");
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{i});
}

VTEST(other_instructions_take_a_register_right_after_the_type) {
  const auto r = run(
      "  mov.u32%r1, %laneid;\n  add.u32%r2,%r1,%r1;\n  mul.lo.u32%r3,%r2,%r1;\n  mov.b32%r4, 7;\n  sub.s32%r5,%r3,%r4;\n"
      "  setp.lt.u32%p1,%r1, 16;\n  selp.b32%r9,%r5,%r4,%p1;\n"
      "  @%p1 add.u32%r9,%r9,%r1;\n  @!%p1 bra DONE;\n  xor.b32%r9,%r9,1;\nDONE:\n");
  for (uint32_t i = 0; i < 32; ++i) {
    uint32_t v = i < 16 ? 2 * i * i - 7 : 7;
    if (i < 16) v += i, v ^= 1;
    VCHECK_EQ(uint64_t{r[i]}, uint64_t{v});
  }
}

VTEST_MAIN
