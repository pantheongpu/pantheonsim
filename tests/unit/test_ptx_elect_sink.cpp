// elect.sync with the sink `_` for either destination: "elect.sync _|%p1, mask;" and "elect.sync %r1|_, mask;".
//   * the parser wanted a register for both parts and stopped with "elect destination must be a register, got '_'"   issue 503
// cubecl's CUDA C++ backend emits the first form for plane_elect. The other half of the pair is still written.
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

// Runs one warp of `body` (it leaves its answer in %r9) and returns the 32 values, one per lane.
std::vector<uint32_t> run(const std::string& body) {
  const std::string ptx = std::string(".version 8.3\n.target sm_90\n.address_size 64\n") +
                          ".visible .entry k(.param .u64 po)\n{\n  .reg .b32 %r<16>;\n  .reg .b64 %rd<4>;\n  .reg .pred %p<4>;\n"
                          "  ld.param.u64 %rd0, [po];\n  cvta.to.global.u64 %rd0, %rd0;\n  mov.u32 %r10, %laneid;\n" + body +
                          "  mul.wide.u32 %rd1, %r10, 4;\n  add.u64 %rd0, %rd0, %rd1;\n  st.global.u32 [%rd0], %r9;\n  ret;\n}\n";
  runtime::Runtime rt(load_gpu("nvidia/h100"));
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

VTEST(elect_with_a_sink_lane_id_writes_the_predicate) {
  // The issue's form: one lane wins, the lane id is thrown away.
  const auto r = run("  elect.sync _|%p1, 0xffffffff;\n  selp.b32 %r9, 1, 0, %p1;\n");
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{i == 0 ? 1u : 0u});
}

VTEST(elect_with_a_sink_predicate_writes_the_lane_id) {
  const auto r = run("  elect.sync %r9|_, 0xffffffff;\n");
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{0});
}

VTEST(elect_with_both_sinks_runs) {
  const auto r = run("  mov.u32 %r9, 77;\n  elect.sync _|_, 0xffffffff;\n");
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{77});
}

VTEST(elect_sink_picks_the_lowest_member_of_a_partial_mask) {
  // Members are the odd lanes: the leader is lane 1, and only lanes 1,3,5,... run the instruction.
  const auto r = run(
      "  mov.u32 %r9, 0xff;\n  and.b32 %r1, %r10, 1;\n  setp.eq.u32 %p2, %r1, 1;\n"
      "  @!%p2 bra DONE;\n  elect.sync _|%p1, 0xaaaaaaaa;\n  selp.b32 %r9, 1, 0, %p1;\n  DONE:\n");
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{i == 1 ? 1u : (i & 1 ? 0u : 0xffu)});
}

VTEST(elect_with_a_real_register_destination_still_works) {
  const auto r = run("  elect.sync %r1|%p1, 0xffffffff;\n  selp.b32 %r9, 1, 0, %p1;\n  add.u32 %r9, %r9, %r1;\n");
  for (uint32_t i = 0; i < 32; ++i) VCHECK_EQ(uint64_t{r[i]}, uint64_t{i == 0 ? 1u : 0u});
}

VTEST_MAIN
