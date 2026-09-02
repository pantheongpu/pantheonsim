// Register pressure and occupancy.
//
// The counts are an estimate: PTX declares virtual registers and ptxas
// allocates physical ones, so this solves liveness over the control-flow graph
// and takes the peak. It matches hardware exactly on a simple kernel and
// over-estimates a register-heavy one -- erring toward "needs more", which is
// the safe direction for a tool that refuses launches. Erring too far is not
// safe, though: an over-estimate that crosses the hardware limit refuses a
// launch that would have worked, which is what a cruder approximation here
// used to do.
#include "vgpu/ptx/regalloc.hpp"

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
const char* kHeader = ".version 8.3\n.target sm_86\n.address_size 64\n";

ptx::RegisterUsage usage_of(const std::string& body, const std::string& params = "") {
  auto m = ptx::parse(std::string(kHeader) + ".visible .entry k(" + params + ")\n{\n" + body + "\n}\n");
  return ptx::analyze_registers(m.entries[0]);
}
}  // namespace

VTEST(overlapping_lifetimes_share_registers) {
  // Four virtual registers used one after another need one physical register,
  // not four: counting declarations would say four.
  auto seq = usage_of(R"(
    .reg .b32 %r<8>;
    mov.u32 %r1, 1;
    add.s32 %r2, %r1, 1;
    add.s32 %r3, %r2, 1;
    add.s32 %r4, %r3, 1;
    ret;
  )");
  // Four values that are all live at once must cost more than the chain above.
  auto par = usage_of(R"(
    .reg .b32 %r<8>;
    mov.u32 %r1, 1;
    mov.u32 %r2, 2;
    mov.u32 %r3, 3;
    mov.u32 %r4, 4;
    add.s32 %r5, %r1, %r2;
    add.s32 %r6, %r3, %r4;
    add.s32 %r7, %r5, %r6;
    ret;
  )");
  VCHECK(par.peak_live > seq.peak_live);
}

VTEST(sixty_four_bit_values_cost_a_register_pair) {
  auto narrow = usage_of(R"(
    .reg .b32 %r<4>;
    mov.u32 %r1, 1;
    mov.u32 %r2, 2;
    add.s32 %r3, %r1, %r2;
    ret;
  )");
  auto wide = usage_of(R"(
    .reg .b64 %rd<4>;
    mov.u64 %rd1, 1;
    mov.u64 %rd2, 2;
    add.s64 %rd3, %rd1, %rd2;
    ret;
  )");
  VCHECK_EQ(wide.peak_live, narrow.peak_live * 2);
}

VTEST(predicates_are_counted_separately) {
  auto u = usage_of(R"(
    .reg .pred %p<4>;
    .reg .b32 %r<4>;
    mov.u32 %r1, 1;
    setp.eq.s32 %p1, %r1, 0;
    setp.ne.s32 %p2, %r1, 0;
    selp.b32 %r2, 1, 0, %p1;
    ret;
  )");
  VCHECK(u.pred_regs >= 2);           // both predicates live at once
  VCHECK(u.regs_per_thread >= 1);     // and they do not inflate the value file
}

VTEST(loop_carried_values_stay_live_across_the_back_edge) {
  // The accumulator is defined before the loop and used inside it, so it is
  // live for the whole body -- a linear scan that ignored the back edge would
  // free it early.
  auto u = usage_of(R"(
    .reg .pred %p<2>;
    .reg .b32 %r<8>;
    mov.u32 %r1, 0;
    mov.u32 %r2, 0;
    mov.u32 %r3, 100;
LOOP:
    setp.ge.u32 %p1, %r2, %r3;
    @%p1 bra DONE;
    add.s32 %r1, %r1, %r2;
    add.s32 %r2, %r2, 1;
    bra LOOP;
DONE:
    ret;
  )");
  VCHECK(u.peak_live >= 3);  // accumulator, counter and bound all live in the loop
}

// The counterpart to the test above, and the one that was missing: a value
// whose whole life is inside one iteration must not be charged for the loop.
// Treating "overlaps the loop" as "live across the loop" reported 328
// registers for a grid-stride kernel that ptxas compiles into 14, and refused
// launches hardware accepts.
VTEST(values_dead_within_an_iteration_do_not_inflate_the_loop) {
  auto u = usage_of(R"(
    .reg .pred %p<2>;
    .reg .b32 %r<40>;
    mov.u32 %r1, 0;
    mov.u32 %r2, 100;
LOOP:
    setp.ge.u32 %p1, %r1, %r2;
    @%p1 bra DONE;
    add.s32 %r10, %r1, 1;
    add.s32 %r11, %r10, 1;
    add.s32 %r12, %r11, 1;
    add.s32 %r13, %r12, 1;
    add.s32 %r14, %r13, 1;
    add.s32 %r15, %r14, 1;
    add.s32 %r1, %r15, 1;
    bra LOOP;
DONE:
    ret;
  )");
  // %r1 and %r2 live across the loop, plus at most one temporary at a time.
  VCHECK(u.peak_live <= 4);
}

VTEST(occupancy_matches_the_standard_calculation) {
  // An RTX 3060: 65536 registers/SM, 1536 threads/SM, 16 blocks/SM.
  // At 256 threads/block (8 warps) the warp ceiling gives 1536/256 = 6 blocks.
  auto o = ptx::compute_occupancy(/*regs*/ 24, /*threads*/ 256, 0, 0,
                                  65536, 1536, 16, 102400, 32);
  VCHECK_EQ(o.warps_per_block, 8u);
  VCHECK_EQ(o.blocks_per_sm, 6u);
  VCHECK_EQ(std::string(o.limited_by), "warps");

  // Heavy register use becomes the binding constraint instead.
  auto heavy = ptx::compute_occupancy(/*regs*/ 200, 256, 0, 0, 65536, 1536, 16, 102400, 32);
  VCHECK(heavy.blocks_per_sm < 6u);
  VCHECK_EQ(std::string(heavy.limited_by), "registers");

  // Shared memory can bind too.
  auto shmem = ptx::compute_occupancy(24, 256, 48 * 1024, 0, 65536, 1536, 16, 102400, 32);
  VCHECK_EQ(shmem.blocks_per_sm, 2u);
  VCHECK_EQ(std::string(shmem.limited_by), "shared memory");
}

VTEST(launch_bounds_are_enforced) {
  // __launch_bounds__ emits .maxntid; hardware refuses a larger block.
  std::string ptx = std::string(kHeader) +
                    ".visible .entry k()\n.maxntid 128, 1, 1\n{\n.reg .b32 %r<2>;\nmov.u32 %r1, 1;\nret;\n}\n";
  auto m = ptx::parse(ptx);
  VCHECK_EQ(m.entries[0].max_ntid[0], 128u);

  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/rtx3060");
  LaunchConfig ok;
  ok.block = {128, 1, 1};
  exec::launch(m.entries[0], ok, {}, mem, prof);  // at the bound: fine

  LaunchConfig too_big;
  too_big.block = {256, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], too_big, {}, mem, prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "__launch_bounds__");
}

VTEST(register_budget_refuses_an_impossible_launch) {
  // A block asking for more registers than the device has must fail the way
  // hardware does, not run and produce numbers.
  // Define many 64-bit values and only consume them at the end, so every one
  // is genuinely live at the same time (a chain of dead defs would not be).
  std::string body = ".reg .b64 %rd<600>;\n";
  for (int i = 1; i < 300; ++i)
    body += "mov.u64 %rd" + std::to_string(i) + ", " + std::to_string(i) + ";\n";
  for (int i = 2; i < 300; ++i)
    body += "add.s64 %rd1, %rd1, %rd" + std::to_string(i) + ";\n";
  body += "ret;\n";
  auto m = ptx::parse(std::string(kHeader) + ".visible .entry k()\n{\n" + body + "}\n");
  auto u = ptx::analyze_registers(m.entries[0]);
  VCHECK(u.regs_per_thread > 255);  // way past any real per-thread ceiling

  MemoryManager mem(1 << 20);
  DeviceProfile prof = load_gpu("nvidia/rtx3060");
  LaunchConfig cfg;
  cfg.block = {256, 1, 1};
  auto err = VCAPTURE(Error, exec::launch(m.entries[0], cfg, {}, mem, prof));
  VCHECK(err.code() == Err::LaunchConfig);
  VCHECK_CONTAINS(err.what(), "registers per thread");
}

VTEST(simple_kernel_matches_hardware_register_count) {
  // A one-store kernel measured 8 registers on a physical RTX 3060; the
  // estimate agrees after rounding to the allocation granularity.
  auto u = usage_of(R"(
    .reg .b32 %r<6>;
    .reg .b64 %rd<6>;
    ld.param.u64 %rd1, [k_param_0];
    cvta.to.global.u64 %rd2, %rd1;
    mov.u32 %r1, %tid.x;
    mul.wide.u32 %rd3, %r1, 4;
    add.s64 %rd4, %rd2, %rd3;
    st.global.u32 [%rd4], %r1;
    ret;
  )", ".param .u64 k_param_0");
  VCHECK_EQ(u.regs_per_thread, 8u);
}

VTEST_MAIN
