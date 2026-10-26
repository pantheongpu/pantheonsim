// The PTX interpreter's batches of a diverged warp's instructions (run_micro_batch on one of several paths) against its
// one-instruction-at-a-time loop.
//
// A warp whose lanes took different branches is a set of paths, and the one with the lowest pc runs next; paths that reach one
// pc merge, a path that has waited more than 1024 instructions gets the next 256 (select_path). The batch may run a path's
// instructions together only while all of that is unchanged, so every case here is a kernel whose result depends on it: loops
// of different lengths per lane, a lane that waits for another's store, a barrier after a diverged region, nested ifs that merge
// and split again, a path parked at a barrier while another passes its pc. Each runs with the fast paths (VGPU_FASTPATH=1) and
// with VGPU_FASTPATH=0, which runs one instruction at a time; the memory they leave, the statistics (every counter and the
// per-opcode counts) and the error, if any, must be the same.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
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

struct Outcome {
  std::vector<uint32_t> out;
  std::vector<uint64_t> counters;
  std::string error;
};

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}
std::vector<uint8_t> arg_u32(uint32_t v) {
  std::vector<uint8_t> b(4);
  std::memcpy(b.data(), &v, 4);
  return b;
}

// Kernel template: %r1 is the thread's number in the block, %r2 the global thread number, %rd_out the thread's word of `out`
// (out[tid]), %rd_flag a word shared by all threads (flag), %r_n the parameter n. The body writes its results to out[tid].
std::string kernel(const std::string& body) {
  return R"(.version 8.3
.target sm_86
.address_size 64
.visible .entry k(.param .u64 pout, .param .u64 pflag, .param .u32 pn)
{
    .reg .pred %p<8>;
    .reg .b32 %r<32>;
    .reg .b64 %rd<8>;
    ld.param.u64 %rd1, [pout];
    ld.param.u64 %rd2, [pflag];
    ld.param.u32 %r20, [pn];
    mov.u32 %r1, %tid.x;
    mov.u32 %r3, %ctaid.x;
    mov.u32 %r4, %ntid.x;
    mad.lo.u32 %r2, %r3, %r4, %r1;
    mul.wide.u32 %rd3, %r2, 4;
    add.u64 %rd4, %rd1, %rd3;
)" + body + R"(
    ret;
}
)";
}

Outcome run(const std::string& body, uint32_t blocks, uint32_t threads, uint32_t n, bool fast) {
  setenv("VGPU_FASTPATH", fast ? "1" : "0", 1);
  setenv("VGPU_THREADS", "1", 1);   // (blocks in a fixed order: a kernel that spins on another block's store would otherwise vary)
  Outcome o;
  MemoryManager mem{1 << 20};
  DeviceProfile prof = load_gpu("nvidia/a10");
  static std::map<std::string, ptx::Module> parsed;
  auto it = parsed.find(body);
  if (it == parsed.end()) it = parsed.emplace(body, ptx::parse(kernel(body))).first;
  const uint64_t total = uint64_t{blocks} * threads;
  const uint64_t pout = mem.alloc(total * 4 + 64), pflag = mem.alloc(64);
  LaunchConfig cfg;
  cfg.grid = {blocks, 1, 1};
  cfg.block = {threads, 1, 1};
  try {
    exec::LaunchStats st = exec::launch(it->second.entries[0], cfg, {arg_u64(pout), arg_u64(pflag), arg_u32(n)}, mem, prof);
    const auto* w = reinterpret_cast<const uint64_t*>(&st);
    o.counters.assign(w, w + exec::kCounterWords);
    o.counters.insert(o.counters.end(), st.inst_by_opcode.begin(), st.inst_by_opcode.end());
  } catch (const Error& e) {
    o.error = e.what();
  }
  unsetenv("VGPU_FASTPATH");
  unsetenv("VGPU_THREADS");
  o.out.resize(total);
  mem.read(pout, o.out.data(), total * 4);
  uint32_t flag = 0;
  mem.read(pflag, &flag, 4);
  o.out.push_back(flag);
  return o;
}

void same(const char* what, const std::string& body, uint32_t blocks, uint32_t threads, uint32_t n) {
  const Outcome a = run(body, blocks, threads, n, true), b = run(body, blocks, threads, n, false);
  const std::string where = std::string(what) + " blocks=" + std::to_string(blocks) + " threads=" + std::to_string(threads) +
                            " n=" + std::to_string(n);
  if (a.error != b.error) throw vtest::Failure(where + ": error differs\n  fast:    " + a.error + "\n  general: " + b.error);
  if (a.out != b.out) {
    size_t i = 0;
    while (i < a.out.size() && i < b.out.size() && a.out[i] == b.out[i]) ++i;
    throw vtest::Failure(where + ": memory differs at word " + std::to_string(i) + ": " +
                         std::to_string(i < a.out.size() ? a.out[i] : 0) + " vs " + std::to_string(i < b.out.size() ? b.out[i] : 0));
  }
  if (a.counters != b.counters) {
    std::string d;
    for (size_t i = 0; i < a.counters.size() && i < b.counters.size(); ++i)
      if (a.counters[i] != b.counters[i])
        d += " [" + std::to_string(i) + "]: " + std::to_string(a.counters[i]) + " vs " + std::to_string(b.counters[i]);
    throw vtest::Failure(where + ": the counters differ:" + d);
  }
}

// Each lane loops a different number of times (a Collatz walk); the paths diverge, merge at the exit, and run on.
const char* kCollatz = R"(
    add.u32 %r5, %r2, 1;
    mov.u32 %r6, 0;
$C:
    setp.eq.u32 %p1, %r5, 1;
    @%p1 bra $CDONE;
    and.b32 %r7, %r5, 1;
    setp.eq.u32 %p2, %r7, 0;
    @%p2 bra $EVEN;
    mad.lo.u32 %r5, %r5, 3, 1;
    add.u32 %r6, %r6, 1;
    bra $C;
$EVEN:
    shr.u32 %r5, %r5, 1;
    add.u32 %r6, %r6, 1;
    bra $C;
$CDONE:
    st.global.u32 [%rd4], %r6;
)";

// Lane 0 stores a flag after a short wait while the other lanes spin on it, in a loop at a lower pc than the store: the spinners
// are the lowest path for 1024 instructions, then lane 0 gets 256. How many rounds a spinner counts before it sees the flag is
// what the starvation rule decides, and what a batch of the spinners' instructions must not move. %r20 (n) spins before lane 0
// stores, a pad of straight-line instructions in lane 0's path makes it longer.
std::string spin_body(int pad, int pre, int hold = 0) {
  std::string pads, pres, holds;
  for (int i = 0; i < hold; ++i) holds += "    add.u32 %r12, %r12, " + std::to_string(i % 7 + 1) + ";\n";
  for (int i = 0; i < pad; ++i) pads += "    add.u32 %r8, %r8, " + std::to_string(i + 1) + ";\n";
  for (int i = 0; i < pre; ++i) pres += "    add.u32 %r11, %r11, " + std::to_string(i + 1) + ";\n";
  return pres + R"(
    setp.eq.u32 %p1, %r1, 0;
    @%p1 bra $HOLDER;
    mov.u32 %r6, 0;
$SPIN:
    ld.volatile.global.u32 %r7, [%rd2];
    add.u32 %r6, %r6, 1;
    setp.eq.u32 %p2, %r7, 0;
    mul.lo.u32 %r8, %r6, 3;
    xor.b32 %r8, %r8, %r6;
)" + pads + R"(    @%p2 bra $SPIN;
    st.global.u32 [%rd4], %r6;
    bra $END;
$HOLDER:
)" + holds + R"(    mov.u32 %r9, 0;
    add.u32 %r9, %r9, 1;
    add.u32 %r9, %r9, 2;
    add.u32 %r9, %r9, 3;
    add.u32 %r9, %r9, 4;
    mov.u32 %r10, 77;
    st.volatile.global.u32 [%rd2], %r10;
    st.global.u32 [%rd4], %r9;
$END:
)";
}


// Nested ifs that split a warp, merge it, and split it again; a barrier after the region (the block's other warps arrive at it
// on other paths' time), and a store from the merged warp.
const char* kNested = R"(
    and.b32 %r5, %r2, 3;
    setp.eq.u32 %p1, %r5, 0;
    @%p1 bra $A;
    setp.eq.u32 %p2, %r5, 1;
    @%p2 bra $B;
    mad.lo.u32 %r6, %r2, 7, 1;
    bra $J1;
$A:
    mad.lo.u32 %r6, %r2, 11, 2;
    add.u32 %r6, %r6, %r6;
    bra $J1;
$B:
    mov.u32 %r6, 5;
    add.u32 %r6, %r6, %r2;
$J1:
    and.b32 %r7, %r2, 8;
    setp.eq.u32 %p3, %r7, 0;
    @%p3 bra $K;
    xor.b32 %r6, %r6, 0x5555;
    add.u32 %r6, %r6, 3;
$K:
    bar.sync 0;
    add.u32 %r6, %r6, 1;
    st.global.u32 [%rd4], %r6;
)";

// A warp split at a bar.sync: the lanes that take the branch go on to the barrier while the others run a longer way round,
// and the barrier's path is parked at a pc the others reach (and merge with).
const char* kBarrier = R"(
    setp.lt.u32 %p1, %r1, 16;
    @%p1 bra $EARLY;
    mov.u32 %r6, 0;
    add.u32 %r6, %r6, 1;
    add.u32 %r6, %r6, 2;
    add.u32 %r6, %r6, 3;
    bra $SYNC;
$EARLY:
    mov.u32 %r6, 100;
$SYNC:
    bar.sync 0;
    add.u32 %r6, %r6, %r1;
    st.global.u32 [%rd4], %r6;
)";

// Lanes leave a loop one by one and return at different pcs (exited lanes, then the rest).
const char* kExit = R"(
    and.b32 %r5, %r2, 31;
    mov.u32 %r6, 0;
$L:
    add.u32 %r6, %r6, 1;
    setp.lt.u32 %p1, %r6, %r5;
    @%p1 bra $L;
    st.global.u32 [%rd4], %r6;
    setp.lt.u32 %p2, %r5, 10;
    @%p2 bra $OUT;
    mul.lo.u32 %r6, %r6, 3;
    add.u32 %r6, %r6, 1;
    st.global.u32 [%rd4], %r6;
$OUT:
)";

// A spinning warp-mate that another warp releases: the slice a warp may run is bounded and the batch stops at it.
const char* kSpinBlock = R"(
    setp.ge.u32 %p1, %r1, 32;
    @%p1 bra $RELEASE;
    mov.u32 %r6, 0;
$W:
    ld.volatile.global.u32 %r7, [%rd2];
    add.u32 %r6, %r6, 1;
    setp.eq.u32 %p2, %r7, 0;
    @%p2 bra $W;
    st.global.u32 [%rd4], %r6;
    bra $FIN;
$RELEASE:
    mov.u32 %r8, 1;
    st.volatile.global.u32 [%rd2], %r8;
$FIN:
)";

}  // namespace

VTEST(collatz_walks_of_different_lengths_match_one_instruction_at_a_time) {
  same("collatz", kCollatz, 1, 32, 0);
  same("collatz", kCollatz, 2, 64, 0);
  same("collatz", kCollatz, 3, 40, 0);   // a warp of 8 lanes
}

VTEST(a_lane_that_waits_for_a_store_sees_it_on_the_same_round_both_ways) {
  // The loop's length moves where in it the 1024th instruction falls.
  // (A path is starved when more than 1024 instructions have been issued since it last ran: a loop of 25 or 41 instructions
  // has an iteration end exactly there.)
  for (int pad = 0; pad < 44; ++pad) {
    same("spin", spin_body(pad, 0), 1, 32, 0);
    same("spin", spin_body(pad, 5), 1, 8, 0);
  }
  // A holder with a long straight run before its store: it has the turn for 256 instructions, then the spinners have it again
  // until it is starved once more, and the store lands in the third or fourth of its turns.
  for (int hold : {200, 300, 700, 1400})
    for (int pad : {0, 3, 9, 20}) {
      const std::string what = "spin with a long holder (" + std::to_string(hold) + " instructions, loop pad " + std::to_string(pad) + ")";
      same(what.c_str(), spin_body(pad, 0, hold), 1, 32, 0);
    }
}

VTEST(nested_ifs_and_a_barrier_after_the_diverged_region) {
  same("nested", kNested, 1, 64, 0);
  same("nested", kNested, 3, 128, 0);
  same("nested", kNested, 2, 96, 0);
}

VTEST(a_path_parked_at_a_barrier_meets_the_path_that_comes_to_it) {
  same("barrier", kBarrier, 1, 32, 0);
  same("barrier", kBarrier, 2, 64, 0);
}

VTEST(lanes_that_leave_a_loop_one_by_one_and_return_at_different_pcs) {
  same("exit", kExit, 1, 32, 0);
  same("exit", kExit, 2, 48, 0);
}

VTEST(a_warp_spinning_on_the_store_of_another_warp) {
  same("spin across warps", kSpinBlock, 1, 64, 0);
  same("spin across warps", kSpinBlock, 2, 96, 0);
}

VTEST_MAIN
