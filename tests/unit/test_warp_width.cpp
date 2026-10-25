// A launch's warp width (LaunchConfig::warp_size) and the warp-level operations at the widths other than 32.
//
// The engine used to run at the profile's width, 32 or 64, and refuse every warp-level PTX operation at 64. A frontend
// whose programming model picks the sub-group size per kernel (SPIR-V on Intel Xe: 8, 16 or 32) needs the width per
// launch, and needs shuffle, vote, ballot, activemask, match, redux and elect to mean what they mean at 32 over fewer
// lanes. Each kernel here runs at 8, 16, 32 and 64 lanes (the warp-level ones at 8, 16 and 32, where the masks are the
// low bits of a .b32; at 64 they are refused by name) and every result is compared with a host reference that is
// written from the ISA, not from the engine.
//
// Also: the per-warp bit sets (named barriers and the race detector) hold the 128 warps that 1024 threads make at 8 lanes.
#include <cstdint>
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

const char* kHeader = ".version 8.3\n.target sm_90\n.address_size 64\n";

std::vector<uint8_t> arg_u64(uint64_t v) {
  std::vector<uint8_t> b(8);
  std::memcpy(b.data(), &v, 8);
  return b;
}

// One kernel that stores `K` words per thread at out[tid * K + j], j = 0..K-1. `body` computes them (it may use
// %r1 = tid, %r2..%r30, %p1..%p6, %rd4 = this thread's output address) and ends with the stores `put(...)` writes.
std::string kernel(const std::string& body, const std::string& decls = "") {
  return std::string(kHeader) + ".visible .entry k(.param .u64 out)\n{\n" + decls +
         "    .reg .pred %p<8>;\n    .reg .b32 %r<48>;\n    .reg .b64 %rd<8>;\n"
         "    ld.param.u64 %rd1, [out];\n    cvta.to.global.u64 %rd2, %rd1;\n    mov.u32 %r1, %tid.x;\n" +
         body + "    ret;\n}\n";
}
// The address of this thread's K words, as %rd4.
std::string base(uint32_t K) {
  return "    mul.wide.u32 %rd3, %r1, " + std::to_string(K * 4) + ";\n    add.s64 %rd4, %rd2, %rd3;\n";
}
std::string put(uint32_t j, const std::string& reg) {
  return "    st.global.u32 [%rd4+" + std::to_string(j * 4) + "], " + reg + ";\n";
}

// Runs `ptx` with `threads` threads at warp width W; returns out[0 .. threads*K).
std::vector<uint32_t> run(const std::string& ptx, uint32_t K, uint32_t W, uint32_t threads) {
  DeviceProfile prof = load_gpu("nvidia/h100");
  const uint32_t before = prof.warp_size;
  auto m = ptx::parse(ptx);
  MemoryManager mem{1 << 20};
  const uint64_t out = mem.alloc(uint64_t{threads} * K * 4);
  for (uint64_t i = 0; i < uint64_t{threads} * K; ++i) mem.store_scalar(out + i * 4, 4, 0xDEADBEEFu);
  LaunchConfig cfg;
  cfg.block = {threads, 1, 1};
  cfg.warp_size = W;
  exec::launch(m.entries[0], cfg, {arg_u64(out)}, mem, prof);
  VCHECK_EQ(prof.warp_size, before);   // the launch ran on a copy: the caller's profile is untouched
  std::vector<uint32_t> r(static_cast<size_t>(threads) * K);
  for (size_t i = 0; i < r.size(); ++i) r[i] = static_cast<uint32_t>(mem.load_scalar(out + i * 4, 4));
  mem.free(out);
  return r;
}

uint32_t live_lanes(uint32_t W, uint32_t threads, uint32_t warp) {
  const uint32_t first = warp * W;
  return first >= threads ? 0 : (threads - first < W ? threads - first : W);
}
// The low `n` bits, n = 0..32.
uint32_t low_bits(uint32_t n) { return n >= 32 ? 0xFFFFFFFFu : (1u << n) - 1u; }

const uint32_t kWidths[] = {8, 16, 32, 64};
const uint32_t kNarrow[] = {8, 16, 32};

}  // namespace

// The ids, at every width: %laneid, %warpid, %nwarpid and WARP_SZ come from the launch's width.
VTEST(warp_ids_follow_the_launch_width) {
  const uint32_t threads = 80;   // not a multiple of 32 or 64: the last warp is partial
  for (uint32_t W : kWidths) {
    const auto r = run(kernel(base(4) + "    mov.u32 %r2, %laneid;\n    mov.u32 %r3, %warpid;\n"
                                        "    mov.u32 %r4, %nwarpid;\n    mov.u32 %r5, WARP_SZ;\n" +
                              put(0, "%r2") + put(1, "%r3") + put(2, "%r4") + put(3, "%r5")),
                       4, W, threads);
    for (uint32_t t = 0; t < threads; ++t) {
      VCHECK_EQ(r[t * 4 + 0], t % W);
      VCHECK_EQ(r[t * 4 + 1], t / W);
      VCHECK_EQ(r[t * 4 + 2], (threads + W - 1) / W);
      VCHECK_EQ(r[t * 4 + 3], W);
    }
  }
}

// %lanemask_eq/lt/le/gt/ge are the low `W` bits of a .b32. (The old `~0u << lane` and `~0u >> (W - lane)` forms were
// right at 32 only.)
VTEST(lane_masks_are_the_low_bits_of_a_32_bit_mask) {
  const uint32_t threads = 80;
  for (uint32_t W : kNarrow) {
    const auto r = run(kernel(base(5) + "    mov.u32 %r2, %lanemask_eq;\n    mov.u32 %r3, %lanemask_lt;\n"
                                        "    mov.u32 %r4, %lanemask_le;\n    mov.u32 %r5, %lanemask_gt;\n"
                                        "    mov.u32 %r6, %lanemask_ge;\n" +
                              put(0, "%r2") + put(1, "%r3") + put(2, "%r4") + put(3, "%r5") + put(4, "%r6")),
                       5, W, threads);
    const uint32_t full = low_bits(W);
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t l = t % W;
      const uint64_t le = (uint64_t{2} << l) - 1;
      VCHECK_EQ(r[t * 5 + 0], static_cast<uint32_t>(uint64_t{1} << l));
      VCHECK_EQ(r[t * 5 + 1], static_cast<uint32_t>((uint64_t{1} << l) - 1));
      VCHECK_EQ(r[t * 5 + 2], static_cast<uint32_t>(le));
      VCHECK_EQ(r[t * 5 + 3], full & ~static_cast<uint32_t>(le));
      VCHECK_EQ(r[t * 5 + 4], full & ~static_cast<uint32_t>((uint64_t{1} << l) - 1));
    }
  }
}

// activemask and the votes, in full warps and in a partial last one.
VTEST(activemask_and_votes_cover_the_warps_lanes) {
  const uint32_t threads = 80;
  for (uint32_t W : kNarrow) {
    const auto r = run(kernel(base(7) +
                              "    rem.u32 %r2, %r1, 3;\n    setp.eq.u32 %p1, %r2, 0;\n"
                              "    activemask.b32 %r3;\n"
                              "    vote.sync.ballot.b32 %r4, %p1, 0xffffffff;\n"
                              "    vote.sync.all.pred %p2, %p1, 0xffffffff;\n"
                              "    vote.sync.any.pred %p3, %p1, 0xffffffff;\n"
                              "    vote.sync.uni.pred %p4, %p1, 0xffffffff;\n"
                              "    selp.u32 %r5, 1, 0, %p2;\n    selp.u32 %r6, 1, 0, %p3;\n"
                              "    selp.u32 %r7, 1, 0, %p4;\n    vote.sync.ballot.b32 %r8, !%p1, 0xffffffff;\n" +
                              put(0, "%r3") + put(1, "%r4") + put(2, "%r5") + put(3, "%r6") + put(4, "%r7") +
                              put(5, "%r8")),
                       7, W, threads);
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t warp = t / W, live = live_lanes(W, threads, warp);
      uint32_t ballot = 0, nballot = 0;
      for (uint32_t l = 0; l < live; ++l) {
        if ((warp * W + l) % 3 == 0) ballot |= 1u << l;
        else nballot |= 1u << l;
      }
      const uint32_t active = low_bits(live);
      VCHECK_EQ(r[t * 7 + 0], active);
      VCHECK_EQ(r[t * 7 + 1], ballot);
      VCHECK_EQ(r[t * 7 + 2], ballot == active ? 1u : 0u);                 // vote.all
      VCHECK_EQ(r[t * 7 + 3], ballot != 0 ? 1u : 0u);                      // vote.any
      VCHECK_EQ(r[t * 7 + 4], (ballot == active || ballot == 0) ? 1u : 0u);   // vote.uni
      VCHECK_EQ(r[t * 7 + 5], nballot);                                    // ballot of the negated predicate
    }
  }
}

namespace {

enum Mode { kUp, kDown, kBfly, kIdx };
const char* kModeName[] = {"up", "down", "bfly", "idx"};

// shfl.sync's lane arithmetic from the PTX ISA (9.7.9.6), then this engine's one addition for warps narrower than
// 32: a source lane past the warp's last is out of range, as a clamped one is (the lane reads its own value and the
// predicate is false). A source lane that is not running the instruction gives 0.
struct ShflOut {
  uint32_t v;
  bool p;
};
ShflOut ref_shfl(Mode mode, uint32_t lane, uint32_t b, uint32_t c, uint32_t W, uint32_t live,
                 const std::vector<uint32_t>& vals) {
  const int bval = b & 31, cval = c & 31, seg = (c >> 8) & 31;
  const int min_lane = lane & seg, max_lane = (lane & seg) | (cval & ~seg & 31);
  int j = 0;
  bool pred = false;
  switch (mode) {
    case kUp:
      j = static_cast<int>(lane) - bval;
      pred = j >= max_lane;
      break;
    case kDown:
      j = static_cast<int>(lane) + bval;
      pred = j <= max_lane;
      break;
    case kBfly:
      j = static_cast<int>(lane) ^ bval;
      pred = j <= max_lane;
      break;
    case kIdx:
      j = min_lane | (bval & ~seg & 31);
      pred = j <= max_lane;
      break;
  }
  if (pred && j >= static_cast<int>(W)) pred = false;
  if (!pred) j = static_cast<int>(lane);
  return {static_cast<uint32_t>(j) < live ? vals[j] : 0u, pred};
}

}  // namespace

// shfl.sync in every mode, for shuffle widths a CUDA program would pass (segmask = 32 - W) and for the plain
// 32-lane form (a source past the warp's width is out of range).
VTEST(shuffles_in_every_mode_at_every_narrow_width) {
  const uint32_t threads = 80;
  struct Combo {
    Mode mode;
    uint32_t b;
    uint32_t c;   // 0xFFFFFFFF: choose from the launch width
  };
  for (uint32_t W : kNarrow) {
    const uint32_t seg = (32 - W) & 31;
    std::vector<Combo> combos;
    for (uint32_t b : {1u, 3u}) {
      combos.push_back({kUp, b, seg << 8});
      combos.push_back({kDown, b, (seg << 8) | 31});
      combos.push_back({kBfly, b, (seg << 8) | 31});
    }
    combos.push_back({kIdx, 5, (seg << 8) | 31});
    combos.push_back({kIdx, 0, (seg << 8) | 31});
    // The plain warp-wide forms: no segment mask, clamp 31 (up: 0). At W < 32 the sources they name past the warp
    // are out of range.
    combos.push_back({kBfly, 16, 31});
    combos.push_back({kDown, 7, 31});
    combos.push_back({kUp, 2, 0});
    combos.push_back({kIdx, 19, 31});
    const uint32_t K = static_cast<uint32_t>(combos.size()) * 2;
    std::string body = base(K) + "    mad.lo.u32 %r2, %r1, 7, 1000;\n    mov.u32 %r5, -1;\n";
    std::string stores;
    for (size_t i = 0; i < combos.size(); ++i) {
      body += "    mov.u32 %r3, " + std::to_string(combos[i].b) + ";\n    mov.u32 %r4, " + std::to_string(combos[i].c) +
              ";\n    shfl.sync." + kModeName[combos[i].mode] + ".b32 %r6|%p1, %r2, %r3, %r4, %r5;\n"
              "    selp.u32 %r7, 1, 0, %p1;\n";
      body += put(static_cast<uint32_t>(2 * i), "%r6") + put(static_cast<uint32_t>(2 * i + 1), "%r7");
    }
    const auto r = run(kernel(body), K, W, threads);
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t warp = t / W, lane = t % W, live = live_lanes(W, threads, warp);
      std::vector<uint32_t> vals(W);
      for (uint32_t l = 0; l < W; ++l) vals[l] = (warp * W + l) * 7 + 1000;
      for (size_t i = 0; i < combos.size(); ++i) {
        const ShflOut e = ref_shfl(combos[i].mode, lane, combos[i].b, combos[i].c, W, live, vals);
        if (r[t * K + 2 * i] != e.v || r[t * K + 2 * i + 1] != (e.p ? 1u : 0u)) {
          VCHECK_EQ(std::string("W=") + std::to_string(W) + " t=" + std::to_string(t) + " shfl." +
                        kModeName[combos[i].mode] + " b=" + std::to_string(combos[i].b) + " c=" +
                        std::to_string(combos[i].c) + " -> " + std::to_string(r[t * K + 2 * i]) + "," +
                        std::to_string(r[t * K + 2 * i + 1]),
                    std::string("W=") + std::to_string(W) + " t=" + std::to_string(t) + " shfl." +
                        kModeName[combos[i].mode] + " b=" + std::to_string(combos[i].b) + " c=" +
                        std::to_string(combos[i].c) + " -> " + std::to_string(e.v) + "," + std::to_string(e.p));
        }
      }
    }
  }
}

// The kernels a program actually writes with shuffles, against sums computed on the host: a butterfly reduction, a
// broadcast from lane 0, and an inclusive scan with shfl.up.
VTEST(warp_reduce_broadcast_and_scan_by_shuffle) {
  const uint32_t threads = 64;
  for (uint32_t W : kNarrow) {
    const uint32_t seg = (32 - W) & 31;
    std::string body = base(3) + "    add.u32 %r2, %r1, 1;\n    mov.u32 %r5, -1;\n    mov.u32 %r20, " +
                       std::to_string((seg << 8) | 31) + ";\n    mov.u32 %r21, " + std::to_string(seg << 8) + ";\n";
    // butterfly sum
    body += "    mov.u32 %r8, %r2;\n";
    for (uint32_t off = W / 2; off >= 1; off /= 2)
      body += "    mov.u32 %r3, " + std::to_string(off) + ";\n    shfl.sync.bfly.b32 %r4, %r8, %r3, %r20, %r5;\n"
              "    add.u32 %r8, %r8, %r4;\n";
    // broadcast of lane 0
    body += "    mov.u32 %r3, 0;\n    shfl.sync.idx.b32 %r9, %r2, %r3, %r20, %r5;\n";
    // inclusive scan: add the value d lanes up when there is one (the predicate)
    body += "    mov.u32 %r10, %r2;\n";
    for (uint32_t d = 1; d < W; d *= 2)
      body += "    mov.u32 %r3, " + std::to_string(d) + ";\n    shfl.sync.up.b32 %r4|%p1, %r10, %r3, %r21, %r5;\n"
              "    @%p1 add.u32 %r10, %r10, %r4;\n";
    body += put(0, "%r8") + put(1, "%r9") + put(2, "%r10");
    const auto r = run(kernel(body), 3, W, threads);
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t warp = t / W, lane = t % W;
      uint32_t sum = 0, scan = 0;
      for (uint32_t l = 0; l < W; ++l) sum += warp * W + l + 1;
      for (uint32_t l = 0; l <= lane; ++l) scan += warp * W + l + 1;
      VCHECK_EQ(r[t * 3 + 0], sum);
      VCHECK_EQ(r[t * 3 + 1], warp * W + 1);
      VCHECK_EQ(r[t * 3 + 2], scan);
    }
  }
}

// match.any, redux.sync and elect.sync over a warp of 8, 16 or 32 lanes.
VTEST(match_redux_and_elect_over_a_narrow_warp) {
  const uint32_t threads = 64;
  for (uint32_t W : kNarrow) {
    const auto r = run(kernel(base(7) +
                              "    and.b32 %r2, %r1, 3;\n"
                              "    match.any.sync.b32 %r3, %r2, -1;\n"
                              "    mov.u32 %r4, %laneid;\n    add.u32 %r4, %r4, 1;\n"
                              "    redux.sync.add.u32 %r5, %r4, -1;\n"
                              "    redux.sync.max.u32 %r6, %r4, -1;\n"
                              "    redux.sync.min.u32 %r7, %r4, -1;\n"
                              "    elect.sync %r8|%p1, -1;\n    selp.u32 %r9, 1, 0, %p1;\n" +
                              put(0, "%r3") + put(1, "%r5") + put(2, "%r6") + put(3, "%r7") + put(4, "%r8") +
                              put(5, "%r9")),
                       7, W, threads);
    for (uint32_t t = 0; t < threads; ++t) {
      const uint32_t warp = t / W, lane = t % W;
      uint32_t same = 0;
      for (uint32_t l = 0; l < W; ++l)
        if (((warp * W + l) & 3) == ((warp * W + lane) & 3)) same |= 1u << l;
      VCHECK_EQ(r[t * 7 + 0], same);
      VCHECK_EQ(r[t * 7 + 1], W * (W + 1) / 2);
      VCHECK_EQ(r[t * 7 + 2], W);
      VCHECK_EQ(r[t * 7 + 3], 1u);
      VCHECK_EQ(r[t * 7 + 4], 0u);                       // the leader is lane 0
      VCHECK_EQ(r[t * 7 + 5], lane == 0 ? 1u : 0u);      // and only it holds the predicate
    }
  }
}

// At 64 lanes the 32-bit-mask operations are refused by name (the engine will not invent a 64-bit meaning for them);
// the tensor-core instructions need exactly 32.
VTEST(the_32_bit_mask_operations_are_refused_at_64_lanes) {
  struct Refused {
    const char* what;
    std::string body;
  };
  const Refused cases[] = {
      {"shfl.sync", "    mov.u32 %r2, 1;\n    shfl.sync.idx.b32 %r3, %r2, %r2, 31, -1;\n"},
      {"vote/ballot", "    setp.eq.u32 %p1, %r1, 0;\n    vote.sync.ballot.b32 %r3, %p1, -1;\n"},
      {"activemask", "    activemask.b32 %r3;\n"},
      {"match.sync", "    match.any.sync.b32 %r3, %r1, -1;\n"},
      {"redux.sync", "    redux.sync.add.u32 %r3, %r1, -1;\n"},
      {"%lanemask_*", "    mov.u32 %r3, %lanemask_lt;\n"},
  };
  for (const Refused& c : cases) {
    auto err = VCAPTURE(Error, run(kernel(base(1) + c.body + put(0, "%r3")), 1, 64, 64));
    VCHECK(err.code() == Err::UnsupportedPtx);
    VCHECK_CONTAINS(err.what(), c.what);
    VCHECK_CONTAINS(err.what(), "64 lanes wide");
  }
}

// A shared-memory tree reduction with bar.sync and a divergent step: nothing warp-level in it, so every width must
// give the host's sum. 256 threads of 8 lanes are 32 warps; of 64, four.
VTEST(a_block_reduction_is_the_same_at_every_width) {
  const std::string decls = "    .shared .align 4 .b8 sm[1024];\n";
  const std::string body =
      "    mov.u32 %r2, sm;\n    shl.b32 %r3, %r1, 2;\n    add.u32 %r4, %r2, %r3;\n"
      "    mad.lo.u32 %r5, %r1, 3, 1;\n    and.b32 %r6, %r1, 1;\n    setp.eq.u32 %p5, %r6, 1;\n"
      "    @%p5 add.u32 %r5, %r5, 0;\n    st.shared.u32 [%r4], %r5;\n    bar.sync 0;\n"
      "    mov.u32 %r10, 128;\n"
      "LOOP:\n"
      "    setp.ge.u32 %p2, %r1, %r10;\n    @%p2 bra SKIP;\n"
      "    add.u32 %r11, %r1, %r10;\n    shl.b32 %r12, %r11, 2;\n    add.u32 %r13, %r2, %r12;\n"
      "    ld.shared.u32 %r14, [%r13];\n    ld.shared.u32 %r15, [%r4];\n    add.u32 %r15, %r15, %r14;\n"
      "    st.shared.u32 [%r4], %r15;\n"
      "SKIP:\n    bar.sync 0;\n    shr.u32 %r10, %r10, 1;\n    setp.ne.u32 %p3, %r10, 0;\n    @%p3 bra LOOP;\n"
      "    setp.ne.u32 %p4, %r1, 0;\n    @%p4 bra DONE;\n"
      "    ld.shared.u32 %r16, [%r2];\n    st.global.u32 [%rd2], %r16;\n"
      "DONE:\n";
  uint32_t expect = 0;
  for (uint32_t t = 0; t < 256; ++t) expect += 3 * t + 1;
  for (uint32_t W : kWidths) {
    const auto r = run(kernel(body, decls), 1, W, 256);
    VCHECK_EQ(r[0], expect);
  }
}

// A barrier with a thread count keeps the set of warps waiting at it: at 8 lanes a 1024-thread block is 128 warps, more
// than a 64-bit set holds (the shift by a warp index past 63 was undefined).
VTEST(a_counted_barrier_holds_128_warps) {
  const std::string decls = "    .shared .align 4 .b8 buf[4096];\n";
  const std::string body = base(1) +
                           "    mov.u32 %r2, buf;\n    shl.b32 %r3, %r1, 2;\n    add.u32 %r4, %r2, %r3;\n"
                           "    add.u32 %r5, %r1, 100;\n    st.shared.u32 [%r4], %r5;\n"
                           "    bar.sync 1, 1024;\n"
                           "    add.u32 %r6, %r1, 37;\n    and.b32 %r6, %r6, 1023;\n    shl.b32 %r7, %r6, 2;\n"
                           "    add.u32 %r8, %r2, %r7;\n    ld.shared.u32 %r9, [%r8];\n" +
                           put(0, "%r9");
  for (uint32_t W : kWidths) {
    const auto r = run(kernel(body, decls), 1, W, 1024);
    for (uint32_t t = 0; t < 1024; ++t) VCHECK_EQ(r[t], ((t + 37) & 1023) + 100);
  }
}

// The race detector names the warps by index, and the set of warps that read a word spans 128 of them at 8 lanes: warp
// 100 reads, warp 101 writes the same word with no barrier between them.
VTEST(the_race_detector_tells_warps_past_the_64th_apart) {
  const std::string decls = "    .shared .align 4 .b8 buf[64];\n";
  auto kernel_for = [&](bool second_writes) {
    return kernel(std::string("    mov.u32 %r2, buf;\n    shr.u32 %r3, %r1, 3;\n"
                              "    setp.eq.u32 %p1, %r3, 100;\n    @%p1 ld.shared.u32 %r4, [%r2];\n"
                              "    setp.eq.u32 %p2, %r3, 101;\n") +
                      (second_writes ? "    mov.u32 %r5, 9;\n    @%p2 st.shared.u32 [%r2], %r5;\n"
                                     : "    @%p2 ld.shared.u32 %r4, [%r2];\n"),
                  decls);
  };
  ::setenv("VGPU_RACE", "1", 1);
  auto err = VCAPTURE(Error, run(kernel_for(true), 1, 8, 1024));
  VCHECK(err.code() == Err::DataRace);
  VCHECK_CONTAINS(err.what(), "warp 101 and warp 100");
  // Both only read: not a race, however many warps read it.
  (void)run(kernel_for(false), 1, 8, 1024);
  ::unsetenv("VGPU_RACE");
}

// The gate: a width the engine has no storage for is refused by the launch, naming it.
VTEST(an_unsupported_warp_width_is_refused) {
  for (uint32_t bad : {1u, 4u, 12u, 24u, 48u, 128u}) {
    auto err = VCAPTURE(Error, run(kernel(base(1) + "    mov.u32 %r2, 1;\n" + put(0, "%r2")), 1, bad, 64));
    VCHECK(err.code() == Err::Unsupported);
    VCHECK_CONTAINS(err.what(), "only 8, 16, 32 and 64");
    VCHECK_CONTAINS(err.what(), std::to_string(bad));
  }
}

VTEST_MAIN
