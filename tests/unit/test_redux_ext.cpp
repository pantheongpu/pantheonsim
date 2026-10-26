// OpRedux, extended (patch E7): float add/mul/min/max, integer mul, inclusive and exclusive scans and clustered
// reductions, over the active lanes, in increasing lane order.
//
// Each case builds a kernel with the AST builder, runs it at warp widths 8, 16 and 32 (a block of 100 threads: the
// last warp of each width is partial), once with every lane taking part and once with a diverged warp (the lanes
// whose block index is a multiple of 5 skip the instruction, so it runs under an irregular active mask), and
// compares every lane with a host fold written from the definition: lane order, the participating lanes only,
// a scan's exclusive first lane getting the identity, a cluster being an aligned run of lanes. Floats are compared
// bit for bit, every NaN being one value (the engine's float add, mul, min and max are the references).
// Also: the PTX redux.sync integer reduce, unchanged, through the parser; and the refusals.
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/ptx/parser.hpp"
#include "ast_run.hpp"
#include "vtest.hpp"

using namespace astrun;
using vgpu::ptx::ReduxMode;
using vgpu::ptx::ReduxOp;

namespace {

const uint32_t kThreads = 100;

const char* op_name(ReduxOp op) {
  switch (op) {
    case ReduxOp::Add: return "add";
    case ReduxOp::Min: return "min";
    case ReduxOp::Max: return "max";
    case ReduxOp::And: return "and";
    case ReduxOp::Or: return "or";
    case ReduxOp::Xor: return "xor";
    case ReduxOp::Mul: return "mul";
  }
  return "?";
}

std::vector<uint8_t> run_redux(ReduxOp op, const TySpec& t, ReduxMode mode, uint32_t cluster, bool diverge, uint32_t W,
                               const std::vector<uint64_t>& in) {
  vgpu::ptx::KernelBuilder kb("redux");
  Prologue p = prologue(kb, 2);
  const Type ty = t.type();
  const Reg x = load_elem(kb, p.ptr[0], p.gid, ty);
  const vgpu::ptx::Label skip = kb.make_label();
  if (diverge) {
    const Reg rem = kb.temp(tu(32));
    kb.emit(ibin(IntBinOp::Rem, tu(32), rem, RO(p.lin), IM(5)), "rem");
    const Reg q = kb.pred();
    kb.emit(setp(CmpOp::Eq, tu(32), q, RO(rem), IM(0)), "setp");
    kb.bra(skip, q);
  }
  const Reg am = kb.temp(tu(32));
  kb.emit(OpActiveMask{am}, "activemask");
  const Reg r = kb.temp(ty);
  OpRedux rd;
  rd.op = op;
  rd.ty = ty;
  rd.dst = r;
  rd.src = RO(x);
  rd.members = RO(am);
  rd.mode = mode;
  rd.cluster = cluster;
  kb.emit(rd, "redux");
  store_elem(kb, p.ptr[1], p.gid, ty, r);
  kb.bind(skip);
  kb.emit(OpRet{}, "ret");

  const uint32_t eb = t.bits / 8;
  std::vector<uint8_t> inb(in.size() * eb), outb(in.size() * eb, 0xCD);
  for (size_t i = 0; i < in.size(); ++i) std::memcpy(&inb[i * eb], &in[i], eb);
  vgpu::exec::LaunchConfig cfg;
  cfg.block = {kThreads, 1, 1};
  cfg.warp_size = W;
  return run(kb.finish(), cfg, {inb, outb})[1];
}

// The host fold. Returns the expected output per thread, or nothing (not written) for the skipped lanes.
struct Expect {
  bool written = false;
  uint64_t v = 0;
};
std::vector<Expect> expected(ReduxOp op, const TySpec& t, ReduxMode mode, uint32_t cluster, bool diverge, uint32_t W,
                             const std::vector<uint64_t>& in) {
  std::vector<Expect> out(kThreads);
  uint32_t cs = cluster ? cluster : W;
  if (cs > W) cs = W;
  for (uint32_t warp0 = 0; warp0 < kThreads; warp0 += W) {
    for (uint32_t base = 0; base < W; base += cs) {
      bool first = true;
      uint64_t acc = 0;
      std::vector<uint32_t> members;
      for (uint32_t lane = base; lane < base + cs && lane < W; ++lane) {
        const uint32_t tid = warp0 + lane;
        if (tid >= kThreads) break;
        if (diverge && tid % 5 == 0) continue;
        members.push_back(tid);
        const uint64_t v = in[tid];
        if (mode == ReduxMode::Exclusive) {
          out[tid] = {true, first ? ref_identity(op, t) : acc};
          acc = first ? v : ref_combine(op, t, acc, v);
        } else {
          acc = first ? v : ref_combine(op, t, acc, v);
          if (mode == ReduxMode::Inclusive) out[tid] = {true, acc};
        }
        first = false;
      }
      if (mode == ReduxMode::Reduce)
        for (uint32_t tid : members) out[tid] = {true, acc};
    }
  }
  return out;
}

}  // namespace

VTEST(float_and_integer_reductions_scans_and_clusters_follow_lane_order) {
  const std::vector<TySpec> types = {{false, true, 8},  {false, false, 16}, {false, true, 32}, {false, false, 32},
                                     {false, true, 64}, {false, false, 64}, {true, true, 16},  {true, true, 32},
                                     {true, true, 64}};
  const ReduxOp int_ops[] = {ReduxOp::Add, ReduxOp::Mul, ReduxOp::Min, ReduxOp::Max, ReduxOp::And, ReduxOp::Or, ReduxOp::Xor};
  const ReduxOp fl_ops[] = {ReduxOp::Add, ReduxOp::Mul, ReduxOp::Min, ReduxOp::Max};
  struct Form { ReduxMode mode; uint32_t cluster; };
  const Form forms[] = {{ReduxMode::Reduce, 0}, {ReduxMode::Inclusive, 0}, {ReduxMode::Exclusive, 0},
                        {ReduxMode::Reduce, 1}, {ReduxMode::Reduce, 2}, {ReduxMode::Reduce, 8}};
  uint64_t cases = 0;
  Rng rng(77);
  for (const TySpec& t : types) {
    const std::vector<ReduxOp> ops = t.fl ? std::vector<ReduxOp>(std::begin(fl_ops), std::end(fl_ops))
                                          : std::vector<ReduxOp>(std::begin(int_ops), std::end(int_ops));
    for (ReduxOp op : ops) {
      std::vector<uint64_t> in(kThreads);
      const bool minmax = op == ReduxOp::Min || op == ReduxOp::Max;
      for (auto& v : in) v = random_value(rng, t, /*zeros=*/!(t.fl && t.bits == 16 && minmax));
      for (const Form& f : forms) {
        for (bool diverge : {false, true}) {
          for (uint32_t W : {8u, 16u, 32u}) {
            const auto got = run_redux(op, t, f.mode, f.cluster, diverge, W, in);
            const auto want = expected(op, t, f.mode, f.cluster, diverge, W, in);
            const uint32_t eb = t.bits / 8;
            for (uint32_t tid = 0; tid < kThreads; ++tid) {
              uint64_t g = 0;
              std::memcpy(&g, &got[static_cast<size_t>(tid) * eb], eb);
              bool ok;
              if (want[tid].written) ok = same_value(t, g, want[tid].v);
              else ok = g == (t.mask() & 0xCDCDCDCDCDCDCDCDull);   // untouched
              if (!ok)
                throw vtest::Failure("redux." + std::string(op_name(op)) + "." + t.name() + " mode " +
                                     std::to_string(static_cast<int>(f.mode)) + " cluster " + std::to_string(f.cluster) +
                                     (diverge ? " diverged" : "") + " W=" + std::to_string(W) + " thread " + std::to_string(tid) +
                                     ": got " + std::to_string(g) + " want " + std::to_string(want[tid].v));
            }
            ++cases;
          }
        }
      }
    }
  }
  VCHECK(cases > 1900);
}

// A float sum depends on the order it is added in; the order is the lane order, so a hand-built sequence with a big
// value and many small ones has exactly one answer.
VTEST(a_float_sum_is_added_in_lane_order) {
  const TySpec f32t{true, true, 32};
  std::vector<uint64_t> in(kThreads);
  for (uint32_t i = 0; i < kThreads; ++i) {
    const float v = (i % 8 == 0) ? 16777216.0f : 1.0f;   // 2^24 swallows a following 1.0f
    in[i] = std::bit_cast<uint32_t>(v);
  }
  for (uint32_t W : {8u, 16u, 32u}) {
    const auto got = run_redux(ReduxOp::Add, f32t, ReduxMode::Reduce, 0, false, W, in);
    const auto want = expected(ReduxOp::Add, f32t, ReduxMode::Reduce, 0, false, W, in);
    for (uint32_t tid = 0; tid < kThreads; ++tid) {
      uint32_t g;
      std::memcpy(&g, &got[tid * 4], 4);
      VCHECK_EQ(g, static_cast<uint32_t>(want[tid].v));
    }
    // warp 0, W = 8: 2^24 then seven 1.0f, each swallowed
    uint32_t g0;
    std::memcpy(&g0, &got[0], 4);
    if (W == 8) VCHECK_EQ(g0, std::bit_cast<uint32_t>(16777216.0f));
  }
}

// Bad forms are refused by name.
VTEST(redux_refusals) {
  const TySpec bad_float{true, true, 32};
  auto err = VCAPTURE(vgpu::Error, run_redux(ReduxOp::And, bad_float, ReduxMode::Reduce, 0, false, 32, std::vector<uint64_t>(kThreads, 1)));
  VCHECK_CONTAINS(err.what(), "float reduction");
  auto err2 = VCAPTURE(vgpu::Error, run_redux(ReduxOp::Add, TySpec{false, false, 32}, ReduxMode::Reduce, 3, false, 32, std::vector<uint64_t>(kThreads, 1)));
  VCHECK_CONTAINS(err2.what(), "power of two");
}

// The PTX instruction is untouched: redux.sync.add/min/max/and/or/xor.u32/.s32 through the parser, at the widths.
VTEST(ptx_redux_sync_still_means_what_it_meant) {
  for (uint32_t W : {8u, 16u, 32u}) {
    const std::string ptx = std::string(".version 8.3\n.target sm_90\n.address_size 64\n") +
        ".visible .entry k(.param .u64 out)\n{\n    .reg .b32 %r<8>;\n    .reg .b64 %rd<4>;\n"
        "    ld.param.u64 %rd1, [out];\n    cvta.to.global.u64 %rd2, %rd1;\n    mov.u32 %r1, %tid.x;\n"
        "    add.u32 %r2, %r1, 1;\n    activemask.b32 %r3;\n"
        "    redux.sync.add.u32 %r4, %r2, %r3;\n    redux.sync.max.s32 %r5, %r2, %r3;\n"
        "    redux.sync.xor.b32 %r6, %r2, %r3;\n"
        "    mul.wide.u32 %rd3, %r1, 12;\n    add.s64 %rd3, %rd2, %rd3;\n"
        "    st.global.u32 [%rd3], %r4;\n    st.global.u32 [%rd3+4], %r5;\n    st.global.u32 [%rd3+8], %r6;\n    ret;\n}\n";
    auto m = vgpu::ptx::parse(ptx);
    vgpu::DeviceProfile prof = vgpu::load_gpu("nvidia/h100");
    vgpu::MemoryManager mem{1 << 20};
    const uint64_t dout = mem.alloc(kThreads * 12);
    for (uint32_t i = 0; i < kThreads * 3; ++i) mem.store_scalar(dout + i * 4, 4, 0);
    vgpu::exec::LaunchConfig cfg;
    cfg.block = {kThreads, 1, 1};
    cfg.warp_size = W;
    std::vector<uint8_t> arg(8);
    std::memcpy(arg.data(), &dout, 8);
    vgpu::exec::launch(m.entries[0], cfg, {arg}, mem, prof);
    std::vector<uint32_t> out(kThreads * 3);
    for (uint32_t i = 0; i < kThreads * 3; ++i) out[i] = static_cast<uint32_t>(mem.load_scalar(dout + i * 4, 4));
    for (uint32_t w0 = 0; w0 < kThreads; w0 += W) {
      uint32_t sum = 0, mx = 0, x = 0;
      for (uint32_t l = 0; l < W && w0 + l < kThreads; ++l) {
        sum += w0 + l + 1;
        mx = std::max(mx, w0 + l + 1);
        x ^= w0 + l + 1;
      }
      for (uint32_t l = 0; l < W && w0 + l < kThreads; ++l) {
        VCHECK_EQ(out[(w0 + l) * 3 + 0], sum);
        VCHECK_EQ(out[(w0 + l) * 3 + 1], mx);
        VCHECK_EQ(out[(w0 + l) * 3 + 2], x);
      }
    }
  }
}

VTEST_MAIN
