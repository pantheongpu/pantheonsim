// OpWgCollective (patch E8): work-group reduce, inclusive/exclusive scan and broadcast on the barrier rendezvous.
//
// Each case builds a kernel with the AST builder and compares every thread with a host fold in increasing linear
// thread id (x fastest), over blocks of different shapes (1-D with a partial last warp, 2-D, 3-D, 1000 threads =
// 125 warps at 8 lanes), several blocks per launch, warp widths 8, 16 and 32, and the three schedulers. The answer must not
// depend on the warp width or the schedule, which is the point of the instruction: a float sum is the sequential sum
// in linear thread order. Also: several collectives in a row (a chain, and in a loop with a bar.sync between),
// a collective that only some threads of a warp take (refused by name) and mismatched collectives (refused).
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "vgpu/error.hpp"
#include "ast_run.hpp"
#include "vtest.hpp"

using namespace astrun;
using vgpu::ptx::ReduxOp;
using vgpu::ptx::WgKind;

namespace {

struct Shape {
  std::array<uint32_t, 3> block;
  uint32_t grid;
  uint32_t n() const { return block[0] * block[1] * block[2]; }
};
const Shape kShapes[] = {{{96, 1, 1}, 2}, {{100, 1, 1}, 1}, {{8, 6, 1}, 3}, {{4, 3, 5}, 2}, {{1000, 1, 1}, 1}, {{1, 1, 1}, 3}};

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

OpWgCollective make(WgKind kind, ReduxOp op, const TySpec& t, const Reg& dst, const Reg& a, Operand b = IM(0)) {
  OpWgCollective c;
  c.kind = kind;
  c.op = op;
  c.ty = t.type();
  c.dst = dst;
  c.a = RO(a);
  c.b = std::move(b);
  return c;
}

std::vector<uint8_t> pack(const std::vector<uint64_t>& v, uint32_t bits) {
  std::vector<uint8_t> o(v.size() * (bits / 8));
  for (size_t i = 0; i < v.size(); ++i) std::memcpy(&o[i * (bits / 8)], &v[i], bits / 8);
  return o;
}
std::vector<uint64_t> unpack(const std::vector<uint8_t>& b, uint32_t bits) {
  std::vector<uint64_t> v(b.size() / (bits / 8), 0);
  for (size_t i = 0; i < v.size(); ++i) std::memcpy(&v[i], &b[i * (bits / 8)], bits / 8);
  return v;
}

std::vector<uint64_t> run_one(const Shape& sh, uint32_t W, vgpu::exec::SchedulerKind sched, uint64_t seed, WgKind kind, ReduxOp op,
                              const TySpec& t, uint32_t src, const std::vector<uint64_t>& in) {
  vgpu::ptx::KernelBuilder kb("wg");
  Prologue p = prologue(kb, 2);
  const Reg x = load_elem(kb, p.ptr[0], p.gid, t.type());
  const Reg r = kb.temp(t.type());
  kb.emit(make(kind, op, t, r, x, IM(src)), "wgcoll");
  store_elem(kb, p.ptr[1], p.gid, t.type(), r);
  kb.emit(OpRet{}, "ret");
  vgpu::exec::LaunchConfig cfg;
  cfg.block = sh.block;
  cfg.grid = {sh.grid, 1, 1};
  cfg.warp_size = W;
  cfg.scheduler = sched;
  cfg.scheduler_seed = seed;
  auto res = run(kb.finish(), cfg, {pack(in, t.bits), std::vector<uint8_t>(in.size() * (t.bits / 8), 0xCD)});
  return unpack(res[1], t.bits);
}

// The host fold for one block's slice.
std::vector<uint64_t> expected(const Shape& sh, WgKind kind, ReduxOp op, const TySpec& t, uint32_t src, const std::vector<uint64_t>& in) {
  std::vector<uint64_t> out(in.size());
  const uint32_t n = sh.n();
  for (uint32_t b = 0; b < sh.grid; ++b) {
    uint64_t acc = 0;
    for (uint32_t i = 0; i < n; ++i) {
      const uint64_t v = in[b * n + i];
      switch (kind) {
        case WgKind::Broadcast: out[b * n + i] = in[b * n + (src < n ? src : 0)]; break;
        case WgKind::Reduce: acc = i == 0 ? v : ref_combine(op, t, acc, v); break;
        case WgKind::InclusiveScan:
          acc = i == 0 ? v : ref_combine(op, t, acc, v);
          out[b * n + i] = acc;
          break;
        case WgKind::ExclusiveScan:
          out[b * n + i] = i == 0 ? ref_identity(op, t) : acc;
          acc = i == 0 ? v : ref_combine(op, t, acc, v);
          break;
      }
    }
    if (kind == WgKind::Reduce)
      for (uint32_t i = 0; i < n; ++i) out[b * n + i] = acc;
  }
  return out;
}

void compare(const char* what, const TySpec& t, const std::vector<uint64_t>& got, const std::vector<uint64_t>& want) {
  VCHECK_EQ(got.size(), want.size());
  for (size_t i = 0; i < got.size(); ++i)
    if (!same_value(t, got[i], want[i]))
      throw vtest::Failure(std::string(what) + " element " + std::to_string(i) + ": got " + std::to_string(got[i]) + " want " +
                           std::to_string(want[i]));
}

}  // namespace

VTEST(reduce_and_scans_follow_linear_thread_order) {
  const std::vector<TySpec> types = {{false, true, 32}, {false, false, 32}, {false, true, 64}, {true, true, 16}, {true, true, 32}, {true, true, 64}};
  uint64_t cases = 0;
  Rng rng(5);
  for (const TySpec& t : types) {
    const ReduxOp ops_int[] = {ReduxOp::Add, ReduxOp::Mul, ReduxOp::Min, ReduxOp::Max, ReduxOp::And, ReduxOp::Or, ReduxOp::Xor};
    const ReduxOp ops_fl[] = {ReduxOp::Add, ReduxOp::Mul, ReduxOp::Min, ReduxOp::Max};
    const std::vector<ReduxOp> ops = t.fl ? std::vector<ReduxOp>(std::begin(ops_fl), std::end(ops_fl))
                                          : std::vector<ReduxOp>(std::begin(ops_int), std::end(ops_int));
    for (ReduxOp op : ops) {
      const bool minmax = op == ReduxOp::Min || op == ReduxOp::Max;
      for (const Shape& sh : kShapes) {
        std::vector<uint64_t> in(static_cast<size_t>(sh.n()) * sh.grid);
        for (auto& v : in) v = random_value(rng, t, !(t.fl && t.bits == 16 && minmax));
        for (WgKind kind : {WgKind::Reduce, WgKind::InclusiveScan, WgKind::ExclusiveScan}) {
          const auto want = expected(sh, kind, op, t, 0, in);
          for (uint32_t W : {8u, 16u, 32u}) {
            // one shape at a time gets the random schedulers, the rest the default
            const bool rand_sched = (cases % 7) == 0;
            const auto got = run_one(sh, W, rand_sched ? vgpu::exec::SchedulerKind::Random : vgpu::exec::SchedulerKind::Deterministic,
                                     cases + 1, kind, op, t, 0, in);
            compare((std::string("wg ") + op_name(op) + "." + t.name() + " kind " + std::to_string(static_cast<int>(kind)) + " W=" +
                     std::to_string(W) + " block " + std::to_string(sh.n())).c_str(),
                    t, got, want);
            ++cases;
          }
        }
      }
    }
  }
  VCHECK(cases > 1000);
}

VTEST(broadcast_takes_the_value_of_the_named_thread) {
  Rng rng(6);
  for (const TySpec& t : {TySpec{false, false, 32}, TySpec{false, true, 64}, TySpec{true, true, 32}, TySpec{true, true, 16}, TySpec{false, false, 8}}) {
    for (const Shape& sh : kShapes) {
      std::vector<uint64_t> in(static_cast<size_t>(sh.n()) * sh.grid);
      for (auto& v : in) v = random_value(rng, t);
      for (uint32_t src : {0u, 1u, sh.n() / 2, sh.n() - 1}) {
        for (uint32_t W : {8u, 16u, 32u}) {
          const auto got = run_one(sh, W, vgpu::exec::SchedulerKind::Deterministic, 0, WgKind::Broadcast, ReduxOp::Add, t, src, in);
          compare("broadcast", t, got, expected(sh, WgKind::Broadcast, ReduxOp::Add, t, src, in));
        }
      }
    }
  }
}

// The answer must not depend on the warp width or the schedule: a float sum in particular.
VTEST(a_float_sum_is_the_same_at_every_warp_width_and_schedule) {
  const TySpec t{true, true, 32};
  const Shape sh{{1000, 1, 1}, 1};
  std::vector<uint64_t> in(1000);
  Rng rng(9);
  for (auto& v : in) {
    const float f = (rng.next() % 11 == 0 ? 1.0e8f : 1.0f) * (static_cast<float>(rng.next() % 2000) / 7.0f - 140.0f);
    v = std::bit_cast<uint32_t>(f);
  }
  const auto first = run_one(sh, 32, vgpu::exec::SchedulerKind::Deterministic, 0, WgKind::Reduce, ReduxOp::Add, t, 0, in);
  compare("sequential", t, first, expected(sh, WgKind::Reduce, ReduxOp::Add, t, 0, in));
  for (uint32_t W : {8u, 16u, 32u})
    for (auto k : {vgpu::exec::SchedulerKind::Deterministic, vgpu::exec::SchedulerKind::Random, vgpu::exec::SchedulerKind::Adversarial})
      for (uint64_t seed : {1ull, 2ull}) {
        const auto again = run_one(sh, W, k, seed, WgKind::Reduce, ReduxOp::Add, t, 0, in);
        VCHECK(again == first);
      }
}

// Collectives in a row: the answers of round k feed round k+1 (so a warp that is released early and races into the next
// one must not corrupt a slower warp's answer), inside a loop with a bar.sync between the rounds.
VTEST(collectives_in_rounds_do_not_mix) {
  const TySpec t{false, false, 32};
  for (uint32_t W : {8u, 16u, 32u}) {
    for (const Shape& sh : {Shape{{100, 1, 1}, 2}, Shape{{4, 3, 5}, 1}}) {
      vgpu::ptx::KernelBuilder kb("rounds");
      Prologue p = prologue(kb, 2);
      Reg v = load_elem(kb, p.ptr[0], p.gid, t.type());
      const Reg acc = kb.temp(tu(32));
      kb.emit(mov(tu(32), acc, RO(v)), "mov");
      const Reg i = kb.temp(tu(32));
      kb.emit(mov(tu(32), i, IM(0)), "mov");
      const vgpu::ptx::Label top = kb.make_label(), end = kb.make_label();
      kb.bind(top);
      const Reg done = kb.pred();
      kb.emit(setp(CmpOp::Ge, tu(32), done, RO(i), IM(3)), "setp");
      kb.bra(end, done);
      // round: scan (inclusive), then reduce of the scan, then a broadcast of thread 3's value; fold them in
      const Reg s = kb.temp(tu(32)), rd = kb.temp(tu(32)), bc = kb.temp(tu(32));
      kb.emit(make(WgKind::InclusiveScan, ReduxOp::Add, t, s, acc), "wgcoll");
      kb.emit(make(WgKind::Reduce, ReduxOp::Xor, t, rd, s), "wgcoll");
      kb.emit(barsync(0), "bar");
      kb.emit(make(WgKind::Broadcast, ReduxOp::Add, t, bc, s, IM(3)), "wgcoll");
      const Reg t1 = kb.temp(tu(32));
      kb.emit(ibin(IntBinOp::Add, tu(32), t1, RO(s), RO(rd)), "add");
      kb.emit(ibin(IntBinOp::Add, tu(32), acc, RO(t1), RO(bc)), "add");
      kb.emit(ibin(IntBinOp::Add, tu(32), i, RO(i), IM(1)), "add");
      kb.bra(top);
      kb.bind(end);
      store_elem(kb, p.ptr[1], p.gid, t.type(), acc);
      kb.emit(OpRet{}, "ret");

      std::vector<uint64_t> in(static_cast<size_t>(sh.n()) * sh.grid);
      Rng rng(40 + W);
      for (auto& x : in) x = rng.next() & 0xFFFF;
      // host
      std::vector<uint64_t> want(in.size());
      for (uint32_t b = 0; b < sh.grid; ++b) {
        std::vector<uint32_t> a(sh.n());
        for (uint32_t k = 0; k < sh.n(); ++k) a[k] = static_cast<uint32_t>(in[b * sh.n() + k]);
        for (int round = 0; round < 3; ++round) {
          std::vector<uint32_t> sc(sh.n());
          uint32_t run_sum = 0, x = 0;
          for (uint32_t k = 0; k < sh.n(); ++k) { run_sum += a[k]; sc[k] = run_sum; }
          for (uint32_t k = 0; k < sh.n(); ++k) x ^= sc[k];
          const uint32_t b3 = sc[3 < sh.n() ? 3 : 0];
          for (uint32_t k = 0; k < sh.n(); ++k) a[k] = sc[k] + x + b3;
        }
        for (uint32_t k = 0; k < sh.n(); ++k) want[b * sh.n() + k] = a[k];
      }
      vgpu::exec::LaunchConfig cfg;
      cfg.block = sh.block;
      cfg.grid = {sh.grid, 1, 1};
      cfg.warp_size = W;
      cfg.scheduler = vgpu::exec::SchedulerKind::Random;
      cfg.scheduler_seed = W;
      auto res = run(kb.finish(), cfg, {pack(in, 32), std::vector<uint8_t>(in.size() * 4, 0)});
      compare("rounds", t, unpack(res[1], 32), want);
    }
  }
}

VTEST(wg_collective_refusals) {
  const TySpec t{false, false, 32};
  // some lanes of a warp take it
  {
    vgpu::ptx::KernelBuilder kb("partial");
    Prologue p = prologue(kb, 2);
    const Reg x = load_elem(kb, p.ptr[0], p.gid, t.type());
    const Reg r = kb.temp(tu(32));
    const Reg odd = kb.temp(tu(32));
    kb.emit(ibin(IntBinOp::And, tu(32), odd, RO(p.lin), IM(1)), "and");
    const Reg q = kb.pred();
    kb.emit(setp(CmpOp::Ne, tu(32), q, RO(odd), IM(0)), "setp");
    kb.emit_if(q, false, make(WgKind::Reduce, ReduxOp::Add, t, r, x), "wgcoll");
    kb.emit(OpRet{}, "ret");
    vgpu::exec::LaunchConfig cfg;
    cfg.block = {64, 1, 1};
    cfg.warp_size = 16;
    auto err = VCAPTURE(vgpu::Error, run(kb.finish(), cfg, {std::vector<uint8_t>(64 * 4, 1), std::vector<uint8_t>(64 * 4, 0)}));
    VCHECK_CONTAINS(err.what(), "work-group collective");
  }
  // float and, bf16-like misuse
  {
    vgpu::ptx::KernelBuilder kb("badop");
    Prologue p = prologue(kb, 2);
    const TySpec f{true, true, 32};
    const Reg x = load_elem(kb, p.ptr[0], p.gid, f.type());
    const Reg r = kb.temp(f.type());
    kb.emit(make(WgKind::Reduce, ReduxOp::And, f, r, x), "wgcoll");
    kb.emit(OpRet{}, "ret");
    vgpu::exec::LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    auto err = VCAPTURE(vgpu::Error, run(kb.finish(), cfg, {std::vector<uint8_t>(32 * 4, 1), std::vector<uint8_t>(32 * 4, 0)}));
    VCHECK_CONTAINS(err.what(), "float reduction");
  }
  // two warps reach different collectives in one round
  {
    vgpu::ptx::KernelBuilder kb("mismatch");
    Prologue p = prologue(kb, 2);
    const Reg x = load_elem(kb, p.ptr[0], p.gid, t.type());
    const Reg r = kb.temp(tu(32));
    const Reg hi = kb.temp(tu(32));
    kb.emit(ibin(IntBinOp::Shr, tu(32), hi, RO(p.lin), IM(4)), "shr");
    const Reg q = kb.pred();
    kb.emit(setp(CmpOp::Ne, tu(32), q, RO(hi), IM(0)), "setp");
    const vgpu::ptx::Label other = kb.make_label(), end = kb.make_label();
    kb.bra(other, q);
    kb.emit(make(WgKind::Reduce, ReduxOp::Add, t, r, x), "wgcoll");
    kb.bra(end);
    kb.bind(other);
    kb.emit(make(WgKind::Reduce, ReduxOp::Max, t, r, x), "wgcoll");
    kb.bind(end);
    kb.emit(OpRet{}, "ret");
    vgpu::exec::LaunchConfig cfg;
    cfg.block = {32, 1, 1};
    cfg.warp_size = 16;
    auto err = VCAPTURE(vgpu::Error, run(kb.finish(), cfg, {std::vector<uint8_t>(32 * 4, 1), std::vector<uint8_t>(32 * 4, 0)}));
    VCHECK_CONTAINS(err.what(), "different work-group collectives");
  }
}

VTEST_MAIN
