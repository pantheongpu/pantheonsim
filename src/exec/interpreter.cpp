// The SIMT warp interpreter — VirtualGPU's CPU execution engine.
//
// Execution model (see ARCHITECTURE.md):
//  - A warp is 32 lanes advancing in lockstep over vector registers
//    (one 32-bit lane mask, one value per lane per register). We do NOT spawn
//    a CPU thread per GPU thread.
//  - Divergence: a branch that splits the active mask parks the not-taken
//    (pc, mask) on a per-warp divergence stack and continues with the taken
//    side; a path that retires pops the next parked path. Paths reconverge
//    implicitly at ret. Barriers inside divergent control flow are rejected
//    with a clear error rather than deadlocking (IPDOM reconvergence is a
//    planned upgrade — see TODO.md).
//  - Warps in a block run under a pluggable Scheduler; a warp yields only at
//    barriers or retirement. Blocks run sequentially in a fixed order.
//    Everything is deterministic by construction.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <unordered_map>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"

namespace vgpu::exec {
namespace {

using namespace vgpu::ptx;

constexpr uint32_t kWarpSize = 32;
using Mask = uint32_t;  // bit i == lane i active
using Lanes = std::array<uint64_t, kWarpSize>;

struct ParamBuffer {
  std::vector<uint8_t> bytes;
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> layout;  // name -> (offset, size)
};

struct BlockCtx {
  std::array<uint32_t, 3> ctaid{};
  std::array<uint32_t, 3> ntid{};
  std::array<uint32_t, 3> nctaid{};
};

struct Warp {
  enum class State { Ready, AtBarrier, Done };
  State state = State::Ready;
  size_t pc = 0;
  Mask active = 0;
  Mask exited = 0;
  std::vector<std::pair<size_t, Mask>> divergence;  // parked (pc, mask)
  std::unordered_map<std::string, Lanes> regs;
  std::unordered_map<std::string, Mask> preds;
  std::array<uint32_t, kWarpSize> tid_x{}, tid_y{}, tid_z{};
};

float f32(uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); }
uint64_t f32bits(float f) { return std::bit_cast<uint32_t>(f); }
double f64(uint64_t bits) { return std::bit_cast<double>(bits); }
uint64_t f64bits(double d) { return std::bit_cast<uint64_t>(d); }

class Interpreter {
 public:
  Interpreter(const EntryFn& fn, const LaunchConfig& cfg, const ParamBuffer& params, MemoryManager& mem,
              const DeviceProfile& profile, LaunchStats& stats)
      : fn_(fn), cfg_(cfg), params_(params), mem_(mem), profile_(profile), stats_(stats) {}

  void run_grid() {
    auto sched = make_scheduler(cfg_.scheduler);
    for (uint32_t bz = 0; bz < cfg_.grid[2]; ++bz)
      for (uint32_t by = 0; by < cfg_.grid[1]; ++by)
        for (uint32_t bx = 0; bx < cfg_.grid[0]; ++bx) {
          BlockCtx ctx;
          ctx.ctaid = {bx, by, bz};
          ctx.ntid = cfg_.block;
          ctx.nctaid = cfg_.grid;
          run_block(ctx, *sched);
          ++stats_.blocks;
        }
  }

 private:
  [[noreturn]] void fault(size_t line, const std::string& msg) {
    throw Error::make(Err::Internal, msg, "\n  in kernel '", fn_.name, "' at line ", line);
  }

  // Re-throws a lower-level error with kernel/instruction context attached.
  [[noreturn]] void rethrow_with_context(const Error& e, const Instr& ins, int lane) {
    throw Error::make(e.code(), e.message(), "\n  in kernel '", fn_.name, "', PTX line ", ins.line,
                      lane >= 0 ? "\n  lane " + std::to_string(lane) : "",
                      "\n  instruction: ", ins.text.empty() ? "?" : ins.text,
                      "\n  GPU profile: ", profile_.id);
  }

  void run_block(const BlockCtx& ctx, Scheduler& sched) {
    uint64_t total = uint64_t{ctx.ntid[0]} * ctx.ntid[1] * ctx.ntid[2];
    size_t nwarps = static_cast<size_t>((total + kWarpSize - 1) / kWarpSize);
    std::vector<Warp> warps(nwarps);
    for (size_t w = 0; w < nwarps; ++w) {
      Warp& warp = warps[w];
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        uint64_t lin = uint64_t{static_cast<uint32_t>(w)} * kWarpSize + lane;
        if (lin >= total) break;
        warp.active |= (1u << lane);
        warp.tid_x[lane] = static_cast<uint32_t>(lin % ctx.ntid[0]);
        warp.tid_y[lane] = static_cast<uint32_t>((lin / ctx.ntid[0]) % ctx.ntid[1]);
        warp.tid_z[lane] = static_cast<uint32_t>(lin / (uint64_t{ctx.ntid[0]} * ctx.ntid[1]));
      }
    }
    stats_.warps += nwarps;

    std::vector<size_t> runnable;
    while (true) {
      runnable.clear();
      for (size_t i = 0; i < warps.size(); ++i)
        if (warps[i].state == Warp::State::Ready) runnable.push_back(i);
      if (runnable.empty()) {
        // Barrier release: every warp is either retired or waiting, which is
        // exactly the functional bar.sync condition.
        bool any_waiting = false;
        for (auto& w : warps)
          if (w.state == Warp::State::AtBarrier) {
            w.state = Warp::State::Ready;
            any_waiting = true;
          }
        if (!any_waiting) return;  // all Done
        continue;
      }
      run_warp_until_yield(warps[sched.pick(runnable)], ctx);
    }
  }

  void run_warp_until_yield(Warp& w, const BlockCtx& ctx) {
    while (w.state == Warp::State::Ready) {
      if (w.pc >= fn_.body.size())
        throw Error::make(Err::PtxParse, "control fell off the end of kernel '", fn_.name,
                          "' (missing ret)");
      const Instr& ins = fn_.body[w.pc];
      if (++stats_.instructions > cfg_.max_steps)
        throw Error::make(Err::ExecLimit, "kernel '", fn_.name, "' exceeded the launch step budget (",
                          cfg_.max_steps, " instructions) — possible infinite loop");
      step(w, ctx, ins);
    }
  }

  // ---- register access ----

  Lanes read_operand(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& op) {
    Lanes out{};
    if (const auto* r = std::get_if<RegOperand>(&op)) {
      auto it = w.regs.find(r->name);
      if (it == w.regs.end()) {
        try {
          throw Error::make(Err::UninitializedRegister, "register ", r->name, " read before any write");
        } catch (const Error& e) {
          rethrow_with_context(e, ins, -1);
        }
      }
      return it->second;
    }
    if (const auto* imm = std::get_if<ImmInt>(&op)) {
      out.fill(static_cast<uint64_t>(imm->value));
      return out;
    }
    if (const auto* immf = std::get_if<ImmFloatBits>(&op)) {
      out.fill(immf->bits);
      return out;
    }
    const auto& s = std::get<SregOperand>(op);
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) out[lane] = sreg_value(s.reg, w, ctx, lane);
    return out;
  }

  uint32_t sreg_value(Sreg s, const Warp& w, const BlockCtx& ctx, uint32_t lane) {
    switch (s) {
      case Sreg::TidX: return w.tid_x[lane];
      case Sreg::TidY: return w.tid_y[lane];
      case Sreg::TidZ: return w.tid_z[lane];
      case Sreg::NtidX: return ctx.ntid[0];
      case Sreg::NtidY: return ctx.ntid[1];
      case Sreg::NtidZ: return ctx.ntid[2];
      case Sreg::CtaidX: return ctx.ctaid[0];
      case Sreg::CtaidY: return ctx.ctaid[1];
      case Sreg::CtaidZ: return ctx.ctaid[2];
      case Sreg::NctaidX: return ctx.nctaid[0];
      case Sreg::NctaidY: return ctx.nctaid[1];
      case Sreg::NctaidZ: return ctx.nctaid[2];
      case Sreg::LaneId: return lane;
    }
    return 0;
  }

  void write_reg(Warp& w, const std::string& name, Mask m, const Lanes& vals, uint32_t bits) {
    Lanes& dst = w.regs[name];  // zero-initialized on first touch
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      if (m & (1u << lane)) dst[lane] = bits == 64 ? vals[lane] : (vals[lane] & 0xFFFFFFFFull);
  }

  Mask read_pred(Warp& w, const Instr& ins, const std::string& name) {
    auto it = w.preds.find(name);
    if (it == w.preds.end()) {
      try {
        throw Error::make(Err::UninitializedRegister, "predicate ", name, " read before any write");
      } catch (const Error& e) {
        rethrow_with_context(e, ins, -1);
      }
    }
    return it->second;
  }

  // ---- the dispatcher ----

  void step(Warp& w, const BlockCtx& ctx, const Instr& ins) {
    Mask m = w.active;
    if (ins.has_pred) {
      Mask p = read_pred(w, ins, ins.pred);
      if (ins.pred_negated) p = ~p;
      m &= p;
    }

    if (const auto* op = std::get_if<OpBra>(&ins.op)) {
      exec_bra(w, *op, m);
      return;
    }
    if (std::holds_alternative<OpRet>(ins.op)) {
      exec_ret(w, m);
      return;
    }
    if (std::holds_alternative<OpBar>(ins.op)) {
      if (ins.has_pred)
        try {
          throw Error::make(Err::UnsupportedPtx, "predicated bar.sync is not supported");
        } catch (const Error& e) {
          rethrow_with_context(e, ins, -1);
        }
      if (!w.divergence.empty())
        try {
          throw Error::make(Err::UnsupportedPtx,
                            "bar.sync inside divergent control flow is not supported yet "
                            "(the warp still has parked execution paths)");
        } catch (const Error& e) {
          rethrow_with_context(e, ins, -1);
        }
      ++w.pc;
      w.state = Warp::State::AtBarrier;
      return;
    }

    // All remaining ops are straight-line: execute under mask m, advance pc.
    if (m != 0) exec_straightline(w, ctx, ins, m);
    ++w.pc;
  }

  void exec_bra(Warp& w, const OpBra& op, Mask m) {
    Mask taken = m;
    Mask fallthrough = w.active & ~taken;
    if (taken == 0) {
      ++w.pc;
      return;
    }
    if (fallthrough == 0) {
      w.pc = op.target;
      return;
    }
    w.divergence.emplace_back(w.pc + 1, fallthrough);
    w.active = taken;
    w.pc = op.target;
  }

  void exec_ret(Warp& w, Mask m) {
    w.exited |= m;
    w.active &= ~m;
    ++w.pc;  // predicated ret: surviving lanes continue at the next instruction
    while (w.active == 0 && !w.divergence.empty()) {
      auto [pc, mask] = w.divergence.back();
      w.divergence.pop_back();
      w.pc = pc;
      w.active = mask & ~w.exited;
    }
    if (w.active == 0) w.state = Warp::State::Done;
  }

  void exec_straightline(Warp& w, const BlockCtx& ctx, const Instr& ins, Mask m) {
    try {
      dispatch(w, ctx, ins, m);
    } catch (const Error& e) {
      // Memory faults etc. thrown mid-instruction get kernel context here.
      if (std::string(e.what()).find("in kernel") == std::string::npos)
        rethrow_with_context(e, ins, -1);
      throw;
    }
  }

  void dispatch(Warp& w, const BlockCtx& ctx, const Instr& ins, Mask m) {
    if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      if (op->ty.kind == Type::Kind::Pred)
        throw Error::make(Err::UnsupportedPtx, "mov.pred is not supported yet");
      Lanes v = read_operand(w, ctx, ins, op->src);
      write_reg(w, op->dst, m, v, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCvtaToGlobal>(&ins.op)) {
      // Flat virtual address space: generic<->global conversion is identity.
      Lanes v = read_operand(w, ctx, ins, op->src);
      write_reg(w, op->dst, m, v, 64);
      return;
    }
    if (const auto* op = std::get_if<OpLd>(&ins.op)) {
      exec_ld(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSt>(&ins.op)) {
      exec_st(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpIntBin>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = int_bin(op->op, op->ty, a[lane], b[lane], ins);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMadLo>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b),
            c = read_operand(w, ctx, ins, op->c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = a[lane] * b[lane] + c[lane];  // wrapping; .lo truncates on write
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMulWide>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          if (op->is_signed)
            r[lane] = static_cast<uint64_t>(int64_t{static_cast<int32_t>(a[lane])} *
                                            int64_t{static_cast<int32_t>(b[lane])});
          else
            r[lane] = uint64_t{static_cast<uint32_t>(a[lane])} * uint64_t{static_cast<uint32_t>(b[lane])};
        }
      write_reg(w, op->dst, m, r, 64);
      return;
    }
    if (const auto* op = std::get_if<OpFloatBin>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = float_bin(op->op, op->ty, a[lane], b[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpFma>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b),
            c = read_operand(w, ctx, ins, op->c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          if (op->ty.bits == 32)
            r[lane] = f32bits(std::fma(f32(a[lane]), f32(b[lane]), f32(c[lane])));
          else
            r[lane] = f64bits(std::fma(f64(a[lane]), f64(b[lane]), f64(c[lane])));
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpSetp>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b);
      Mask& p = w.preds[op->dst];
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          bool t = compare(op->cmp, op->ty, a[lane], b[lane]);
          p = t ? (p | (1u << lane)) : (p & ~(1u << lane));
        }
      return;
    }
    if (const auto* op = std::get_if<OpSelp>(&ins.op)) {
      Lanes a = read_operand(w, ctx, ins, op->a), b = read_operand(w, ctx, ins, op->b);
      Mask p = read_pred(w, ins, op->pred);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = (p & (1u << lane)) ? a[lane] : b[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    throw Error::make(Err::Internal, "interpreter has no handler for a parsed instruction");
  }

  uint64_t int_bin(IntBinOp op, Type ty, uint64_t a, uint64_t b, const Instr& ins) {
    bool sixty_four = ty.bits == 64;
    bool sig = ty.is_signed();
    auto s = [&](uint64_t v) -> int64_t {
      return sixty_four ? static_cast<int64_t>(v) : int64_t{static_cast<int32_t>(v)};
    };
    auto u = [&](uint64_t v) -> uint64_t { return sixty_four ? v : (v & 0xFFFFFFFFull); };
    switch (op) {
      case IntBinOp::Add: return a + b;
      case IntBinOp::Sub: return a - b;
      case IntBinOp::Mul: return a * b;
      case IntBinOp::Min: return sig ? static_cast<uint64_t>(std::min(s(a), s(b))) : std::min(u(a), u(b));
      case IntBinOp::Max: return sig ? static_cast<uint64_t>(std::max(s(a), s(b))) : std::max(u(a), u(b));
      case IntBinOp::Div:
      case IntBinOp::Rem: {
        if (u(b) == 0) {
          // Real GPUs produce an undefined value here; VirtualGPU traps instead,
          // because a div-by-zero in a kernel is almost always a bug (documented
          // deliberate divergence — see ARCHITECTURE.md).
          try {
            throw Error::make(Err::InvalidValue, "integer division by zero");
          } catch (const Error& e) {
            rethrow_with_context(e, ins, -1);
          }
        }
        if (op == IntBinOp::Div)
          return sig ? static_cast<uint64_t>(s(a) / s(b)) : u(a) / u(b);
        return sig ? static_cast<uint64_t>(s(a) % s(b)) : u(a) % u(b);
      }
      case IntBinOp::And: return a & b;
      case IntBinOp::Or: return a | b;
      case IntBinOp::Xor: return a ^ b;
      case IntBinOp::Shl: {
        uint64_t sh = u(b);
        return sh >= ty.bits ? 0 : a << sh;  // PTX clamps shift amounts
      }
      case IntBinOp::Shr: {
        uint64_t sh = u(b);
        if (sig) {
          int64_t v = s(a);
          if (sh >= ty.bits) return static_cast<uint64_t>(v < 0 ? -1 : 0);
          return static_cast<uint64_t>(v >> sh);
        }
        return sh >= ty.bits ? 0 : u(a) >> sh;
      }
    }
    return 0;
  }

  uint64_t float_bin(FloatBinOp op, Type ty, uint64_t a, uint64_t b) {
    if (ty.bits == 32) {
      float x = f32(a), y = f32(b), r = 0;
      switch (op) {
        case FloatBinOp::Add: r = x + y; break;
        case FloatBinOp::Sub: r = x - y; break;
        case FloatBinOp::Mul: r = x * y; break;
        case FloatBinOp::Div: r = x / y; break;
        case FloatBinOp::Min: r = std::fmin(x, y); break;
        case FloatBinOp::Max: r = std::fmax(x, y); break;
      }
      return f32bits(r);
    }
    double x = f64(a), y = f64(b), r = 0;
    switch (op) {
      case FloatBinOp::Add: r = x + y; break;
      case FloatBinOp::Sub: r = x - y; break;
      case FloatBinOp::Mul: r = x * y; break;
      case FloatBinOp::Div: r = x / y; break;
      case FloatBinOp::Min: r = std::fmin(x, y); break;
      case FloatBinOp::Max: r = std::fmax(x, y); break;
    }
    return f64bits(r);
  }

  bool compare(CmpOp cmp, Type ty, uint64_t a, uint64_t b) {
    auto do_cmp = [&](auto x, auto y) {
      switch (cmp) {
        case CmpOp::Eq: return x == y;
        case CmpOp::Ne: return x != y;
        case CmpOp::Lt: return x < y;
        case CmpOp::Le: return x <= y;
        case CmpOp::Gt: return x > y;
        case CmpOp::Ge: return x >= y;
      }
      return false;
    };
    if (ty.is_float())
      return ty.bits == 32 ? do_cmp(f32(a), f32(b)) : do_cmp(f64(a), f64(b));
    if (ty.is_signed())
      return ty.bits == 64 ? do_cmp(static_cast<int64_t>(a), static_cast<int64_t>(b))
                           : do_cmp(static_cast<int32_t>(a), static_cast<int32_t>(b));
    return ty.bits == 64 ? do_cmp(a, b) : do_cmp(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
  }

  // ---- memory ops ----

  void exec_ld(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpLd& op, Mask m) {
    uint32_t size = op.ty.bytes();
    Lanes r{};
    if (op.space == Space::Param) {
      auto it = params_.layout.find(op.addr.base);
      if (it == params_.layout.end() || !op.addr.base_is_param)
        throw Error::make(Err::PtxParse, "ld.param from non-parameter '", op.addr.base, "'");
      auto [off, psize] = it->second;
      int64_t at = int64_t{off} + op.addr.offset;
      if (at < 0 || static_cast<uint64_t>(at) + size > off + psize)
        throw Error::make(Err::OutOfBounds, "ld.param reads past parameter '", op.addr.base, "'");
      uint64_t v = 0;
      std::memcpy(&v, params_.bytes.data() + at, size);
      r.fill(v);
    } else {
      Lanes base = addr_base(w, ctx, ins, op.addr);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t addr = base[lane] + static_cast<uint64_t>(op.addr.offset);
          try {
            r[lane] = mem_.load_scalar(addr, size);
          } catch (const Error& e) {
            rethrow_with_context(e, ins, static_cast<int>(lane));
          }
        }
    }
    write_reg(w, op.dst, m, r, op.ty.bits == 64 ? 64 : 32);
  }

  void exec_st(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSt& op, Mask m) {
    uint32_t size = op.ty.bytes();
    Lanes vals = read_operand(w, ctx, ins, op.src);
    Lanes base = addr_base(w, ctx, ins, op.addr);
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      if (m & (1u << lane)) {
        uint64_t addr = base[lane] + static_cast<uint64_t>(op.addr.offset);
        uint64_t v = size == 8 ? vals[lane] : (vals[lane] & ((1ull << (size * 8)) - 1));
        try {
          mem_.store_scalar(addr, size, v);
        } catch (const Error& e) {
          rethrow_with_context(e, ins, static_cast<int>(lane));
        }
      }
  }

  Lanes addr_base(Warp& w, const BlockCtx& ctx, const Instr& ins, const Addr& a) {
    if (a.base_is_param)
      throw Error::make(Err::UnsupportedPtx, "parameter-based addressing outside ld.param");
    return read_operand(w, ctx, ins, Operand{RegOperand{a.base}});
  }

  const EntryFn& fn_;
  const LaunchConfig& cfg_;
  const ParamBuffer& params_;
  MemoryManager& mem_;
  const DeviceProfile& profile_;
  LaunchStats& stats_;
};

ParamBuffer build_params(const EntryFn& fn, const std::vector<std::vector<uint8_t>>& args) {
  if (args.size() != fn.params.size())
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' expects ", fn.params.size(),
                      " parameters, launch supplied ", args.size());
  ParamBuffer pb;
  for (size_t i = 0; i < args.size(); ++i) {
    uint32_t size = fn.params[i].ty.bytes();
    if (args[i].size() != size)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' parameter '", fn.params[i].name,
                        "' is ", size, " bytes, launch supplied ", args[i].size(), " bytes");
    uint32_t off = static_cast<uint32_t>((pb.bytes.size() + size - 1) / size * size);  // natural alignment
    pb.bytes.resize(off + size);
    std::memcpy(pb.bytes.data() + off, args[i].data(), size);
    pb.layout[fn.params[i].name] = {off, size};
  }
  return pb;
}

void validate(const EntryFn& fn, const LaunchConfig& cfg, const DeviceProfile& p) {
  if (p.warp_size != kWarpSize)
    throw Error::make(Err::Unsupported, "profile ", p.id, " has warp size ", p.warp_size,
                      "; only 32 is implemented");
  uint64_t threads = 1;
  for (int i = 0; i < 3; ++i) {
    if (cfg.block[i] == 0 || cfg.grid[i] == 0)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': grid/block dimensions must be >= 1");
    if (cfg.block[i] > p.limits.max_block_dim[i])
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': block dim ", i, " is ", cfg.block[i],
                        ", profile ", p.id, " allows at most ", p.limits.max_block_dim[i]);
    if (cfg.grid[i] > p.limits.max_grid_dim[i])
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': grid dim ", i, " is ", cfg.grid[i],
                        ", profile ", p.id, " allows at most ", p.limits.max_grid_dim[i]);
    threads *= cfg.block[i];
  }
  if (threads > p.limits.max_threads_per_block)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': ", threads,
                      " threads per block exceeds profile limit ", p.limits.max_threads_per_block, " (",
                      p.id, ")");
  if (cfg.shared_bytes != 0)
    throw Error::make(Err::Unsupported, "dynamic shared memory is not implemented yet (requested ",
                      cfg.shared_bytes, " bytes)");
}

}  // namespace

LaunchStats launch(const EntryFn& fn, const LaunchConfig& cfg,
                   const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                   const DeviceProfile& profile) {
  validate(fn, cfg, profile);
  ParamBuffer pb = build_params(fn, args);
  LaunchStats stats;
  Interpreter interp(fn, cfg, pb, mem, profile, stats);
  interp.run_grid();
  return stats;
}

}  // namespace vgpu::exec
