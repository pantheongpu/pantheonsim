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
//  - Address spaces: device globals live in the MemoryManager VA range;
//    per-thread .local frames live in a reserved window (kLocalVaBase) that
//    generic loads/stores route to the executing lane's private buffer —
//    which is exactly PTX .local semantics. cvta is identity everywhere.
//  - Atomics are read-modify-write in fixed lane order — trivially atomic
//    and deterministic in this sequential engine.
//  - Warps in a block run under a pluggable Scheduler; a warp yields only at
//    barriers or retirement. Blocks run sequentially in a fixed order.
//    Everything is deterministic by construction.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "vgpu/error.hpp"
#include "vgpu/exec/launch.hpp"

namespace vgpu::exec {
namespace {

using namespace vgpu::ptx;

constexpr uint32_t kWarpSize = 32;
// Per-thread .local window: distinct from both host pointers and device
// globals. Addresses here are lane-relative (each lane sees its own frame).
constexpr uint64_t kLocalVaBase = 0x6fff'0000'0000ull;
constexpr uint64_t kLocalVaSize = 1ull << 30;
// Per-block .shared window, likewise distinct from host and device-global VAs.
constexpr uint64_t kSharedVaBase = 0x6ffe'0000'0000ull;
constexpr uint64_t kSharedVaSize = 1ull << 30;

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
  // Per-block shared memory. Zero-initialized at block start: real hardware
  // leaves it undefined, VirtualGPU makes it deterministic (documented).
  std::vector<uint8_t>* shared = nullptr;
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
  std::unordered_map<std::string, Lanes> slots;  // call-argument slots
  std::vector<std::vector<uint8_t>> local;       // per-lane .local frames (lazy)
  std::array<uint32_t, kWarpSize> tid_x{}, tid_y{}, tid_z{};
};

// IEEE 754 binary16 <-> double, implemented in software so the engine needs no
// host f16 support. Round-to-nearest-even, with subnormals and inf/NaN.
double f16_to_double(uint64_t bits) {
  uint16_t h = static_cast<uint16_t>(bits);
  uint32_t sign = (h >> 15) & 0x1;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  double mag;
  if (exp == 0) {
    mag = std::ldexp(static_cast<double>(mant), -24);  // subnormal
  } else if (exp == 31) {
    mag = mant ? std::numeric_limits<double>::quiet_NaN()
               : std::numeric_limits<double>::infinity();
  } else {
    mag = std::ldexp(1.0 + static_cast<double>(mant) / 1024.0, static_cast<int>(exp) - 15);
  }
  return sign ? -mag : mag;
}

uint64_t double_to_f16(double d) {
  if (std::isnan(d)) return 0x7E00;
  uint32_t sign = std::signbit(d) ? 0x8000u : 0u;
  double a = std::fabs(d);
  if (std::isinf(a) || a >= 65520.0) return sign | 0x7C00;  // overflow -> inf
  if (a < std::ldexp(1.0, -24)) return sign;                // underflow -> zero
  int exp;
  double frac = std::frexp(a, &exp);  // a = frac * 2^exp, frac in [0.5, 1)
  int e16 = exp - 1 + 15;             // unbiased exponent + bias
  if (e16 <= 0) {                     // subnormal
    uint32_t mant = static_cast<uint32_t>(std::nearbyint(std::ldexp(a, 24)));
    return sign | (mant & 0x3FF);
  }
  uint32_t mant = static_cast<uint32_t>(std::nearbyint((frac * 2.0 - 1.0) * 1024.0));
  if (mant == 1024) {  // rounding carried into the exponent
    mant = 0;
    ++e16;
  }
  if (e16 >= 31) return sign | 0x7C00;
  return sign | (static_cast<uint32_t>(e16) << 10) | (mant & 0x3FF);
}

float f32(uint64_t bits) { return std::bit_cast<float>(static_cast<uint32_t>(bits)); }
uint64_t f32bits(float f) { return std::bit_cast<uint32_t>(f); }
double f64(uint64_t bits) { return std::bit_cast<double>(bits); }
uint64_t f64bits(double d) { return std::bit_cast<uint64_t>(d); }

uint64_t mask_to_bits(uint64_t v, uint32_t bits) {
  return bits >= 64 ? v : (v & ((1ull << bits) - 1));
}

class Interpreter {
 public:
  Interpreter(const EntryFn& fn, const LaunchConfig& cfg, const ParamBuffer& params, MemoryManager& mem,
              const DeviceProfile& profile, const SymbolTable* symbols, LaunchStats& stats)
      : fn_(fn), cfg_(cfg), params_(params), mem_(mem), profile_(profile), symbols_(symbols),
        stats_(stats) {}

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
  // Re-throws a lower-level error with kernel/instruction context attached.
  [[noreturn]] void rethrow_with_context(const Error& e, const Instr& ins, int lane) {
    throw Error::make(e.code(), e.message(), "\n  in kernel '", fn_.name, "', PTX line ", ins.line,
                      lane >= 0 ? "\n  lane " + std::to_string(lane) : "",
                      "\n  instruction: ", ins.text.empty() ? "?" : ins.text,
                      "\n  GPU profile: ", profile_.id);
  }

  [[noreturn]] void ctx_fail(const Instr& ins, int lane, Err code, const std::string& msg) {
    try {
      throw Error::make(code, msg);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, lane);
    }
  }

  void run_block(BlockCtx& ctx, Scheduler& sched) {
    // Fresh, zeroed shared memory per block (static declarations + the
    // launch's dynamic bytes).
    std::vector<uint8_t> shared(fn_.static_shared_size + cfg_.shared_bytes, 0);
    ctx.shared = &shared;
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

  // ---- symbols / registers / operands ----

  uint64_t resolve_symbol(const Instr& ins, const std::string& name) {
    // .local/.shared variables name an offset within their address space, not
    // a generic address; cvta converts when the kernel needs a generic pointer.
    if (auto it = fn_.locals.find(name); it != fn_.locals.end()) return it->second.offset;
    if (auto it = fn_.shared.find(name); it != fn_.shared.end()) return it->second.offset;
    if (symbols_) {
      if (auto it = symbols_->find(name); it != symbols_->end()) return it->second;
    }
    ctx_fail(ins, -1, Err::NotFound,
             "unknown symbol '" + name + "' (not a .local depot or module .global variable)");
  }

  // Returns a reference to the operand's lane vector. Register operands alias
  // the warp's register file directly; everything else is materialized into
  // `scratch`. Returning a reference keeps a 256-byte copy off the hot path.
  const Lanes& read_operand(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& op,
                            Lanes& scratch) {
    if (const auto* r = std::get_if<RegOperand>(&op)) {
      auto it = w.regs.find(r->name);
      if (it == w.regs.end())
        ctx_fail(ins, -1, Err::UninitializedRegister,
                 "register " + r->name + " read before any write");
      return it->second;
    }
    if (const auto* imm = std::get_if<ImmInt>(&op)) {
      scratch.fill(static_cast<uint64_t>(imm->value));
      return scratch;
    }
    if (const auto* immf = std::get_if<ImmFloatBits>(&op)) {
      scratch.fill(immf->bits);
      return scratch;
    }
    if (const auto* sym = std::get_if<SymbolOperand>(&op)) {
      scratch.fill(resolve_symbol(ins, sym->name));
      return scratch;
    }
    const auto& sr = std::get<SregOperand>(op);
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) scratch[lane] = sreg_value(sr.reg, w, ctx, lane);
    return scratch;
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
      if (m & (1u << lane)) dst[lane] = mask_to_bits(vals[lane], bits);
  }

  Mask read_pred(Warp& w, const Instr& ins, const std::string& name) {
    auto it = w.preds.find(name);
    if (it == w.preds.end())
      ctx_fail(ins, -1, Err::UninitializedRegister, "predicate " + name + " read before any write");
    return it->second;
  }

  // ---- routed memory access (device global VA range vs .local window) ----

  // Window base for an address space. Global/generic already alias the flat
  // device VA range, so their base is zero.
  static uint64_t space_base(Space sp) {
    switch (sp) {
      case Space::Shared: return kSharedVaBase;
      case Space::Local: return kLocalVaBase;
      default: return 0;
    }
  }

  bool is_local(uint64_t addr) const {
    return addr >= kLocalVaBase && addr < kLocalVaBase + kLocalVaSize;
  }

  bool is_shared(uint64_t addr) const {
    return addr >= kSharedVaBase && addr < kSharedVaBase + kSharedVaSize;
  }

  void check_shared(const BlockCtx& ctx, const Instr& ins, int lane, uint64_t addr, uint32_t size) {
    uint64_t off = addr - kSharedVaBase;
    size_t have = ctx.shared ? ctx.shared->size() : 0;
    if (off + size > have)
      ctx_fail(ins, lane, Err::OutOfBounds,
               "shared memory access at offset " + std::to_string(off) + " (+" +
                   std::to_string(size) + " bytes) exceeds the " + std::to_string(have) +
                   "-byte shared allocation for this block");
  }

  std::vector<uint8_t>& lane_local(Warp& w, uint32_t lane) {
    if (w.local.empty()) w.local.resize(kWarpSize);
    auto& buf = w.local[lane];
    if (buf.size() < fn_.local_frame_size) buf.resize(fn_.local_frame_size, 0);
    return buf;
  }

  void check_local(const Instr& ins, int lane, uint64_t addr, uint32_t size) {
    uint64_t off = addr - kLocalVaBase;
    if (off + size > fn_.local_frame_size)
      ctx_fail(ins, lane, Err::OutOfBounds,
               "local memory access at frame offset " + std::to_string(off) + " (+" +
                   std::to_string(size) + " bytes) exceeds the " +
                   std::to_string(fn_.local_frame_size) + "-byte .local frame");
  }

  uint64_t load_routed(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane, uint64_t addr,
                       uint32_t size) {
    if (is_shared(addr)) {
      check_shared(ctx, ins, static_cast<int>(lane), addr, size);
      uint64_t v = 0;
      std::memcpy(&v, ctx.shared->data() + (addr - kSharedVaBase), size);
      return v;
    }
    if (is_local(addr)) {
      check_local(ins, static_cast<int>(lane), addr, size);
      uint64_t v = 0;
      std::memcpy(&v, lane_local(w, lane).data() + (addr - kLocalVaBase), size);
      return v;
    }
    try {
      return mem_.load_scalar(addr, size);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, static_cast<int>(lane));
    }
  }

  void store_routed(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane, uint64_t addr,
                    uint32_t size, uint64_t value) {
    if (is_shared(addr)) {
      check_shared(ctx, ins, static_cast<int>(lane), addr, size);
      std::memcpy(ctx.shared->data() + (addr - kSharedVaBase), &value, size);
      return;
    }
    if (is_local(addr)) {
      check_local(ins, static_cast<int>(lane), addr, size);
      std::memcpy(lane_local(w, lane).data() + (addr - kLocalVaBase), &value, size);
      return;
    }
    try {
      mem_.store_scalar(addr, size, value);
    } catch (const Error& e) {
      rethrow_with_context(e, ins, static_cast<int>(lane));
    }
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
      if (ins.has_pred) ctx_fail(ins, -1, Err::UnsupportedPtx, "predicated bar.sync is not supported");
      if (!w.divergence.empty())
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "bar.sync inside divergent control flow is not supported yet "
                 "(the warp still has parked execution paths)");
      ++w.pc;
      w.state = Warp::State::AtBarrier;
      return;
    }

    if (m != 0) {
      try {
        dispatch(w, ctx, ins, m);
      } catch (const Error& e) {
        if (std::string(e.what()).find("in kernel") == std::string::npos)
          rethrow_with_context(e, ins, -1);
        throw;
      }
    }
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

  void dispatch(Warp& w, const BlockCtx& ctx, const Instr& ins, Mask m) {
    if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      write_reg(w, op->dst, m, v, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMovPack>(&ins.op)) {
      uint32_t n = static_cast<uint32_t>(op->srcs.size());
      uint32_t piece = op->ty.bits / n;
      std::vector<Lanes> vals;
      vals.reserve(n);
      for (const auto& src : op->srcs) {
        Lanes tmp;
        vals.push_back(read_operand(w, ctx, ins, src, tmp));
      }
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t out = 0;
          for (uint32_t i = 0; i < n; ++i)
            out |= mask_to_bits(vals[i][lane], piece) << (piece * i);
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMovUnpack>(&ins.op)) {
      uint32_t n = static_cast<uint32_t>(op->dsts.size());
      uint32_t piece = op->ty.bits / n;
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      for (uint32_t i = 0; i < n; ++i) {
        Lanes r{};
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = mask_to_bits(v[lane] >> (piece * i), piece);
        write_reg(w, op->dsts[i], m, r, piece);
      }
      return;
    }
    if (const auto* op = std::get_if<OpCvta>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      uint64_t base = space_base(op->space);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane))
          r[lane] = op->to_space ? v[lane] - base : v[lane] + base;
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCvt>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = convert(op, v[lane]);
      write_reg(w, op->dst, m, r, op->dst_ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpNot>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = ~v[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpNeg>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          if (op->ty.kind == Type::Kind::F)
            r[lane] = op->ty.bits == 32 ? f32bits(-f32(v[lane])) : f64bits(-f64(v[lane]));
          else
            r[lane] = static_cast<uint64_t>(-static_cast<int64_t>(v[lane]));
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpPrmt>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint8_t bytes[8];
          uint32_t lo = static_cast<uint32_t>(a[lane]), hi = static_cast<uint32_t>(b[lane]);
          for (int i = 0; i < 4; ++i) bytes[i] = (lo >> (8 * i)) & 0xFF;
          for (int i = 0; i < 4; ++i) bytes[4 + i] = (hi >> (8 * i)) & 0xFF;
          uint32_t sel = static_cast<uint32_t>(c[lane]);
          uint32_t out = 0;
          for (int i = 0; i < 4; ++i) {
            uint32_t nib = (sel >> (4 * i)) & 0xF;
            uint8_t byte = bytes[nib & 0x7];
            if (nib & 0x8) byte = (byte & 0x80) ? 0xFF : 0x00;  // sign-replicate mode
            out |= static_cast<uint32_t>(byte) << (8 * i);
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpAbs>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          if (op->ty.is_float())
            r[lane] = op->ty.bits == 32 ? f32bits(std::fabs(f32(v[lane])))
                                        : f64bits(std::fabs(f64(v[lane])));
          else {
            int64_t x = op->ty.bits == 64 ? static_cast<int64_t>(v[lane])
                                          : int64_t{static_cast<int32_t>(v[lane])};
            r[lane] = static_cast<uint64_t>(x < 0 ? -x : x);
          }
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMath>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          double x = op->ty.bits == 32 ? static_cast<double>(f32(v[lane])) : f64(v[lane]);
          double y = 0;
          switch (op->op) {
            case MathOp::Ex2: y = std::exp2(x); break;
            case MathOp::Lg2: y = std::log2(x); break;
            case MathOp::Sin: y = std::sin(x); break;
            case MathOp::Cos: y = std::cos(x); break;
            case MathOp::Sqrt: y = std::sqrt(x); break;
            case MathOp::Rsqrt: y = 1.0 / std::sqrt(x); break;
            case MathOp::Rcp: y = 1.0 / x; break;
            case MathOp::Tanh: y = std::tanh(x); break;
          }
          r[lane] = op->ty.bits == 32 ? f32bits(static_cast<float>(y)) : f64bits(y);
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBfe>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = bfe(op->ty, a[lane], b[lane], c[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBfi>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes _s_d;
      const Lanes& d = read_operand(w, ctx, ins, op->d, _s_d);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = bfi(op->ty, a[lane], b[lane], c[lane], d[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBrev>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t x = mask_to_bits(v[lane], op->ty.bits), out = 0;
          for (uint32_t i = 0; i < op->ty.bits; ++i)
            if (x & (1ull << i)) out |= 1ull << (op->ty.bits - 1 - i);
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpPopcClz>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t x = mask_to_bits(v[lane], op->ty.bits);
          if (op->popc) {
            r[lane] = static_cast<uint64_t>(__builtin_popcountll(x));
          } else {
            uint32_t n = 0;
            for (int i = static_cast<int>(op->ty.bits) - 1; i >= 0 && !(x & (1ull << i)); --i) ++n;
            r[lane] = n;
          }
        }
      write_reg(w, op->dst, m, r, 32);  // popc/clz always produce a .u32
      return;
    }
    if (const auto* op = std::get_if<OpShfl>(&ins.op)) {
      exec_shfl(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpVote>(&ins.op)) {
      Mask p = read_pred(w, ins, op->src);
      if (op->negate_src) p = ~p;
      Mask voters = p & m;
      if (op->ballot) {
        Lanes r{};
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = voters;
        write_reg(w, op->dst, m, r, 32);
      } else {
        bool all = (voters == m), any = (voters != 0);
        bool val = op->mode == VoteMode::All   ? all
                   : op->mode == VoteMode::Any ? any
                                               : (voters == m || voters == 0);  // uni
        Mask& dp = w.preds[op->dst];
        dp = val ? (dp | m) : (dp & ~m);
      }
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
    if (const auto* op = std::get_if<OpAtom>(&ins.op)) {
      exec_atom(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpIntBin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = int_bin(op->op, op->ty, a[lane], b[lane], ins);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMadLo>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = a[lane] * b[lane] + c[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMadWide>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t prod;
          if (op->is_signed)
            prod = static_cast<uint64_t>(int64_t{static_cast<int32_t>(a[lane])} *
                                         int64_t{static_cast<int32_t>(b[lane])});
          else
            prod = uint64_t{static_cast<uint32_t>(a[lane])} * uint64_t{static_cast<uint32_t>(b[lane])};
          r[lane] = prod + c[lane];
        }
      write_reg(w, op->dst, m, r, 64);
      return;
    }
    if (const auto* op = std::get_if<OpMulWide>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
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
    if (const auto* op = std::get_if<OpShf>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t hi = static_cast<uint32_t>(b[lane]);
          uint64_t lo = static_cast<uint32_t>(a[lane]);
          uint32_t n = op->wrap ? (static_cast<uint32_t>(c[lane]) & 31u)
                                : std::min<uint32_t>(static_cast<uint32_t>(c[lane]), 32u);
          uint64_t funnel = (hi << 32) | lo;
          r[lane] = op->left ? static_cast<uint32_t>((funnel << n) >> 32)
                             : static_cast<uint32_t>(funnel >> n);
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpFloatBin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = float_bin(op->op, op->ty, a[lane], b[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpFma>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
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
    if (const auto* op = std::get_if<OpF16x2Bin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t out = 0;
          for (int h = 0; h < 2; ++h) {
            double x = f16_to_double((a[lane] >> (16 * h)) & 0xFFFF);
            double y = f16_to_double((b[lane] >> (16 * h)) & 0xFFFF);
            double v = op->op == FloatBinOp::Add   ? x + y
                       : op->op == FloatBinOp::Sub ? x - y
                                                   : x * y;
            out |= double_to_f16(v) << (16 * h);
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpF16x2Fma>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint64_t out = 0;
          for (int h = 0; h < 2; ++h) {
            double x = f16_to_double((a[lane] >> (16 * h)) & 0xFFFF);
            double y = f16_to_double((b[lane] >> (16 * h)) & 0xFFFF);
            double z = f16_to_double((c[lane] >> (16 * h)) & 0xFFFF);
            out |= double_to_f16(std::fma(x, y, z)) << (16 * h);
          }
          r[lane] = out;
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpF16x2Neg>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = v[lane] ^ 0x80008000ull;  // flip both sign bits
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpWmmaMma>(&ins.op)) {
      exec_wmma_mma(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpWmmaStore>(&ins.op)) {
      exec_wmma_store(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSetp>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Mask& p = w.preds[op->dst];
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          bool t = compare(op->cmp, op->ty, a[lane], b[lane]);
          p = t ? (p | (1u << lane)) : (p & ~(1u << lane));
        }
      return;
    }
    if (const auto* op = std::get_if<OpSelp>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Mask p = read_pred(w, ins, op->pred);
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = (p & (1u << lane)) ? a[lane] : b[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpPredBin>(&ins.op)) {
      Mask a = read_pred(w, ins, op->a);
      Mask b = read_pred(w, ins, op->b);
      Mask r = op->op == PredBinOp::And ? (a & b) : op->op == PredBinOp::Or ? (a | b) : (a ^ b);
      Mask& p = w.preds[op->dst];
      p = (p & ~m) | (r & m);
      return;
    }
    if (const auto* op = std::get_if<OpNotPred>(&ins.op)) {
      Mask s = read_pred(w, ins, op->src);
      Mask& p = w.preds[op->dst];
      p = (p & ~m) | (~s & m);
      return;
    }
    if (const auto* op = std::get_if<OpDeclSlot>(&ins.op)) {
      w.slots[op->name].fill(0);
      return;
    }
    if (const auto* op = std::get_if<OpStSlot>(&ins.op)) {
      if (op->offset != 0)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "st.param at a non-zero slot offset");
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes& slot = w.slots[op->slot];
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) slot[lane] = mask_to_bits(v[lane], op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpLdSlot>(&ins.op)) {
      auto it = w.slots.find(op->slot);
      if (it == w.slots.end())
        ctx_fail(ins, -1, Err::UninitializedRegister, "call slot '" + op->slot + "' read before write");
      if (op->offset != 0)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "ld.param at a non-zero slot offset");
      write_reg(w, op->dst, m, it->second, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCall>(&ins.op)) {
      exec_vprintf(w, ctx, ins, *op, m);
      return;
    }
    throw Error::make(Err::Internal, "interpreter has no handler for a parsed instruction");
  }

  static uint64_t bfe(Type ty, uint64_t a, uint64_t bpos, uint64_t clen) {
    uint32_t pos = static_cast<uint32_t>(bpos) & 0xFF;
    uint32_t len = static_cast<uint32_t>(clen) & 0xFF;
    uint32_t bits = ty.bits;
    if (pos >= bits || len == 0) {
      // Signed extract of an empty//out-of-range field replicates the sign bit.
      if (!ty.is_signed() || len == 0) return 0;
      uint64_t sign = (a >> (bits - 1)) & 1;
      return sign ? mask_to_bits(~0ull, bits) : 0;
    }
    if (len > bits - pos) len = bits - pos;
    uint64_t field = (a >> pos) & ((len >= 64) ? ~0ull : ((1ull << len) - 1));
    if (ty.is_signed() && (field & (1ull << (len - 1)))) field |= ~((1ull << len) - 1);
    return mask_to_bits(field, bits);
  }

  static uint64_t bfi(Type ty, uint64_t a, uint64_t b, uint64_t cpos, uint64_t dlen) {
    uint32_t pos = static_cast<uint32_t>(cpos) & 0xFF;
    uint32_t len = static_cast<uint32_t>(dlen) & 0xFF;
    uint32_t bits = ty.bits;
    if (pos >= bits || len == 0) return mask_to_bits(b, bits);
    if (len > bits - pos) len = bits - pos;
    uint64_t field_mask = ((len >= 64) ? ~0ull : ((1ull << len) - 1)) << pos;
    return mask_to_bits((b & ~field_mask) | ((a << pos) & field_mask), bits);
  }

  // Warp shuffle: a lane reads another lane's register. This is a plain index
  // into the warp's lane vector — the operation the warp-vectorized register
  // file makes trivial (see ARCHITECTURE.md D1).
  void exec_shfl(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpShfl& op, Mask m) {
    Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op.a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op.b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op.c, _s_c);
    Lanes r{};
    Mask pred_out = 0;
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      uint32_t bval = static_cast<uint32_t>(b[lane]) & 0x1F;
      uint32_t cval = static_cast<uint32_t>(c[lane]) & 0x1F;
      uint32_t segmask = (static_cast<uint32_t>(c[lane]) >> 8) & 0x1F;
      uint32_t min_lane = lane & segmask;
      uint32_t max_lane = (lane & segmask) | (cval & ~segmask);
      uint32_t j = lane;
      bool pred;
      switch (op.mode) {
        case ShflMode::Up:
          j = lane - bval;
          pred = static_cast<int32_t>(lane - bval) >= static_cast<int32_t>(min_lane);
          break;
        case ShflMode::Down:
          j = lane + bval;
          pred = j <= max_lane;
          break;
        case ShflMode::Bfly:
          j = lane ^ bval;
          pred = j <= max_lane;
          break;
        case ShflMode::Idx:
          j = min_lane | (bval & ~segmask);
          pred = j <= max_lane;
          break;
      }
      if (!pred) j = lane;  // out-of-range source: the lane reads its own value
      r[lane] = a[j & 0x1F];
      if (pred) pred_out |= (1u << lane);
    }
    write_reg(w, op.dst, m, r, 32);
    if (!op.pred_dst.empty()) {
      Mask& p = w.preds[op.pred_dst];
      p = (p & ~m) | (pred_out & m);
    }
  }

  // ---- tensor cores (wmma) ----
  //
  // VirtualGPU's fragment layout. PTX leaves the element->register mapping
  // unspecified, so we define a self-consistent one:
  //   A/B (f16): 8 x .b32 per lane = 16 halves. Lane L, register r, half h
  //              holds element (row = L % 16, col = r * 2 + h). Lanes 16-31
  //              mirror lanes 0-15, matching the hardware's 2x redundancy.
  //   C/D (f32): 8 x .f32 per lane = 256 slots exactly. Lane L register r
  //              holds linear element L * 8 + r (row-major 16x16).
  // A ".col" layout transposes the matrix on the way in/out.
  static constexpr uint32_t kMmaDim = 16;

  void exec_wmma_mma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaMma& op, Mask m) {
    // Gather A and B (16x16 f16 each) and C (16x16 f32) from the warp.
    double A[kMmaDim][kMmaDim] = {}, B[kMmaDim][kMmaDim] = {}, C[kMmaDim][kMmaDim] = {};
    for (int reg = 0; reg < 8; ++reg) {
      Lanes _s_av;
      const Lanes& av = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s_av);
      Lanes _s_bv;
      const Lanes& bv = read_operand(w, ctx, ins, Operand{RegOperand{op.b[reg]}}, _s_bv);
      for (uint32_t lane = 0; lane < kMmaDim; ++lane) {
        for (int h = 0; h < 2; ++h) {
          uint32_t col = static_cast<uint32_t>(reg * 2 + h);
          double a = f16_to_double((av[lane] >> (16 * h)) & 0xFFFF);
          double b = f16_to_double((bv[lane] >> (16 * h)) & 0xFFFF);
          // .row: element (lane, col); .col: transposed.
          if (op.alayout == MatLayout::Row) A[lane][col] = a; else A[col][lane] = a;
          if (op.blayout == MatLayout::Row) B[lane][col] = b; else B[col][lane] = b;
        }
      }
    }
    for (int reg = 0; reg < 8; ++reg) {
      Lanes _s_cv;
      const Lanes& cv = read_operand(w, ctx, ins, Operand{RegOperand{op.c[reg]}}, _s_cv);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        C[linear / kMmaDim][linear % kMmaDim] = f32(cv[lane]);
      }
    }
    // D = A x B + C, accumulated in f32 (matching the .f32 accumulate type).
    float D[kMmaDim][kMmaDim];
    for (uint32_t i = 0; i < kMmaDim; ++i)
      for (uint32_t j = 0; j < kMmaDim; ++j) {
        float acc = static_cast<float>(C[i][j]);
        for (uint32_t k = 0; k < kMmaDim; ++k)
          acc += static_cast<float>(A[i][k]) * static_cast<float>(B[k][j]);
        D[i][j] = acc;
      }
    // Scatter D back into the destination fragment.
    for (int reg = 0; reg < 8; ++reg) {
      Lanes r{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        r[lane] = f32bits(D[linear / kMmaDim][linear % kMmaDim]);
      }
      write_reg(w, op.d[reg], m, r, 32);
    }
  }

  void exec_wmma_store(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaStore& op,
                       Mask m) {
    const Lanes& base = addr_base(w, ctx, ins, op.addr);
    Lanes _s_stride;
      const Lanes& stride = read_operand(w, ctx, ins, op.stride, _s_stride);
    uint64_t sbase = space_base(op.space);
    // The whole warp cooperates; lane 0's address and stride describe the tile.
    uint32_t lead = 0;
    while (lead < kWarpSize && !(m & (1u << lead))) ++lead;
    if (lead == kWarpSize) return;
    uint64_t addr0 = sbase + base[lead] + static_cast<uint64_t>(op.addr.offset);
    uint64_t ld = stride[lead];
    for (int reg = 0; reg < 8; ++reg) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op.src[reg], _s_v);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        if (!(m & (1u << lane))) continue;
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        uint32_t i = linear / kMmaDim, j = linear % kMmaDim;
        uint64_t elem = op.layout == MatLayout::Row ? (uint64_t{i} * ld + j)
                                                    : (uint64_t{j} * ld + i);
        store_routed(w, ctx, ins, lane, addr0 + elem * 4, 4, v[lane] & 0xFFFFFFFFull);
      }
    }
  }

  // Applies an integer rounding mode to a real value.
  static double round_int(double x, Round rnd) {
    switch (rnd) {
      case Round::Rzi: return std::trunc(x);
      case Round::Rm: return std::floor(x);
      case Round::Rp: return std::ceil(x);
      case Round::Rni: [[fallthrough]];
      default: return std::nearbyint(x);  // round-to-nearest-even
    }
  }

  uint64_t convert(const OpCvt* op, uint64_t in) {
    const Type& s = op->src_ty;
    const Type& d = op->dst_ty;
    // Read the source as a real number (float src) or integer (int src).
    if (s.is_float()) {
      double x = s.bits == 16 ? f16_to_double(in)
                 : s.bits == 32 ? static_cast<double>(f32(in))
                                : f64(in);
      if (d.is_float()) {
        if (d.bits == 16) return double_to_f16(x);
        return d.bits == 32 ? f32bits(static_cast<float>(x)) : f64bits(x);
      }
      // float -> int: round then clamp to the destination range.
      double rounded = round_int(x, op->round == Round::None ? Round::Rzi : op->round);
      if (d.is_signed()) {
        long double lo = -std::pow(2.0L, d.bits - 1), hi = std::pow(2.0L, d.bits - 1) - 1;
        if (std::isnan(rounded)) return 0;
        if (rounded < lo) rounded = static_cast<double>(lo);
        if (rounded > hi) rounded = static_cast<double>(hi);
        return mask_to_bits(static_cast<uint64_t>(static_cast<int64_t>(rounded)), d.bits);
      }
      long double hi = std::pow(2.0L, d.bits) - 1;
      if (std::isnan(rounded) || rounded < 0) return 0;
      if (rounded > hi) rounded = static_cast<double>(hi);
      return mask_to_bits(static_cast<uint64_t>(rounded), d.bits);
    }
    // Integer source: sign/zero-extend to 64 bits first.
    uint64_t sv = mask_to_bits(in, s.bits);
    if (s.is_signed() && s.bits < 64) {
      uint64_t sign_bit = 1ull << (s.bits - 1);
      if (sv & sign_bit) sv |= ~((sign_bit << 1) - 1);
    }
    if (d.is_float()) {
      double x = s.is_signed() ? static_cast<double>(static_cast<int64_t>(sv))
                               : static_cast<double>(sv);
      if (d.bits == 16) return double_to_f16(x);
      return d.bits == 32 ? f32bits(static_cast<float>(x)) : f64bits(x);
    }
    return mask_to_bits(sv, d.bits);  // int -> int: truncate/extend
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
        if (u(b) == 0)
          // Real GPUs produce an undefined value here; VirtualGPU traps instead,
          // because a div-by-zero in a kernel is almost always a bug (documented
          // deliberate divergence — see ARCHITECTURE.md).
          ctx_fail(ins, -1, Err::InvalidValue, "integer division by zero");
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
    if (ty.bits == 16) {
      double x = f16_to_double(a), y = f16_to_double(b), r = 0;
      switch (op) {
        case FloatBinOp::Add: r = x + y; break;
        case FloatBinOp::Sub: r = x - y; break;
        case FloatBinOp::Mul: r = x * y; break;
        case FloatBinOp::Div: r = x / y; break;
        case FloatBinOp::Min: r = std::fmin(x, y); break;
        case FloatBinOp::Max: r = std::fmax(x, y); break;
      }
      return double_to_f16(r);
    }
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

  // Float comparison including the NaN-aware forms: the ordered ops are false
  // if either operand is NaN, the "u" (unordered) ops are true.
  static bool compare_float(CmpOp cmp, double x, double y) {
    bool unordered = std::isnan(x) || std::isnan(y);
    switch (cmp) {
      case CmpOp::Eq: return !unordered && x == y;
      case CmpOp::Ne: return !unordered && x != y;
      case CmpOp::Lt: return !unordered && x < y;
      case CmpOp::Le: return !unordered && x <= y;
      case CmpOp::Gt: return !unordered && x > y;
      case CmpOp::Ge: return !unordered && x >= y;
      case CmpOp::Equ: return unordered || x == y;
      case CmpOp::Neu: return unordered || x != y;
      case CmpOp::Ltu: return unordered || x < y;
      case CmpOp::Leu: return unordered || x <= y;
      case CmpOp::Gtu: return unordered || x > y;
      case CmpOp::Geu: return unordered || x >= y;
      case CmpOp::Num: return !unordered;
      case CmpOp::Nan: return unordered;
    }
    return false;
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
        default: return false;  // unordered forms are float-only; handled above
      }
    };
    if (ty.is_float()) {
      if (ty.bits == 16) return compare_float(cmp, f16_to_double(a), f16_to_double(b));
      return ty.bits == 32 ? compare_float(cmp, f32(a), f32(b)) : compare_float(cmp, f64(a), f64(b));
    }
    if (ty.is_signed())
      return ty.bits == 64 ? do_cmp(static_cast<int64_t>(a), static_cast<int64_t>(b))
                           : do_cmp(static_cast<int32_t>(a), static_cast<int32_t>(b));
    return ty.bits == 64 ? do_cmp(a, b) : do_cmp(static_cast<uint32_t>(a), static_cast<uint32_t>(b));
  }

  // ---- memory ops ----

  // Address bases are always register names (the parser guarantees it for
  // Base::Reg), so look the register up directly and alias the register file
  // rather than routing through read_operand's scratch path.
  const Lanes& addr_base(Warp& w, const BlockCtx& ctx, const Instr& ins, const Addr& a) {
    (void)ctx;
    auto it = w.regs.find(a.base);
    if (it == w.regs.end())
      ctx_fail(ins, -1, Err::UninitializedRegister,
               "address register " + a.base + " read before any write");
    return it->second;
  }

  void exec_ld(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpLd& op, Mask m) {
    uint32_t size = op.ty.bytes();
    size_t n = op.dsts.size();
    if (op.space == Space::Param) {
      auto it = params_.layout.find(op.addr.base);
      if (it == params_.layout.end())
        ctx_fail(ins, -1, Err::PtxParse, "ld.param from non-parameter '" + op.addr.base + "'");
      auto [off, psize] = it->second;
      for (size_t e = 0; e < n; ++e) {
        int64_t at = int64_t{off} + op.addr.offset + static_cast<int64_t>(e * size);
        if (at < 0 || static_cast<uint64_t>(at) + size > uint64_t{off} + psize)
          ctx_fail(ins, -1, Err::OutOfBounds,
                   "ld.param reads past parameter '" + op.addr.base + "'");
        uint64_t v = 0;
        std::memcpy(&v, params_.bytes.data() + at, size);
        Lanes r{};
        r.fill(v);
        write_reg(w, op.dsts[e], m, r, op.ty.bits);
      }
      return;
    }
    const Lanes& base = addr_base(w, ctx, ins, op.addr);
    uint64_t sbase = space_base(op.space);
    uint32_t va = size * static_cast<uint32_t>(n);  // vector accesses need vector alignment
    std::vector<Lanes> results(n);
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      if (m & (1u << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        if (addr % va != 0)
          ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                   "vector load requires " + std::to_string(va) + "-byte alignment");
        for (size_t e = 0; e < n; ++e)
          results[e][lane] = load_routed(w, ctx, ins, lane, addr + e * size, size);
      }
    for (size_t e = 0; e < n; ++e) write_reg(w, op.dsts[e], m, results[e], op.ty.bits);
  }

  void exec_st(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSt& op, Mask m) {
    uint32_t size = op.ty.bytes();
    size_t n = op.srcs.size();
    std::vector<Lanes> vals;
    vals.reserve(n);
    for (const auto& src : op.srcs) {
      Lanes tmp;
      vals.push_back(read_operand(w, ctx, ins, src, tmp));
    }
    const Lanes& base = addr_base(w, ctx, ins, op.addr);
    uint64_t sbase = space_base(op.space);
    uint32_t va = size * static_cast<uint32_t>(n);
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      if (m & (1u << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        if (addr % va != 0)
          ctx_fail(ins, static_cast<int>(lane), Err::MisalignedAccess,
                   "vector store requires " + std::to_string(va) + "-byte alignment");
        for (size_t e = 0; e < n; ++e)
          store_routed(w, ctx, ins, lane, addr + e * size, size, mask_to_bits(vals[e][lane], op.ty.bits));
      }
  }

  void exec_atom(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpAtom& op, Mask m) {
    uint32_t size = op.ty.bytes();
    if (size != 4 && size != 8)
      ctx_fail(ins, -1, Err::UnsupportedPtx, "atomics are only implemented for 32/64-bit types");
    const Lanes& base = addr_base(w, ctx, ins, op.addr);
    uint64_t sbase = space_base(op.space);
    Lanes _s_bv;
      const Lanes& bv = read_operand(w, ctx, ins, op.b, _s_bv);
    Lanes cv{};
    Lanes _s_cv;
    if (op.op == AtomOp::Cas) cv = read_operand(w, ctx, ins, op.c, _s_cv);
    Lanes r{};
    // Fixed lane order: trivially atomic and deterministic in this engine.
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      if (m & (1u << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        uint64_t old = load_routed(w, ctx, ins, lane, addr, size);
        uint64_t b = mask_to_bits(bv[lane], op.ty.bits);
        uint64_t nv = old;
        switch (op.op) {
          case AtomOp::Add: nv = old + b; break;
          case AtomOp::And: nv = old & b; break;
          case AtomOp::Or: nv = old | b; break;
          case AtomOp::Xor: nv = old ^ b; break;
          case AtomOp::Exch: nv = b; break;
          case AtomOp::Min:
            nv = op.ty.is_signed()
                     ? (size == 8 ? static_cast<uint64_t>(std::min(static_cast<int64_t>(old),
                                                                   static_cast<int64_t>(b)))
                                  : static_cast<uint64_t>(static_cast<uint32_t>(
                                        std::min(static_cast<int32_t>(old), static_cast<int32_t>(b)))))
                     : std::min(old, b);
            break;
          case AtomOp::Max:
            nv = op.ty.is_signed()
                     ? (size == 8 ? static_cast<uint64_t>(std::max(static_cast<int64_t>(old),
                                                                   static_cast<int64_t>(b)))
                                  : static_cast<uint64_t>(static_cast<uint32_t>(
                                        std::max(static_cast<int32_t>(old), static_cast<int32_t>(b)))))
                     : std::max(old, b);
            break;
          case AtomOp::Cas: nv = (old == mask_to_bits(cv[lane], op.ty.bits)) ? b : old; break;
        }
        store_routed(w, ctx, ins, lane, addr, size, mask_to_bits(nv, op.ty.bits));
        r[lane] = old;
      }
    write_reg(w, op.dst, m, r, op.ty.bits);
  }

  // ---- device printf (the vprintf builtin) ----

  std::string read_cstring(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane,
                           uint64_t addr) {
    std::string out;
    for (size_t i = 0; i < 8192; ++i) {
      uint64_t b = load_routed(w, ctx, ins, lane, addr + i, 1);
      if (b == 0) return out;
      out += static_cast<char>(b);
    }
    ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
             "printf format string exceeds 8 KiB (missing NUL terminator?)");
  }

  void exec_vprintf(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCall& op, Mask m) {
    if (op.param_slots.size() != 2)
      ctx_fail(ins, -1, Err::UnsupportedPtx, "vprintf expects exactly 2 arguments (format, valist)");
    auto fmt_it = w.slots.find(op.param_slots[0]);
    auto va_it = w.slots.find(op.param_slots[1]);
    if (fmt_it == w.slots.end() || va_it == w.slots.end())
      ctx_fail(ins, -1, Err::UninitializedRegister, "vprintf argument slot read before write");

    Lanes counts{};
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      std::string fmt = read_cstring(w, ctx, ins, lane, fmt_it->second[lane]);
      uint64_t valist = va_it->second[lane];
      uint64_t cursor = 0;
      std::string out;
      char buf[256];

      auto fetch = [&](uint32_t size) -> uint64_t {
        cursor = (cursor + size - 1) / size * size;  // natural alignment in the valist
        if (valist == 0)
          ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                   "printf format consumes arguments but no argument buffer was passed");
        uint64_t v = load_routed(w, ctx, ins, lane, valist + cursor, size);
        cursor += size;
        return v;
      };

      for (size_t i = 0; i < fmt.size(); ++i) {
        if (fmt[i] != '%') {
          out += fmt[i];
          continue;
        }
        size_t start = i++;
        if (i < fmt.size() && fmt[i] == '%') {
          out += '%';
          continue;
        }
        while (i < fmt.size() && std::string("-+ #0123456789.").find(fmt[i]) != std::string::npos) ++i;
        int longs = 0;
        while (i < fmt.size() && (fmt[i] == 'l' || fmt[i] == 'h' || fmt[i] == 'z')) {
          if (fmt[i] == 'l') ++longs;
          if (fmt[i] == 'z') longs = 2;
          ++i;
        }
        if (i >= fmt.size())
          ctx_fail(ins, static_cast<int>(lane), Err::InvalidValue,
                   "printf format ends inside a % specifier");
        char conv = fmt[i];
        // Spec with normalized length: 64-bit integers always use "ll".
        std::string flags = fmt.substr(start + 1, (i - (longs ? longs : 0)) - start - 1);
        // Strip any length chars that slipped into flags capture.
        while (!flags.empty() && (flags.back() == 'l' || flags.back() == 'h' || flags.back() == 'z'))
          flags.pop_back();
        bool is64 = longs >= 1 || conv == 'p';  // %l.. and pointers are 8 bytes on this ABI
        switch (conv) {
          case 'd': case 'i': case 'u': case 'o': case 'x': case 'X': case 'c': {
            uint64_t v = fetch(is64 ? 8 : 4);
            std::string spec = "%" + flags + (is64 ? "ll" : "") + conv;
            if (is64)
              std::snprintf(buf, sizeof buf, spec.c_str(), static_cast<unsigned long long>(v));
            else if (conv == 'd' || conv == 'i' || conv == 'c')
              std::snprintf(buf, sizeof buf, spec.c_str(), static_cast<int>(v));
            else
              std::snprintf(buf, sizeof buf, spec.c_str(), static_cast<unsigned>(v));
            out += buf;
            break;
          }
          case 'f': case 'F': case 'e': case 'E': case 'g': case 'G': {
            uint64_t v = fetch(8);
            std::string spec = "%" + flags + conv;
            std::snprintf(buf, sizeof buf, spec.c_str(), f64(v));
            out += buf;
            break;
          }
          case 'p': {
            uint64_t v = fetch(8);
            std::snprintf(buf, sizeof buf, "0x%llx", static_cast<unsigned long long>(v));
            out += buf;
            break;
          }
          case 's': {
            uint64_t v = fetch(8);
            out += read_cstring(w, ctx, ins, lane, v);
            break;
          }
          default:
            ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                     std::string("printf conversion '%") + conv + "' is not implemented");
        }
      }
      std::fwrite(out.data(), 1, out.size(), stdout);
      counts[lane] = out.size();
    }
    std::fflush(stdout);
    if (!op.retval_slot.empty()) {
      Lanes& slot = w.slots[op.retval_slot];
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) slot[lane] = counts[lane];
    }
  }

  const EntryFn& fn_;
  const LaunchConfig& cfg_;
  const ParamBuffer& params_;
  MemoryManager& mem_;
  const DeviceProfile& profile_;
  const SymbolTable* symbols_;
  LaunchStats& stats_;
};

ParamBuffer build_params(const EntryFn& fn, const std::vector<std::vector<uint8_t>>& args) {
  if (args.size() != fn.params.size())
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' expects ", fn.params.size(),
                      " parameters, launch supplied ", args.size());
  ParamBuffer pb;
  for (size_t i = 0; i < args.size(); ++i) {
    uint32_t size = fn.params[i].size;
    if (args[i].size() != size)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' parameter '", fn.params[i].name,
                        "' is ", size, " bytes, launch supplied ", args[i].size(), " bytes");
    uint32_t align = fn.params[i].align ? fn.params[i].align : (size < 8 ? size : 8);
    uint32_t off = static_cast<uint32_t>((pb.bytes.size() + align - 1) / align * align);
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
  uint64_t total_shared = uint64_t{fn.static_shared_size} + cfg.shared_bytes;
  uint32_t shared_limit = std::max(p.limits.shared_mem_per_block, p.limits.shared_mem_per_block_optin);
  if (total_shared > shared_limit)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' needs ", total_shared,
                      " bytes of shared memory (", fn.static_shared_size, " static + ",
                      cfg.shared_bytes, " dynamic); profile ", p.id, " allows at most ", shared_limit);
}

}  // namespace

LaunchStats launch(const EntryFn& fn, const LaunchConfig& cfg,
                   const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                   const DeviceProfile& profile, const SymbolTable* symbols) {
  validate(fn, cfg, profile);
  ParamBuffer pb = build_params(fn, args);
  LaunchStats stats;
  Interpreter interp(fn, cfg, pb, mem, profile, symbols, stats);
  interp.run_grid();
  return stats;
}

}  // namespace vgpu::exec
