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
#include <chrono>
#include <thread>
#include <deque>
#include <memory>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cerrno>
#include <limits>
#include <map>
#include <mutex>
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
// Kernel parameters get an address window of their own so a kernel can take a
// parameter's address and load through it -- CUB's segmented sort does exactly
// that. Nothing else may be addressed here, so a stray pointer into this range
// is still diagnosable.
constexpr uint64_t kParamVaBase = 0x6ffd'0000'0000ull;
constexpr uint64_t kParamVaSize = 1ull << 20;
constexpr uint64_t kSharedVaBase = 0x6ffe'0000'0000ull;
constexpr uint64_t kSharedVaSize = 1ull << 30;

using Mask = uint32_t;  // bit i == lane i active
// Operands are passed around as 64-bit lanes so every handler sees one type.
using Lanes = std::array<uint64_t, kWarpSize>;
// Storage, however, is split by declared width: a 32-bit register costs
// 128 bytes per warp instead of 256. Most registers in real kernels are
// 32-bit, so this halves register-file traffic for the common case.
using Lanes32 = std::array<uint32_t, kWarpSize>;

struct ParamBuffer {
  std::vector<uint8_t> bytes;
  std::unordered_map<std::string, std::pair<uint32_t, uint32_t>> layout;  // name -> (offset, size)
};

// State for bar.red, which needs every warp in the block to arrive before it
// can produce a value. Held behind a pointer for the same reason shared memory
// is: the context is passed by const reference.
struct BarrierReduction {
  uint64_t acc = 0;
  uint32_t arrived = 0;   // warps that have contributed and not yet collected
  bool complete = false;  // set when the barrier released; cleared when drained
};

// Shadow state for shared memory, one entry per 4-byte word, used only when
// race detection is on.
//
// The rule a CUDA block promises is simple: two warps may touch the same shared
// word without a barrier between them only if both are reading. Anything else
// is a race, and the answer depends on an order the program never specified.
// Hardware usually hides that -- warps advance together and the window is
// small -- which is exactly why it is worth checking here.
//
// An epoch is the count of barriers the block has completed, so "no barrier
// between them" is "same epoch". Warps per block cap at 32 (1024 threads), so
// the set of warps that read a word in an epoch fits in a uint32_t exactly.
struct WordShadow {
  uint32_t readers = 0;             // bitmask of warps that read it this epoch
  uint32_t read_epoch = 0xFFFFFFFF;
  uint32_t write_epoch = 0xFFFFFFFF;
  uint16_t writer = 0xFFFF;         // warp that wrote it, 0xFFFF for none
};

struct SharedShadow {
  std::vector<WordShadow> words;
  uint32_t epoch = 0;
};

struct BlockCtx {
  std::array<uint32_t, 3> ctaid{};
  std::array<uint32_t, 3> ntid{};
  std::array<uint32_t, 3> nctaid{};
  // Per-block shared memory. Zero-initialized at block start: real hardware
  // leaves it undefined, VirtualGPU makes it deterministic (documented).
  std::vector<uint8_t>* shared = nullptr;
  BarrierReduction* bar_red = nullptr;
  SharedShadow* shadow = nullptr;   // non-null only when race detection is on
};

// One diverged execution path: a set of lanes sharing a program counter.
struct Path {
  size_t pc = 0;
  Mask mask = 0;
};

// One cp.async copy that has been issued but not yet awaited.
//
// The source is read when the instruction issues and the destination is
// written when the thread waits. Both are points the hardware is allowed to
// pick, and splitting them this way is what makes the instruction mean
// anything: a kernel that reads the destination before its wait sees the old
// contents, exactly as it would on a device, instead of data that a
// synchronous copy would have put there early and hidden the bug.
struct PendingCopy {
  uint64_t dst = 0;                  // shared-window address
  uint32_t bytes = 0;                // 4, 8 or 16
  std::array<uint8_t, 16> data{};    // read at issue; zero past src-size
};

// Per-lane copy state, allocated only for warps that actually use cp.async --
// which is almost none of them, and this is 32 lanes of container otherwise.
struct AsyncCopies {
  // Issued but not yet committed to a group.
  std::array<std::vector<PendingCopy>, kWarpSize> open;
  // Committed groups, oldest first. wait_group N drains until N remain.
  std::array<std::deque<std::vector<PendingCopy>>, kWarpSize> groups;
};

struct Warp {
  enum class State { Ready, AtBarrier, Done };
  State state = State::Ready;
  // Set between contributing to a bar.red and collecting its result. The
  // instruction re-executes when the barrier releases, and this is how it knows
  // to collect rather than contribute a second time.
  bool bar_red_waiting = false;
  // Live paths. Reconvergence is by *lowest program counter*: the path with
  // the smallest pc always runs next, and paths that arrive at the same pc are
  // merged. For the structured control flow compilers emit, that reconverges
  // an if/else at its join point and lets a loop's lanes iterate until they
  // reach the exit -- which is what makes bar.sync after a divergent region
  // work, since every lane has merged back into one path by then.
  std::vector<Path> paths;
  Mask exited = 0;
  // Register files indexed by the parser's dense ids: narrow registers live
  // in regs32, 64-bit ones in regs64. `written*` tracks first assignment so a
  // read-before-write is still diagnosed.
  std::vector<Lanes32> regs32;
  std::vector<Lanes> regs64;
  std::vector<Mask> preds;
  std::vector<uint8_t> written32, written64;
  // Widening a 32-bit register into the 64-bit operand form needs somewhere to
  // land; reused per read to avoid touching the allocator.
  mutable std::array<Lanes, 4> widen_scratch;
  mutable uint32_t widen_next = 0;
  std::unordered_map<std::string, Lanes> slots;  // call-argument slots
  std::vector<std::vector<uint8_t>> local;       // per-lane .local frames (lazy)
  std::array<uint32_t, kWarpSize> tid_x{}, tid_y{}, tid_z{};
  // PTX's condition-code carry bit, one per lane. Written by ".cc" arithmetic
  // and read by addc/subc/madc; nothing else in the ISA touches it.
  Mask carry = 0;
  std::unique_ptr<AsyncCopies> cp;  // created on the first cp.async
};

// Strict mode, sampled once per launch: consulting it is a relaxed atomic load
// rather than a getenv on a hot path, and a process that changes the variable
// between launches gets what it asked for.
std::atomic<bool> g_strict{false};
std::atomic<bool> g_race{false};
// VGPU_RACE=2: also report writes that leave the bytes unchanged. Off by
// default -- see the comment on `unobservable_write` for why those are not
// races -- but the strict definition has its uses, so it stays reachable.
std::atomic<bool> g_race_strict{false};

// Sampled together, once per launch. Caching either in a function-local static
// makes it depend on which kernel in a process ran first -- which is how the
// race detector's own test came to pass a racy kernel: an earlier launch had
// already frozen the flag to false.
void refresh_modes() {
  const char* strict = std::getenv("VGPU_STRICT");
  g_strict.store(strict && strict[0] == '1', std::memory_order_relaxed);
  const char* race = std::getenv("VGPU_RACE");
  g_race.store(race && (race[0] == '1' || race[0] == '2'), std::memory_order_relaxed);
  g_race_strict.store(race && race[0] == '2', std::memory_order_relaxed);
}

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

inline constexpr Mask kAllLanes = 0xFFFFFFFFu;

// Applies `f` to each active lane. The full-warp case is a straight loop the
// compiler can vectorize; a partial mask walks only the set bits instead of
// testing all 32. Both matter: this runs once per instruction per warp.
template <class F>
inline void for_active(Mask m, F&& f) {
  if (m == kAllLanes) {
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) f(lane);
  } else {
    Mask rest = m;
    while (rest) {
      uint32_t lane = static_cast<uint32_t>(__builtin_ctz(rest));
      rest &= rest - 1;
      f(lane);
    }
  }
}

uint64_t mask_to_bits(uint64_t v, uint32_t bits) {
  return bits >= 64 ? v : (v & ((1ull << bits) - 1));
}

class Interpreter {
 public:
  Interpreter(const EntryFn& fn, const LaunchConfig& cfg, const ParamBuffer& params, MemoryManager& mem,
              const DeviceProfile& profile, const SymbolTable* symbols, LaunchStats& stats,
              const ProgressFn& progress)
      : fn_(fn), cfg_(cfg), params_(params), mem_(mem), profile_(profile), symbols_(symbols),
        stats_(stats), progress_(progress) {
    if (progress_) last_progress_ = std::chrono::steady_clock::now();
  }

  void set_concurrent(bool v) { concurrent_ = v; }

  void run_grid() {
    const uint64_t total = uint64_t{cfg_.grid[0]} * cfg_.grid[1] * cfg_.grid[2];
    if (cfg_.cooperative) run_grid_cooperative(total);
    else run_block_range(0, total);
  }

  // A cooperative launch: every block resident at once, interleaved.
  //
  // `cg::this_grid().sync()` is not an instruction. It compiles to an atomic
  // increment of a counter in global memory and then a spin on that counter --
  // so it only terminates if the blocks that have not arrived yet are still
  // able to run. Running blocks one at a time, which is what the ordinary
  // scheduler does and what the programming model permits, deadlocks on the
  // first block to arrive.
  //
  // So every block is set up before any of them runs, and each gets a bounded
  // turn in round-robin order. Deterministic, and the same order every time:
  // the point of this simulator is that a run is reproducible, which rules out
  // handing the blocks to OS threads and letting them race.
  void run_grid_cooperative(uint64_t total) {
    auto sched = make_scheduler(cfg_.scheduler);
    const uint64_t gx = cfg_.grid[0], gy = cfg_.grid[1];
    std::vector<BlockState> blocks(static_cast<size_t>(total));
    for (uint64_t i = 0; i < total; ++i) {
      BlockState& b = blocks[static_cast<size_t>(i)];
      b.ctx.ctaid = {static_cast<uint32_t>(i % gx), static_cast<uint32_t>((i / gx) % gy),
                     static_cast<uint32_t>(i / (gx * gy))};
      b.ctx.ntid = cfg_.block;
      b.ctx.nctaid = cfg_.grid;
      setup_block(b);
    }
    // One warp-turn per block per round. Long enough that a block making real
    // progress is not paying scheduler overhead per instruction, short enough
    // that a spinning warp hands the grid back promptly.
    constexpr uint64_t kSlice = 256;
    size_t live = blocks.size();
    while (live) {
      live = 0;
      for (BlockState& b : blocks) {
        if (b.done) continue;
        if (step_block(b, *sched, kSlice)) ++live;
        else ++stats_.blocks;
      }
    }
  }

  // Runs the blocks with linear indices [first, last). CUDA blocks are
  // independent -- that is the programming model's central promise -- so a
  // range can run on its own thread with nothing shared but device memory.
  void run_block_range(uint64_t first, uint64_t last) {
    auto sched = make_scheduler(cfg_.scheduler);
    const uint64_t gx = cfg_.grid[0], gy = cfg_.grid[1];
    for (uint64_t i = first; i < last; ++i) {
      BlockCtx ctx;
      ctx.ctaid = {static_cast<uint32_t>(i % gx), static_cast<uint32_t>((i / gx) % gy),
                   static_cast<uint32_t>(i / (gx * gy))};
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

  // Everything a block owns while it is resident. An ordinary launch keeps one
  // of these on the stack and runs it to completion; a cooperative launch keeps
  // one per block and interleaves them, which is the whole difference between
  // the two scheduling modes.
  struct BlockState {
    BlockCtx ctx;
    std::vector<uint8_t> shared;
    BarrierReduction bar_red;
    SharedShadow shadow;
    std::vector<Warp> warps;
    bool done = false;
  };

  void setup_block(BlockState& b) {
    // Fresh, zeroed shared memory per block (static declarations + the
    // launch's dynamic bytes).
    b.shared.assign(fn_.static_shared_size + cfg_.shared_bytes, 0);
    b.ctx.shared = &b.shared;
    b.ctx.bar_red = &b.bar_red;
    if (detect_races() && !b.shared.empty()) {
      b.shadow.words.assign(b.shared.size() / 4 + 1, WordShadow{});
      b.ctx.shadow = &b.shadow;
    }
    const uint64_t total = uint64_t{b.ctx.ntid[0]} * b.ctx.ntid[1] * b.ctx.ntid[2];
    const size_t nwarps = static_cast<size_t>((total + kWarpSize - 1) / kWarpSize);
    // resize, not assign: a Warp owns a unique_ptr and so is move-only.
    b.warps.clear();
    b.warps.resize(nwarps);
    for (size_t w = 0; w < nwarps; ++w) {
      Warp& warp = b.warps[w];
      warp.regs32.assign(fn_.num_regs32, Lanes32{});
      warp.regs64.assign(fn_.num_regs64, Lanes{});
      warp.preds.assign(fn_.num_regs32 + fn_.num_regs64, 0);
      warp.written32.assign(fn_.num_regs32, 0);
      warp.written64.assign(fn_.num_regs64, 0);
      Mask live = 0;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        uint64_t lin = uint64_t{static_cast<uint32_t>(w)} * kWarpSize + lane;
        if (lin >= total) break;
        live |= (1u << lane);
        warp.tid_x[lane] = static_cast<uint32_t>(lin % b.ctx.ntid[0]);
        warp.tid_y[lane] = static_cast<uint32_t>((lin / b.ctx.ntid[0]) % b.ctx.ntid[1]);
        warp.tid_z[lane] = static_cast<uint32_t>(lin / (uint64_t{b.ctx.ntid[0]} * b.ctx.ntid[1]));
      }
      if (live) warp.paths.push_back({0, live});
      else warp.state = Warp::State::Done;
    }
    stats_.warps += nwarps;
  }

  // Gives the block one turn: picks a runnable warp and runs it, or releases a
  // barrier when every warp has arrived. Returns false once the block is
  // finished.
  //
  // `slice` bounds how many instructions the chosen warp may execute before
  // returning here. An ordinary launch leaves it unbounded, because nothing
  // else is waiting. A cooperative launch must bound it: a grid barrier is a
  // spin on a global flag another block has to set, and a warp spinning without
  // a bound would never give that block a turn.
  bool step_block(BlockState& b, Scheduler& sched, uint64_t slice) {
    std::vector<size_t> runnable;
    for (size_t i = 0; i < b.warps.size(); ++i)
      if (b.warps[i].state == Warp::State::Ready) runnable.push_back(i);
    if (runnable.empty()) {
      bool any_waiting = false;
      for (auto& w : b.warps)
        if (w.state == Warp::State::AtBarrier) {
          w.state = Warp::State::Ready;
          any_waiting = true;
        }
      // Every warp has now arrived, so a bar.red in flight has its answer.
      if (any_waiting) b.bar_red.complete = true;
      // ...and the barrier they arrived at orders everything before it
      // against everything after, which is what ends the epoch.
      if (any_waiting && b.ctx.shadow) ++b.ctx.shadow->epoch;
      if (!any_waiting) {
        b.done = true;
        return false;  // all Done
      }
      return true;
    }
    const size_t picked = sched.pick(runnable);
    cur_warp_ = static_cast<uint32_t>(picked);
    run_warp_until_yield(b.warps[picked], b.ctx, slice);
    return true;
  }

  void run_block(BlockCtx& ctx, Scheduler& sched) {
    BlockState b;
    b.ctx = ctx;
    setup_block(b);
    while (step_block(b, sched, /*slice=*/0)) {
    }
  }

  // Merges paths sitting at the same pc and returns the index of the one with
  // the lowest pc, which is the path that runs next.
  size_t select_path(Warp& w) {
    if (w.paths.size() == 1) return 0;  // no divergence: nothing to merge
    for (size_t i = 0; i < w.paths.size(); ++i) {
      for (size_t j = w.paths.size(); j-- > i + 1;) {
        if (w.paths[j].pc == w.paths[i].pc) {
          w.paths[i].mask |= w.paths[j].mask;
          w.paths.erase(w.paths.begin() + static_cast<long>(j));
        }
      }
    }
    size_t best = 0;
    for (size_t i = 1; i < w.paths.size(); ++i)
      if (w.paths[i].pc < w.paths[best].pc) best = i;
    return best;
  }

  // Runs one warp until it yields: a barrier, a return, or -- when `slice` is
  // non-zero -- that many instructions. Zero means no bound.
  void run_warp_until_yield(Warp& w, const BlockCtx& ctx, uint64_t slice = 0) {
    uint64_t issued = 0;
    while (w.state == Warp::State::Ready) {
      if (slice && issued++ >= slice) return;
      if (w.paths.empty()) {
        w.state = Warp::State::Done;
        return;
      }
      size_t idx = select_path(w);
      if (w.paths[idx].pc >= fn_.body.size())
        throw Error::make(Err::PtxParse, "control fell off the end of kernel '", fn_.name,
                          "' (missing ret)");
      const Instr& ins = fn_.body[w.paths[idx].pc];
      if (progress_ && (stats_.instructions & 0xFFFFF) == 0) report_progress();
      // Warp-level issue count, plus the per-lane total: their ratio is the
      // average lane utilisation, which is divergence measured directly.
      const uint64_t lanes = static_cast<uint64_t>(popcount_mask(w.paths[idx].mask));
      stats_.thread_instructions += lanes;
      // Counted against the same mask as thread_instructions, so the classes
      // partition it exactly. Predication narrows the mask inside step(); it is
      // deliberately not subtracted here, for the same reason.
      const InstClass cls = class_of_pc(w.paths[idx].pc);
      stats_.inst_by_class[static_cast<size_t>(cls)] += lanes;
      if (cls == InstClass::Tensor) ++stats_.tensor_instructions;
      if (++stats_.instructions > cfg_.max_steps)
        throw Error::make(Err::ExecLimit, "kernel '", fn_.name, "' exceeded the launch step budget (",
                          cfg_.max_steps, " instructions) — possible infinite loop");
      step(w, ctx, idx, ins);
    }
  }

  InstClass class_of_pc(size_t pc) const {
    if (class_by_pc_.empty()) {
      class_by_pc_.resize(fn_.body.size());
      for (size_t i = 0; i < fn_.body.size(); ++i)
        class_by_pc_[i] = static_cast<uint8_t>(classify(fn_.body[i]));
    }
    return static_cast<InstClass>(class_by_pc_[pc]);
  }

  // Which category an instruction falls in. Worked out once per kernel rather
  // than per execution: a kernel is parsed once and its instructions run
  // millions of times, and a chain of variant tests on that path would cost
  // more than the counting is worth.
  static InstClass classify(const Instr& ins) {
    auto by_width = [](const Type& t) {
      if (t.bits == 16) return InstClass::Fp16;
      if (t.bits == 64) return InstClass::Fp64;
      return InstClass::Fp32;
    };
    // Arithmetic that carries a type: float by width, otherwise integer.
    if (const auto* o = std::get_if<OpFloatBin>(&ins.op)) return by_width(o->ty);
    if (const auto* o = std::get_if<OpFma>(&ins.op)) return by_width(o->ty);
    if (const auto* o = std::get_if<OpMath>(&ins.op)) return by_width(o->ty);
    if (const auto* o = std::get_if<OpCopysign>(&ins.op)) return by_width(o->ty);
    if (std::holds_alternative<OpF16x2Bin>(ins.op) ||
        std::holds_alternative<OpF16x2Fma>(ins.op) ||
        std::holds_alternative<OpF16x2Neg>(ins.op))
      return InstClass::Fp16;
    for (const Type* t : {std::get_if<OpNeg>(&ins.op) ? &std::get_if<OpNeg>(&ins.op)->ty : nullptr,
                          std::get_if<OpAbs>(&ins.op) ? &std::get_if<OpAbs>(&ins.op)->ty : nullptr})
      if (t) return t->is_real() ? by_width(*t) : InstClass::Integer;

    if (std::holds_alternative<OpIntBin>(ins.op) || std::holds_alternative<OpMadLo>(ins.op) ||
        std::holds_alternative<OpMulWide>(ins.op) || std::holds_alternative<OpMadWide>(ins.op) ||
        std::holds_alternative<OpMulHi>(ins.op) || std::holds_alternative<OpMadHi>(ins.op) ||
        std::holds_alternative<OpShf>(ins.op) || std::holds_alternative<OpBfe>(ins.op) ||
        std::holds_alternative<OpBfi>(ins.op) || std::holds_alternative<OpBrev>(ins.op) ||
        std::holds_alternative<OpPopcClz>(ins.op) || std::holds_alternative<OpPrmt>(ins.op) ||
        std::holds_alternative<OpDp4a>(ins.op) || std::holds_alternative<OpBmsk>(ins.op) ||
        std::holds_alternative<OpNot>(ins.op))
      return InstClass::Integer;

    if (std::holds_alternative<OpCvt>(ins.op) || std::holds_alternative<OpCvtF16x2>(ins.op) ||
        std::holds_alternative<OpCvta>(ins.op) || std::holds_alternative<OpMovPack>(ins.op) ||
        std::holds_alternative<OpMovUnpack>(ins.op))
      return InstClass::BitConvert;

    if (std::holds_alternative<OpBra>(ins.op) || std::holds_alternative<OpRet>(ins.op) ||
        std::holds_alternative<OpBar>(ins.op) || std::holds_alternative<OpBarRed>(ins.op) ||
        std::holds_alternative<OpTrap>(ins.op) || std::holds_alternative<OpCall>(ins.op))
      return InstClass::Control;

    if (std::holds_alternative<OpLd>(ins.op) || std::holds_alternative<OpSt>(ins.op) ||
        std::holds_alternative<OpAtom>(ins.op) || std::holds_alternative<OpCpAsync>(ins.op) ||
        std::holds_alternative<OpCpAsyncGroup>(ins.op) ||
        std::holds_alternative<OpLdMatrix>(ins.op) ||
        std::holds_alternative<OpWmmaStore>(ins.op) ||
        std::holds_alternative<OpLdSlot>(ins.op) || std::holds_alternative<OpStSlot>(ins.op))
      return InstClass::Memory;

    if (std::holds_alternative<OpMma>(ins.op) || std::holds_alternative<OpWmmaMma>(ins.op) ||
        std::holds_alternative<OpMovMatrix>(ins.op))
      return InstClass::Tensor;

    return InstClass::Misc;
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
    // A kernel may take a parameter's address rather than loading it by name:
    // CUB's segmented sort does "mov.b64 %rd, kernel_param_10" and then loads
    // through the register. Parameters live in their own window so those loads
    // resolve back to the parameter buffer.
    if (auto it = params_.layout.find(name); it != params_.layout.end())
      return kParamVaBase + it->second.first;
    ctx_fail(ins, -1, Err::NotFound,
             "unknown symbol '" + name + "' (not a .local depot, module .global variable, or "
             "kernel parameter)");
  }

  // Returns a reference to the operand's lane vector. Register operands alias
  // the warp's register file directly; everything else is materialized into
  // `scratch`. Returning a reference keeps a 256-byte copy off the hot path.
  const Lanes& read_operand(Warp& w, const BlockCtx& ctx, const Instr& ins, const Operand& op,
                            Lanes& scratch) {
    if (const auto* r = std::get_if<RegOperand>(&op)) {
      // Reading a register that has not been written yet is undefined in PTX,
      // and compilers emit it on purpose. ggml's flash-attention kernels
      // unroll "pick the value belonging to my lane" into a chain of selp, each
      // step overwriting the accumulator only for the lane whose index matches;
      // the chain is seeded with a register nothing has written, and every
      // lane's copy of that seed is dead by the time the chain ends. NVIDIA's
      // own compute-sanitizer checks uninitialized *memory* and not registers,
      // for this reason.
      //
      // So an undefined register reads as zero here -- deterministically, which
      // is more than hardware promises -- and the diagnostic is kept where it
      // still means something: addr_base, read_pred and exec_st below refuse a
      // register nothing has written, because an address, a branch condition or
      // a stored value made of nothing is a bug in any program.
      if (r->reg.wide) {
        if (r->reg.id >= w.regs64.size() || !w.written64[r->reg.id]) {
          scratch.fill(0);
          return scratch;
        }
        return w.regs64[r->reg.id];
      }
      if (r->reg.id >= w.regs32.size() || !w.written32[r->reg.id]) {
        scratch.fill(0);
        return scratch;
      }
      return widen(w, w.regs32[r->reg.id]);
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
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      scratch[lane] = sreg_value(sr.reg, sr.index, w, ctx, lane);
    return scratch;
  }

  uint32_t sreg_value(Sreg s, uint32_t index, const Warp& w, const BlockCtx& ctx, uint32_t lane) {
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
      case Sreg::LaneMaskEq: return 1u << lane;
      case Sreg::LaneMaskLt: return lane == 0 ? 0u : (~0u >> (kWarpSize - lane));
      case Sreg::LaneMaskLe: return static_cast<uint32_t>((uint64_t{2} << lane) - 1);
      case Sreg::LaneMaskGt: return lane == kWarpSize - 1
                                        ? 0u
                                        : ~static_cast<uint32_t>((uint64_t{2} << lane) - 1);
      case Sreg::LaneMaskGe: return ~0u << lane;
      // Warps are numbered within the block, which is how a kernel uses this:
      // to index a per-warp slot in shared memory.
      case Sreg::WarpId: {
        const uint32_t linear = w.tid_x[lane] + w.tid_y[lane] * ctx.ntid[0] +
                                w.tid_z[lane] * ctx.ntid[0] * ctx.ntid[1];
        return linear / kWarpSize;
      }
      // The driver's parameter bank. Only two entries mean anything here: a
      // cooperative launch passes the address of its grid-barrier workspace as
      // %envreg1 (low half) and %envreg2 (high half). cooperative_groups reads
      // exactly those, and traps if the pair is zero -- which is how it detects
      // a grid.sync() outside a cooperative launch, and why every other entry
      // must stay zero rather than being given a plausible-looking value.
      // %envreg1 is the *high* half and %envreg2 the low one. That is not a
      // guess: the generated code reassembles them with
      // "bfi.b64 %rd1, %rd7, %rd6, 32, 32", which inserts %envreg1 into bits
      // 63:32 of %envreg2.
      case Sreg::EnvReg:
        if (index == 1) return static_cast<uint32_t>(cfg_.coop_workspace >> 32);
        if (index == 2) return static_cast<uint32_t>(cfg_.coop_workspace);
        return 0;
      case Sreg::NWarpId:
        return (ctx.ntid[0] * ctx.ntid[1] * ctx.ntid[2] + kWarpSize - 1) / kWarpSize;
    }
    return 0;
  }

  // Widens a 32-bit register into the 64-bit operand form. A small rotating
  // set of buffers keeps several operands of one instruction alive at once.
  const Lanes& widen(Warp& w, const Lanes32& src) {
    Lanes& out = w.widen_scratch[w.widen_next];
    w.widen_next = (w.widen_next + 1) % w.widen_scratch.size();
    for (uint32_t l = 0; l < kWarpSize; ++l) out[l] = src[l];
    return out;
  }

  // True when an operand can be read as 32-bit lanes with no widening: a
  // narrow register, or an immediate (which is materialized either way).
  static bool narrow_operand(const Operand& o) {
    if (const auto* r = std::get_if<RegOperand>(&o)) return !r->reg.wide;
    return std::holds_alternative<ImmInt>(o) || std::holds_alternative<ImmFloatBits>(o);
  }

  // Reads an operand as 32-bit lanes. Only valid when narrow_operand() holds.
  const Lanes32& read_narrow(Warp& w, const Instr& ins, const Operand& o, Lanes32& scratch) {
    if (const auto* r = std::get_if<RegOperand>(&o)) {
      // Undefined reads as zero, for the reason given on read_operand.
      if (r->reg.id >= w.regs32.size() || !w.written32[r->reg.id]) {
        (void)ins;
        scratch.fill(0);
        return scratch;
      }
      return w.regs32[r->reg.id];
    }
    uint32_t v = std::holds_alternative<ImmInt>(o)
                     ? static_cast<uint32_t>(std::get<ImmInt>(o).value)
                     : static_cast<uint32_t>(std::get<ImmFloatBits>(o).bits);
    scratch.fill(v);
    return scratch;
  }

  // Writes 32-bit lanes straight into the narrow file -- no widening, and half
  // the memory traffic of the 64-bit path.
  void write_narrow(Warp& w, const Reg& reg, Mask m, const Lanes32& vals) {
    Lanes32& dst = w.regs32[reg.id];
    for_active(m, [&](uint32_t l) { dst[l] = vals[l]; });
    w.written32[reg.id] = 1;
  }

  void write_reg(Warp& w, const Reg& reg, Mask m, const Lanes& vals, uint32_t bits) {
    if (reg.wide) {
      Lanes& dst = w.regs64[reg.id];
      if (bits >= 64)
        for_active(m, [&](uint32_t l) { dst[l] = vals[l]; });
      else {
        const uint64_t keep = (1ull << bits) - 1;
        for_active(m, [&](uint32_t l) { dst[l] = vals[l] & keep; });
      }
      w.written64[reg.id] = 1;
      return;
    }
    Lanes32& dst = w.regs32[reg.id];
    if (bits >= 32)
      for_active(m, [&](uint32_t l) { dst[l] = static_cast<uint32_t>(vals[l]); });
    else {
      const uint32_t keep = (1u << bits) - 1;
      for_active(m, [&](uint32_t l) { dst[l] = static_cast<uint32_t>(vals[l]) & keep; });
    }
    w.written32[reg.id] = 1;
  }

  // Predicates are declared .pred, so they live in the narrow numbering; the
  // slot index is offset past the 64-bit file to keep one predicate vector.
  uint32_t pred_index(const Reg& reg) const {
    return reg.wide ? fn_.num_regs32 + reg.id : reg.id;
  }

  Mask read_pred(Warp& w, const Instr& ins, const Reg& reg) {
    uint32_t idx = pred_index(reg);
    bool ok = reg.wide ? (reg.id < w.written64.size() && w.written64[reg.id])
                       : (reg.id < w.written32.size() && w.written32[reg.id]);
    if (idx >= w.preds.size() || !ok)
      ctx_fail(ins, -1, Err::UninitializedRegister,
               "predicate " + reg.name + " read before any write");
    return w.preds[idx];
  }

  static uint32_t popcount_mask(Mask m) {
    return static_cast<uint32_t>(__builtin_popcount(static_cast<unsigned>(m)));
  }

  // Memory traffic, counted per active lane. Space matters: a shared access and
  // a global one cost very different things on hardware, and lumping them would
  // make the numbers useless for the comparison people actually want.
  // Sectors and bank conflicts, counted from the addresses the lanes actually
  // used. A device derives both by sampling; here they are exact, because every
  // lane's address is in hand at the moment of the access.
  //
  // `addrs` holds one address per lane, valid where `m` is set. `bytes` is the
  // per-lane access width, `count` the number of consecutive elements a vector
  // access touches from that address.
  void count_addresses(Space space, const Lanes& addrs, Mask m, uint32_t bytes, size_t count) {
    if (m == 0) return;
    const uint64_t span = static_cast<uint64_t>(bytes) * count;

    if (space == Space::Shared) {
      // 32 banks of 4 bytes. Lanes reaching different words in one bank
      // serialize; lanes reaching the same word are broadcast and free. An
      // access wider than a word touches consecutive words, which the hardware
      // takes in separate passes -- counting each word is what makes a
      // conflict-free 8-byte access come out as conflict-free.
      uint32_t worst = 1;
      for (uint32_t bank = 0; bank < 32; ++bank) {
        std::array<uint64_t, kWarpSize * 4> words{};
        uint32_t distinct = 0;
        for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
          if (!(m & (1u << lane))) continue;
          for (uint64_t off = 0; off < span; off += 4) {
            const uint64_t word = (addrs[lane] + off) / 4;
            if (word % 32 != bank) continue;
            bool seen = false;
            for (uint32_t i = 0; i < distinct; ++i)
              if (words[i] == word) { seen = true; break; }
            if (!seen && distinct < words.size()) words[distinct++] = word;
          }
        }
        if (distinct > worst) worst = distinct;
      }
      stats_.shared_bank_conflicts += worst - 1;
      ++stats_.shared_requests;
      return;
    }

    if (space == Space::Param) return;  // the constant bank, not device memory

    // Everything else moves in 32-byte sectors. Count the distinct ones the
    // warp touched: four for a coalesced 32-lane 4-byte load, up to 32 when
    // every lane lands in its own sector.
    std::array<uint64_t, kWarpSize * 8> sectors{};
    uint32_t distinct = 0;
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      const uint64_t first = addrs[lane] / 32;
      const uint64_t last = (addrs[lane] + (span ? span - 1 : 0)) / 32;
      for (uint64_t sec = first; sec <= last; ++sec) {
        bool seen = false;
        for (uint32_t i = 0; i < distinct; ++i)
          if (sectors[i] == sec) { seen = true; break; }
        if (!seen && distinct < sectors.size()) sectors[distinct++] = sec;
      }
    }
    if (space == Space::Local) {
      stats_.local_sectors += distinct;
      ++stats_.local_requests;
    } else {
      stats_.global_sectors += distinct;
      ++stats_.global_requests;
    }
  }

  void count_memory(Space space, uint32_t bytes, uint32_t lanes, bool is_store) {
    const uint64_t n = lanes;
    const uint64_t b = n * bytes;
    switch (space) {
      case Space::Shared:
        if (is_store) { stats_.shared_stores += n; stats_.shared_bytes_written += b; }
        else { stats_.shared_loads += n; stats_.shared_bytes_read += b; }
        break;
      case Space::Local:
        if (is_store) stats_.local_stores += n; else stats_.local_loads += n;
        break;
      case Space::Param:
        // Kernel parameters are read from the constant bank, not from device
        // memory. Counting them as global traffic inflated every kernel's load
        // count by one per thread and would make the byte totals wrong.
        break;
      case Space::Global:
      case Space::Generic:
        if (is_store) { stats_.global_stores += n; stats_.global_bytes_written += b; }
        else { stats_.global_loads += n; stats_.global_bytes_read += b; }
        break;
    }
  }

  Mask& pred_slot(Warp& w, const Reg& reg) {
    if (reg.wide) w.written64[reg.id] = 1;
    else w.written32[reg.id] = 1;
    return w.preds[pred_index(reg)];
  }

  // ---- routed memory access (device global VA range vs .local window) ----

  // Window base for an address space. Global/generic already alias the flat
  // device VA range, so their base is zero.
  static uint64_t space_base(Space sp) {
    switch (sp) {
      case Space::Shared: return kSharedVaBase;
      case Space::Local: return kLocalVaBase;
      case Space::Param: return kParamVaBase;
      default: return 0;
    }
  }

  bool is_local(uint64_t addr) const {
    return addr >= kLocalVaBase && addr < kLocalVaBase + kLocalVaSize;
  }

  bool is_shared(uint64_t addr) const {
    return addr >= kSharedVaBase && addr < kSharedVaBase + kSharedVaSize;
  }

  // Race detection, off unless VGPU_RACE=1.
  static bool detect_races() { return g_race.load(std::memory_order_relaxed); }
  static bool races_strict() { return g_race_strict.load(std::memory_order_relaxed); }

  [[noreturn]] void report_race(const Instr& ins, const char* what, uint64_t word,
                                uint32_t other_warp) {
    ctx_fail(ins, -1, Err::DataRace,
             std::string(what) + " on shared memory at byte offset " +
                 std::to_string(word * 4) + ": warp " + std::to_string(cur_warp_) +
                 " and warp " + std::to_string(other_warp) +
                 " both reach it with no bar.sync between them, so which one wins is not "
                 "something the program decided. Hardware usually hides this because warps "
                 "advance together; it is a real race either way");
  }

  // True when the store leaves shared memory exactly as it found it.
  //
  // Such a store cannot be observed by anyone: no reader and no other writer
  // can tell whether it happened before or after, because the bytes are the
  // same either way. So it does not turn a conflicting access into a race.
  //
  // This is not a heuristic -- it is checked against the bytes actually there,
  // and the conflicting writer is what put them there. Real kernels do this on
  // purpose: llama.cpp's mul_mat_q clamps out-of-range tile rows with
  // `i = min(i, i_max)`, so several warps recompute the same source pointer and
  // write the same value to the same word. Reporting that as a bug makes the
  // detector useless on the code it exists to check.
  //
  // The write is still *recorded* as a write, so a later store of a different
  // value is still caught against it. Only the report is suppressed.
  bool unobservable_write(const BlockCtx& ctx, uint64_t addr, uint32_t bytes,
                          uint64_t value) const {
    if (races_strict()) return false;
    if (bytes == 0 || bytes > 8) return false;
    uint64_t existing = 0;
    std::memcpy(&existing, ctx.shared->data() + (addr - kSharedVaBase), bytes);
    uint64_t incoming = 0;
    std::memcpy(&incoming, &value, bytes);
    return existing == incoming;
  }

  void note_shared_access(const BlockCtx& ctx, const Instr& ins, uint64_t addr, uint32_t bytes,
                          bool is_write, uint64_t value = 0) {
    if (!ctx.shadow) return;
    SharedShadow& sh = *ctx.shadow;
    const bool silent = is_write && unobservable_write(ctx, addr, bytes, value);
    const uint64_t first = (addr - kSharedVaBase) / 4;
    const uint64_t last = (addr - kSharedVaBase + (bytes ? bytes - 1 : 0)) / 4;
    for (uint64_t w = first; w <= last && w < sh.words.size(); ++w) {
      WordShadow& s = sh.words[w];
      if (is_write) {
        if (!silent && s.write_epoch == sh.epoch && s.writer != 0xFFFF && s.writer != cur_warp_)
          report_race(ins, "write-write race", w, s.writer);
        if (!silent && s.read_epoch == sh.epoch) {
          const uint32_t others = s.readers & ~(1u << cur_warp_);
          if (others) report_race(ins, "read-write race", w, __builtin_ctz(others));
        }
        s.writer = static_cast<uint16_t>(cur_warp_);
        s.write_epoch = sh.epoch;
      } else {
        if (s.write_epoch == sh.epoch && s.writer != 0xFFFF && s.writer != cur_warp_)
          report_race(ins, "write-read race", w, s.writer);
        if (s.read_epoch != sh.epoch) {
          s.readers = 0;
          s.read_epoch = sh.epoch;
        }
        s.readers |= (1u << cur_warp_);
      }
    }
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
      note_shared_access(ctx, ins, addr, size, /*is_write=*/false);
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
      note_shared_access(ctx, ins, addr, size, /*is_write=*/true, value);
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

  void step(Warp& w, const BlockCtx& ctx, size_t idx, const Instr& ins) {
    Mask active = w.paths[idx].mask;
    Mask m = active;
    if (ins.has_pred) {
      Mask p = read_pred(w, ins, ins.pred);
      if (ins.pred_negated) p = ~p;
      m &= p;
    }

    if (const auto* op = std::get_if<OpBra>(&ins.op)) {
      exec_bra(w, idx, *op, m);
      return;
    }
    if (std::holds_alternative<OpRet>(ins.op)) {
      exec_ret(w, ctx, ins, idx, m);
      return;
    }
    if (const auto* op = std::get_if<OpBarRed>(&ins.op)) {
      if (ins.has_pred)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "predicated bar.red is not supported");
      if (!ctx.bar_red)
        ctx_fail(ins, -1, Err::UnsupportedPtx, "bar.red outside a block context");
      BarrierReduction& red = *ctx.bar_red;
      if (!w.bar_red_waiting) {
        // Contribute and wait. The result is not known until every warp in the
        // block has arrived, so the pc stays put and this re-executes on
        // release rather than advancing now.
        if (red.arrived == 0) {
          red.acc = op->op == BarRedOp::And ? ~uint64_t{0} : 0;
          red.complete = false;
        }
        Mask p = read_pred(w, ins, op->src);
        if (op->negate_src) p = ~p;
        const Mask voters = p & m;
        switch (op->op) {
          case BarRedOp::And:
            // True only if every participating lane of every warp voted true.
            red.acc &= (voters == m) ? 1u : 0u;
            break;
          case BarRedOp::Or:
            red.acc |= (voters != 0) ? 1u : 0u;
            break;
          case BarRedOp::Popc:
            red.acc += static_cast<uint64_t>(popcount_mask(voters));
            break;
        }
        ++red.arrived;
        ++stats_.barriers;
        w.bar_red_waiting = true;
        w.state = Warp::State::AtBarrier;
        return;
      }
      // Released: collect the block-wide result.
      w.bar_red_waiting = false;
      if (red.arrived) --red.arrived;
      if (op->op == BarRedOp::Popc) {
        Lanes r;
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = red.acc;
        write_reg(w, op->dst, m, r, 32);
      } else {
        Mask& dp = pred_slot(w, op->dst);
        dp = (red.acc & 1u) ? (dp | m) : (dp & ~m);
      }
      ++w.paths[idx].pc;
      return;
    }
    if (std::holds_alternative<OpBar>(ins.op)) {
      if (ins.has_pred) ctx_fail(ins, -1, Err::UnsupportedPtx, "predicated bar.sync is not supported");
      ++stats_.barriers;
      // Every live lane must arrive before the warp yields. Lanes still on
      // other paths have a higher pc and will merge here first; if any path
      // can never reach this barrier the kernel is malformed, and the step
      // budget catches it rather than deadlocking silently.
      if (w.paths.size() > 1) {
        // Let the other paths run until they merge at this barrier.
        w.paths[idx].pc = w.paths[idx].pc;  // stay put; a lower-pc path runs next
        size_t other = select_other_runnable(w, idx);
        if (other != idx) return;
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "bar.sync cannot be reached by every lane of the warp: some lanes are on a path "
                 "that never arrives at this barrier");
      }
      ++w.paths[idx].pc;
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
    ++w.paths[idx].pc;
  }

  // Is there another path that can still run (a strictly lower pc)? Used to
  // decide whether a barrier is merely waiting for stragglers.
  size_t select_other_runnable(Warp& w, size_t idx) {
    for (size_t i = 0; i < w.paths.size(); ++i)
      if (i != idx && w.paths[i].pc < w.paths[idx].pc) return i;
    return idx;
  }

  void exec_bra(Warp& w, size_t idx, const OpBra& op, Mask m) {
    Mask taken = m;
    Mask fallthrough = w.paths[idx].mask & ~taken;
    if (taken == 0) {
      ++w.paths[idx].pc;
      return;
    }
    if (fallthrough == 0) {
      w.paths[idx].pc = op.target;
      return;
    }
    // Diverge: both halves become live paths, and whichever has the lower pc
    // runs first. They merge again as soon as they reach the same pc. Only this
    // case is divergence -- a branch every lane agrees on took one of the two
    // early returns above and costs nothing.
    ++stats_.divergent_branches;
    size_t fall_pc = w.paths[idx].pc + 1;
    w.paths[idx].pc = op.target;
    w.paths[idx].mask = taken;
    w.paths.push_back({fall_pc, fallthrough});
  }

  void exec_ret(Warp& w, const BlockCtx& ctx, const Instr& ins, size_t idx, Mask m) {
    w.exited |= m;
    Mask survivors = w.paths[idx].mask & ~m;
    if (survivors == 0) {
      w.paths.erase(w.paths.begin() + static_cast<long>(idx));
    } else {
      // A predicated ret retires some lanes; the rest carry on.
      w.paths[idx].mask = survivors;
      ++w.paths[idx].pc;
    }
    if (w.paths.empty()) {
      w.state = Warp::State::Done;
      drain_async_copies(w, ctx, ins);
    }
  }

  void dispatch(Warp& w, const BlockCtx& ctx, const Instr& ins, Mask m) {
    if (const auto* op = std::get_if<OpMov>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      write_reg(w, op->dst, m, v, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpIntBin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      if (op->carry_in || op->carry_out) {
        exec_carry_add_sub(w, *op, a, b, m, r);
      } else {
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = int_bin(op->op, op->ty, a[lane], b[lane], ins);
      }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (std::holds_alternative<OpNop>(ins.op)) return;
    if (const auto* op = std::get_if<OpActiveMask>(&ins.op)) {
      Lanes r;  // every active lane sees the same membership
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = m;
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCpAsync>(&ins.op)) {
      exec_cp_async(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpCpAsyncGroup>(&ins.op)) {
      exec_cp_async_group(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpLd>(&ins.op)) {
      count_memory(op->space, op->ty.bytes(), popcount_mask(m), /*is_store=*/false);
      exec_ld(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSt>(&ins.op)) {
      count_memory(op->space, op->ty.bytes(), popcount_mask(m), /*is_store=*/true);
      exec_st(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpSetp>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Mask& p = pred_slot(w, op->dst);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          bool t = compare(op->cmp, op->ty, a[lane], b[lane]);
          p = t ? (p | (1u << lane)) : (p & ~(1u << lane));
        }
      return;
    }
    if (const auto* op = std::get_if<OpFloatBin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      // An explicit rounding mode changes the result, so it has to be applied
      // rather than assumed to be round-to-nearest: quantization kernels use
      // .rz specifically because truncation is what they want. Set the host
      // mode around the arithmetic and put it back, so nothing else observes
      // the change.
      const int prev_round = std::fegetround();
      switch (op->round) {
        case FRound::Zero: std::fesetround(FE_TOWARDZERO); break;
        case FRound::MinusInf: std::fesetround(FE_DOWNWARD); break;
        case FRound::PlusInf: std::fesetround(FE_UPWARD); break;
        case FRound::Nearest: break;
      }
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = float_bin(op->op, op->ty, a[lane], b[lane]);
      if (op->round != FRound::Nearest) std::fesetround(prev_round);
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
      Lanes r;  // written for every active lane below
      if (op->carry_in || op->carry_out) {
        Lanes prod;
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) prod[lane] = a[lane] * b[lane];
        exec_carry_mad(w, op->ty.bits, op->carry_in, op->carry_out, prod, c, m, r);
      } else {
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = a[lane] * b[lane] + c[lane];
      }
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
      Lanes r;  // written for every active lane below
      if (op->ty.bits == 32)
        for_active(m, [&](uint32_t l) { r[l] = f32bits(std::fma(f32(a[l]), f32(b[l]), f32(c[l]))); });
      else
        for_active(m, [&](uint32_t l) { r[l] = f64bits(std::fma(f64(a[l]), f64(b[l]), f64(c[l]))); });
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpMulWide>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          if (op->src_bits == 16) {
            if (op->is_signed)
              r[lane] = static_cast<uint64_t>(static_cast<uint32_t>(
                  int32_t{static_cast<int16_t>(a[lane])} * int32_t{static_cast<int16_t>(b[lane])}));
            else
              r[lane] = uint32_t{static_cast<uint16_t>(a[lane])} *
                        uint32_t{static_cast<uint16_t>(b[lane])};
          } else if (op->is_signed)
            r[lane] = static_cast<uint64_t>(int64_t{static_cast<int32_t>(a[lane])} *
                                            int64_t{static_cast<int32_t>(b[lane])});
          else
            r[lane] = uint64_t{static_cast<uint32_t>(a[lane])} * uint64_t{static_cast<uint32_t>(b[lane])};
        }
      write_reg(w, op->dst, m, r, op->src_bits * 2);  // .wide doubles the width
      return;
    }
    if (const auto* op = std::get_if<OpSelp>(&ins.op)) {
      Mask p = read_pred(w, ins, op->pred);
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = (p & (1u << lane)) ? a[lane] : b[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCvta>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      uint64_t base = space_base(op->space);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane))
          r[lane] = op->to_space ? v[lane] - base : v[lane] + base;
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
      Lanes r;  // written for every active lane below
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
    if (const auto* op = std::get_if<OpCvt>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = convert(op, v[lane]);
      write_reg(w, op->dst, m, r, op->dst_ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpAtom>(&ins.op)) {
      exec_atom(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMulHi>(&ins.op)) {
      Lanes _s_a; const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b; const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = mul_hi(op->ty, a[lane], b[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
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
      Lanes r;  // written for every active lane below
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
        Lanes r;  // written for every active lane below
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = mask_to_bits(v[lane] >> (piece * i), piece);
        write_reg(w, op->dsts[i], m, r, piece);
      }
      return;
    }
    if (const auto* op = std::get_if<OpNot>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = ~v[lane];
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpNeg>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
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
    if (const auto* op = std::get_if<OpMovPred>(&ins.op)) {
      Mask& p = pred_slot(w, op->dst);
      if (const auto* imm = std::get_if<ImmInt>(&op->src)) {
        // A constant applies to every active lane; inactive lanes keep theirs.
        if (imm->value != 0) p |= m;
        else p &= ~m;
      } else if (const auto* r = std::get_if<RegOperand>(&op->src)) {
        const Mask src = w.preds[pred_index(r->reg)];
        p = (p & ~m) | (src & m);
      } else {
        ctx_fail(ins, -1, Err::UnsupportedPtx,
                 "mov.pred source must be an immediate or a predicate register");
      }
      return;
    }
    if (const auto* op = std::get_if<OpLdMatrix>(&ins.op)) {
      exec_ldmatrix(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMovMatrix>(&ins.op)) {
      exec_movmatrix(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpMma>(&ins.op)) {
      exec_mma(w, ctx, ins, *op, m);
      return;
    }
    if (const auto* op = std::get_if<OpRedux>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->src, _s_a);
      // Every participating lane contributes and every one receives the result.
      // The active mask is the membership: a lane not executing this
      // instruction is not in the warp's reduction.
      bool first = true;
      uint64_t acc = 0;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        if (!(m & (1u << lane))) continue;
        const uint64_t v = a[lane];
        if (first) { acc = v; first = false; continue; }
        switch (op->op) {
          case ReduxOp::Add: acc = acc + v; break;
          case ReduxOp::And: acc = acc & v; break;
          case ReduxOp::Or: acc = acc | v; break;
          case ReduxOp::Xor: acc = acc ^ v; break;
          case ReduxOp::Min:
            acc = op->ty.is_signed()
                      ? static_cast<uint64_t>(std::min(narrow_s(acc, op->ty.bits),
                                                       narrow_s(v, op->ty.bits)))
                      : std::min(narrow_u(acc, op->ty.bits), narrow_u(v, op->ty.bits));
            break;
          case ReduxOp::Max:
            acc = op->ty.is_signed()
                      ? static_cast<uint64_t>(std::max(narrow_s(acc, op->ty.bits),
                                                       narrow_s(v, op->ty.bits)))
                      : std::max(narrow_u(acc, op->ty.bits), narrow_u(v, op->ty.bits));
            break;
        }
      }
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = acc;
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpCvtF16x2>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          // The first source goes in the high half, the second in the low.
          const double x = static_cast<double>(f32(a[lane]));
          const double y = static_cast<double>(f32(b[lane]));
          const uint64_t hi = op->bf16 ? double_to_bf16(x) : double_to_f16(x);
          const uint64_t lo = op->bf16 ? double_to_bf16(y) : double_to_f16(y);
          r[lane] = ((hi & 0xffffull) << 16) | (lo & 0xffffull);
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpCopysign>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          // Magnitude of b, sign of a -- and it must be bit-exact for zeros and
          // NaNs, so move the sign bit rather than going through comparisons.
          if (op->ty.bits == 64) {
            const uint64_t sign = a[lane] & (1ull << 63);
            r[lane] = sign | (b[lane] & ~(1ull << 63));
          } else {
            const uint32_t av = static_cast<uint32_t>(a[lane]);
            const uint32_t bv = static_cast<uint32_t>(b[lane]);
            r[lane] = (av & 0x80000000u) | (bv & 0x7fffffffu);
          }
        }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpDp4a>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          const uint32_t av = static_cast<uint32_t>(a[lane]);
          const uint32_t bv = static_cast<uint32_t>(b[lane]);
          // Each operand contributes four bytes, sign- or zero-extended
          // according to its own type; the four products accumulate into c.
          int64_t acc = op->a_signed || op->b_signed
                            ? static_cast<int64_t>(static_cast<int32_t>(c[lane]))
                            : static_cast<int64_t>(static_cast<uint32_t>(c[lane]));
          for (int byte = 0; byte < 4; ++byte) {
            const uint8_t ab = static_cast<uint8_t>(av >> (byte * 8));
            const uint8_t bb = static_cast<uint8_t>(bv >> (byte * 8));
            const int64_t ax = op->a_signed ? static_cast<int8_t>(ab) : static_cast<int64_t>(ab);
            const int64_t bx = op->b_signed ? static_cast<int8_t>(bb) : static_cast<int64_t>(bb);
            acc += ax * bx;
          }
          r[lane] = static_cast<uint32_t>(static_cast<int32_t>(acc));
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpBmsk>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          uint32_t base = static_cast<uint32_t>(a[lane]);
          uint32_t width = static_cast<uint32_t>(b[lane]);
          // .wrap takes both operands modulo 32; .clamp caps them at 32. These
          // are different: wrapping the start position for .clamp turns a
          // position of 40 into 8 and produces a mask in the wrong place,
          // rather than the empty mask the clamp is supposed to give.
          if (op->wrap) {
            base &= 31u;
            width &= 31u;
          } else {
            if (base > 32u) base = 32u;
            if (width > 32u) width = 32u;
          }
          // Build and shift in 64 bits: a width or shift of 32 is undefined on
          // a 32-bit type.
          const uint64_t bits = width >= 32u ? 0xffffffffull : ((1ull << width) - 1ull);
          const uint64_t shifted = base >= 32u ? 0ull : (bits << base);
          r[lane] = static_cast<uint32_t>(shifted & 0xffffffffull);
        }
      write_reg(w, op->dst, m, r, 32);
      return;
    }
    if (const auto* op = std::get_if<OpPrmt>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) r[lane] = bfi(op->ty, a[lane], b[lane], c[lane], d[lane]);
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpBrev>(&ins.op)) {
      Lanes _s_v;
      const Lanes& v = read_operand(w, ctx, ins, op->src, _s_v);
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
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
        Lanes r;  // written for every active lane below
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = voters;
        write_reg(w, op->dst, m, r, 32);
      } else {
        bool all = (voters == m), any = (voters != 0);
        bool val = op->mode == VoteMode::All   ? all
                   : op->mode == VoteMode::Any ? any
                                               : (voters == m || voters == 0);  // uni
        Mask& dp = pred_slot(w, op->dst);
        dp = val ? (dp | m) : (dp & ~m);
      }
      return;
    }
    if (const auto* op = std::get_if<OpMadHi>(&ins.op)) {
      Lanes _s_a; const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b; const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c; const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
      if (op->carry_in || op->carry_out) {
        Lanes prod;
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) prod[lane] = mul_hi(op->ty, a[lane], b[lane]);
        exec_carry_mad(w, op->ty.bits, op->carry_in, op->carry_out, prod, c, m, r);
      } else {
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) r[lane] = mul_hi(op->ty, a[lane], b[lane]) + c[lane];
      }
      write_reg(w, op->dst, m, r, op->ty.bits);
      return;
    }
    if (const auto* op = std::get_if<OpShf>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes _s_c;
      const Lanes& c = read_operand(w, ctx, ins, op->c, _s_c);
      Lanes r;  // written for every active lane below
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
    if (const auto* op = std::get_if<OpF16x2Bin>(&ins.op)) {
      Lanes _s_a;
      const Lanes& a = read_operand(w, ctx, ins, op->a, _s_a);
      Lanes _s_b;
      const Lanes& b = read_operand(w, ctx, ins, op->b, _s_b);
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
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
      Lanes r;  // written for every active lane below
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
    if (const auto* op = std::get_if<OpPredBin>(&ins.op)) {
      Mask a = read_pred(w, ins, op->a);
      Mask b = read_pred(w, ins, op->b);
      Mask r = op->op == PredBinOp::And ? (a & b) : op->op == PredBinOp::Or ? (a | b) : (a ^ b);
      Mask& p = pred_slot(w, op->dst);
      p = (p & ~m) | (r & m);
      return;
    }
    if (const auto* op = std::get_if<OpNotPred>(&ins.op)) {
      Mask s = read_pred(w, ins, op->src);
      Mask& p = pred_slot(w, op->dst);
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
    if (std::holds_alternative<OpTrap>(ins.op)) {
      // "trap" ends the kernel with an unrecoverable device-side error. On
      // hardware the launch fails and the context is left unusable; here it is
      // an error with the line that did it, which is more use and no less true.
      //
      // Compilers put it on paths that are supposed to be unreachable --
      // cooperative_groups emits one where the grid is not cooperative, and
      // assert() lowers to it -- so reaching one is a fact about the program
      // worth reporting rather than a gap in this interpreter.
      ctx_fail(ins, -1, Err::Trap,
               "the kernel executed 'trap', which ends it with a device-side error. This is what "
               "a failed device assert(), an unreachable path, or a cooperative-groups call "
               "outside a cooperative launch compiles to");
    }
    // The parser accepted it and nothing here runs it -- a gap between the two
    // halves, not a bad program. Naming the instruction is the difference
    // between a five-minute fix and a bisection.
    ctx_fail(ins, -1, Err::Internal,
             "the parser accepts this instruction but the interpreter has no handler for it");
  }

  // High half of a same-width product. 64-bit needs a 128-bit intermediate.
  static uint64_t mul_hi(Type ty, uint64_t a, uint64_t b) {
    if (ty.bits == 64) {
      if (ty.is_signed()) {
        __int128 prod = static_cast<__int128>(static_cast<int64_t>(a)) *
                        static_cast<__int128>(static_cast<int64_t>(b));
        return static_cast<uint64_t>(static_cast<unsigned __int128>(prod) >> 64);
      }
      unsigned __int128 prod = static_cast<unsigned __int128>(a) * static_cast<unsigned __int128>(b);
      return static_cast<uint64_t>(prod >> 64);
    }
    uint32_t bits = ty.bits;
    if (ty.is_signed()) {
      int64_t x = static_cast<int64_t>(sign_extend(a, bits));
      int64_t y = static_cast<int64_t>(sign_extend(b, bits));
      return mask_to_bits(static_cast<uint64_t>((x * y) >> bits), bits);
    }
    uint64_t prod = mask_to_bits(a, bits) * mask_to_bits(b, bits);
    return mask_to_bits(prod >> bits, bits);
  }

  static uint64_t sign_extend(uint64_t v, uint32_t bits) {
    if (bits >= 64) return v;
    uint64_t sign = 1ull << (bits - 1);
    v = mask_to_bits(v, bits);
    return (v & sign) ? (v | ~((sign << 1) - 1)) : v;
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
    Lanes r;  // written for every active lane below
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
    if (op.pred_dst.id != kNoReg) {
      Mask& p = pred_slot(w, op.pred_dst);
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

  // ldmatrix: row r of matrix i comes from the address supplied by lane i*8+r,
  // and every lane leaves with two consecutive 16-bit elements of one row. The
  // distribution is the layout an mma A/B fragment expects, which is the whole
  // point of the instruction.
  void exec_ldmatrix(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpLdMatrix& op, Mask m) {
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    // Without ".shared" the register already holds an address in the shared
    // window -- it comes from cvta, which has done the space conversion, and
    // adding the window base again lands at twice it. With ".shared" the
    // instruction names the space itself and the register is a bare offset, so
    // the base has to be applied here instead.
    const uint64_t window = op.shared_space ? space_base(Space::Shared) : 0;
    for (uint32_t mat = 0; mat < op.count; ++mat) {
      // Pull the 8x8 matrix in, a row at a time.
      uint16_t tile[8][8] = {};
      for (uint32_t r = 0; r < 8; ++r) {
        const uint32_t src_lane = mat * 8 + r;
        if (!(m & (1u << src_lane)))
          ctx_fail(ins, static_cast<int>(src_lane), Err::UnsupportedPtx,
                   "ldmatrix needs every lane that supplies a row address to be active");
        const uint64_t addr = window + base[src_lane] + static_cast<uint64_t>(op.addr.offset);
        for (uint32_t c = 0; c < 8; ++c)
          tile[r][c] = static_cast<uint16_t>(
              load_routed(w, ctx, ins, src_lane, addr + c * 2, 2));
      }
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          const uint32_t row = lane / 4, colpair = lane % 4;
          uint16_t lo, hi;
          if (op.trans) {
            lo = tile[colpair * 2][row];
            hi = tile[colpair * 2 + 1][row];
          } else {
            lo = tile[row][colpair * 2];
            hi = tile[row][colpair * 2 + 1];
          }
          r[lane] = (static_cast<uint64_t>(hi) << 16) | lo;
        }
      write_reg(w, op.dsts[mat], m, r, 32);
    }
  }

  // The warp holds an 8x8 matrix of 16-bit elements in the same layout ldmatrix
  // produces: lane L holds row L/4, columns 2*(L%4) and 2*(L%4)+1, packed into
  // one 32-bit register. Transposing it is a matter of re-gathering, since
  // every element already lives somewhere in the warp.
  // The addend half of mad.{lo,hi}.cc / madc: the multiply has already been
  // reduced to one value per lane, and only its addition with c touches the
  // carry bit. The product's own overflow is discarded -- ".lo" and ".hi" have
  // each already chosen which half of it survives.
  void exec_carry_mad(Warp& w, uint32_t bits, bool carry_in, bool carry_out, const Lanes& prod,
                      const Lanes& c, Mask m, Lanes& r) {
    Mask out = w.carry;
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      const uint64_t p = mask_to_bits(prod[lane], bits);
      const uint64_t addend = mask_to_bits(c[lane], bits);
      const uint64_t cin = carry_in ? ((w.carry >> lane) & 1u) : 0u;
      const uint64_t sum = mask_to_bits(p + addend + cin, bits);
      r[lane] = sum;
      if (carry_out) {
        const bool cout = (sum < p) || (cin && sum == p);
        out = (out & ~(Mask{1} << lane)) | (Mask{cout} << lane);
      }
    }
    if (carry_out) w.carry = out;
  }

  // add/sub with the condition-code carry bit. PTX defines subtraction's carry
  // as the carry-out of (a + ~b + 1), so a borrow clears the bit rather than
  // setting it -- which is what makes "sub.cc" then "subc" chain correctly into
  // a wider subtract.
  void exec_carry_add_sub(Warp& w, const OpIntBin& op, const Lanes& a, const Lanes& b, Mask m,
                          Lanes& r) {
    const uint32_t bits = op.ty.bits;
    const bool sub = (op.op == IntBinOp::Sub);
    Mask out = w.carry;
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      const uint64_t x = mask_to_bits(a[lane], bits);
      // For a subtract the carry-in defaults to 1 (the +1 of two's complement);
      // subc supplies the previous carry-out in its place.
      const uint64_t cin =
          op.carry_in ? ((w.carry >> lane) & 1u) : (sub ? 1u : 0u);
      const uint64_t y = sub ? mask_to_bits(~b[lane], bits) : mask_to_bits(b[lane], bits);
      const uint64_t sum = mask_to_bits(x + y + cin, bits);
      r[lane] = sum;
      if (op.carry_out) {
        // Unsigned overflow: the sum wrapped below either addend, or landed
        // exactly on one because the carry-in pushed it around.
        const bool cout = (sum < x) || (cin && sum == x);
        out = (out & ~(Mask{1} << lane)) | (Mask{cout} << lane);
      }
    }
    if (op.carry_out) w.carry = out;
  }

  void exec_movmatrix(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpMovMatrix& op,
                      Mask m) {
    Lanes _s_a;
    const Lanes& a = read_operand(w, ctx, ins, op.src, _s_a);
    uint16_t tile[8][8] = {};
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      // .aligned means the whole warp executes this together; a partial warp
      // would be reading registers no lane wrote.
      if (!(m & (1u << lane)))
        ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                 "movmatrix is warp-aligned: every lane must be active, because the "
                 "matrix is spread across all 32 of them");
      const uint32_t row = lane / 4, colpair = lane % 4;
      tile[row][colpair * 2] = static_cast<uint16_t>(a[lane] & 0xFFFFu);
      tile[row][colpair * 2 + 1] = static_cast<uint16_t>((a[lane] >> 16) & 0xFFFFu);
    }
    Lanes r{};
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      const uint32_t row = lane / 4, colpair = lane % 4;
      // d[row][c] = a[c][row]
      const uint16_t lo = tile[colpair * 2][row];
      const uint16_t hi = tile[colpair * 2 + 1][row];
      r[lane] = (static_cast<uint64_t>(hi) << 16) | lo;
    }
    write_reg(w, op.dst, m, r, 32);
  }

  // Decodes one element of an mma A/B fragment from a lane's register.
  double mma_elem(MmaElem t, bool is_signed, uint64_t reg, uint32_t slot) {
    switch (t) {
      case MmaElem::F16: return f16_to_double((reg >> (16 * slot)) & 0xFFFF);
      case MmaElem::BF16: return bf16_to_double((reg >> (16 * slot)) & 0xFFFF);
      case MmaElem::TF32:
        // tf32 occupies a full 32-bit register; its reduced mantissa is a
        // hardware precision detail, and computing exactly stays inside it.
        return static_cast<double>(f32(reg));
      case MmaElem::S8:
      case MmaElem::U8: {
        const uint8_t byte = static_cast<uint8_t>(reg >> (8 * slot));
        return is_signed ? static_cast<double>(static_cast<int8_t>(byte))
                         : static_cast<double>(byte);
      }
    }
    return 0.0;
  }

  void exec_mma(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpMma& op, Mask m) {
    constexpr uint32_t kM = 16, kN = 8;
    const uint32_t K = op.k;
    const bool sixteen_bit = op.ab_type == MmaElem::F16 || op.ab_type == MmaElem::BF16;
    const bool eight_bit = op.ab_type == MmaElem::S8 || op.ab_type == MmaElem::U8;
    // Elements each lane holds per register: 2 for 16-bit, 4 for 8-bit, 1 for
    // tf32. The register counts follow from the shape.
    const uint32_t per_reg = sixteen_bit ? 2u : (eight_bit ? 4u : 1u);
    const uint32_t a_regs = (kM * K) / (kWarpSize * per_reg);
    const uint32_t b_regs = (K * kN) / (kWarpSize * per_reg);
    if (op.a.size() != a_regs || op.b.size() != b_regs)
      ctx_fail(ins, -1, Err::UnsupportedPtx, "mma fragment arity does not match the shape");

    std::vector<double> A(kM * K, 0.0), B(K * kN, 0.0), C(kM * kN, 0.0);
    // A: lane (groupID, tid) holds rows {groupID, groupID+8} at the column
    // block the register index selects.
    for (uint32_t reg = 0; reg < a_regs; ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.a[reg]}}, _s);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        const uint32_t group = lane / 4, tid = lane % 4;
        const uint32_t row = group + (reg % 2) * 8;
        const uint32_t col0 = tid * per_reg + (reg / 2) * (per_reg * 4);
        for (uint32_t e = 0; e < per_reg; ++e)
          A[row * K + col0 + e] = mma_elem(op.ab_type, op.ab_signed, v[lane], e);
      }
    }
    // B is K x N: the lane's group selects the column, the register and tid
    // select the rows.
    for (uint32_t reg = 0; reg < b_regs; ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.b[reg]}}, _s);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        const uint32_t group = lane / 4, tid = lane % 4;
        const uint32_t col = group;
        const uint32_t row0 = tid * per_reg + reg * (per_reg * 4);
        for (uint32_t e = 0; e < per_reg; ++e)
          B[(row0 + e) * kN + col] = mma_elem(op.ab_type, op.ab_signed, v[lane], e);
      }
    }
    // C/D: four values per lane, two rows by two columns.
    auto cd_index = [](uint32_t lane, uint32_t slot) {
      const uint32_t group = lane / 4, tid = lane % 4;
      const uint32_t row = group + (slot / 2) * 8;
      const uint32_t col = tid * 2 + (slot % 2);
      return row * kN + col;
    };
    for (uint32_t reg = 0; reg < op.c.size(); ++reg) {
      Lanes _s;
      const Lanes& v = read_operand(w, ctx, ins, Operand{RegOperand{op.c[reg]}}, _s);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        if (op.acc_f16) {
          // Two halves per register.
          C[cd_index(lane, reg * 2)] = f16_to_double(v[lane] & 0xFFFF);
          C[cd_index(lane, reg * 2 + 1)] = f16_to_double((v[lane] >> 16) & 0xFFFF);
        } else if (op.acc_int) {
          C[cd_index(lane, reg)] = static_cast<double>(static_cast<int32_t>(v[lane]));
        } else {
          C[cd_index(lane, reg)] = static_cast<double>(f32(v[lane]));
        }
      }
    }
    // D = A x B + C. Float shapes accumulate in f32 and integer shapes in s32,
    // matching the accumulate type the instruction names.
    std::vector<double> D(kM * kN, 0.0);
    for (uint32_t i = 0; i < kM; ++i)
      for (uint32_t j = 0; j < kN; ++j) {
        if (op.acc_int) {
          int64_t acc = static_cast<int64_t>(C[i * kN + j]);
          for (uint32_t k = 0; k < K; ++k)
            acc += static_cast<int64_t>(A[i * K + k]) * static_cast<int64_t>(B[k * kN + j]);
          D[i * kN + j] = static_cast<double>(static_cast<int32_t>(acc));
        } else {
          float acc = static_cast<float>(C[i * kN + j]);
          for (uint32_t k = 0; k < K; ++k)
            acc += static_cast<float>(A[i * K + k]) * static_cast<float>(B[k * kN + j]);
          D[i * kN + j] = static_cast<double>(acc);
        }
      }
    for (uint32_t reg = 0; reg < op.d.size(); ++reg) {
      Lanes r;
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          if (op.acc_f16) {
            const uint64_t lo = double_to_f16(D[cd_index(lane, reg * 2)]);
            const uint64_t hi = double_to_f16(D[cd_index(lane, reg * 2 + 1)]);
            r[lane] = ((hi & 0xFFFF) << 16) | (lo & 0xFFFF);
          } else if (op.acc_int) {
            r[lane] = static_cast<uint32_t>(static_cast<int32_t>(D[cd_index(lane, reg)]));
          } else {
            r[lane] = f32bits(static_cast<float>(D[cd_index(lane, reg)]));
          }
        }
      write_reg(w, op.d[reg], m, r, 32);
    }
  }

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
      Lanes r;  // written for every active lane below
      for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
        uint32_t linear = lane * 8 + static_cast<uint32_t>(reg);
        r[lane] = f32bits(D[linear / kMmaDim][linear % kMmaDim]);
      }
      write_reg(w, op.d[reg], m, r, 32);
    }
  }

  void exec_wmma_store(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpWmmaStore& op,
                       Mask m) {
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
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
      case Round::Rmi: [[fallthrough]];
      case Round::Rm: return std::floor(x);
      case Round::Rpi: [[fallthrough]];
      case Round::Rp: return std::ceil(x);
      case Round::Rni: [[fallthrough]];
      default: return std::nearbyint(x);  // round-to-nearest-even
    }
  }

  // bfloat16 is the top 16 bits of an f32: same exponent, mantissa truncated to
  // 7 bits. Converting in rounds to nearest even on the discarded half, which is
  // what cvt.rn asks for; converting out is exact.
  static double bf16_to_double(uint64_t in) {
    const uint32_t bits = static_cast<uint32_t>(in & 0xffffu) << 16;
    return static_cast<double>(std::bit_cast<float>(bits));
  }
  static uint64_t double_to_bf16(double x) {
    const float f = static_cast<float>(x);
    const uint32_t bits = std::bit_cast<uint32_t>(f);
    if (std::isnan(f)) return (bits >> 16) | 0x0040u;  // keep it quiet
    // Round to nearest, ties to even, on the 16 bits being dropped.
    const uint32_t lsb = (bits >> 16) & 1u;
    const uint32_t rounded = bits + 0x7fffu + lsb;
    return rounded >> 16;
  }

  uint64_t convert(const OpCvt* op, uint64_t in) {
    const Type& s = op->src_ty;
    const Type& d = op->dst_ty;
    // Read the source as a real number (float src) or integer (int src).
    if (s.is_real()) {
      double x = s.is_bfloat()  ? bf16_to_double(in)
                 : s.bits == 16 ? f16_to_double(in)
                 : s.bits == 32 ? static_cast<double>(f32(in))
                                : f64(in);
      if (d.is_real()) {
        // An "i" rounding mode rounds to an integral value and keeps the float
        // type: cvt.rpi.f32.f32 is ceilf. Skipping it here made ceilf, floorf
        // and truncf return their argument, and roundf return x + 0.5.
        switch (op->round) {
          case Round::Rni:
          case Round::Rzi:
          case Round::Rmi:
          case Round::Rpi:
            x = round_int(x, op->round);
            break;
          default:
            break;
        }
        if (d.is_bfloat()) return double_to_bf16(x);
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
    if (d.is_real()) {
      double x = s.is_signed() ? static_cast<double>(static_cast<int64_t>(sv))
                               : static_cast<double>(sv);
      if (d.is_bfloat()) return double_to_bf16(x);
      if (d.bits == 16) return double_to_f16(x);
      return d.bits == 32 ? f32bits(static_cast<float>(x)) : f64bits(x);
    }
    return mask_to_bits(sv, d.bits);  // int -> int: truncate/extend
  }

  // Registers are 32 or 64 bits wide, but an operand type can be narrower. The
  // bits above the operand's width belong to whatever was in the register
  // before and must take no part: shr.u16 of a register holding 0xFFFFFFFF has
  // to shift 0xFFFF, and shr.s16 has to take its sign from bit 15, not bit 31.
  // Treating every non-64-bit type as 32-bit got both wrong, which is why the
  // i-quant kernels -- almost entirely 16-bit bit manipulation -- produced
  // wrong values while the 32-bit paths around them were fine.
  static uint64_t narrow_u(uint64_t v, uint32_t bits) {
    return bits >= 64 ? v : (v & ((1ull << bits) - 1));
  }
  static int64_t narrow_s(uint64_t v, uint32_t bits) {
    if (bits >= 64) return static_cast<int64_t>(v);
    const uint64_t mask = (1ull << bits) - 1;
    uint64_t m = v & mask;
    const uint64_t sign = 1ull << (bits - 1);
    if (m & sign) m |= ~mask;
    return static_cast<int64_t>(m);
  }

  // Strict mode: the checks that hunt for bugs hardware would hide, but which
  // real compiler output trips over. Integer division by zero and a store of a
  // register nothing has written are both undefined-but-harmless in code ptxas
  // and CUB emit every day, so neither can be on by default -- and both are
  // worth having when you are looking for a bug rather than running a workload.
  static bool strict() { return g_strict.load(std::memory_order_relaxed); }

  uint64_t int_bin(IntBinOp op, Type ty, uint64_t a, uint64_t b, const Instr& ins) {
    bool sig = ty.is_signed();
    auto s = [&](uint64_t v) -> int64_t { return narrow_s(v, ty.bits); };
    auto u = [&](uint64_t v) -> uint64_t { return narrow_u(v, ty.bits); };
    switch (op) {
      case IntBinOp::Add: return a + b;
      case IntBinOp::Sub: return a - b;
      case IntBinOp::Mul: return a * b;
      case IntBinOp::Min: return sig ? static_cast<uint64_t>(std::min(s(a), s(b))) : std::min(u(a), u(b));
      case IntBinOp::Max: return sig ? static_cast<uint64_t>(std::max(s(a), s(b))) : std::max(u(a), u(b));
      case IntBinOp::Div:
      case IntBinOp::Rem: {
        if (u(b) == 0) {
          // PTX leaves this undefined and real GPUs do not trap. VirtualGPU
          // used to, on the reasoning that a div-by-zero is almost always a
          // bug -- but real code falsifies that: ggml's flash-attention passes
          // zero for a stride its configuration does not use, computes a
          // remainder from it, and discards the answer. Trapping made those
          // kernels unrunnable over arithmetic that was never going to matter.
          //
          // So the default now follows the hardware, deterministically: a
          // quotient of all-ones and a remainder of the dividend, which is what
          // the usual expansion of these instructions leaves behind. The trap
          // is still available for a debugging run, since it does find real
          // bugs -- it just cannot be the default.
          if (strict())
            ctx_fail(ins, -1, Err::InvalidValue, "integer division by zero");
          return op == IntBinOp::Div ? narrow_u(~uint64_t{0}, ty.bits) : u(a);
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
    // Same narrowing as the arithmetic: setp.gt.s16 compares 16-bit values, so
    // the bits above them must not decide the result.
    if (ty.is_signed()) return do_cmp(narrow_s(a, ty.bits), narrow_s(b, ty.bits));
    return do_cmp(narrow_u(a, ty.bits), narrow_u(b, ty.bits));
  }

  // ---- memory ops ----

  // Resolves an address operand's base. Registers alias the register file
  // directly; a named variable resolves through the symbol table into
  // `scratch`.
  const Lanes& addr_base(Warp& w, const BlockCtx& ctx, const Instr& ins, const Addr& a,
                         Lanes& scratch) {
    (void)ctx;
    if (a.base_kind == Addr::Base::Symbol) {
      scratch.fill(resolve_symbol(ins, a.base));
      return scratch;
    }
    // Address registers are .b64 in 64-bit PTX, but shared/local addressing
    // legitimately uses 32-bit registers, so honor whichever file it is in.
    if (a.base_wide) {
      if (a.base_id >= w.regs64.size() || !w.written64[a.base_id])
        ctx_fail(ins, -1, Err::UninitializedRegister,
                 "address register " + a.base + " read before any write");
      return w.regs64[a.base_id];
    }
    if (a.base_id >= w.regs32.size() || !w.written32[a.base_id])
      ctx_fail(ins, -1, Err::UninitializedRegister,
               "address register " + a.base + " read before any write");
    return widen(w, w.regs32[a.base_id]);
  }

  // ---- cp.async ----

  // Issue: read the source now, hold the bytes, and leave the destination
  // untouched until the thread waits for the group this copy lands in.
  void exec_cp_async(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpCpAsync& op, Mask m) {
    if (!w.cp) w.cp = std::make_unique<AsyncCopies>();
    Lanes _s_dst, _s_src;
    const Lanes& dstb = addr_base(w, ctx, ins, op.dst, _s_dst);
    const Lanes& srcb = addr_base(w, ctx, ins, op.src, _s_src);
    Lanes _s_size;
    const Lanes* size_lanes = nullptr;
    if (op.have_src_size) size_lanes = &read_operand(w, ctx, ins, op.src_size, _s_size);

    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      PendingCopy pc;
      pc.bytes = op.bytes;
      // The destination is a shared-space address; the source a global one.
      pc.dst = kSharedVaBase + dstb[lane] + static_cast<uint64_t>(op.dst.offset);
      const uint64_t src = srcb[lane] + static_cast<uint64_t>(op.src.offset);
      // Bytes past src-size are zero-filled rather than read, which is how a
      // tile that runs off the end of a tensor is handled without a branch.
      uint64_t readable = op.bytes;
      if (size_lanes) readable = std::min<uint64_t>((*size_lanes)[lane], op.bytes);
      // Fault now if the destination could not take the write: the instruction
      // that named the address is far more useful to report than the wait that
      // happens to drain it.
      check_shared(ctx, ins, static_cast<int>(lane), pc.dst, op.bytes);
      for (uint64_t off = 0; off < readable;) {
        const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(8, readable - off));
        const uint64_t v = load_routed(w, ctx, ins, lane, src + off, chunk);
        std::memcpy(pc.data.data() + off, &v, chunk);
        off += chunk;
      }
      w.cp->open[lane].push_back(pc);
    }
    count_memory(Space::Global, op.bytes, popcount_mask(m), /*is_store=*/false);
  }

  // Land one group's copies in shared memory.
  void complete_group(Warp& w, const BlockCtx& ctx, const Instr& ins, uint32_t lane,
                      const std::vector<PendingCopy>& group) {
    (void)w;
    for (const PendingCopy& pc : group) {
      check_shared(ctx, ins, static_cast<int>(lane), pc.dst, pc.bytes);
      std::memcpy(ctx.shared->data() + (pc.dst - kSharedVaBase), pc.data.data(), pc.bytes);
      // The shared write lands here, not where the copy was issued, so this is
      // where it is counted.
      count_memory(Space::Shared, pc.bytes, 1, /*is_store=*/true);
    }
  }

  void exec_cp_async_group(Warp& w, const BlockCtx& ctx, const Instr& ins,
                           const OpCpAsyncGroup& op, Mask m) {
    if (!w.cp) {
      // Waiting with nothing outstanding is legal and common -- a loop's first
      // iteration waits before it has issued anything.
      return;
    }
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      if (!(m & (1u << lane))) continue;
      auto& open = w.cp->open[lane];
      auto& groups = w.cp->groups[lane];
      // wait_all is defined as commit_group followed by wait_group 0, so both
      // it and commit close whatever is open first.
      if (op.kind != OpCpAsyncGroup::Kind::WaitGroup) {
        if (!open.empty()) {
          groups.push_back(std::move(open));
          open.clear();
        }
      }
      if (op.kind == OpCpAsyncGroup::Kind::Commit) continue;
      const size_t keep = op.kind == OpCpAsyncGroup::Kind::WaitAll ? 0 : op.keep;
      while (groups.size() > keep) {
        complete_group(w, ctx, ins, lane, groups.front());
        groups.pop_front();
      }
    }
  }

  // A thread that ends with copies still in flight still performs them: the
  // hardware does not cancel an issued copy because the thread finished, and
  // another warp in the block may yet read what it wrote.
  void drain_async_copies(Warp& w, const BlockCtx& ctx, const Instr& ins) {
    if (!w.cp) return;
    for (uint32_t lane = 0; lane < kWarpSize; ++lane) {
      for (auto& g : w.cp->groups[lane]) complete_group(w, ctx, ins, lane, g);
      w.cp->groups[lane].clear();
      if (!w.cp->open[lane].empty()) {
        complete_group(w, ctx, ins, lane, w.cp->open[lane]);
        w.cp->open[lane].clear();
      }
    }
  }

  void exec_ld(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpLd& op, Mask m) {
    uint32_t size = op.ty.bytes();
    size_t n = op.dsts.size();
    if (op.space == Space::Param && op.addr.base_kind == Addr::Base::Reg) {
      // Address form: the register holds a parameter-window address.
      Lanes _s_base;
      const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
      std::vector<Lanes> results(n);
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) {
          const uint64_t addr =
              base[lane] + static_cast<uint64_t>(op.addr.offset);
          for (size_t e = 0; e < n; ++e) {
            const uint64_t at = addr + e * size;
            if (at < kParamVaBase || at + size > kParamVaBase + params_.bytes.size())
              ctx_fail(ins, static_cast<int>(lane), Err::OutOfBounds,
                       "ld.param through a register reads outside the parameter buffer");
            uint64_t v = 0;
            std::memcpy(&v, params_.bytes.data() + (at - kParamVaBase), size);
            results[e][lane] = v;
          }
        }
      for (size_t e = 0; e < n; ++e) write_reg(w, op.dsts[e], m, results[e], op.ty.bits);
      return;
    }
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
        Lanes r;  // written for every active lane below
        r.fill(v);
        write_reg(w, op.dsts[e], m, r, op.ty.bits);
      }
      return;
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    uint64_t sbase = space_base(op.space);
    uint32_t va = size * static_cast<uint32_t>(n);  // vector accesses need vector alignment
    {
      // Sectors and bank conflicts need the addresses, which only exist here.
      Lanes at{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane))
          at[lane] = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      count_addresses(op.space, at, m, size, n);
    }
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
    // A signed narrow load sign-extends into the destination register: PTX says
    // ld.s8 delivers the byte's value, not its bit pattern. Masking to the type
    // width instead turns -1 into 255, and the cvt that follows reads the
    // positive number -- which is how a quantized weight of -1 became +255 and
    // corrupted every dequantized tensor while still looking like a plain copy.
    for (size_t e = 0; e < n; ++e) {
      const uint32_t dst_bits = op.dsts[e].wide ? 64u : 32u;
      if (op.ty.is_signed() && op.ty.bits < dst_bits) {
        Lanes ext = results[e];
        const uint64_t sign_bit = 1ull << (op.ty.bits - 1);
        const uint64_t value_mask = (sign_bit << 1) - 1;
        for (uint32_t lane = 0; lane < kWarpSize; ++lane)
          if (m & (1u << lane)) {
            uint64_t v = ext[lane] & value_mask;
            if (v & sign_bit) v |= ~value_mask;
            ext[lane] = v;
          }
        write_reg(w, op.dsts[e], m, ext, dst_bits);
      } else {
        write_reg(w, op.dsts[e], m, results[e], op.ty.bits);
      }
    }
  }

  void exec_st(Warp& w, const BlockCtx& ctx, const Instr& ins, const OpSt& op, Mask m) {
    uint32_t size = op.ty.bytes();
    size_t n = op.srcs.size();
    std::vector<Lanes> vals;
    vals.reserve(n);
    for (const auto& src : op.srcs) {
      // A value nothing has written, on its way to memory, looks like a bug --
      // but CUB's radix sort stores exactly that into the unused part of a
      // shared tile, and never reads it back. So this is a strict-mode check
      // rather than a default one; see strict().
      if (const auto* r = strict() ? std::get_if<RegOperand>(&src) : nullptr) {
        const bool init = r->reg.wide ? (r->reg.id < w.regs64.size() && w.written64[r->reg.id])
                                      : (r->reg.id < w.regs32.size() && w.written32[r->reg.id]);
        if (!init)
          ctx_fail(ins, -1, Err::UninitializedRegister,
                   "store writes register " + r->reg.name + ", which nothing has written");
      }
      Lanes tmp;
      vals.push_back(read_operand(w, ctx, ins, src, tmp));
    }
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    uint64_t sbase = space_base(op.space);
    uint32_t va = size * static_cast<uint32_t>(n);
    {
      Lanes at{};
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane))
          at[lane] = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
      count_addresses(op.space, at, m, size, n);
    }
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
    Lanes _s_base;
    const Lanes& base = addr_base(w, ctx, ins, op.addr, _s_base);
    uint64_t sbase = space_base(op.space);
    Lanes _s_bv;
      const Lanes& bv = read_operand(w, ctx, ins, op.b, _s_bv);
    Lanes cv;
    Lanes _s_cv;
    if (op.op == AtomOp::Cas) cv = read_operand(w, ctx, ins, op.c, _s_cv);
    Lanes r;  // written for every active lane below
    // A fixed lane order makes this atomic within a warp, and within a block,
    // because a block runs on one thread. Across blocks it does not: when the
    // grid is spread over several threads, two blocks can read-modify-write the
    // same global address at once and lose an update. The stripe lock closes
    // that, and is skipped entirely when the launch is single-threaded, where
    // the fixed order was already enough.
    const bool lock_needed = concurrent_ && op.space != Space::Shared && op.space != Space::Local;
    for (uint32_t lane = 0; lane < kWarpSize; ++lane)
      if (m & (1u << lane)) {
        uint64_t addr = sbase + base[lane] + static_cast<uint64_t>(op.addr.offset);
        std::unique_lock<std::mutex> guard;
        if (lock_needed)
          guard = std::unique_lock<std::mutex>(atomic_lock_for(addr));
        ++stats_.atomics;
        uint64_t old = load_routed(w, ctx, ins, lane, addr, size);
        uint64_t b = mask_to_bits(bv[lane], op.ty.bits);
        uint64_t nv = old;
        if (op.ty.is_float()) {
          // Reinterpret and operate in the float domain: adding the bit
          // patterns of two floats produces a number unrelated to their sum.
          // Min/max follow CUDA and use the fmin/fmax ordering rather than <,
          // so a NaN operand yields the other value.
          if (size == 4) {
            const float x = std::bit_cast<float>(static_cast<uint32_t>(old));
            const float y = std::bit_cast<float>(static_cast<uint32_t>(b));
            float res = x;
            switch (op.op) {
              case AtomOp::Add: res = x + y; break;
              case AtomOp::Exch: res = y; break;
              case AtomOp::Min: res = std::fmin(x, y); break;
              case AtomOp::Max: res = std::fmax(x, y); break;
              default:
                ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                         "this atomic operation has no float form");
            }
            nv = std::bit_cast<uint32_t>(res);
          } else {
            const double x = std::bit_cast<double>(old);
            const double y = std::bit_cast<double>(b);
            double res = x;
            switch (op.op) {
              case AtomOp::Add: res = x + y; break;
              case AtomOp::Exch: res = y; break;
              case AtomOp::Min: res = std::fmin(x, y); break;
              case AtomOp::Max: res = std::fmax(x, y); break;
              default:
                ctx_fail(ins, static_cast<int>(lane), Err::UnsupportedPtx,
                         "this atomic operation has no float form");
            }
            nv = std::bit_cast<uint64_t>(res);
          }
          store_routed(w, ctx, ins, lane, addr, size, mask_to_bits(nv, op.ty.bits));
          r[lane] = old;
          continue;
        }
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
      // One lock for the whole line: blocks run on several threads, and
      // interleaving two device printfs mid-line makes both unreadable.
      {
        static std::mutex printf_mu;
        std::lock_guard<std::mutex> guard(printf_mu);
        std::fwrite(out.data(), 1, out.size(), stdout);
        std::fflush(stdout);
      }
      counts[lane] = out.size();
    }
    if (!op.retval_slot.empty()) {
      Lanes& slot = w.slots[op.retval_slot];
      for (uint32_t lane = 0; lane < kWarpSize; ++lane)
        if (m & (1u << lane)) slot[lane] = counts[lane];
    }
  }

  // Publishes elapsed busy time mid-launch so telemetry stays live during a
  // long kernel instead of freezing until the launch returns.
  void report_progress() {
    auto now = std::chrono::steady_clock::now();
    double dt = std::chrono::duration<double>(now - last_progress_).count();
    last_progress_ = now;
    progress_(dt);
  }

  // Striped locks for device-side atomics when the grid is running on more
  // than one thread. Sized well above the core count so unrelated addresses
  // rarely collide.
  static std::mutex& atomic_lock_for(uint64_t addr) {
    static std::array<std::mutex, 251> locks;
    return locks[(addr >> 2) % locks.size()];
  }

  const EntryFn& fn_;
  const LaunchConfig& cfg_;
  const ParamBuffer& params_;
  MemoryManager& mem_;
  const DeviceProfile& profile_;
  const SymbolTable* symbols_;
  LaunchStats& stats_;
  ProgressFn progress_;
  // One entry per instruction, filled on first use. Classifying costs a chain
  // of variant tests, and the instruction stream is the hottest path there is.
  mutable std::vector<uint8_t> class_by_pc_;
  uint32_t cur_warp_ = 0;     // which warp of the block is running, for race reports
  bool concurrent_ = false;   // set when the grid is split across threads
  std::chrono::steady_clock::time_point last_progress_;
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

}  // namespace

KernelResources kernel_resources(const EntryFn& fn, const DeviceProfile& profile,
                                 uint32_t block_threads, uint32_t dynamic_shared) {
  // The register analysis depends only on the kernel, so memoize it on the
  // kernel itself. Occupancy also depends on the launch shape and is cheap.
  ptx::RegisterUsage usage;
  if (fn.regs_analyzed) {
    usage.regs_per_thread = fn.cached_regs_per_thread;
    usage.pred_regs = fn.cached_pred_regs;
    usage.peak_live = fn.cached_peak_live;
    usage.local_bytes = fn.local_frame_size;
  } else {
    usage = ptx::analyze_registers(fn);
    fn.cached_regs_per_thread = usage.regs_per_thread;
    fn.cached_pred_regs = usage.pred_regs;
    fn.cached_peak_live = usage.peak_live;
    fn.regs_analyzed = true;
  }
  // A thread cannot occupy more architectural registers than the ISA has. When
  // the data flow needs more, ptxas does not give up -- it spills the excess to
  // local memory and the kernel runs, slower. Treating the analysis figure as a
  // hard requirement instead turned a register-hungry but perfectly legal
  // kernel into a launch that could never happen: ggml's flash-attention
  // kernels want 272 by this measure, and refusing them meant the whole
  // attention path was unreachable.
  //
  // The interpreter has no architectural register file to run out of, so this
  // only corrects what is *reported* -- the occupancy figure and the launch
  // decision, which are exactly the things the number exists to inform.
  const uint32_t arch_max = profile.limits.max_registers_per_thread;
  if (arch_max && usage.regs_per_thread > arch_max) {
    const uint32_t spilled = usage.regs_per_thread - arch_max;
    usage.spilled_regs = spilled;
    usage.regs_per_thread = arch_max;
    usage.local_bytes += spilled * 4u;  // a spilled 32-bit register
  }
  KernelResources r;
  r.usage = usage;
  r.occupancy = ptx::compute_occupancy(
      usage.regs_per_thread, block_threads, fn.static_shared_size, dynamic_shared,
      profile.limits.registers_per_sm, profile.limits.max_threads_per_sm,
      profile.limits.max_blocks_per_sm, profile.limits.shared_mem_per_sm, profile.warp_size);
  return r;
}

namespace {

void validate(const EntryFn& fn, const LaunchConfig& cfg, const DeviceProfile& p) {
  if (p.warp_size != kWarpSize)
    throw Error::make(Err::Unsupported, "profile ", p.id, " has warp size ", p.warp_size,
                      "; only 32 is implemented");
  uint64_t threads = 1;
  for (int i = 0; i < 3; ++i) {
    if (cfg.block[i] == 0 || cfg.grid[i] == 0)
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "': grid/block dimensions must be "
                        ">= 1, got grid ", cfg.grid[0], "x", cfg.grid[1], "x", cfg.grid[2],
                        " block ", cfg.block[0], "x", cfg.block[1], "x", cfg.block[2]);
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
  // Launch bounds declared by the kernel (__launch_bounds__). Hardware
  // refuses a block larger than the kernel was compiled for.
  for (int i = 0; i < 3; ++i) {
    if (fn.req_ntid[i] && cfg.block[i] != fn.req_ntid[i])
      throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' requires exactly ",
                        fn.req_ntid[0], "x", fn.req_ntid[1], "x", fn.req_ntid[2],
                        " threads per block (.reqntid), launch asked for ", cfg.block[0], "x",
                        cfg.block[1], "x", cfg.block[2]);
  }
  // .maxntid bounds the *product* of the block dimensions, not each one
  // separately: __launch_bounds__(128) emits ".maxntid 128, 1, 1", and a launch
  // of 32x4x1 is 128 threads, which hardware accepts. Comparing dimension by
  // dimension rejected shapes that are within the bound -- every block whose y
  // extent was greater than 1, which is most of them.
  const uint64_t max_ntid_total = uint64_t{fn.max_ntid[0]} * fn.max_ntid[1] * fn.max_ntid[2];
  const uint64_t block_total = uint64_t{cfg.block[0]} * cfg.block[1] * cfg.block[2];
  if (max_ntid_total && block_total > max_ntid_total)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' was compiled for at most ",
                      max_ntid_total, " threads per block (__launch_bounds__), launch asked for ",
                      cfg.block[0], "x", cfg.block[1], "x", cfg.block[2], " = ", block_total);

  // Register budget. A block whose threads collectively need more registers
  // than the device provides cannot be launched -- the same
  // "too many resources requested for launch" a real driver reports.
  KernelResources res = kernel_resources(fn, p, static_cast<uint32_t>(threads),
                                         cfg.shared_bytes);
  uint64_t block_regs = uint64_t{res.usage.regs_per_thread} * threads;
  if (block_regs > p.limits.registers_per_block)
    throw Error::make(Err::LaunchConfig, "too many resources requested for launch: kernel '",
                      fn.name, "' uses ", res.usage.regs_per_thread, " registers per thread and ",
                      threads, " threads per block (", block_regs,
                      " registers), but profile ", p.id, " allows ", p.limits.registers_per_block,
                      " per block. Use a smaller block or fewer registers.");
  if (res.occupancy.blocks_per_sm == 0)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name,
                      "' cannot place a single block on a multiprocessor of ", p.id,
                      " (limited by ", res.occupancy.limited_by, ")");

  uint64_t total_shared = uint64_t{fn.static_shared_size} + cfg.shared_bytes;
  uint32_t shared_limit = std::max(p.limits.shared_mem_per_block, p.limits.shared_mem_per_block_optin);
  if (total_shared > shared_limit)
    throw Error::make(Err::LaunchConfig, "kernel '", fn.name, "' needs ", total_shared,
                      " bytes of shared memory (", fn.static_shared_size, " static + ",
                      cfg.shared_bytes, " dynamic); profile ", p.id, " allows at most ", shared_limit);
}

}  // namespace

namespace {

// How many host threads to spread the grid over. One thread reproduces the old
// strictly serial block order exactly, which is what a kernel with a data race
// needs to stay reproducible; more threads is faster and is what CUDA's own
// model already allows, since blocks may run in any order and concurrently.
unsigned worker_count(uint64_t blocks) {
  unsigned want = 0;
  if (const char* t = std::getenv("VGPU_THREADS")) {
    const int v = std::atoi(t);
    want = v > 0 ? static_cast<unsigned>(v) : 1;
  } else {
    want = std::thread::hardware_concurrency();
    if (want == 0) want = 1;
  }
  if (want > blocks) want = static_cast<unsigned>(blocks);
  return want ? want : 1;
}

// The step budget is a guard against a kernel that loops forever. A kernel that
// is legitimately long -- a soak or bake kernel that runs for minutes on
// hardware -- trips it too, and the interpreter is slow enough that this is not
// rare. VGPU_MAX_STEPS raises it without a rebuild; 0 turns the guard off, for
// when you know the kernel terminates and only need it to finish.
uint64_t effective_max_steps(uint64_t configured) {
  const char* e = std::getenv("VGPU_MAX_STEPS");
  if (!e || !*e) return configured;
  // strtoull accepts a leading sign and wraps a negative around, so "-5" would
  // parse as an enormous budget and quietly remove the very guard this is
  // protecting. Reject a sign before parsing.
  const char* p = e;
  while (*p == ' ' || *p == '\t') ++p;
  if (*p == '-' || *p == '+') return configured;
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(e, &end, 10);
  // Anything that is not a whole number leaves the guard as configured rather
  // than silently disabling it.
  if (errno != 0 || end == e || *end != '\0') return configured;
  return v == 0 ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(v);
}

}  // namespace

LaunchStats launch(const EntryFn& fn, const LaunchConfig& cfg,
                   const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                   const DeviceProfile& profile, const SymbolTable* symbols,
                   const ProgressFn& progress) {
  refresh_modes();
  validate(fn, cfg, profile);
  LaunchConfig eff = cfg;
  eff.max_steps = effective_max_steps(cfg.max_steps);

  // A cooperative launch needs the grid-barrier workspace the driver would
  // reserve on hardware: cg::this_grid().sync() finds it through %envreg1 and
  // %envreg2, and the barrier counter it spins on lives a few bytes in. It must
  // start zeroed, because the counter's sign is what the barrier reads.
  //
  // Freed on the way out however that happens -- a kernel that faults must not
  // leak a buffer the caller never asked for.
  struct CoopWorkspace {
    MemoryManager& mem;
    uint64_t addr = 0;
    ~CoopWorkspace() { if (addr) mem.free(addr); }
  } coop{mem};
  if (eff.cooperative) {
    constexpr uint64_t kCoopWorkspaceBytes = 64;
    coop.addr = mem.alloc(kCoopWorkspaceBytes);
    for (uint64_t i = 0; i < kCoopWorkspaceBytes; i += 8) mem.store_scalar(coop.addr + i, 8, 0);
    eff.coop_workspace = coop.addr;
  }
  ParamBuffer pb = build_params(fn, args);
  const uint64_t blocks = uint64_t{cfg.grid[0]} * cfg.grid[1] * cfg.grid[2];
  // A cooperative launch runs on one worker whatever VGPU_THREADS says. Its
  // blocks wait on each other, so they cannot be split into independent ranges
  // -- that is precisely the promise a cooperative launch does not make.
  const unsigned nthreads = eff.cooperative ? 1 : worker_count(blocks);

  if (nthreads <= 1) {
    LaunchStats stats;
    Interpreter interp(fn, eff, pb, mem, profile, symbols, stats, progress);
    interp.run_grid();
    return stats;
  }

  // Each worker gets its own interpreter and its own statistics; the only thing
  // they share is device memory, whose chunk table is lock-free for exactly
  // this. Progress reporting stays on one worker so the callback is never
  // re-entered.
  std::vector<LaunchStats> per_thread(nthreads);
  std::vector<std::thread> workers;
  std::mutex err_mu;
  std::exception_ptr first_error;
  workers.reserve(nthreads);
  for (unsigned t = 0; t < nthreads; ++t) {
    const uint64_t begin = blocks * t / nthreads;
    const uint64_t end = blocks * (t + 1) / nthreads;
    workers.emplace_back([&, t, begin, end] {
      try {
        Interpreter interp(fn, eff, pb, mem, profile, symbols, per_thread[t],
                           t == 0 ? progress : ProgressFn{});
        interp.set_concurrent(true);
        interp.run_block_range(begin, end);
      } catch (...) {
        std::lock_guard<std::mutex> lock(err_mu);
        if (!first_error) first_error = std::current_exception();
      }
    });
  }
  for (auto& w : workers) w.join();
  if (first_error) std::rethrow_exception(first_error);

  LaunchStats total;
  for (const auto& s : per_thread) total.add(s);
  return total;
}

}  // namespace vgpu::exec
