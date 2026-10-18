// The SASS executor: runs a kernel's machine code over a grid on the CPU.
//
// A warp is 32 lanes with the architectural state of sm_70-and-later GPUs:
// R0-R254 per lane (RZ reads zero), P0-P6 (PT is true), the uniform
// registers and predicates one per warp, sixteen convergence barriers, and a
// program counter per lane -- Volta's independent thread scheduling, where
// lanes that took different branches run separately until a BSYNC or
// WARPSYNC brings them together. Each step runs the group of lanes at the
// lowest address (so the side of a branch that falls through runs first and
// reaches the join), except that a group that keeps branching backwards while
// others wait gives way, so one lane spinning on another cannot starve it.
//
// Every instruction completes before the next: the control bits' stall counts
// and scoreboards are timing, which VirtualGPU does not model.
#include "vgpu/exec/devrt.hpp"
#include "vgpu/exec/lockstep.hpp"
#include "vgpu/exec/worker_pool.hpp"
#include "vgpu/sass/exec.hpp"
#include "tier.hpp"

#if defined(__SSE2__)
#include <emmintrin.h>
#endif
#include <algorithm>
#include <array>
#include <atomic>
#include <optional>
#include <utility>
#include <bit>
#include <bitset>
#include <cerrno>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <functional>
#include <map>
#include <unordered_map>
#include <mutex>
#include <thread>
#include <tuple>

#include "vgpu/error.hpp"
#include "vgpu/exec/device_printf.hpp"
#include "vgpu/exec/host_atomic.hpp"
#include "vgpu/exec/ldmatrix.hpp"
#include "vgpu/exec/mem_counters.hpp"
#include "vgpu/exec/window_check.hpp"
#include "vgpu/exec/numerics.hpp"
#include "vgpu/exec/tc_ampere.hpp"
#include "vgpu/exec/texture_grad.hpp"
#include "vgpu/exec/tma.hpp"
#include "vgpu/exec/tcgen05.hpp"
#include "vgpu/exec/wgmma.hpp"
#include "vgpu/simd/fp16.hpp"
#include "vgpu/simd/simd.hpp"
#if __has_include("vgpu/host_cpus.hpp")
#include "vgpu/host_cpus.hpp"
#define VGPU_HAVE_HOST_CPUS 1
#endif

namespace vgpu::sass {

namespace {

using Mask = uint32_t;
constexpr Mask kAll = 0xffffffffu;

float as_f32(uint32_t b) { float f; std::memcpy(&f, &b, 4); return f; }
uint32_t f32_bits(float f) { uint32_t b; std::memcpy(&b, &f, 4); return b; }
double as_f64(uint64_t b) { double d; std::memcpy(&d, &b, 8); return d; }
uint64_t f64_bits(double d) { uint64_t b; std::memcpy(&b, &d, 8); return b; }

// The number of set bits of a lane mask. std::popcount in this build (baseline
// x86-64, no POPCNT instruction) is a call into libgcc, and every instruction
// counts its active lanes: vgpu/simd/simd.hpp has the one implementation.
inline unsigned popc(uint32_t v) { return simd::popcount32(v); }

// Why a lane is not runnable.
enum class Wait : uint8_t { None, BSync, WarpSync, Bar, Cluster };

struct Warp {
  uint32_t index = 0;                 // within the block
  uint32_t nregs = 0;
  std::vector<uint32_t> r;            // r[reg * 32 + lane]
  Mask p[8] = {0, 0, 0, 0, 0, 0, 0, kAll};
  uint32_t ur[256] = {};
  bool up[8] = {false, false, false, false, false, false, false, true};
  uint64_t pc[32] = {};               // per-lane code address
  uint32_t device_error[32] = {};     // the device runtime's per-thread last error: a failed call sets it, a successful one leaves it
  Mask alive = 0;                     // lanes that are threads of the block
  Mask exited = 0;
  Mask waiting = 0;
  // Lanes whose WARPSYNC is satisfied but that wait for the lanes released
  // before them (named another mask) to run, so the two sets do not execute
  // together as one convergent group.
  Mask ws_held = 0;
  Wait wait_kind[32] = {};
  uint32_t wait_arg[32] = {};         // the barrier a lane waits on
  Mask b[16] = {};                    // convergence barriers
  // BMOV.PQUAD MACTIVE: the threads every instruction runs on until MACTIVE is put back (0: the issuing group).
  Mask mactive_force = 0;
  uint64_t bssy_stamp[16] = {};       // when (Block::bssy_clock) each barrier was last set up: higher is more deeply nested
  uint64_t rpc[32] = {};              // the return-address register (RPCMOV)
  uint64_t gmma_issued = 0;           // warpgroup MMAs this warp has issued
  const Code* code_hint = nullptr;    // the code section this warp last ran in (step_warp, and the tier)
  // The tier's chain (tier_ops.inc): how many more calls of step_warp the caller's turn has after this one, and how many the
  // tier has taken over. Set by the caller before each call; the caller adds `extra` to its count and clears it.
  uint32_t budget = 0, extra = 0;
  std::vector<uint8_t> local;         // per-lane local memory, local_size bytes each
  uint32_t local_size = 0;
  uint64_t steps = 0;
  // Fairness: how many backward branches the running group has taken while
  // other lanes waited to run.
  uint32_t spins = 0;
  bool give_way = false;
  uint32_t spin_yields = 0;   // YIELDs in a row with no other lane group able to run
  // YIELDs since the warp last wrote memory (a store, or an atomic that changed a word), with several lane groups
  // taking turns or not: a warp whose groups all spin on words only another warp's lanes can change is not
  // making progress either, and spin_yields alone never sees it.
  uint32_t idle_yields = 0;
  // The group that last gave way (a YIELD, or a backward branch taken often enough while other lanes waited) and
  // where it stands now: the next group to run is the one after it in address order, wrapping round.
  uint64_t yield_pc = 0;
  Mask slice = 0;             // the group let in by a YIELD: it runs until its own next YIELD or until it blocks
  uint32_t b2r = 0;                   // the last block barrier reduction's result (B2R.RESULT)
  bool b2r_pred = false;
  // Set by a wait that came back unsatisfied (an mbarrier phase check): the
  // warp's turn ends there, so the warps it waits on get to run.
  bool yield = false;
  // Armed until the warp reaches its first store to global memory, where its turn ends once (see
  // step_warp): by then it has issued every load it makes before storing anything.
  bool store_gate = true;
  // The cluster barrier (UCGABAR): lanes that have arrived in its current
  // phase, and the phase each lane's wait waits for.
  Mask cga_arrived = 0;
  uint64_t cga_target[32] = {};

  Mask runnable() const { return alive & ~exited & ~waiting; }
  uint32_t& reg(unsigned n, unsigned lane) {
    if (n >= nregs) [[unlikely]] past_file(n);
    return r[n * 32 + lane];
  }
  [[noreturn, gnu::cold, gnu::noinline]] void past_file(unsigned n) const {
    throw std::out_of_range("register R" + std::to_string(n) + " past the warp's " + std::to_string(nregs));
  }
};

// A warpgroup MMA's shared-memory operands, read by the first of its four
// warps (exec_gmma).
struct GmmaSnapshot {
  std::vector<double> A, B;
  uint8_t taken = 0;   // the warps that have used it
};

struct Block;

// The blocks of one thread-block cluster, resident and run together, by
// rank; and its barrier (barrier.cluster), whose phase advances when every
// thread of every block that has not exited has arrived. A launch without
// clusters gives each block a cluster of its own.
struct Cluster {
  std::vector<Block*> blocks;
  uint64_t phase = 0;
};

struct Block {
  uint32_t ctaid[3] = {};
  uint32_t rank = 0;            // in its cluster (SR_CgaCtaId)
  uint32_t worker = 0;          // the launch's host thread running it (SR_VIRTUALSMID)
  Cluster* cluster = nullptr;
  Cluster own;                  // the cluster when the launch has none
  std::vector<Warp> warps;
  uint64_t bssy_clock = 0;      // counts BSSYs, to order the warps' barriers by nesting
  std::vector<uint8_t> shared;
  std::map<std::pair<uint32_t, uint64_t>, GmmaSnapshot> gmma;   // (warpgroup, n-th MMA)
  // A block barrier (BAR) in progress: threads arrived, the count it waits
  // for, and the reduction it computes.
  struct Barrier {
    uint32_t arrived = 0;
    uint32_t expected = 0;
    uint32_t mode = 0, red = 0;
    uint32_t popc = 0;
    bool and_ = true, or_ = false;
  };
  Barrier bars[16];
  // What an mbarrier's word in shared memory has no room for, by its offset:
  // the transaction bytes its phase still waits for (raised by expect-tx,
  // lowered as copies complete; it may dip below zero when a copy lands
  // before its expect-tx), and whether every arrival is in while bytes are
  // still due.
  struct MbarSide {
    int64_t tx = 0;
    bool arrived_all = false;
  };
  std::unordered_map<uint32_t, MbarSide> mbar;
  exec::TensorMemory tmem;      // sm_100's Tensor Memory (tcgen05)
  exec::LaunchStats st;   // this block's counts, folded into the launch's
};

struct Fault : std::exception {
  std::string msg;
  explicit Fault(std::string m) : msg(std::move(m)) {}
  const char* what() const noexcept override { return msg.c_str(); }
};

// Dynamic parallelism, as the PTX engine runs it (src/exec/interpreter.cpp):
// a thread gets a parameter buffer for a kernel of the module, fills it, and
// launches it; the child grid runs after the parent grid has finished and
// before the launch as a whole returns -- a schedule CUDA allows for every
// device-side stream, since it promises no concurrency between a parent and
// its children -- except a tail launch, which waits for the grid's other
// children too. Children run in the order they were launched, parent block by
// parent block, so the order does not depend on how blocks were spread over
// host threads. The half both engines share, and what each call does and
// returns, is vgpu/exec/devrt.hpp.
using Child = exec::devrt::Child<size_t>;     // the kernel is an index in the module's cubin.kernels
using Launches = exec::devrt::Launches<size_t>;

// A kernel's parameter bytes: where its last parameter ends.
uint32_t param_bytes(const CubinKernel& k) {
  uint32_t end = 0;
  for (const CubinParam& p : k.params) end = std::max(end, p.offset + p.size);
  return end;
}

std::mutex& atomic_lock(uint64_t addr) {
  static std::mutex locks[64];
  return locks[(addr >> 3) % 64];
}

void tier_add_total(uint64_t n);   // tier_ops.inc

// The budget of steps a warp may take: the launch's, or VGPU_MAX_STEPS (0: none), by the rule the PTX engine applies
// (effective_max_steps in src/exec/interpreter.cpp). Read at the start of each launch.
uint64_t sass_max_steps(uint64_t configured) {
  static const char* const e = std::getenv("VGPU_MAX_STEPS");   // (once per process: a launch of a few instructions costs less than a getenv)
  if (!e || !*e) return configured;
  const char* p = e;
  while (*p == ' ' || *p == '\t') ++p;
  if (*p == '-' || *p == '+') return configured;   // (strtoull would wrap a negative into a huge budget)
  errno = 0;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(e, &end, 10);
  if (errno != 0 || end == e || *end != '\0') return configured;
  return v == 0 ? std::numeric_limits<uint64_t>::max() : static_cast<uint64_t>(v);
}

// The lanes whose program counter is `p`, compared two to a vector register where SSE2 is there.
inline Mask tier_lanes_at(const Warp& w, uint64_t p) {
  Mask eq = 0;
#if defined(__SSE2__)
  const __m128i want = _mm_set1_epi64x(static_cast<long long>(p));
  for (unsigned i = 0; i < 32; i += 2) {
    const __m128i v = _mm_loadu_si128(reinterpret_cast<const __m128i*>(&w.pc[i]));
    const __m128i e = _mm_cmpeq_epi32(v, want);
    const __m128i both = _mm_and_si128(e, _mm_shuffle_epi32(e, 0xB1));   // a 64-bit lane is equal when both halves are
    eq |= static_cast<Mask>(_mm_movemask_pd(_mm_castsi128_pd(both))) << i;
  }
#else
  for (unsigned i = 0; i < 32; ++i) eq |= static_cast<Mask>(w.pc[i] == p) << i;
#endif
  return eq;
}

// ---- kernels whose blocks wait on one another (vgpu/exec/lockstep.hpp) ----------------------------
//
// A kernel polls global memory when it has a loop with an atomic that returns its old value, or a
// strong load (a volatile or relaxed one, CTA scope aside), whose address and operands are the same
// on every turn of the loop, and no store to global memory anywhere in the loop: `while (atomicCAS(lock,
// 0, 1)) {}`, `while (*(volatile int*)flag != n) {}`. A compare-and-swap retry loop is not one: the value
// it swaps in comes from the old value, which changes. The test is cheap and errs towards "no": a
// kernel it misses runs as before, and one it takes for a poller only loses some speed.
//
// Going through the loop's instructions in address order, a register is "variable" when what it
// holds can differ from one turn to the next: it was set from what a load, atomic, shuffle, vote or
// clock returned, or from a variable register, or it is read before the loop sets it and the loop
// does set it (it comes round from the last turn: a counter, an old value). A register the loop only
// reads, or sets from constants and other fixed registers before it reads it, is fixed.
bool kernel_polls_loop(const std::vector<const Instr*>& body) {
  enum : uint8_t { Undef, Fixed, Variable };
  uint8_t state[256] = {};
  std::bitset<256> set_in_loop;
  const auto touch = [](const Operand& o, const auto& f) {
    if (o.kind == Kind::Reg && o.reg != kRZ) {
      for (unsigned i = 0; i < std::max<unsigned>(1, o.width) && o.reg + i < 255; ++i) f(o.reg + i);
    } else if (o.kind == Kind::Mem && o.reg != kRZ) {
      f(o.reg);   // the base: a 64-bit address takes two
      if (o.reg + 1 < 255) f(o.reg + 1);
    }
  };
  for (const Instr* ins : body)
    for (const Operand& d : ins->dst) touch(d, [&](unsigned r) { set_in_loop.set(r); });
  const auto is_variable = [&](unsigned r) {
    if (state[r] == Undef) return set_in_loop.test(r);
    return state[r] == Variable;
  };
  const auto generates = [](const Instr& ins) {
    switch (ins.op) {
      case Op::LDG: case Op::LD: case Op::LDL: case Op::LDS: case Op::LDSM: case Op::LDGSTS: case Op::ATOM:
      case Op::ATOMG: case Op::ATOMS: case Op::CS2R: case Op::SHFL: case Op::VOTE: case Op::MATCH: case Op::REDUX:
      case Op::TEX: case Op::LDGMC:
        return true;
      case Op::S2R:   // thread and block ids are the same on every turn; the clocks are not
        return !ins.src.empty() && ins.src[0].text.find("CLOCK") != std::string::npos;
      default:
        return false;
    }
  };
  bool polls = false;
  for (const Instr* ins : body) {
    switch (ins->op) {
      case Op::STG: case Op::ST: case Op::RED: case Op::CALL: case Op::UTMASTG: case Op::UTMAREDG:
      case Op::UBLKRED: case Op::REDAS:
        return false;   // the loop writes global memory, or calls something that might
      default:
        break;
    }
    bool var_src = false;
    for (const Operand& s : ins->src) touch(s, [&](unsigned r) { var_src = var_src || is_variable(r); });
    bool poll = false;
    if (ins->op == Op::ATOM || ins->op == Op::ATOMG) {
      if (ins->dst.empty() || (ins->dst.back().kind == Kind::Reg && ins->dst.back().reg == kRZ)) return false;   // no result: a store
      poll = true;
    } else if (ins->op == Op::LDG || ins->op == Op::LD) {
      bool strong = false, cta = false;
      for (const std::string& m : ins->mods) {
        strong = strong || m == "STRONG";
        cta = cta || m == "CTA";
      }
      poll = strong && !cta;
    }
    polls = polls || (poll && !var_src);
    const uint8_t now = (generates(*ins) || var_src) ? Variable : Fixed;
    for (const Operand& d : ins->dst) touch(d, [&](unsigned r) { state[r] = now; });
  }
  return polls;
}

bool kernel_polls(const Module& m, const CubinKernel& k) {
  {
    std::lock_guard<std::mutex> g(m.polling_mu);
    if (const auto it = m.polling.find(k.name); it != m.polling.end()) return it->second;
  }
  std::vector<std::string> names = reachable(m, k.name);
  names.push_back(k.name);
  bool found = false;
  for (const std::string& f : names) {
    const auto it = m.code_index.find(f == k.name ? k.text_section : ".text." + f);
    if (it == m.code_index.end()) continue;
    const Code& code = m.code[it->second];
    for (size_t i = 0; i < code.count && !found; ++i) {
      const Instr& br = code.instr(i);
      if (br.op != Op::BRA || br.src.empty()) continue;
      const int64_t target = br.src.back().imm;
      if (target < 0 || static_cast<uint64_t>(target) > br.pc) continue;   // a branch back
      const size_t first = static_cast<size_t>(target) / 16;
      if (i - first > 400) continue;
      std::vector<const Instr*> body;
      for (size_t j = first; j <= i; ++j) body.push_back(&code.instr(j));
      found = kernel_polls_loop(body);
    }
    if (found) break;
  }
  std::lock_guard<std::mutex> g(m.polling_mu);
  return m.polling.emplace(k.name, found).first->second;
}

// The lowest address among the runnable lanes' (`first` is the first one's): the same minimum the plain loop finds. With many lanes
// running, as four independent chains over all 32 lanes, a lane that does not run counting as the largest address (an all-ones OR,
// no branch on the lane's bit: those were mispredicted in a diverged warp); with few, the loop over just those lanes.
inline uint64_t tier_lowest_pc(const Warp& w, Mask run, uint64_t first) {
  if (simd::popcount32(run) <= 5) {
    uint64_t lo = first;
    for (Mask m = run & (run - 1); m; m &= m - 1) lo = std::min(lo, w.pc[std::countr_zero(m)]);
    return lo;
  }
  const auto pcv = [&](unsigned l) { return w.pc[l] | ((static_cast<uint64_t>((run >> l) & 1)) - 1); };
  uint64_t a = ~uint64_t{0}, b = a, c = a, d = a;
  for (unsigned l = 0; l < 32; l += 4) {
    a = std::min(a, pcv(l));
    b = std::min(b, pcv(l + 1));
    c = std::min(c, pcv(l + 2));
    d = std::min(d, pcv(l + 3));
  }
  return std::min(std::min(a, b), std::min(c, d));
}

class Runner {
 public:
  Runner(const Module& m, const CubinKernel& k, const exec::LaunchConfig& cfg,
         const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem, const DeviceProfile& profile)
      : m_(m), k_(k), cfg_(cfg), mem_(mem), profile_(profile), max_steps_(sass_max_steps(cfg.max_steps)) {
    // A cooperative launch's grid-barrier workspace, as the PTX interpreter
    // makes it: 64 zeroed bytes, freed however the launch ends.
    if (cfg.cooperative) {
      coop_ws_ = mem.alloc(64);
      const uint8_t zero = 0;
      mem.fill(coop_ws_, &zero, 1, 64);
    }
    for (int d = 0; d < 3; ++d) {
      cshape_[d] = cfg.cluster[d] ? cfg.cluster[d] : 1;
      csize_ *= cshape_[d];
      clustered_ = clustered_ || cshape_[d] > 1;
    }
    // Each thread's local window: the kernel's frame, or the stack the
    // device gives every thread (cudaLimitStackSize), which alloca grows
    // into below the frame, whichever is larger.
    local_size_ = std::max<uint32_t>({k.min_stack, k.frame_size, 16,
                                      static_cast<uint32_t>(std::min<uint64_t>(cfg.stack_bytes, 1u << 24))});
    local_size_ = (local_size_ + 15) & ~15u;
    shared_size_ = static_cast<uint64_t>(k.shared_bytes) + cfg.shared_bytes;
    if (m.sm >= 100) {
      for (const CubinSymbol& sym : m.cubin.symbols)
        if (sym.name == ".nv.reservedSmem.offset0") alloc_handshake_ = static_cast<uint32_t>(sym.value);
        else if (sym.name == "__nv_reservedSMEM_tmem_allocation_pipeline_mbarrier") alloc_mbar_ = static_cast<uint32_t>(sym.value);
    }
    // From compute capability 8.0 the driver reserves 1 KiB of shared memory
    // behind every block's own, which it rounds to its 128-byte allocation
    // unit (cooperative_groups keeps the scratch of multi-warp tiles there).
    // SR_SMEMSZ is the whole allocation, and the reserved region's begin is
    // SR_SMEMSZ less the reserved size the constant bank holds.
    if (const uint32_t r = profile_.reserved_smem_per_block()) {
      // From sm_90 ptxas puts the reserved region first, at offset 0, and
      // counts it in the kernel's .nv.shared section (400 bytes of
      // __shared__ make a 0x590-byte section, the variables from 0x400), so
      // there it is already inside the block's own.
      const uint64_t own = m.sm >= 90 && k.shared_bytes >= r ? k.shared_bytes - r : k.shared_bytes;
      kernel_shared_ = (own + cfg.shared_bytes + 127) / 128 * 128;
      shared_size_ = kernel_shared_ + r;
      smemsz_ = shared_size_;
    } else {
      kernel_shared_ = shared_size_;
      // Before 8.0 there is no reserved part, but the allocation is still
      // counted in its unit (256 bytes, cuda_occupancy.h), as the PTX path's
      // %total_smem_size counts it; ptxas reads that register as SR_SMEMSZ.
      // The window itself stays exact, so an overrun into the rounding is
      // still caught.
      smemsz_ = (shared_size_ + 255) / 256 * 256;
    }
    // VGPU_FASTPATH=0 sends every instruction down the general path, as it does for the PTX interpreter: the
    // plain-form ALU kernels (fast_alu), coalesced global accesses and shared-memory accesses inside the
    // block. For checking that the fast paths compute what the general one does.
    if (const char* f = std::getenv("VGPU_FASTPATH"); f && f[0] == '0') fast_ = false;
    vgpu::exec::tc86::refresh_exact();   // VGPU_TC_EXACT=0: the earlier sequential tensor sums
    if (tier::requested()) {
      const char* tr = std::getenv("VGPU_SASS_TRACE");
      tier_ = fast_ && simd::enabled() && !(tr && *tr);
      const char* ts = std::getenv("VGPU_TIER_STATS");
      tier_stats_ = tier_ && ts && (ts[0] == '1' || ts[0] == '2');
      const char* tc = std::getenv("VGPU_TIER_CHAIN");
      tier_chain_ = !(tc && tc[0] == '0');
    }
    build_bank0(args);
    for (const auto& [name, va] : m.bank_va) {
      // ".nv.constant<N>" for the module; ".nv.constant<N>.<kernel>" for ours.
      const size_t dot = name.find('.', 12);
      const unsigned bank = static_cast<unsigned>(std::stoul(name.substr(12, dot == std::string::npos ? std::string::npos : dot - 12)));
      if (dot != std::string::npos && name.substr(dot + 1) != k.name) continue;
      bank_va_[bank] = va;
      bank_size_[bank] = m.cubin.section(name)->size;
    }
    const auto it = m.code_index.find(k.text_section);
    entry_ = m.code[it->second].base;
  }

  exec::LaunchStats run();
  void set_launches(Launches* dl) { dl_ = dl; }
  // For the tier's handlers (tier_ops.inc): the memory the launch runs on, and the interpreter's memory
  // instructions for the cases a handler leaves to it.
  MemoryManager& memory() { return mem_; }
  uint32_t tier_sreg(const Block& blk, const Warp& w, unsigned idx, unsigned lane) const { return sreg(blk, w, idx, lane); }
  void tier_exec_mem(Block& blk, Warp& w, const Instr& ins, Mask ex) { exec_mem(blk, w, ins, ex); }
  uint32_t tier_cbank32(unsigned bank, uint64_t offset) const { return cbank32(bank, offset); }
  void tier_exec_float(Block& blk, Warp& w, const Instr& ins, Mask ex) { exec_float(blk, w, ins, ex); }
  // A float instruction as Runner::execute runs it: the 32-lane fast path first, then the general rule. The tier's handlers for the
  // forms whose result rules are more than a call of a SIMD kernel use this (so that the rules stay in one place, exec_ops.inc).
  void tier_float_op(Block& blk, Warp& w, const Instr& ins, Mask ex) {
    if (!fast_alu(w, ins, ex)) exec_float(blk, w, ins, ex);
  }
  // The `kind` argument of simd::Kernels::hmma for an f16 (0), bf16 (1) or tf32 (2) HMMA: an sm_86 cubin's sum is the RTX 3060 tensor core's
  // (bit 4) unless VGPU_TC_EXACT=0. exec_mma (exec_ops.inc) and the tier's HMMA handler both call this, so that the choice is made in one place.
  unsigned hmma_kernel_kind(unsigned kind) const { return kind | (m_.sm == 86 && vgpu::exec::tc86::exact_enabled() ? 16u : 0u); }
  ~Runner() {
    if (coop_ws_) {
      try {
        mem_.free(coop_ws_);
      } catch (const Error&) {
      }
    }
  }

 private:
  // ---- setup ----
  void build_bank0(const std::vector<std::vector<uint8_t>>& args);

  // ---- per-block ----
  void run_block(Block& blk);
  bool block_round(Block& blk, int slice);
  uint64_t fingerprint(const Block& blk) const;
  bool break_warpsync_standoff(Block& blk);
  bool break_bsync_standoff(Block& blk);
  void init_block(Block& blk, uint64_t linear);
  bool step_warp(Block& blk, Warp& w);
  // A warp's turn: up to `limit` calls of step_warp, ending early when it cannot run or yields. With the tier, a call may do the work
  // of several (the chain, tier_ops.inc): `budget` tells it how many calls the turn has left, `extra` how many it took over.
  template <bool Tier>
  void turn(Block& blk, Warp& w, int limit, bool& progress) {
    for (int n = 0; n < limit; ++n) {
      if (!(w.alive & ~w.exited)) break;
      if constexpr (Tier) w.budget = static_cast<uint32_t>(limit - 1 - n);
      if (!step_warp(blk, w)) break;
      progress = true;
      if constexpr (Tier) {
        n += static_cast<int>(w.extra);
        w.extra = 0;
      }
      if (w.yield) break;
    }
  }
  bool tier_run(Block& blk, Warp& w, Mask group, uint64_t pick, const Code& code, const tier::Program& tp);   // tier_ops.inc
  bool break_spin_wait(const std::vector<Block*>& blocks);
  void execute(Block& blk, Warp& w, const Instr& ins, Mask group, bool advance = true);
  bool fast_alu(Warp& w, const Instr& ins, Mask ex);   // exec_ops.inc
  bool fast_wide(Warp& w, const Instr& ins, Mask ex);   // exec_ops.inc: the wider forms fast_alu hands over
  bool fast_float(Warp& w, const Instr& ins, Mask ex);  // exec_ops.inc: the float forms fast_alu hands over

  // ---- state access ----
  const Instr& fetch(uint64_t addr, const Code** code_out = nullptr);
  uint32_t cbank32(unsigned bank, uint64_t offset) const;
  // 64-bit operands: a register pair inline, as read32/write32 do for one register; the rest out of line.
  [[gnu::always_inline]] uint64_t read64(Warp& w, const Operand& o, unsigned lane) {
    if (o.kind == Kind::Reg && o.reg != kRZ) [[likely]]
      return w.reg(o.reg, lane) | (static_cast<uint64_t>(w.reg(o.reg + 1, lane)) << 32);
    return read64_slow(w, o, lane);
  }
  [[gnu::always_inline]] void write64(Warp& w, const Operand& o, unsigned lane, uint64_t v) {
    if (o.kind == Kind::Reg && o.reg != kRZ) [[likely]] {
      w.reg(o.reg, lane) = static_cast<uint32_t>(v);
      if (o.reg + 1 != kRZ) w.reg(o.reg + 1, lane) = static_cast<uint32_t>(v >> 32);   // R254's partner is RZ
      return;
    }
    write64_slow(w, o, lane, v);
  }
  uint64_t read64_slow(Warp& w, const Operand& o, unsigned lane);
  void write64_slow(Warp& w, const Operand& o, unsigned lane, uint64_t v);
  bool pred(Warp& w, const Operand& o, unsigned lane) const;
  void set_pred(Warp& w, const Operand& o, unsigned lane, bool v);
  uint32_t sreg(const Block& blk, const Warp& w, unsigned idx, unsigned lane) const;

  // ---- memory ----
  enum class Space { Global, Shared, Local, Param };
  Space classify(uint64_t generic, uint64_t* offset) const;
  void mem_read(Block& blk, Warp& w, unsigned lane, Space s, uint64_t addr, void* out, uint32_t n);
  void mem_write(Block& blk, Warp& w, unsigned lane, Space s, uint64_t addr, const void* in, uint32_t n);
  uint64_t generic_addr(Warp& w, const Instr& ins, unsigned lane, const Operand& addr_op, bool wide);

  [[noreturn]] void fault(const Warp& w, const Instr& ins, unsigned lane, const std::string& why,
                          Err code = Err::OutOfBounds) const;

  // ---- instruction groups (exec_ops.inc) ----
  void exec_int(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_float(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_mem(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_warp(Block& blk, Warp& w, const Instr& ins, Mask ex, Mask group);
  bool exec_control(Block& blk, Warp& w, const Instr& ins, Mask ex, Mask group);   // true: it set the pcs
  void exec_mma(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_gmma(Block& blk, Warp& w, const Instr& ins, Mask ex);   // warpgroup MMA
  void exec_tex(Block& blk, Warp& w, const Instr& ins, Mask ex);   // textures and surfaces
  void exec_syncs(Block& blk, Warp& w, const Instr& ins, Mask ex);   // mbarriers
  uint32_t mbar_offset(Block& blk, Warp& w, const Instr& ins, unsigned lane, bool ur_only, Block** owner);
  uint64_t mbar_arrive(Block& blk, uint32_t off, uint32_t arrivals, int64_t tx, bool drop);
  uint64_t mbar_arrive80(Block& blk, uint32_t off);
  void exec_tma(Block& blk, Warp& w, const Instr& ins, Mask ex);   // TMA and bulk copies
  void exec_tcgen05(Block& blk, Warp& w, const Instr& ins, Mask ex);   // sm_100's tensor core
  void exec_async_store(Block& blk, Warp& w, const Instr& ins, Mask ex);   // STAS, REDAS
  void exec_clc(Block& blk, Warp& w, const Instr& ins, Mask ex);           // UGETNEXTWORKID
  Block& shared_block(Block& blk, uint64_t addr, uint32_t* off);
  Block& cluster_block(Block& blk, uint32_t rank);
  void run_cluster(uint64_t k, unsigned worker);        // one cluster's blocks, together
  void build_cluster(uint64_t k, unsigned worker, Block* blocks, Cluster& cl);   // cluster k's blocks, set up
  void cluster_barrier_check(Block& blk);                // UCGABAR: complete the phase if all are in
  void builtin_call(Block& blk, Warp& w, const Instr& ins, Mask ex, const std::string& name);
  bool device_runtime_call(Block& blk, Warp& w, const Instr& ins, Mask ex, const std::string& name);
  const size_t* kernel_at(uint64_t code_addr);   // the module's kernel whose code starts there
  const exec::KernelRef* ptx_kernel(size_t idx);   // the PTX engine's record of that kernel, if it has one
  uint32_t ival(Warp& w, const Operand& o, unsigned lane);   // with -/~ applied
  // Operand access, inline for the common case -- a register -- since every
  // instruction makes it once per lane per operand; the rest out of line.
  [[gnu::always_inline]] uint32_t read32(Warp& w, const Operand& o, unsigned lane) {
    if (o.kind == Kind::Reg) [[likely]] return o.reg == kRZ ? 0 : w.reg(o.reg, lane);
    return read32_slow(w, o, lane);
  }
  [[gnu::always_inline]] void write32(Warp& w, const Operand& o, unsigned lane, uint32_t v) {
    if (o.kind == Kind::Reg) [[likely]] {
      if (o.reg != kRZ) w.reg(o.reg, lane) = v;
      return;
    }
    write32_slow(w, o, lane, v);
  }
  uint32_t read32_slow(Warp& w, const Operand& o, unsigned lane);
  void write32_slow(Warp& w, const Operand& o, unsigned lane, uint32_t v);

  const Module& m_;
  const CubinKernel& k_;
  const exec::LaunchConfig& cfg_;
  MemoryManager& mem_;
  const DeviceProfile& profile_;
  const uint64_t max_steps_;   // the step budget of a warp (cfg_.max_steps, or VGPU_MAX_STEPS)
  bool fast_ = true;   // VGPU_FASTPATH, sampled at launch
  // The pre-decoded tier (src/sass/tier.hpp): on unless VGPU_TIER=0, with the fast paths and the SIMD kernels it is held to on and
  // no per-instruction trace. tier_stats_: VGPU_TIER_STATS=1 counts what ran where.
  bool tier_ = false;
  bool tier_stats_ = false;
  bool tier_chain_ = true;   // VGPU_TIER_CHAIN=0: one run per call of step_warp, as the interpreter's (tier_ops.inc)
  std::vector<uint8_t> bank0_;
  std::map<unsigned, uint64_t> bank_va_, bank_size_;
  uint64_t entry_ = 0;
  uint64_t coop_ws_ = 0;
  uint32_t local_size_ = 0;
  uint64_t shared_size_ = 0;
  uint64_t kernel_shared_ = 0;   // the block's own shared memory, rounded (see shared_size_)
  // Where ptxas keeps its two-CTA Tensor Memory allocation handshake in the
  // driver's reserved shared memory (.nv.reservedSmem.offset0), or ~0u.
  uint32_t alloc_handshake_ = ~0u;
  // CUDA 12.8's ptxas names the handshake's mbarrier (__nv_reservedSMEM_tmem_
  // allocation_pipeline_mbarrier, at 0x58 beside separate phase, mask and
  // parity words) where CUDA 13's keeps it eight bytes into one 32-byte
  // struct; ~0u when the module is the latter's.
  uint32_t alloc_mbar_ = ~0u;
  void seed_alloc_handshake(Block& blk);   // by the block's rank in its pair
  uint64_t smemsz_ = 0;          // SR_SMEMSZ: the whole allocation, in allocation units
  uint32_t block_threads_ = 0;
  // Thread-block clusters: the shape (1x1x1 without one), its size, and
  // whether the launch has clusters at all (any dimension over 1, as the PTX
  // engine counts %is_explicit_cluster).
  std::array<uint32_t, 3> cshape_{1, 1, 1};
  uint32_t csize_ = 1;
  bool clustered_ = false;
  // The launch's units of work -- blocks, or clusters -- handed out in order;
  // cluster launch control's try_cancel takes the next one, which then never
  // runs.
  std::atomic<uint64_t> next_unit_{0};
  uint64_t units_ = 0;
  exec::LaunchStats stats_;
  std::mutex stats_mu_;
  Launches* dl_ = nullptr;   // where device-side launches go (dynamic parallelism)
  std::map<uint64_t, size_t> kernel_by_code_;
  std::map<size_t, const exec::KernelRef*> ptx_refs_;
  std::once_flag kernel_by_code_once_;
  // The heap malloc() draws from, shared by every launch on the device.
};

// Runs one child grid: what devrt's run_children calls for each. `base` is the
// launch of the grid whose children these are (a child differs from it in its
// shape); `sink` takes the counters.
struct ChildRunner {
  const Module& m;
  const exec::LaunchConfig& base;
  MemoryManager& mem;
  const DeviceProfile& profile;
  exec::LaunchStats* sink;
  void add_stats(const exec::LaunchStats& s) {
    if (sink) sink->add(s);
  }
  void operator()(Child& c, Launches& out);
};

// ---- setup -------------------------------------------------------------------

// Bank 0 as the driver fills it for this launch (layout read from what ptxas
// emits for these fields; nvidia/docs/sass.md):
//   0x00 block size x, y, z        0x0c grid size x, y, z
//   0x18 the shared window (64)    0x20 the local window (64)
//   0x28 the initial stack pointer (each thread's local memory size)
//   0x118 the global memory descriptor (64), which .E loads name
//   param_base: the parameters, laid out as .nv.info says
void Runner::build_bank0(const std::vector<std::vector<uint8_t>>& args) {
  const CubinSection* tmpl = m_.cubin.section(".nv.constant0." + k_.name);
  bank0_.assign(std::max<size_t>(tmpl ? tmpl->size : 0, k_.param_base + k_.param_size + 16), 0);
  if (tmpl && !tmpl->bytes.empty()) std::memcpy(bank0_.data(), tmpl->bytes.data(), tmpl->bytes.size());
  const auto put32 = [&](size_t off, uint32_t v) { std::memcpy(&bank0_[off], &v, 4); };
  const auto put64 = [&](size_t off, uint64_t v) { std::memcpy(&bank0_[off], &v, 8); };
  // The driver's fields, where each generation keeps them (read from what
  // ptxas emits for blockDim, gridDim, the windows, the stack, the memory
  // descriptor and %envreg1/2):
  //                      ntid   nctaid  shared  local   stack  window-lo  desc   envreg1/2
  //   sm_75-sm_89        0x0    0xc     0x18    0x20    0x28   -          0x118  0x8c/0x90
  //   sm_90              0x0    0xc     SWINHI  0x20    0x28   0xd0       0x208  0x44/0x48
  //   sm_100, sm_120     0x360  0x370   SWINHI  0x2f8   0x37c  0x120      0x358  0x254/0x258
  // (SWINHI: the shared window's high word comes from a special register.)
  const int sm = m_.sm;
  const size_t ntid = sm >= 100 ? 0x360 : 0x0, nctaid = sm >= 100 ? 0x370 : 0xc;
  const size_t local = sm >= 100 ? 0x2f8 : 0x20, stackf = sm >= 100 ? 0x37c : 0x28;
  if (bank0_.size() < 0x400) bank0_.resize(std::max<size_t>(bank0_.size(), sm >= 100 ? 0x380 : 0x220), 0);
  for (int i = 0; i < 3; ++i) {
    put32(ntid + 4 * i, cfg_.block[i]);
    put32(nctaid + 4 * i, cfg_.grid[i]);
  }
  if (sm < 90) put64(0x18, kSharedWindow);
  // %nsmid, which ptxas reads from bank 0 rather than a special register:
  // the SM count, which SR_VIRTUALSMID stays below.
  put32(sm >= 100 ? 0x2d0 : 0x10c, profile_.limits.multiprocessors);
  // The reserved shared memory (see shared_size_): its size, which ptxas
  // subtracts from SR_SMEMSZ for %reserved_smem_offset_begin, and before
  // sm_90 %reserved_smem_offset_end (the 0x120 bytes the driver uses past
  // the begin, as an RTX 3060 reports) and _1, the start of what is left.
  // The shared-memory size registers, as ptxas compiles them (checked on the
  // SASS CUDA 13.0's ptxas writes for sm_75 to sm_120): %dynamic_smem_size is
  // a bank-0 word, the launch's dynamic bytes; %total_smem_size is SR_SMEMSZ
  // less the reserved size below; and from sm_90 %aggr_smem_size is SR_SMEMSZ
  // clamped by another bank-0 word. What the hardware keeps there was not
  // measurable here (no sm_90 card), so it holds the allocation itself and
  // the clamp gives the PTX ISA's answer, which the PTX path gives too. sm_100
  // indexes both words by SR_CgaSize, which is 0 here.
  put32(sm >= 100 ? 0x2ac : 0x2c, cfg_.shared_bytes);
  if (sm >= 90) put32(sm >= 100 ? 0x2bc : 0x13c, static_cast<uint32_t>(smemsz_));
  if (const uint32_t r = profile_.reserved_smem_per_block()) {
    put32(sm >= 100 ? 0x16c : 0x114, r);
    if (sm < 90) {
      put32(0x120, static_cast<uint32_t>(kernel_shared_ + 0x120));
      put32(0x124, static_cast<uint32_t>(kernel_shared_));
    }
  }
  // Where the non-global windows start (a pointer at or past it is not
  // global), and the parameter window: bank 0's own address before sm_90,
  // the parameters' from it.
  const uint64_t bound = std::min({kConstWindow, kSharedWindow, kLocalWindow, kParamWindow});
  if (sm >= 90) put64(sm >= 100 ? 0x120 : 0xd0, bound);
  if (sm < 90) {
    put64(0x40, kParamWindow);
    put64(0x50, bound);
  } else {
    put64(sm >= 100 ? 0x348 : 0x198, kParamWindow + k_.param_base);
  }
  put64(local, kLocalWindow);
  put32(stackf, local_size_);
  put64(sm >= 100 ? 0x358 : sm >= 90 ? 0x208 : 0x118, 0);   // the descriptor's bits are a cache policy
  // The cluster's shape (sm_90+), which ptxas reads to compute %cluster_*
  // and %clusterid -- dividing by multiplying with the shape's reciprocals,
  // as floats rounded up so that the truncated products are exact:
  //                  explicit  shape  1/shape  clusters  CTAs
  //   sm_90          0x140     0x144  0x150    0x15c     0x188
  //   sm_100, 120    0x36c     0x2a0  0x2b0    0x2c0     0x2cc
  // (sm_100 indexes the table by SR_CgaSize, which is 0 here.) explicit is
  // 1 with a cluster: %is_explicit_cluster, and cluster.sync()'s choice
  // between the cluster barrier and a block barrier, test it for 1.
  if (sm >= 90) {
    const size_t shape = sm >= 100 ? 0x2a0 : 0x144, recip = sm >= 100 ? 0x2b0 : 0x150;
    const size_t count = sm >= 100 ? 0x2c0 : 0x15c, ctas = sm >= 100 ? 0x2cc : 0x188;
    put32(sm >= 100 ? 0x36c : 0x140, clustered_ ? 1 : 0);
    for (int i = 0; i < 3; ++i) {
      put32(shape + 4 * i, cshape_[i]);
      float r = static_cast<float>(1.0 / cshape_[i]);
      if (static_cast<double>(r) < 1.0 / cshape_[i]) r = std::nextafter(r, 2.0f);
      std::memcpy(&bank0_[recip + 4 * i], &r, 4);
      put32(count + 4 * i, cfg_.grid[i] / cshape_[i]);
    }
    put32(ctas, csize_);
  }
  // %current_graph_exec (cudaGetCurrentGraphExec): the device graph this
  // kernel runs in, or 0. ptxas loads it from bank 0 at 0x120 on sm_75, 0x130
  // on sm_80-sm_89, 0x190 on sm_90 and 0x2e8 from sm_100 (CUDA 12.0's and
  // 13.0's alike). On an RTX 3060 the sm_86 load returned the executable
  // graph's own handle, and 0 in a kernel outside a device graph.
  put64(sm >= 100 ? 0x2e8 : sm >= 90 ? 0x190 : sm >= 80 ? 0x130 : 0x120, cfg_.current_graph_exec);
  // PTX's %envreg1/%envreg2: a cooperative launch's grid barrier workspace,
  // high word first. cg::this_grid().sync() traps when it is zero, which is
  // what an ordinary launch of a grid-sync kernel gets.
  const uint64_t ws = coop_ws_ ? coop_ws_ : cfg_.coop_workspace;
  const size_t env1 = sm >= 100 ? 0x254 : sm >= 90 ? 0x44 : 0x8c;
  put32(env1, static_cast<uint32_t>(ws >> 32));
  put32(env1 + 4, static_cast<uint32_t>(ws));
  if (args.size() != k_.params.size())
    throw Error(Err::InvalidValue, "kernel " + k_.name + " takes " + std::to_string(k_.params.size()) +
                                       " parameters, the launch passed " + std::to_string(args.size()));
  for (size_t i = 0; i < args.size(); ++i) {
    const CubinParam& p = k_.params[i];
    if (args[i].size() != p.size)
      throw Error(Err::InvalidValue, "kernel " + k_.name + " parameter " + std::to_string(i) + " is " +
                                         std::to_string(p.size) + " bytes, the launch passed " +
                                         std::to_string(args[i].size()));
    std::memcpy(&bank0_[k_.param_base + p.offset], args[i].data(), p.size);
  }
}

uint32_t Runner::cbank32(unsigned bank, uint64_t offset) const {
  if (bank == 0) {
    if (offset + 4 > bank0_.size()) return 0;
    uint32_t v;
    std::memcpy(&v, &bank0_[offset], 4);
    return v;
  }
  const auto it = bank_va_.find(bank);
  if (it == bank_va_.end() || offset + 4 > bank_size_.at(bank)) return 0;   // unbound: reads zero
  return static_cast<uint32_t>(mem_.load_scalar(it->second + offset, 4));
}

// ---- fetch and operands --------------------------------------------------------

const Instr& Runner::fetch(uint64_t addr, const Code** code_out) {
  const Code* c = m_.code_at(addr);
  if (!c) throw Fault("jump to 0x" + [&] { char b[24]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)addr); return std::string(b); }() + ", which is not code");
  if (code_out) *code_out = c;
  return c->instr((addr - c->base) / 16);
}

uint32_t Runner::read32_slow(Warp& w, const Operand& o, unsigned lane) {
  switch (o.kind) {
    case Kind::Reg: return o.reg == kRZ ? 0 : w.reg(o.reg, lane);
    case Kind::UReg: return o.reg >= (m_.sm >= 100 ? 255u : 63u) ? 0 : w.ur[o.reg];
    case Kind::Imm: return static_cast<uint32_t>(o.imm);
    case Kind::FImm: return o.fbits;
    case Kind::CBank: return cbank32(o.reg, static_cast<uint64_t>(o.imm));
    case Kind::UCBank: {
      // cx[URn][off]: the bank is URn's upper half, the offset its low 16 bits plus the immediate.
      const uint32_t u = w.ur[o.reg];
      return cbank32(u >> 16, (u & 0xffff) + static_cast<uint64_t>(o.imm));
    }
    default: return 0;
  }
}

uint64_t Runner::read64_slow(Warp& w, const Operand& o, unsigned lane) {
  switch (o.kind) {
    case Kind::Reg:
      if (o.reg == kRZ) return 0;
      return w.reg(o.reg, lane) | (static_cast<uint64_t>(w.reg(o.reg + 1, lane)) << 32);
    case Kind::UReg: {
      const unsigned z = m_.sm >= 100 ? 255u : 63u;
      if (o.reg >= z) return 0;
      return w.ur[o.reg] | (static_cast<uint64_t>(w.ur[o.reg + 1]) << 32);
    }
    case Kind::CBank:
      return cbank32(o.reg, static_cast<uint64_t>(o.imm)) |
             (static_cast<uint64_t>(cbank32(o.reg, static_cast<uint64_t>(o.imm) + 4)) << 32);
    case Kind::Imm:
      // A 64-bit immediate (sm_120's MOV.64, SEL.64), or a 32-bit one
      // extended as the op's signedness decoded it: nvdisasm prints
      // ISETP.GT.S64's -0x1, and the compare is against -1, not 0xffffffff.
      return static_cast<uint64_t>(o.imm);
    default: return static_cast<uint64_t>(read32(w, o, lane));
  }
}

void Runner::write32_slow(Warp& w, const Operand& o, unsigned /*lane*/, uint32_t v) {
  if (o.kind == Kind::UReg) {
    if (o.reg >= (m_.sm >= 100 ? 255u : 63u)) return;
    w.ur[o.reg] = v;
  }
}

void Runner::write64_slow(Warp& w, const Operand& o, unsigned lane, uint64_t v) {
  write32(w, o, lane, static_cast<uint32_t>(v));
  if (o.kind == Kind::Reg && o.reg == kRZ) return;
  Operand hi = o;
  hi.reg = o.reg + 1;
  write32(w, hi, lane, static_cast<uint32_t>(v >> 32));
}

bool Runner::pred(Warp& w, const Operand& o, unsigned lane) const {
  bool v;
  if (o.kind == Kind::UPred) v = o.reg == kPT ? true : w.up[o.reg];
  else v = o.reg == kPT ? true : ((w.p[o.reg] >> lane) & 1) != 0;
  return v != o.neg;
}

void Runner::set_pred(Warp& w, const Operand& o, unsigned lane, bool v) {
  if (o.reg == kPT) return;
  if (o.kind == Kind::UPred) {
    w.up[o.reg] = v;
    return;
  }
  if (v) w.p[o.reg] |= Mask{1} << lane;
  else w.p[o.reg] &= ~(Mask{1} << lane);
}

uint32_t Runner::sreg(const Block& blk, const Warp& w, unsigned idx, unsigned lane) const {
  const uint32_t tid = w.index * 32 + lane;
  const uint32_t bx = cfg_.block[0], by = cfg_.block[1];
  switch (idx) {
    case 0x00: return lane;                                             // SR_LANEID
    case 0x21: return tid % bx;                                         // SR_TID.X
    case 0x22: return (tid / bx) % by;                                  // SR_TID.Y
    case 0x23: return tid / (bx * by);                                  // SR_TID.Z
    case 0x25: return blk.ctaid[0];
    case 0x26: return blk.ctaid[1];
    case 0x27: return blk.ctaid[2];
    case 0x38: return Mask{1} << lane;                                  // SR_EQMASK
    case 0x39: return (Mask{1} << lane) - 1;                            // SR_LTMASK
    case 0x3a: return lane == 31 ? kAll : (Mask{2} << lane) - 1;        // SR_LEMASK
    case 0x3b: return lane == 31 ? 0 : ~((Mask{2} << lane) - 1);        // SR_GTMASK
    case 0x3c: return ~((Mask{1} << lane) - 1);                         // SR_GEMASK
    case 0x50: case 0x51: case 0x52: case 0x53: case 0x01: {
      // The clocks: a count of instructions this warp has run, which is the
      // only honest clock a functional simulator has.
      const uint64_t t = w.steps;
      return (idx == 0x51 || idx == 0x53) ? static_cast<uint32_t>(t >> 32) : static_cast<uint32_t>(t);
    }
    case 0x32: return static_cast<uint32_t>(smemsz_);                   // SR_SMEMSZ
    case 0x2f: return static_cast<uint32_t>(kSharedWindow >> 32);       // SR_SWINHI
    case 0x88: return blk.rank;                                         // SR_CgaCtaId: the rank in the cluster
    case 0x43: {   // SR_VIRTUALSMID (%smid), by the PTX engine's rule
      // Distinct among the blocks resident at once, as a CTA's SM is on the
      // hardware: a grid that fits the device (or a cooperative one) by its
      // linear order, a larger one by the host thread running it -- thread
      // t's rank-r block is SM t * cluster size + r, and the launch caps its
      // threads so that fits. CUTLASS's grouped GEMMs keep a tensor map per
      // SM; read as 0, every CTA rewrote the same one.
      const uint32_t sms = profile_.limits.multiprocessors;
      if (!sms) return 0;
      const uint64_t linear = blk.ctaid[0] + uint64_t{blk.ctaid[1]} * cfg_.grid[0] +
                              uint64_t{blk.ctaid[2]} * cfg_.grid[0] * cfg_.grid[1];
      const uint64_t blocks = uint64_t{cfg_.grid[0]} * cfg_.grid[1] * cfg_.grid[2];
      if (cfg_.cooperative || blocks <= sms) return static_cast<uint32_t>(linear % sms);
      const uint64_t size = clustered_ ? csize_ : 1;
      return static_cast<uint32_t>((uint64_t{blk.worker} * size + blk.rank) % sms);
    }
    case 0x8a: return 0;                                                // SR_CgaSize (bank 0's envregs at +0)
    case 0x28: return block_threads_;                                   // SR_NTID
    default: return 0;
  }
}

// ---- memory --------------------------------------------------------------------

Runner::Space Runner::classify(uint64_t g, uint64_t* off) const {
  if (g >= kSharedWindow && g < kSharedWindow + (uint64_t{1} << 32)) {
    *off = g - kSharedWindow;
    return Space::Shared;
  }
  if (g >= kLocalWindow && g < kLocalWindow + (uint64_t{1} << 32)) {
    *off = g - kLocalWindow;
    return Space::Local;
  }
  if (g >= kParamWindow && g < kParamWindow + (uint64_t{1} << 32)) {
    *off = g - kParamWindow;
    return Space::Param;
  }
  if (g >= kConstWindow && g < kConstWindow + (uint64_t{1} << 32)) {
    // __constant__ data: a global address of the allocation that holds bank 3
    const auto it = bank_va_.find(3);
    *off = it == bank_va_.end() ? g : it->second + (g - kConstWindow);
    return Space::Global;
  }
  *off = g;
  return Space::Global;
}

// The block whose shared memory a shared address names: ptxas keeps the
// block's rank in its cluster in bits 24 up of every shared address (its
// own included: shared::cta addresses are SR_CgaCtaId << 24 | offset, and
// mapa swaps the rank), and the offset below.
Block& Runner::shared_block(Block& blk, uint64_t addr, uint32_t* off) {
  *off = static_cast<uint32_t>(addr) & 0xffffff;
  const uint32_t rank = static_cast<uint32_t>(addr >> 24);
  if (rank == blk.rank) return blk;
  if (!blk.cluster || rank >= blk.cluster->blocks.size()) {
    char b[160];
    std::snprintf(b, sizeof b, "a shared address (0x%llx) naming block %u of a cluster of %zu",
                  static_cast<unsigned long long>(addr), rank, blk.cluster ? blk.cluster->blocks.size() : size_t{1});
    throw Fault(b);
  }
  Block& to = *blk.cluster->blocks[rank];
  bool live = false;
  for (const Warp& w : to.warps) live = live || (w.alive & ~w.exited);
  // A block that exits takes its shared memory with it: cluster.sync()
  // before exiting is what keeps a peer's reads valid.
  if (!live) throw Fault("shared memory of block " + std::to_string(rank) + " of the cluster, which has exited");
  return to;
}

void Runner::mem_read(Block& blk, Warp& w, unsigned lane, Space s, uint64_t a, void* out, uint32_t n) {
  switch (s) {
    case Space::Shared: {
      uint32_t off;
      Block& b = shared_block(blk, a, &off);
      if (uint64_t{off} + n > b.shared.size())
        throw Fault("shared read of " + std::to_string(n) + " bytes at 0x" + [&] { char t[24]; std::snprintf(t, sizeof t, "%llx", (unsigned long long)a); return std::string(t); }() +
                    ", past the block's " + std::to_string(b.shared.size()) + " bytes");
      std::memcpy(out, &b.shared[off], n);
      return;
    }
    case Space::Local:
      if (a + n > w.local_size)
        throw Fault("local read of " + std::to_string(n) + " bytes at 0x" + [&] { char b[24]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)a); return std::string(b); }() +
                    ", past the thread's " + std::to_string(w.local_size) + " bytes");
      if (w.local.empty()) w.local.assign(size_t{w.local_size} * 32, 0);
      std::memcpy(out, &w.local[lane * w.local_size + a], n);
      return;
    case Space::Global:
      if (n <= 8 && (a % n) == 0) {
        const uint64_t v = mem_.load_scalar(a, n);
        std::memcpy(out, &v, n);
      } else {
        mem_.read(a, out, n);
      }
      return;
    case Space::Param:
      if (a + n > bank0_.size())
        throw Fault("parameter read of " + std::to_string(n) + " bytes past the kernel's " +
                    std::to_string(bank0_.size()) + "-byte constant bank");
      std::memcpy(out, &bank0_[a], n);
      return;
  }
}

void Runner::mem_write(Block& blk, Warp& w, unsigned lane, Space s, uint64_t a, const void* in, uint32_t n) {
  w.idle_yields = 0;   // a store is progress (an atomic that changed nothing puts the count back)
  switch (s) {
    case Space::Shared: {
      uint32_t off;
      Block& b = shared_block(blk, a, &off);
      if (uint64_t{off} + n > b.shared.size())
        throw Fault("shared write of " + std::to_string(n) + " bytes past the block's " + std::to_string(b.shared.size()) + " bytes");
      std::memcpy(&b.shared[off], in, n);
      return;
    }
    case Space::Local:
      if (a + n > w.local_size)
        throw Fault("local write of " + std::to_string(n) + " bytes past the thread's " + std::to_string(w.local_size) + " bytes");
      if (w.local.empty()) w.local.assign(size_t{w.local_size} * 32, 0);
      std::memcpy(&w.local[lane * w.local_size + a], in, n);
      return;
    case Space::Global:
      if (n <= 8 && (a % n) == 0) {
        uint64_t v = 0;
        std::memcpy(&v, in, n);
        mem_.store_scalar(a, n, v);
      } else {
        mem_.write(a, in, n);
      }
      return;
    case Space::Param:
      throw Fault("a store into the kernel's parameters (constant memory)");
  }
}

[[noreturn]] void Runner::fault(const Warp& w, const Instr& ins, unsigned lane, const std::string& why,
                                Err code) const {
  char at[64];
  std::snprintf(at, sizeof at, ", warp %u lane %u, at 0x%llx (", w.index, lane, static_cast<unsigned long long>(ins.pc));
  throw Error(code, "SASS kernel " + k_.name + at + to_text(ins) + "): " + why);
}

// ---- the grid ------------------------------------------------------------------

void Runner::seed_alloc_handshake(Block& blk) {
  // tcgen05.alloc.cta_group::2 is a handshake ptxas writes over the reserved
  // region .nv.reservedSmem.offset0 names: an mbarrier at +8 counting one
  // arrival, the phase each side waits for at +0x10 (flipped every round),
  // the column masks at +0x14 and +0x18. The pair's leader (its even CTA)
  // waits for the peer to have acknowledged the round before, claims the
  // columns and signals the peer (an arrive expecting 4 bytes, and st.async
  // of them); the peer waits for that, then acknowledges on the leader's
  // barrier. So the leader's barrier starts with its first phase complete and
  // the peer's does not -- the only start under which both of the kernel's
  // waits end, whichever CTA gets there first. Left zero, every 2-SM
  // kernel waits for ever.
  const uint32_t at = alloc_mbar_ != ~0u ? alloc_mbar_ : alloc_handshake_ != ~0u ? alloc_handshake_ + 8 : ~0u;
  if (at != ~0u && at + 8 <= blk.shared.size()) {
    const bool leader = (blk.rank & 1) == 0;
    const uint64_t bar = 0x001ffffeull | (uint64_t{0x7ffff800u | (leader ? 0x80000000u : 0u)} << 32);
    std::memcpy(&blk.shared[at], &bar, 8);
  }
}

void Runner::init_block(Block& blk, uint64_t linear) {
  blk.rank = 0;
  blk.own.blocks.assign(1, &blk);
  blk.own.phase = 0;
  blk.cluster = &blk.own;
  blk.ctaid[0] = static_cast<uint32_t>(linear % cfg_.grid[0]);
  blk.ctaid[1] = static_cast<uint32_t>((linear / cfg_.grid[0]) % cfg_.grid[1]);
  blk.ctaid[2] = static_cast<uint32_t>(linear / (static_cast<uint64_t>(cfg_.grid[0]) * cfg_.grid[1]));
  blk.shared.assign(shared_size_, 0);
  seed_alloc_handshake(blk);
  blk.st = exec::LaunchStats{};
  const uint32_t nwarps = (block_threads_ + 31) / 32;
  blk.st.warps = nwarps;
  blk.warps.resize(nwarps);
  const uint32_t nregs = std::max<uint32_t>(k_.regs + 8, 16);
  for (uint32_t i = 0; i < nwarps; ++i) {
    Warp& w = blk.warps[i];
    w = Warp{};
    w.index = i;
    w.nregs = std::min<uint32_t>(nregs, 256);
    w.r.assign(static_cast<size_t>(w.nregs) * 32, 0);
    const uint32_t first = i * 32;
    const uint32_t n = std::min<uint32_t>(32, block_threads_ - first);
    w.alive = n == 32 ? kAll : ((Mask{1} << n) - 1);
    for (unsigned l = 0; l < 32; ++l) w.pc[l] = entry_;
    w.local_size = local_size_;
    w.local.clear();   // made on the warp's first local access
  }
  for (Block::Barrier& b : blk.bars) b = Block::Barrier{};
  blk.mbar.clear();
  blk.gmma.clear();
  blk.tmem = exec::TensorMemory{};
}

exec::LaunchStats Runner::run() {
  block_threads_ = cfg_.block[0] * cfg_.block[1] * cfg_.block[2];
  if (block_threads_ == 0 || block_threads_ > 1024)
    throw Error(Err::LaunchConfig, "block of " + std::to_string(block_threads_) + " threads");
  const uint64_t blocks = static_cast<uint64_t>(cfg_.grid[0]) * cfg_.grid[1] * cfg_.grid[2];
  // The unit of work: a block, or with clusters a cluster, whose blocks must
  // be resident together.
  const uint64_t units = clustered_ ? blocks / csize_ : blocks;
  units_ = cfg_.cooperative ? 0 : units;   // a cooperative launch has every block running already
  std::atomic<uint64_t>& next = next_unit_;
  std::exception_ptr failure;
  std::mutex fail_mu;
  const auto worker = [&](unsigned t) {
    Block blk;
    for (;;) {
      const uint64_t i = next.fetch_add(1);
      if (i >= units) return;
      {
        std::lock_guard<std::mutex> g(fail_mu);
        if (failure) return;
      }
      try {
        if (clustered_) {
          run_cluster(i, t);
          continue;
        }
        init_block(blk, i);
        blk.worker = t;
        run_block(blk);
        std::lock_guard<std::mutex> g(stats_mu_);
        stats_.add(blk.st);
      } catch (...) {
        std::lock_guard<std::mutex> g(fail_mu);
        if (!failure) failure = std::current_exception();
        return;
      }
    }
  };
  unsigned threads = 1;
  if (!cfg_.cooperative) {
    if (const char* v = std::getenv("VGPU_THREADS"); v && *v) threads = static_cast<unsigned>(std::max(1, std::atoi(v)));
    else {
#ifdef VGPU_HAVE_HOST_CPUS
      threads = vgpu::host_cpus();
#else
      threads = std::max(1u, std::thread::hardware_concurrency());
#endif
    }
    threads = static_cast<unsigned>(std::min<uint64_t>(threads, units));
    // A kernel that can allocate while it runs (malloc/free) runs on one
    // host thread: the allocator's table is read unlocked by every access, so
    // an allocation in one block beside a load in another is a data race.
    for (const std::string& e : m_.cubin.externs)
      if (e == "malloc" || e == "free") threads = 1;
    // So does one that can launch kernels: its parameter buffers are
    // allocations too.
    if (m_.device_launches) threads = 1;
    // No more threads than the device has SMs for their clusters, so the
    // blocks resident at once can all have distinct SM numbers (sreg 0x43),
    // as the PTX engine caps its own.
    if (const uint32_t sms = profile_.limits.multiprocessors)
      threads = std::min<unsigned>(threads, std::max<unsigned>(1, sms / static_cast<unsigned>(clustered_ ? csize_ : 1)));
  }
  // Blocks that may wait on one another (vgpu/exec/lockstep.hpp): a kernel with a loop that polls
  // global memory runs the blocks the device holds at once together, taking turns.
  bool lockstep = false;
  uint64_t capacity = 0;
  if (!cfg_.cooperative && !clustered_) {
    const exec::LockstepMode mode = exec::lockstep_mode();
    if (mode == exec::LockstepMode::All || (mode == exec::LockstepMode::Auto && cfg_.resident_blocks && kernel_polls(m_, k_))) {
      lockstep = true;
      capacity = cfg_.resident_blocks ? cfg_.resident_blocks : units;
    }
  }
  if (cfg_.cooperative) {
    // Every block resident at once, taking turns, so one may wait on another.
    // With clusters, each cluster's blocks are built as run_cluster builds them (cluster k's
    // blocks consecutive here), every cluster resident at once.
    std::vector<Block> all(blocks);
    std::vector<Cluster> clusters(clustered_ ? units : 0);
    if (clustered_) {
      for (uint64_t k = 0; k < units; ++k) build_cluster(k, static_cast<unsigned>(k), &all[k * csize_], clusters[k]);
    } else {
      for (uint64_t i = 0; i < blocks; ++i) init_block(all[i], i);
    }
    for (bool live = true; live;) {
      live = false;
      bool progress = false;
      for (Block& blk : all)
        for (Warp& w : blk.warps) {
          if (!(w.alive & ~w.exited)) continue;
          live = true;
          if (tier_) turn<true>(blk, w, 256, progress);
          else turn<false>(blk, w, 256, progress);
          w.yield = false;
        }
      if (live && !progress) {
        bool broke = false;
        if (clustered_)
          for (Block& blk : all) broke = break_warpsync_standoff(blk) || broke;
        if (broke) continue;
        throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": every warp of the grid is waiting (deadlock)");
      }
    }
    for (Block& blk : all) stats_.add(blk.st);
  } else if (lockstep) {
    const exec::LockstepShape shape = exec::lockstep_shape(units, capacity, threads);
    exec::lockstep_note(k_.name.c_str(), units, shape);
    std::atomic<uint64_t> beat{0};
    std::atomic<bool> failed{false};
    const auto resident = [&](unsigned t) {
      try {
        exec::lockstep_worker<Block>(
            next, beat, units, shape.window, shape.initial, shape.chunk,
            [&](Block& b, uint64_t unit, size_t slot) {
              init_block(b, unit);
              // The SM number a block reads (SR_VIRTUALSMID): distinct among the blocks that are resident together.
              b.worker = static_cast<uint32_t>((uint64_t{t} * shape.window + slot) % std::max<uint32_t>(1, profile_.limits.multiprocessors));
            },
            [&](Block& b) { return !block_round(b, static_cast<int>(exec::kLockstepSlice)); },
            [](const Block& b) { return b.st.global_stores + b.st.shared_stores; },
            [this](const Block& b) { return fingerprint(b); },
            [&](Block& b) {
              std::lock_guard<std::mutex> g(stats_mu_);
              stats_.add(b.st);
            },
            [&] { return failed.load(std::memory_order_relaxed); });
      } catch (...) {
        failed.store(true, std::memory_order_relaxed);
        std::lock_guard<std::mutex> g(fail_mu);
        if (!failure) failure = std::current_exception();
      }
    };
    if (shape.workers <= 1) resident(0);
    else exec::run_parallel(shape.workers, resident);
  } else if (threads <= 1) {
    worker(0);
  } else {
    // The host threads are the pool's (vgpu/exec/worker_pool.hpp), parked between launches.
    exec::run_parallel(threads, worker);
  }
  if (failure) std::rethrow_exception(failure);
  stats_.blocks = blocks;
  if (tier_stats_) tier_add_total(stats_.instructions);
  return stats_;
}

// Lanes parked at a BSYNC whose barrier no longer names them. A callee that shares a Bx with its caller
// saves the caller's mask (BMOV.CLEAR) and puts it back on return, and BSYNC releases only the lanes the
// barrier holds while the callee runs. When a warp's lanes reach the callee's epilogue together from
// different call levels (ptxas's shared ABI wrapper, as in PyTorch's bessel_j1 kernels), the masks they
// restore differ and the lanes can be parked at different BSYNCs, each waiting for a mask the other level
// owns. Nothing can move then, so the group of lanes parked at the lowest pc goes on, as every BSYNC did
// before a callee's BSYNC stopped taking the caller's lanes with it.
bool Runner::break_bsync_standoff(Block& blk) {
  bool any = false;
  for (Warp& w : blk.warps) {
    int best = -1;
    uint64_t best_pc = ~uint64_t{0};
    for (unsigned l = 0; l < 32; ++l)
      if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::BSync && w.pc[l] < best_pc) {
        best_pc = w.pc[l];
        best = static_cast<int>(w.wait_arg[l]);
      }
    if (best < 0) continue;
    Mask released = 0;
    for (unsigned l = 0; l < 32; ++l)
      if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::BSync && static_cast<int>(w.wait_arg[l]) == best) {
        w.waiting &= ~(Mask{1} << l);
        w.wait_kind[l] = Wait::None;
        w.pc[l] += 16;
        released |= Mask{1} << l;
      }
    w.b[best] &= ~released;
    any = true;
  }
  return any;
}

// A WARPSYNC waits for every lane its mask names, and a BSYNC waits for every lane of its barrier. A
// warp can hold both at once: lane 0 alone in an `if (i < n)` block calls __syncthreads(), which ptxas
// compiles to WARPSYNC 0xffffffff; BAR.SYNC, while lanes 1 to 31 skip the block and park at the BSYNC
// that closes it, waiting for lane 0. Each waits for the other. An RTX 3080 Ti runs this (CUB's for_each
// does it in the last tile, and cupoch's kd-tree builder inside it), so when nothing can move the
// WARPSYNC lanes whose absent partners are all parked at a BSYNC go on; the rest of the warp follows
// when they reach the BSYNC.
bool Runner::break_warpsync_standoff(Block& blk) {
  bool any = false;
  for (Warp& w : blk.warps) {
    Mask ws = 0, bs = 0;
    for (unsigned l = 0; l < 32; ++l) {
      if (!((w.waiting >> l) & 1)) continue;
      if (w.wait_kind[l] == Wait::WarpSync) ws |= Mask{1} << l;
      else if (w.wait_kind[l] == Wait::BSync) bs |= Mask{1} << l;
    }
    if (!ws || !bs) continue;
    for (unsigned l = 0; l < 32; ++l) {
      if (!((ws >> l) & 1)) continue;
      const Mask need = w.wait_arg[l] & w.alive & ~w.exited;
      if ((need & ~ws & ~bs) != 0) continue;   // a named lane is neither at a WARPSYNC nor parked at a BSYNC
      w.waiting &= ~(Mask{1} << l);
      w.wait_kind[l] = Wait::None;
      w.pc[l] += 16;
      any = true;
    }
  }
  // The block barrier the lanes go on to counts live lanes, and the ones parked at the BSYNC cannot
  // arrive until the others have gone past it. The hardware counts the warp as arrived, so when nothing
  // can move, a barrier that every other live lane has reached is complete.
  for (Block::Barrier& b : blk.bars) {
    if (!b.arrived || b.expected) continue;
    uint32_t live = 0;
    for (const Warp& w : blk.warps) {
      Mask parked = 0;
      for (unsigned l = 0; l < 32; ++l)
        if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::BSync) parked |= Mask{1} << l;
      live += static_cast<uint32_t>(std::popcount(w.alive & ~w.exited & ~parked));
    }
    if (b.arrived < live) continue;
    for (Warp& w : blk.warps) {
      for (unsigned l = 0; l < 32; ++l)
        if (((w.waiting >> l) & 1) && w.wait_kind[l] == Wait::Bar) {
          w.waiting &= ~(Mask{1} << l);
          w.wait_kind[l] = Wait::None;
          w.pc[l] += 16;
        }
      w.b2r = b.red == 0 ? b.popc : b.red == 1 ? b.and_ : b.or_;
      w.b2r_pred = b.red == 0 ? b.popc != 0 : b.red == 1 ? b.and_ : b.or_;
    }
    b = Block::Barrier{};
    any = true;
  }
  return any;
}

// Where the live lanes of a block stand, for the message when none can move: lanes grouped by
// what they wait on (or "running") and by code address, so a deadlock names the instructions
// the lanes are stuck at instead of only saying that they are.
static std::string stall_summary(const Block& blk) {
  static const char* const kind[] = {"running", "BSYNC", "WARPSYNC", "BAR", "cluster"};
  std::string out;
  unsigned shown = 0;
  for (const Warp& w : blk.warps) {
    const Mask live = w.alive & ~w.exited;
    if (!live) continue;
    std::map<std::tuple<unsigned, uint64_t, unsigned>, Mask> groups;   // what, where, and which Bx for a BSYNC
    for (unsigned l = 0; l < 32; ++l)
      if ((live >> l) & 1) {
        const bool wait = (w.waiting >> l) & 1;
        const unsigned k = wait ? static_cast<unsigned>(w.wait_kind[l]) : 0u;
        groups[{k, w.pc[l], wait && k == static_cast<unsigned>(Wait::BSync) ? static_cast<unsigned>(w.wait_arg[l]) : 0u}] |= Mask{1} << l;
      }
    for (const auto& [at, lanes] : groups) {
      if (++shown > 40) return out + "; ...";
      char b[128];
      if (std::get<0>(at) == static_cast<unsigned>(Wait::BSync))
        std::snprintf(b, sizeof b, "; warp %u lanes %08x BSYNC B%u (members %08x) at 0x%llx", w.index, lanes, std::get<2>(at),
                      w.b[std::get<2>(at)] & live, static_cast<unsigned long long>(std::get<1>(at)));
      else
        std::snprintf(b, sizeof b, "; warp %u lanes %08x %s at 0x%llx", w.index, lanes, kind[std::get<0>(at)],
                      static_cast<unsigned long long>(std::get<1>(at)));
      out += b;
    }
  }
  return out;
}

void Runner::run_block(Block& blk) {
  while (block_round(blk, 4096)) {
  }
}

// One round of a block's warps: each runs until it waits, exits, or has had a turn of `slice`
// instructions, so warps that spin on one another all move. Says whether the block is still going.
// (A block alone gets a long turn; blocks that share a host thread, which may wait on one another,
// get a short one, see run().)
bool Runner::block_round(Block& blk, int slice) {
  bool live = false, progress = false;
  for (Warp& w : blk.warps) {
    if (!(w.alive & ~w.exited)) continue;
    live = true;
    if (tier_) turn<true>(blk, w, slice, progress);
    else turn<false>(blk, w, slice, progress);
    if (!w.yield) w.spin_yields = w.idle_yields = 0;
    w.yield = false;
  }
  if (!live) return false;
  if (break_spin_wait({&blk})) return true;
  if (!progress && break_warpsync_standoff(blk)) return true;
  if (!progress && break_bsync_standoff(blk)) return true;
  if (!progress)
    throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": every warp of block (" + std::to_string(blk.ctaid[0]) +
                                  ", " + std::to_string(blk.ctaid[1]) + ", " + std::to_string(blk.ctaid[2]) +
                                  ") is waiting (a barrier some threads never reach)" + stall_summary(blk));
  return true;
}

// A hash of what a block's warps hold: registers, predicates, program counters and the block's shared
// memory. Equal at two moments a long run of turns apart, the block has been going round a loop that
// changes nothing (a wait), not computing (see exec/lockstep.hpp).
uint64_t Runner::fingerprint(const Block& blk) const {
  uint64_t h = 0xcbf29ce484222325ull;
  for (const Warp& w : blk.warps) {
    h = exec::lockstep_fold(h, w.r.data(), w.r.size() * sizeof(uint32_t));
    h = exec::lockstep_fold(h, w.p, sizeof w.p);
    h = exec::lockstep_fold(h, w.ur, sizeof w.ur);
    h = exec::lockstep_fold(h, w.pc, sizeof w.pc);
    const Mask m[2] = {w.waiting, w.exited};
    h = exec::lockstep_fold(h, m, sizeof m);
    if (!w.local.empty()) h = exec::lockstep_fold(h, w.local.data(), w.local.size());
  }
  return exec::lockstep_fold(h, blk.shared.data(), blk.shared.size());
}

// Cluster k's blocks, resident together and taking turns as a block's warps
// do, so that one may wait on another (barrier.cluster, a peer's mbarrier,
// its shared memory). Ranks run x fastest, as %cluster_ctarank numbers them;
// clusters tile the grid x fastest.
void Runner::build_cluster(uint64_t k, unsigned worker, Block* blocks, Cluster& cl) {
  const uint64_t ncx = cfg_.grid[0] / cshape_[0], ncy = cfg_.grid[1] / cshape_[1];
  const uint64_t kx = k % ncx, ky = (k / ncx) % ncy, kz = k / (ncx * ncy);
  for (uint32_t r = 0; r < csize_; ++r) {
    const uint64_t x = kx * cshape_[0] + r % cshape_[0], y = ky * cshape_[1] + (r / cshape_[0]) % cshape_[1],
                   z = kz * cshape_[2] + r / (cshape_[0] * cshape_[1]);
    init_block(blocks[r], x + y * cfg_.grid[0] + z * uint64_t{cfg_.grid[0]} * cfg_.grid[1]);
    blocks[r].rank = r;
    blocks[r].worker = worker;
    seed_alloc_handshake(blocks[r]);   // now that the block knows its rank
    blocks[r].cluster = &cl;
    cl.blocks.push_back(&blocks[r]);
  }
}

void Runner::run_cluster(uint64_t k, unsigned worker) {
  std::vector<Block> blocks(csize_);
  Cluster cl;
  build_cluster(k, worker, blocks.data(), cl);
  for (;;) {
    bool live = false, progress = false;
    for (Block& blk : blocks)
      for (Warp& w : blk.warps) {
        if (!(w.alive & ~w.exited)) continue;
        live = true;
        if (tier_) turn<true>(blk, w, 4096, progress);
        else turn<false>(blk, w, 4096, progress);
        if (!w.yield) w.spin_yields = w.idle_yields = 0;
        w.yield = false;
      }
    if (!live) break;
    {
      std::vector<Block*> all;
      for (Block& blk : blocks) all.push_back(&blk);
      if (break_spin_wait(all)) continue;
    }
    if (!progress) {
      bool broke = false;
      for (Block& blk : blocks) broke = break_warpsync_standoff(blk) || broke;
      if (broke) continue;
    }
    if (!progress)
      throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": every warp of cluster " + std::to_string(k) +
                                    " is waiting (a barrier some threads never reach)");
  }
  std::lock_guard<std::mutex> g(stats_mu_);
  for (Block& blk : blocks) stats_.add(blk.st);
}

exec::InstClass inst_class(Op op);   // exec_ops.inc
bool compute_straight(const Instr& ins);   // see annotate_instr

// Every warp of these blocks is parked or spinning (a YIELD loop with no other lane group to run), and some
// lanes stand at a BSYNC: a lane that won a spin lock waits at the BSYNC closing the loop for the lanes that
// spin because it holds the lock, or the lock is held by a lane of another warp that waits the same way.
// NVIDIA's scheduler guarantees forward progress for a thread in a YIELD loop, so after a while some parked
// lanes go on. Which ones? The lock's holder is inside its critical section, the most deeply nested region
// there is, so the lanes parked at the barrier that was set up last go, and the others stay until the next
// stall: letting the outer region's lanes go too runs them on past a barrier the spinners still count them
// in, and the barriers then wait for each other.
bool Runner::break_spin_wait(const std::vector<Block*>& blocks) {
  constexpr uint32_t kIdleYields = 64;   // two turns of every lane group of a warp, and then some
  Warp* best = nullptr;
  unsigned bar = 0;
  uint64_t stamp = 0;
  for (Block* blk : blocks)
    for (Warp& w : blk->warps) {
      if (!(w.alive & ~w.exited)) continue;
      // A warp still doing something other than spinning: its one lane group has not yielded twice in a row, and
      // several groups taking turns have not yielded kIdleYields times without a write between them.
      if (w.runnable() && w.spin_yields < 2 && w.idle_yields < kIdleYields) return false;
      for (unsigned l = 0; l < 32; ++l) {
        if (!((w.waiting >> l) & 1) || w.wait_kind[l] != Wait::BSync) continue;
        const unsigned x = static_cast<unsigned>(w.wait_arg[l]) & 15;
        if (!best || w.bssy_stamp[x] > stamp) {
          best = &w;
          bar = x;
          stamp = w.bssy_stamp[x];
        }
      }
    }
  if (!best) return false;
  Warp& w = *best;
  for (unsigned l = 0; l < 32; ++l) {
    if (!((w.waiting >> l) & 1) || w.wait_kind[l] != Wait::BSync || (static_cast<unsigned>(w.wait_arg[l]) & 15) != bar) continue;
    w.b[bar] &= ~(Mask{1} << l);
    w.waiting &= ~(Mask{1} << l);
    w.wait_kind[l] = Wait::None;
    w.pc[l] += 16;
  }
  w.spin_yields = 0;
  for (Block* blk : blocks)
    for (Warp& o : blk->warps) o.idle_yields = 0;   // the released lanes get their turn before the stall is called again
  return true;
}

// One instruction for one group of lanes. False when no lane can run.
bool Runner::step_warp(Block& blk, Warp& w) {
  Mask run = w.runnable();
  if (w.ws_held && (!run || w.give_way)) {
    // Everything released so far has run as far as it can, or is spinning
    // (a YIELD, a backward branch it has taken a good many times): the next
    // set of lanes that named another mask goes (see Op::WARPSYNC).
    const uint32_t arg = w.wait_arg[std::countr_zero(w.ws_held)];
    Mask next = 0;
    for (Mask m = w.ws_held; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      if (w.wait_arg[l] == arg) next |= Mask{1} << l;
    }
    w.ws_held &= ~next;
    for (Mask m = next; m; m &= m - 1) {
      const unsigned l = static_cast<unsigned>(std::countr_zero(m));
      w.waiting &= ~(Mask{1} << l);
      w.wait_kind[l] = Wait::None;
      w.pc[l] += 16;
    }
    run = w.runnable();
  }
  if (!run) return false;
  // The group at the lowest address -- or, when the running group has been
  // spinning, the next group up, so it gives the others a turn. One pass
  // over the runnable lanes finds the lowest address and its lanes (every
  // instruction comes through here, so it is kept short).
  // The comparisons are kept out of branches: in a diverged warp the lanes' addresses differ
  // unpredictably, and a mispredicted branch per lane was the largest cost of this loop.
  uint64_t lo = w.pc[std::countr_zero(run)];
  Mask group = 0;
  // With the tier on, a warp whose runnable lanes all stand at one address (the usual case) is found by comparing
  // the 32 addresses at once: the same lo and group the pass below finds.
  if (tier_) {
    if (const Mask at = tier_lanes_at(w, lo); (run & ~at) == 0) {
      group = run;
    } else {   // diverged: the lowest address among the runnable lanes, and the lanes there
      lo = tier_lowest_pc(w, run, lo);
      group = tier_lanes_at(w, lo) & run;
    }
  } else
  for (Mask m = run; m; m &= m - 1) {
    const unsigned l = static_cast<unsigned>(std::countr_zero(m));
    const uint64_t p = w.pc[l];
    const bool lower = p < lo;
    group &= lower ? Mask{0} : kAll;
    lo = lower ? p : lo;
    group |= static_cast<Mask>(p == lo) << l;
  }
  uint64_t pick = lo;
  // The group a YIELD let in keeps the warp until it yields in turn or can no longer run, so that the
  // warp's uniform registers (a VOTEU, POPC, UFLO sequence, say) are not shared out between two groups
  // one instruction at a time.
  if (w.slice) {
    const Mask g = w.slice & run;
    if (g) {
      uint64_t slo = ~uint64_t{0};
      for (Mask m = g; m; m &= m - 1) slo = std::min(slo, w.pc[std::countr_zero(m)]);
      pick = slo;
      group = 0;
      for (Mask m = g; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(std::countr_zero(m));
        if (w.pc[l] == slo) group |= Mask{1} << l;
      }
      w.give_way = false;
    } else {
      w.slice = 0;
    }
  }
  if (w.give_way) {
    w.give_way = false;
    // The group to let in is the next one up from the group that gave way, wrapping round to the lowest address:
    // every group spinning here gets its turn. (Measured from the lowest address instead, a lane that had won a
    // spin lock and branched past the loop was the "current" group, nothing lay above it, and the lanes still
    // spinning below it ran for ever. Measured from the lowest address in every case, as this first did, a
    // group that was let in and yielded in turn was followed by itself again: with two groups spinning, a
    // third that held the lock they wait for never ran.)
    uint64_t next = ~uint64_t{0}, lowest = ~uint64_t{0};
    for (Mask m = run; m; m &= m - 1) {
      const uint64_t p = w.pc[std::countr_zero(m)];
      if (p == w.yield_pc) continue;   // the group that gave way
      lowest = std::min(lowest, p);
      if (p > w.yield_pc) next = std::min(next, p);
    }
    if (next == ~uint64_t{0}) next = lowest;
    if (next == ~uint64_t{0}) {
      // Nothing else can run. A group that yields over and over like this is spinning on something only
      // the lanes parked at a BSYNC can supply (a lane that won a spin lock stands at the BSYNC closing
      // the loop, waiting for the lanes that are spinning because it holds the lock). NVIDIA's scheduler
      // guarantees forward progress for a thread in a YIELD loop, so after a while the parked lanes go on
      // (break_spin_wait).
      ++w.spin_yields;   // run_block decides, with every warp in view, whether the parked lanes go on
    } else {
      w.spin_yields = 0;
    }
    if (next != ~uint64_t{0}) {
      pick = next;
      group = 0;
      for (Mask m = run; m; m &= m - 1) {
        const unsigned l = static_cast<unsigned>(std::countr_zero(m));
        if (w.pc[l] == pick) group |= Mask{1} << l;
      }
      w.slice = group;
    }
  }
  // A straight-line run: arithmetic, conversions and plain loads and stores
  // leave the lanes' program counters alone, so the group runs on through
  // them without coming back here or moving its 32 counters each time; they
  // are set once, where the run stops. Control flow, waits and anything that
  // reads its own pc (LEPC) end the run and take the general path.
  uint64_t pc = pick;
  const Code* code = nullptr;
  // The section the warp ran in last is most often the one it is in, so Module::code_at is asked only when it moves
  // to another (a call into a device function, a return). A pc in no section is fetch's fault, as before.
  if (w.code_hint && pc - w.code_hint->base < 16 * w.code_hint->count) {
    code = w.code_hint;
  } else {
    fetch(pc, &code);
    w.code_hint = code;
  }
  if (tier_ && !w.mactive_force) {
    if (const tier::Program* tp = code->tier_program()) return tier_run(blk, w, group, pick, *code, *tp);
  }
  const Instr* ins = &code->instr((pc - code->base) / 16);
  const uint64_t end = code->base + 16 * code->count;
  for (unsigned n = 0;; ++n) {
    // A warp's first store to global memory ends its turn, once, so that the block's other warps issue the loads
    // their prologues make before it lands. On a card every warp of a block starts within a few cycles of the
    // others, and a store posts only after a load's round trip, so a warp never sees a store of a warp-mate that
    // started with it: Triton's `step += 1` on a one-element tensor has every warp load, add and store the same
    // word, and is right only because of that. Run to the end one warp at a time, warp k would load what warps
    // 0..k-1 stored and add one more each. Any order is a legal one for code with no synchronization; this is
    // the one hardware gives. A block of one warp has no warp-mates to wait for.
    if (w.store_gate && (ins->op == Op::STG || ins->op == Op::ST)) {
      w.store_gate = false;
      if (blk.warps.size() > 1) {
        if (pc != pick)
          for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = pc;
        w.yield = true;
        return true;
      }
    }
    if (++w.steps > max_steps_)
      throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": a warp ran " + std::to_string(max_steps_) +
                                    " instructions (an infinite loop?)");
    if (n < 256 && pc + 16 < end && ins->straight) {
      execute(blk, w, *ins, group, /*advance=*/false);
      pc += 16;
      ins = &code->instr((pc - code->base) / 16);
      continue;
    }
    if (pc != pick)
      for (Mask m = group; m; m &= m - 1) w.pc[std::countr_zero(m)] = pc;
    execute(blk, w, *ins, group);
    return true;
  }
}

// Whether an instruction leaves every lane's pc to its caller (see step_warp).
bool compute_straight(const Instr& ins) {
  switch (ins.op) {
    case Op::Unknown:
    // control flow, barriers and waits
    case Op::BRA: case Op::BRX: case Op::JMP: case Op::CALL: case Op::RET: case Op::EXIT:
    case Op::BSSY: case Op::BSYNC: case Op::BREAK: case Op::WARPSYNC: case Op::BAR:
    case Op::NOP: case Op::YIELD: case Op::NANOSLEEP: case Op::BPT: case Op::DEPBAR:
    case Op::MEMBAR: case Op::ERRBAR: case Op::CCTL: case Op::BMOV: case Op::ENDCOLLECTIVE: case Op::UCGABAR:
    // what waits on, or wakes, other lanes or warps
    case Op::ATOMS: case Op::SYNCS: case Op::ARRIVES: case Op::LDGDEPBAR:
    case Op::SHFL: case Op::VOTE: case Op::VOTEU: case Op::MATCH: case Op::REDUX:
      return false;
    case Op::MOV:
      return ins.mnemonic != "LEPC";
    default:
      break;
  }
  // The arithmetic, conversion and memory families only.
  switch (ins.op) {
    case Op::LDG: case Op::STG: case Op::LDS: case Op::STS: case Op::LDL: case Op::STL:
    case Op::LD: case Op::ST: case Op::LDC:
      return true;
    default:
      return inst_class(ins.op) == exec::InstClass::Integer || inst_class(ins.op) == exec::InstClass::Fp32 ||
             inst_class(ins.op) == exec::InstClass::Fp64 || inst_class(ins.op) == exec::InstClass::Fp16 ||
             inst_class(ins.op) == exec::InstClass::BitConvert;
  }
}

// Mask of the group's lanes whose guard predicate holds.
Mask guard_mask(const Warp& w, const Instr& ins, Mask group) {
  if (ins.guard == kPT && !ins.guard_not) return group;
  Mask g;
  if (ins.guard_uniform) g = (ins.guard == kPT || w.up[ins.guard]) ? kAll : 0;
  else g = ins.guard == kPT ? kAll : w.p[ins.guard];
  if (ins.guard_not) g = ~g;
  return group & g;
}

#include "exec_ops.inc"
#include "tier_ops.inc"

}  // namespace

// The tier of a code section: its instructions analysed once (tier.cpp), each bound to its handler (tier_ops.inc).
const tier::Program* Code::tier_program() const {
  Decoded& d = *decoded_;
  if (const tier::Program* p = d.tier.load(std::memory_order_acquire)) return p;
  std::call_once(d.tier_once, [&] {
    if (count == 0) return;
    (void)instr(0);   // decode the section if nobody has yet
    auto prog = std::make_shared<tier::Program>();
    prog->ops.resize(count);
    for (size_t i = 0; i < count; ++i) {
      tier::MOp& op = prog->ops[i];
      tier::analyze(instr(i), sm, *prog, op);
      op.fn = tier_handler_for(op.form);
      // VGPU_TIER_OFF=form,form,... (names as VGPU_TIER_STATS shows them in tier::form_name: iadd3, bsync, ...) leaves those forms to the
      // interpreter, for finding which form a difference comes from.
      static const std::string off = [] { const char* v = std::getenv("VGPU_TIER_OFF"); return v && *v ? std::string(",") + v + "," : std::string(); }();
      if (!off.empty() && off.find(std::string(",") + tier::form_name(op.form) + ",") != std::string::npos) {
        op.form = tier::Form::None;
        op.fn = nullptr;
      }
      if (op.fn) {
        ++prog->native;
        prog->max_reg = std::max(prog->max_reg, op.maxreg);
      }
    }
    prog->const_index.clear();
    d.tier_owner = prog;
    d.tier.store(prog.get(), std::memory_order_release);
  });
  return d.tier.load(std::memory_order_acquire);
}

// For sass::unsupported (module.cpp).
bool runs_instr(const Instr& ins) { return executes(ins); }

// Called once on each instruction as its code section is decoded (module.cpp):
// sets the facts Instr::cls, ::tensor and ::straight hold, which execute() and
// step_warp() used to work out again from the opcode on every execution.
void annotate_instr(Instr& ins) {
  ins.cls = static_cast<uint8_t>(inst_class(ins.op));
  ins.tensor = ins.op == Op::HMMA || ins.op == Op::IMMA || ins.op == Op::BMMA || ins.op == Op::DMMA ||
               ins.op == Op::QMMA || ins.op == Op::HGMMA || ins.op == Op::IGMMA || ins.op == Op::BGMMA ||
               ins.op == Op::QGMMA;
  ins.straight = compute_straight(ins);
}

namespace {

void ChildRunner::operator()(Child& c, Launches& out) {
  const CubinKernel& k = m.cubin.kernels[c.kernel];
  // The parameters back into one argument per parameter, from where the
  // cubin puts each.
  std::vector<std::vector<uint8_t>> args;
  for (const CubinParam& p : k.params) {
    if (p.offset + p.size > c.params.size()) c.params.resize(p.offset + p.size, 0);
    args.emplace_back(c.params.begin() + p.offset, c.params.begin() + p.offset + p.size);
  }
  exec::LaunchConfig cc = base;
  cc.grid = c.grid;
  cc.block = c.block;
  cc.shared_bytes = c.shared;
  cc.cluster = k.cluster;   // __cluster_dims__, as a host launch applies it
  cc.cooperative = false;
  cc.coop_workspace = 0;
  cc.resident_blocks = 0;   // the parent's figure is for the parent's kernel
  Runner r(m, k, cc, args, mem, profile);
  r.set_launches(&out);
  add_stats(r.run());
}

}  // namespace

exec::LaunchStats launch(const Module& m, const std::string& kernel, const exec::LaunchConfig& cfg,
                         const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                         const DeviceProfile& profile) {
  const auto it = m.kernel_index.find(kernel);
  if (it == m.kernel_index.end()) throw Error(Err::NotFound, "no kernel " + kernel + " in the module");
  Launches dl;
  dl.mem = &mem;
  exec::LaunchStats stats;
  {
    Runner r(m, m.cubin.kernels[it->second], cfg, args, mem, profile);
    r.set_launches(&dl);
    stats = r.run();
  }
  if (!dl.queue.empty() || !dl.by_buffer.empty() || dl.extra.blocks) {
    ChildRunner runner{m, cfg, mem, profile, &stats};
    exec::devrt::complete_children(dl, mem, runner);
  }
  return stats;
}

}  // namespace vgpu::sass
