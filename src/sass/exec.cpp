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
#include "vgpu/sass/exec.hpp"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cfenv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>

#include "vgpu/error.hpp"
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

// Why a lane is not runnable.
enum class Wait : uint8_t { None, BSync, WarpSync, Bar };

struct Warp {
  uint32_t index = 0;                 // within the block
  uint32_t nregs = 0;
  std::vector<uint32_t> r;            // r[reg * 32 + lane]
  Mask p[8] = {0, 0, 0, 0, 0, 0, 0, kAll};
  uint32_t ur[256] = {};
  bool up[8] = {false, false, false, false, false, false, false, true};
  uint64_t pc[32] = {};               // per-lane code address
  Mask alive = 0;                     // lanes that are threads of the block
  Mask exited = 0;
  Mask waiting = 0;
  Wait wait_kind[32] = {};
  uint32_t wait_arg[32] = {};         // the barrier a lane waits on
  Mask b[16] = {};                    // convergence barriers
  std::vector<uint8_t> local;         // per-lane local memory, local_size bytes each
  uint32_t local_size = 0;
  uint64_t steps = 0;
  // Fairness: how many backward branches the running group has taken while
  // other lanes waited to run.
  uint32_t spins = 0;
  bool give_way = false;
  uint32_t b2r = 0;                   // the last block barrier reduction's result (B2R.RESULT)
  bool b2r_pred = false;

  Mask runnable() const { return alive & ~exited & ~waiting; }
  uint32_t& reg(unsigned n, unsigned lane) { return r[n * 32 + lane]; }
};

struct Block {
  uint32_t ctaid[3] = {};
  std::vector<Warp> warps;
  std::vector<uint8_t> shared;
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
};

struct Fault : std::exception {
  std::string msg;
  explicit Fault(std::string m) : msg(std::move(m)) {}
  const char* what() const noexcept override { return msg.c_str(); }
};

std::mutex& atomic_lock(uint64_t addr) {
  static std::mutex locks[64];
  return locks[(addr >> 3) % 64];
}

class Runner {
 public:
  Runner(const Module& m, const CubinKernel& k, const exec::LaunchConfig& cfg,
         const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem, const DeviceProfile& profile)
      : m_(m), k_(k), cfg_(cfg), mem_(mem), profile_(profile) {
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
    local_size_ = std::max<uint32_t>({k.min_stack, k.frame_size, 16});
    local_size_ = (local_size_ + 15) & ~15u;
    shared_size_ = static_cast<uint64_t>(k.shared_bytes) + cfg.shared_bytes;
  }

  exec::LaunchStats run();

 private:
  // ---- setup ----
  void build_bank0(const std::vector<std::vector<uint8_t>>& args);

  // ---- per-block ----
  void run_block(Block& blk);
  void init_block(Block& blk, uint64_t linear);
  bool step_warp(Block& blk, Warp& w);
  void execute(Block& blk, Warp& w, const Instr& ins, Mask group);

  // ---- state access ----
  const Instr& fetch(uint64_t addr, const Code** code_out = nullptr);
  uint32_t cbank32(unsigned bank, uint64_t offset) const;
  uint32_t read32(Warp& w, const Operand& o, unsigned lane);
  uint64_t read64(Warp& w, const Operand& o, unsigned lane);
  void write32(Warp& w, const Operand& o, unsigned lane, uint32_t v);
  void write64(Warp& w, const Operand& o, unsigned lane, uint64_t v);
  bool pred(Warp& w, const Operand& o, unsigned lane) const;
  void set_pred(Warp& w, const Operand& o, unsigned lane, bool v);
  uint32_t sreg(const Block& blk, const Warp& w, unsigned idx, unsigned lane) const;

  // ---- memory ----
  enum class Space { Global, Shared, Local };
  Space classify(uint64_t generic, uint64_t* offset) const;
  void mem_read(Block& blk, Warp& w, unsigned lane, Space s, uint64_t addr, void* out, uint32_t n);
  void mem_write(Block& blk, Warp& w, unsigned lane, Space s, uint64_t addr, const void* in, uint32_t n);
  uint64_t generic_addr(Warp& w, const Instr& ins, unsigned lane, const Operand& addr_op, bool wide);

  [[noreturn]] void fault(const Warp& w, const Instr& ins, unsigned lane, const std::string& why) const;

  // ---- instruction groups (exec_ops.inc) ----
  void exec_int(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_float(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_mem(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void exec_warp(Block& blk, Warp& w, const Instr& ins, Mask ex, Mask group);
  bool exec_control(Block& blk, Warp& w, const Instr& ins, Mask ex, Mask group);   // true: it set the pcs
  void exec_mma(Block& blk, Warp& w, const Instr& ins, Mask ex);
  void builtin_call(Block& blk, Warp& w, const Instr& ins, Mask ex, const std::string& name);
  uint32_t ival(Warp& w, const Operand& o, unsigned lane);   // with -/~ applied

  const Module& m_;
  const CubinKernel& k_;
  const exec::LaunchConfig& cfg_;
  MemoryManager& mem_;
  const DeviceProfile& profile_;
  std::vector<uint8_t> bank0_;
  std::map<unsigned, uint64_t> bank_va_, bank_size_;
  uint64_t entry_ = 0;
  uint32_t local_size_ = 0;
  uint64_t shared_size_ = 0;
  uint32_t block_threads_ = 0;
  exec::LaunchStats stats_;
  std::mutex stats_mu_;
  // The heap malloc() draws from, shared by every launch on the device.
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
  put32(0x00, cfg_.block[0]);
  put32(0x04, cfg_.block[1]);
  put32(0x08, cfg_.block[2]);
  put32(0x0c, cfg_.grid[0]);
  put32(0x10, cfg_.grid[1]);
  put32(0x14, cfg_.grid[2]);
  put64(0x18, kSharedWindow);
  put64(0x20, kLocalWindow);
  const uint32_t stack = std::max<uint32_t>({k_.min_stack, k_.frame_size, 16});
  put32(0x28, (stack + 15) & ~15u);
  put64(0x118, 0);   // the descriptor's bits are a cache policy: nothing here reads them
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
  return c->instrs[(addr - c->base) / 16];
}

uint32_t Runner::read32(Warp& w, const Operand& o, unsigned lane) {
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

uint64_t Runner::read64(Warp& w, const Operand& o, unsigned lane) {
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
    default: return static_cast<uint64_t>(read32(w, o, lane));
  }
}

void Runner::write32(Warp& w, const Operand& o, unsigned lane, uint32_t v) {
  if (o.kind == Kind::Reg) {
    if (o.reg == kRZ) return;
    w.reg(o.reg, lane) = v;
  } else if (o.kind == Kind::UReg) {
    if (o.reg >= (m_.sm >= 100 ? 255u : 63u)) return;
    w.ur[o.reg] = v;
  }
}

void Runner::write64(Warp& w, const Operand& o, unsigned lane, uint64_t v) {
  write32(w, o, lane, static_cast<uint32_t>(v));
  Operand hi = o;
  if (o.kind == Kind::Reg && o.reg == kRZ) return;
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
    case 0x32: return static_cast<uint32_t>(shared_size_);              // SR_SMEMSZ
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
  *off = g;
  return Space::Global;
}

void Runner::mem_read(Block& blk, Warp& w, unsigned lane, Space s, uint64_t a, void* out, uint32_t n) {
  switch (s) {
    case Space::Shared:
      if (a + n > blk.shared.size())
        throw Fault("shared read of " + std::to_string(n) + " bytes at 0x" + [&] { char b[24]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)a); return std::string(b); }() +
                    ", past the block's " + std::to_string(blk.shared.size()) + " bytes");
      std::memcpy(out, &blk.shared[a], n);
      if (std::getenv("VGPU_SASS_DBGMEM")) std::fprintf(stderr, "    shared read %u bytes at 0x%llx lane %u: %08x\n", n, (unsigned long long)a, lane, *static_cast<uint32_t*>(out));
      return;
    case Space::Local:
      if (a + n > w.local_size)
        throw Fault("local read of " + std::to_string(n) + " bytes at 0x" + [&] { char b[24]; std::snprintf(b, sizeof b, "%llx", (unsigned long long)a); return std::string(b); }() +
                    ", past the thread's " + std::to_string(w.local_size) + " bytes");
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
  }
}

void Runner::mem_write(Block& blk, Warp& w, unsigned lane, Space s, uint64_t a, const void* in, uint32_t n) {
  switch (s) {
    case Space::Shared:
      if (a + n > blk.shared.size())
        throw Fault("shared write of " + std::to_string(n) + " bytes past the block's " + std::to_string(blk.shared.size()) + " bytes");
      std::memcpy(&blk.shared[a], in, n);
      if (std::getenv("VGPU_SASS_DBGMEM")) std::fprintf(stderr, "    shared write %u bytes at 0x%llx lane %u: %08x\n", n, (unsigned long long)a, lane, *static_cast<const uint32_t*>(in));
      return;
    case Space::Local:
      if (a + n > w.local_size)
        throw Fault("local write of " + std::to_string(n) + " bytes past the thread's " + std::to_string(w.local_size) + " bytes");
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
  }
}

[[noreturn]] void Runner::fault(const Warp& w, const Instr& ins, unsigned lane, const std::string& why) const {
  char b[160];
  std::snprintf(b, sizeof b, "SASS kernel %s, warp %u lane %u, at 0x%llx (%s): ", k_.name.c_str(), w.index, lane,
                static_cast<unsigned long long>(ins.pc), to_text(ins).c_str());
  throw Error(Err::OutOfBounds, b + why);
}

// ---- the grid ------------------------------------------------------------------

void Runner::init_block(Block& blk, uint64_t linear) {
  blk.ctaid[0] = static_cast<uint32_t>(linear % cfg_.grid[0]);
  blk.ctaid[1] = static_cast<uint32_t>((linear / cfg_.grid[0]) % cfg_.grid[1]);
  blk.ctaid[2] = static_cast<uint32_t>(linear / (static_cast<uint64_t>(cfg_.grid[0]) * cfg_.grid[1]));
  blk.shared.assign(shared_size_, 0);
  const uint32_t nwarps = (block_threads_ + 31) / 32;
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
    w.local.assign(static_cast<size_t>(local_size_) * 32, 0);
  }
  for (Block::Barrier& b : blk.bars) b = Block::Barrier{};
}

exec::LaunchStats Runner::run() {
  block_threads_ = cfg_.block[0] * cfg_.block[1] * cfg_.block[2];
  if (block_threads_ == 0 || block_threads_ > 1024)
    throw Error(Err::InvalidValue, "block of " + std::to_string(block_threads_) + " threads");
  const uint64_t blocks = static_cast<uint64_t>(cfg_.grid[0]) * cfg_.grid[1] * cfg_.grid[2];
  std::atomic<uint64_t> next{0};
  std::exception_ptr failure;
  std::mutex fail_mu;
  const auto worker = [&] {
    Block blk;
    for (;;) {
      const uint64_t i = next.fetch_add(1);
      if (i >= blocks) return;
      {
        std::lock_guard<std::mutex> g(fail_mu);
        if (failure) return;
      }
      try {
        init_block(blk, i);
        run_block(blk);
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
    threads = static_cast<unsigned>(std::min<uint64_t>(threads, blocks));
  }
  if (cfg_.cooperative) {
    // Every block resident at once, taking turns, so one may wait on another.
    std::vector<Block> all(blocks);
    for (uint64_t i = 0; i < blocks; ++i) init_block(all[i], i);
    for (bool live = true; live;) {
      live = false;
      bool progress = false;
      for (Block& blk : all)
        for (Warp& w : blk.warps) {
          if (!(w.alive & ~w.exited)) continue;
          live = true;
          for (int n = 0; n < 256 && (w.alive & ~w.exited); ++n)
            if (step_warp(blk, w)) progress = true;
            else break;
        }
      if (live && !progress) throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": every warp of the grid is waiting (deadlock)");
    }
  } else if (threads <= 1) {
    worker();
  } else {
    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back(worker);
    for (std::thread& t : pool) t.join();
  }
  if (failure) std::rethrow_exception(failure);
  stats_.blocks = blocks;
  return stats_;
}

void Runner::run_block(Block& blk) {
  // Warps take turns: each runs until it waits, exits, or has had a turn of
  // a few thousand instructions, so warps that spin on one another all move.
  for (;;) {
    bool live = false, progress = false;
    for (Warp& w : blk.warps) {
      if (!(w.alive & ~w.exited)) continue;
      live = true;
      for (int n = 0; n < 4096; ++n) {
        if (!(w.alive & ~w.exited)) break;
        if (!step_warp(blk, w)) break;
        progress = true;
      }
    }
    if (!live) return;
    if (!progress)
      throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": every warp of block (" + std::to_string(blk.ctaid[0]) +
                                    ", " + std::to_string(blk.ctaid[1]) + ", " + std::to_string(blk.ctaid[2]) +
                                    ") is waiting (a barrier some threads never reach)");
  }
}

// One instruction for one group of lanes. False when no lane can run.
bool Runner::step_warp(Block& blk, Warp& w) {
  const Mask run = w.runnable();
  if (!run) return false;
  // The group at the lowest address -- or, when the running group has been
  // spinning, the next group up, so it gives the others a turn.
  uint64_t lo = ~uint64_t{0}, cur = w.pc[std::countr_zero(run)];
  for (unsigned l = 0; l < 32; ++l)
    if ((run >> l) & 1) lo = std::min(lo, w.pc[l]);
  uint64_t pick = lo;
  if (w.give_way) {
    uint64_t next = ~uint64_t{0};
    for (unsigned l = 0; l < 32; ++l)
      if (((run >> l) & 1) && w.pc[l] > cur) next = std::min(next, w.pc[l]);
    pick = next != ~uint64_t{0} ? next : lo;
    w.give_way = false;
  }
  Mask group = 0;
  for (unsigned l = 0; l < 32; ++l)
    if (((run >> l) & 1) && w.pc[l] == pick) group |= Mask{1} << l;
  const Instr& ins = fetch(pick);
  if (++w.steps > cfg_.max_steps)
    throw Error(Err::ExecLimit, "SASS kernel " + k_.name + ": a warp ran " + std::to_string(cfg_.max_steps) +
                                  " instructions (an infinite loop?)");
  execute(blk, w, ins, group);
  return true;
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

}  // namespace

exec::LaunchStats launch(const Module& m, const std::string& kernel, const exec::LaunchConfig& cfg,
                         const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                         const DeviceProfile& profile) {
  const auto it = m.kernel_index.find(kernel);
  if (it == m.kernel_index.end()) throw Error(Err::NotFound, "no kernel " + kernel + " in the module");
  Runner r(m, m.cubin.kernels[it->second], cfg, args, mem, profile);
  return r.run();
}

}  // namespace vgpu::sass
