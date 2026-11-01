// Host-GPU mode, the logic that needs no card (docs/perf-design/09-host-gpu-spike.md).
//
// With `VGPU_HOST_GPU=exact` a launch may run on the real GPU of the machine instead of in the simulator, when its result is
// the same bits either way. Three decisions are made here, with no CUDA and no simulator state, so they are unit tested on any
// machine (tests/unit/test_host_gpu_logic.cpp):
//   * exact_opcode_allowed: may a kernel that holds this SASS instruction run on a card with the simulator's bits?
//   * Residency: which 64 KiB chunks of an allocation the simulator holds, the card holds, or both hold (the design for a
//     persistent card copy; the first prototype moves everything on every launch and does not need it, see the doc).
//   * should_offload / SimTimeTable: does the card path (launch, copy in, copy out) beat the simulator's estimate?
// The part that talks to the driver is src/runtime/host_gpu.cpp.
#pragma once

#include <algorithm>
#include <cstdint>
#include <map>
#include <optional>
#include <utility>
#include <string>
#include <vector>

namespace vgpu::hostgpu {

enum class Mode { Off, Exact };

// "exact" turns the mode on; anything else, empty or null is off. (`approx` is not built: a typo must not turn it on.)
inline Mode mode_from_text(const char* text) {
  return text && std::string(text) == "exact" ? Mode::Exact : Mode::Off;
}

// ---- eligibility ------------------------------------------------------------------------------------------------------------

// The instructions whose result is defined bit for bit by the architecture and that the simulator computes as the card does:
// integer and IEEE float arithmetic and conversions, predicates, moves, loads and stores, barriers and plain control flow.
// Measured in the spike (09 doc, M2): every kernel of the Rodinia, PolyBench and HeCBench launches sampled whose code stayed
// inside this set agreed with the card to the bit, and every kernel that differed held one instruction outside it (MUFU, a
// compare-and-swap, a floating-point atomic). A mnemonic that is not named here is refused: an unknown instruction is not
// known to be exact.
inline bool exact_mnemonic_allowed(const std::string& m) {
  static const char* const kAllowed[] = {
      // integer
      "IMAD", "IADD3", "IADD", "LOP3", "LOP", "LEA", "SHF", "SHL", "SHR", "ISETP", "IMNMX", "IABS", "POPC", "FLO", "BREV", "PRMT",
      "SEL", "MOV", "R2P", "P2R", "PLOP3", "UPLOP3", "PSETP", "ISET", "FSET", "DSET", "I2I", "I2IP", "SGXT", "BMSK", "VIADD", "VIMNMX",
      // IEEE float and double (no SFU)
      "FADD", "FMUL", "FFMA", "FSETP", "FSEL", "FMNMX", "DADD", "DMUL", "DFMA", "DSETP", "DMNMX", "I2F", "I2FP", "F2I", "F2F", "F2FP",
      "FRND", "FCHK",
      // memory (plain loads and stores; integer reductions below)
      "LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST", "LDC", "ULDC",
      // special registers, uniform datapath
      "S2R", "S2UR", "CS2R", "UMOV", "UIADD3", "UIMAD", "ULOP3", "USHF", "ULEA", "UISETP", "USEL", "UPRMT", "UFLO", "UPOPC", "UP2UR",
      "R2UR", "VOTEU",
      // control, barriers, warp
      "BRA", "BRX", "BSSY", "BSYNC", "BREAK", "BMOV", "EXIT", "NOP", "YIELD", "WARPSYNC", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL",
      "SHFL", "VOTE", "MATCH",
      // integer reductions with no returned value (order of the adds does not matter)
      "RED",
  };
  for (const char* a : kAllowed)
    if (m == a) return true;
  return false;
}

// Modifiers that turn an allowed mnemonic into an order- or implementation-dependent one: a floating-point reduction adds in
// whatever order the blocks arrive, and a reduction on a float is not the integer one.
inline bool exact_mods_allowed(const std::string& mnemonic, const std::vector<std::string>& mods) {
  if (mnemonic == "RED") {
    for (const std::string& m : mods)
      if (m == "F16" || m == "F32" || m == "F64" || m == "BF16" || m == "FTZ" || m == "CAS" || m == "EXCH" || m == "INC" || m == "DEC" || m == "SPIN" ||
          m == "CAST" || m == "POPC")
        return false;
  }
  return true;
}

inline bool exact_opcode_allowed(const std::string& mnemonic, const std::vector<std::string>& mods) {
  return exact_mnemonic_allowed(mnemonic) && exact_mods_allowed(mnemonic, mods);
}

// ---- the IEEE division idiom (MUFU.RCP + FCHK + a call to the slow path) --------------------------------------------------------------
// MUFU.RCP alone is NOT in the exact set: the card's reciprocal differs from the simulator's correctly rounded one by one ulp on about 13% of the mantissas
// (tools/probes/mufu_rcp_probe.cu, round 5), and 09 doc section 2 saw 1 to 171 ulp results downstream of an SFU instruction. But ptxas builds the IEEE
// quotient a / b (div.rn.f32, also `/` in CUDA C++) from MUFU.RCP, a few FFMAs that refine it, an FCHK that sends the rare operands to a slow path
// (CALL.REL.NOINC to a subroutine that also uses MUFU.RCP, ending in RET), and the result is the correctly rounded quotient whatever the last bit of the
// approximation: the PTX ISA requires div.rn.f32 to be IEEE-compliant, so the card and the simulator both give the one correctly rounded answer.
// Tested here without a card: the refinement over every mantissa of the divisor with the simulator's reciprocal and with it one ulp off either way
// (tests/unit/test_host_gpu_ref.cpp); on a card, by tools/probes/hostgpu_sfu_exhaustive.cu div (all 2^32 divisors), which is PENDING. Until that run is
// clean the recognizer is behind VGPU_HOST_GPU_DIV=1 and the default exact set is unchanged.
//
// What the recognizer accepts (division_idiom_only): every MUFU is a plain RCP; each one outside a subroutine has an FCHK of the same divisor register within
// kIdiomMufuToFchk instructions after it; every CALL goes to a subroutine (from its target to the first RET at or after it) and has an FCHK within
// kIdiomFchkToCall instructions before it; a subroutine holds arithmetic only (no memory, barrier or warp operation, no call); every RET is in a subroutine.
// A kernel with a bare rcp.approx (no FCHK), a device function that is not the slow path, or any other MUFU fails the check and stays refused.
struct CodeInsn {
  enum class Kind : uint8_t { Other, MufuRcp, MufuOther, Fchk, Call, Ret, Memory };
  Kind kind = Kind::Other;
  uint64_t pc = 0;
  int64_t target = -1;    // CALL: the address of the subroutine
  int src0 = -1;          // MUFU: the register read; FCHK: the dividend's
  int src1 = -1;          // FCHK: the divisor's register
};
inline constexpr unsigned kIdiomMufuToFchk = 6;
inline constexpr unsigned kIdiomFchkToCall = 12;

inline bool division_idiom_only(const std::vector<CodeInsn>& code) {
  using K = CodeInsn::Kind;
  struct Region { uint64_t lo, hi; };
  std::vector<Region> regions;
  bool any = false;
  for (size_t i = 0; i < code.size(); ++i) {
    if (code[i].kind != K::Call) continue;
    any = true;
    // the subroutine: from the target to the first RET at or after it
    size_t t = code.size();
    for (size_t j = 0; j < code.size(); ++j)
      if (static_cast<int64_t>(code[j].pc) == code[i].target) { t = j; break; }
    if (t == code.size() || t <= i) return false;   // no such address, or a call backwards (a loop of calls is not the idiom)
    size_t r = t;
    while (r < code.size() && code[r].kind != K::Ret) {
      if (code[r].kind == K::Call || code[r].kind == K::Memory) return false;
      ++r;
    }
    if (r == code.size()) return false;
    bool fchk_before = false;
    for (size_t j = i > kIdiomFchkToCall ? i - kIdiomFchkToCall : 0; j < i; ++j) fchk_before = fchk_before || code[j].kind == K::Fchk;
    if (!fchk_before) return false;
    regions.push_back({code[t].pc, code[r].pc});
  }
  const auto in_region = [&](uint64_t pc) {
    for (const Region& g : regions)
      if (pc >= g.lo && pc <= g.hi) return true;
    return false;
  };
  for (size_t i = 0; i < code.size(); ++i) {
    switch (code[i].kind) {
      case K::MufuOther: return false;
      case K::Ret:
        if (!in_region(code[i].pc)) return false;
        break;
      case K::MufuRcp: {
        if (in_region(code[i].pc)) break;
        bool checked = false;
        for (size_t j = i + 1; j < code.size() && j <= i + kIdiomMufuToFchk && !checked; ++j)
          checked = code[j].kind == K::Fchk && code[j].src1 >= 0 && code[j].src1 == code[i].src0;
        if (!checked) return false;
        break;
      }
      default: break;
    }
  }
  (void)any;
  return true;
}

// The kind of an instruction for the check above, from its mnemonic and modifiers. Anything that touches memory, a barrier or other threads is Memory
// (not allowed inside the slow path); an instruction that is not in the exact set at all is Memory too, so a subroutine holding one fails.
inline CodeInsn::Kind classify_for_idiom(const std::string& mnemonic, const std::vector<std::string>& mods) {
  using K = CodeInsn::Kind;
  if (mnemonic == "MUFU") {
    bool rcp = false, other = false;
    for (const std::string& m : mods) (m == "RCP" ? rcp : other) = true;
    return rcp && !other ? K::MufuRcp : K::MufuOther;
  }
  if (mnemonic == "FCHK") return K::Fchk;
  if (mnemonic == "CALL") return K::Call;
  if (mnemonic == "RET") return K::Ret;
  static const char* const kTouchesOthers[] = {"LDG", "STG", "LDS", "STS", "LDL", "STL", "LD", "ST", "BAR", "MEMBAR", "ERRBAR", "DEPBAR", "CCTL", "SHFL",
                                               "VOTE", "MATCH", "RED", "WARPSYNC", "VOTEU"};
  for (const char* t : kTouchesOthers)
    if (mnemonic == t) return K::Memory;
  return exact_mnemonic_allowed(mnemonic) ? K::Other : K::Memory;
}

// ---- residency (the design for a card copy that outlives a launch) ---------------------------------------------------------

enum class Where : uint8_t {
  Host,   // the simulator's bytes are the truth; a card copy, if any, is stale
  Card,   // the card's bytes are the truth; the simulator's are stale (never read them)
  Both,   // the two are equal
};

// The residency of the chunks of one allocation. A new allocation is Host (zeros live in the simulator's sparse table), a
// simulator write makes a chunk Host, a launch on the card makes every chunk of every allocation it may reach Card (it may
// write any of them: pointers inside memory cannot be followed), a push makes Host chunks Both, and a pull makes Card chunks
// Both. The protocol's whole content: a reader or writer on one side first calls the "need" function for its range and
// copies what it returns; a copy in the other direction is never needed for correctness, only for speed.
class Residency {
 public:
  explicit Residency(uint64_t chunks = 0) : state_(chunks, Where::Host) {}
  uint64_t chunks() const { return state_.size(); }
  Where at(uint64_t c) const { return state_[c]; }

  // The simulator is about to write chunks [first, last]: anything the card holds of them goes stale. The chunks must be
  // current in the simulator first, unless the whole chunk is overwritten (the caller says so with `whole`).
  // Returns the chunks that have to be pulled from the card before the write (partial writes only).
  std::vector<uint64_t> sim_write(uint64_t first, uint64_t last, bool whole) {
    std::vector<uint64_t> pull;
    for (uint64_t c = first; c <= last && c < state_.size(); ++c) {
      if (state_[c] == Where::Card && !whole) pull.push_back(c);
      state_[c] = Where::Host;
    }
    return pull;
  }
  // The simulator is about to read chunks [first, last]: the Card ones have to come back first (they become Both).
  std::vector<uint64_t> sim_read(uint64_t first, uint64_t last) {
    std::vector<uint64_t> pull;
    for (uint64_t c = first; c <= last && c < state_.size(); ++c)
      if (state_[c] == Where::Card) {
        pull.push_back(c);
        state_[c] = Where::Both;
      }
    return pull;
  }
  // A launch on the card is about to run: the Host chunks have to be pushed (they become Both). Every chunk is then Card,
  // for the launch may write any of them; call `card_launched` after it.
  std::vector<uint64_t> card_prepare() {
    std::vector<uint64_t> push;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Host) {
        push.push_back(c);
        state_[c] = Where::Both;
      }
    return push;
  }
  void card_launched() { std::fill(state_.begin(), state_.end(), Where::Card); }
  // After a launch whose written chunks are known (a host-side diff, or a kernel whose stores are bounded).
  void card_launched_writing(uint64_t first, uint64_t last) {
    for (uint64_t c = first; c <= last && c < state_.size(); ++c) state_[c] = Where::Card;
  }
  uint64_t count(Where w) const { return static_cast<uint64_t>(std::count(state_.begin(), state_.end(), w)); }
  // Chunks whose card copy is valid (Card or Both): an allocation with any is kept out of the memory manager's lookup memories.
  uint64_t card_valid() const { return count(Where::Card) + count(Where::Both); }
  // The chunks a launch on the card has to push (the simulator's bytes are the truth for them).
  std::vector<uint64_t> host_chunks() const {
    std::vector<uint64_t> out;
    for (uint64_t c = 0; c < state_.size(); ++c)
      if (state_[c] == Where::Host) out.push_back(c);
    return out;
  }
  // A launch the simulator runs: it may read or write anything, so the Card chunks come back first and every chunk is then Host.
  std::vector<uint64_t> sim_launch() {
    std::vector<uint64_t> pull;
    for (uint64_t c = 0; c < state_.size(); ++c) {
      if (state_[c] == Where::Card) pull.push_back(c);
      state_[c] = Where::Host;
    }
    return pull;
  }
  // The same, for the chunks a simulator launch is known not to write (not used by the runtime: pointers inside memory cannot be followed).
  void set(uint64_t c, Where w) { state_[c] = w; }

 private:
  std::vector<Where> state_;
};

// Chunk indices to [first, count) runs, for one copy call per run of adjacent chunks.
inline std::vector<std::pair<uint64_t, uint64_t>> chunk_runs(const std::vector<uint64_t>& idx) {
  std::vector<std::pair<uint64_t, uint64_t>> out;
  for (uint64_t c : idx) {
    if (!out.empty() && out.back().first + out.back().second == c) ++out.back().second;
    else out.emplace_back(c, 1);
  }
  return out;
}

// The chunks a host access of [addr, addr + len) touches in the allocation [base, base + size), clamped to the allocation, and
// whether the access covers its first and its last chunk completely (a write that does not needs the chunk's old bytes first).
struct ChunkSpan {
  bool any = false;
  uint64_t first = 0, last = 0;
  bool first_whole = false, last_whole = false;
};
inline ChunkSpan chunk_span(uint64_t base, uint64_t size, uint64_t addr, uint64_t len, uint64_t chunk_bytes = 64 * 1024) {
  ChunkSpan s;
  if (!len || !size || addr >= base + size || addr + len <= base) return s;
  const uint64_t lo = addr < base ? 0 : addr - base;
  const uint64_t hi = std::min<uint64_t>(addr + len - base, size);   // exclusive
  s.any = true;
  s.first = lo / chunk_bytes;
  s.last = (hi - 1) / chunk_bytes;
  const auto chunk_end = [&](uint64_t c) { return std::min<uint64_t>((c + 1) * chunk_bytes, size); };
  s.first_whole = lo <= s.first * chunk_bytes && hi >= chunk_end(s.first);
  s.last_whole = lo <= s.last * chunk_bytes && hi >= chunk_end(s.last);
  return s;
}

// ---- the size rule ---------------------------------------------------------------------------------------------------------

// Measured on a g5 rig (A10G, PCIe 4.0): 09 doc, M3. One launch and its synchronisation 7 us; contiguous pageable copies 12 GB/s
// in and 10 GB/s out (1 MiB and up; 64 KiB chunk-at-a-time copies 4.4 and 4.0 GB/s, gathered through pinned staging 5.8).
struct Costs {
  double launch_s = 7e-6;
  double push_bytes_per_s = 12e9;
  double pull_bytes_per_s = 10e9;
  double kernel_s_default = 100e-6;   // before the card has run this kernel once
  double margin = 2.0;                // offload only when the simulator is expected to take this many times the card path
  double sim_warp_instr_per_s = 100e6;   // a lower bound of the simulator's speed in warp instructions per second (tens to ~150 M per thread in the benchmarks)
  double min_sim_s = 200e-6;          // never below this, whatever the arithmetic says (timer noise, the estimate's error)
};

struct Estimate {
  double sim_s = 0;                   // what the simulator is expected to need
  std::optional<double> card_kernel_s;
  uint64_t push_bytes = 0;
  uint64_t pull_bytes = 0;
};

inline double card_path_seconds(const Estimate& e, const Costs& c) {
  return c.launch_s + e.card_kernel_s.value_or(c.kernel_s_default) + static_cast<double>(e.push_bytes) / c.push_bytes_per_s +
         static_cast<double>(e.pull_bytes) / c.pull_bytes_per_s;
}

inline bool should_offload(const Estimate& e, const Costs& c) {
  return e.sim_s >= c.min_sim_s && e.sim_s > c.margin * card_path_seconds(e, c);
}

// ---- start-up of the real driver ---------------------------------------------------------------------------------------------------
// Loading the driver and making its context costs 0.15 to 0.35 s (09 doc, section 5). It starts, on a thread, at the first launch of a kernel in the exact set
// that is big enough to be a candidate; a program with no such launch never loads the driver. A launch of fewer than kStartMinWarps warps in all runs in well
// under a millisecond in the simulator and cannot repay a start by itself; a kernel whose small grid runs long gets its start when the card path is first needed.
inline constexpr uint64_t kStartMinWarps = 32;
inline bool start_worthwhile(uint64_t blocks, uint32_t threads_per_block) {
  const uint64_t warps_per_block = (static_cast<uint64_t>(threads_per_block) + 31) / 32;
  return blocks * warps_per_block >= kStartMinWarps;
}

// An estimate for a launch the simulator has not run yet: every warp of the grid executes the kernel's static instruction count, at the
// simulator's rate. It is NOT a lower bound: a thread that leaves at a bounds check runs a few instructions (PolyBench 3DCONV, a stencil
// whose grid covers the array and whose threads mostly exit, simulated in 3 ms and estimated at 60 ms), and a loop adds to the count.
// The runtime uses it only when asked (VGPU_HOST_GPU_STATIC=1); by default a kernel's first launch is the simulator's.
inline double static_sim_seconds(uint64_t blocks, uint32_t threads_per_block, uint64_t static_instrs, const Costs& c) {
  const uint64_t warps_per_block = (threads_per_block + 31) / 32;
  return static_cast<double>(blocks * warps_per_block * static_instrs) / c.sim_warp_instr_per_s;
}

// What the simulator took for a kernel before, by kernel, per block of the grid. The first two launches of a kernel run in the simulator: the first one
// pays decoding and set-up that the later ones do not (PolyBench 3DCONV's first launch took far longer than the 252 after it, and an estimate from it sent
// 253 launches of 3 ms to a card path of 100 ms: 38x slower), so the estimate is the smallest per-block time seen, which errs towards the simulator, as the
// rule should. It is available after two samples.
//
// One exception, for the launches that cost the most (sp15): a first sample that took kOneSampleMinSeconds or more is a launch whose simulator time is far above any
// decode or set-up cost, and running the kernel a second time in the simulator just to learn what it already showed would cost as much again. Such a kernel
// has an estimate after ONE sample, for at most kOneSampleOffloads offloaded launches (the estimate may still hold set-up time, which would blind the
// offload_paid_off check below): after that it needs a second, simulated sample like any other kernel, and the ban rules judge the card path against that.
inline constexpr double kOneSampleMinSeconds = 0.25;
inline constexpr unsigned kOneSampleOffloads = 2;

class SimTimeTable {
 public:
  void record(const std::string& kernel, uint64_t blocks, double seconds) {
    Entry& e = table_[kernel];
    const double per_block = blocks ? seconds / static_cast<double>(blocks) : seconds;
    if (!e.samples) e.first_total_s = seconds;
    e.per_block_s = e.samples ? std::min(e.per_block_s, per_block) : per_block;
    ++e.samples;
  }
  std::optional<double> predict(const std::string& kernel, uint64_t blocks) const {
    const auto it = table_.find(kernel);
    if (it == table_.end()) return std::nullopt;
    const Entry& e = it->second;
    const bool one_sample_ok = e.samples == 1 && e.first_total_s >= kOneSampleMinSeconds && e.one_sample_offloads < kOneSampleOffloads;
    if (e.samples < kSamplesNeeded && !one_sample_ok) return std::nullopt;
    return e.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));
  }
  // An offloaded launch used an estimate from `kernel`'s samples: counts against the one-sample allowance.
  void note_offload(const std::string& kernel) {
    const auto it = table_.find(kernel);
    if (it != table_.end() && it->second.samples < kSamplesNeeded) ++it->second.one_sample_offloads;
  }
  unsigned samples(const std::string& kernel) const {
    const auto it = table_.find(kernel);
    return it == table_.end() ? 0 : it->second.samples;
  }
  static constexpr unsigned kSamplesNeeded = 2;

 private:
  struct Entry {
    double per_block_s = 0;
    double first_total_s = 0;
    unsigned samples = 0;
    unsigned one_sample_offloads = 0;
  };
  std::map<std::string, Entry> table_;
};

// ---- the first launch of a kernel (docs/perf-design/09-host-gpu-spike.md 6, step 2) --------------------------------------------------
// The history rule needs two simulator launches of a kernel, so a program that launches each kernel once (PolyBench GEMM, SYRK, SYR2K, 3MM) was never
// offloaded. A static count of the code cannot say how long threads run (3DCONV, a stencil with no loop whose threads mostly leave at a bounds check, was
// 38x slower with VGPU_HOST_GPU_STATIC=1), so the first launch of a kernel goes to the card only when ALL of these hold:
//  * the kernel has a loop (a branch back to an earlier instruction): threads cannot all leave at a bounds check, they have work that grows with the code's loop;
//  * the grid covers at least kFirstLaunchBlocksPerSm blocks per multiprocessor: a grid that does not fill the machine is small however long a thread runs;
//  * the live bytes (the most the launch can move each way) are at most kFirstLaunchMaxBytes: a mistaken first launch is bounded;
//  * the static estimate beats the card path by the usual margin, with the pushes and pulls of the whole live memory.
// Only the kernel's first launch is decided this way: the next ones are the simulator's until its history exists, as before.
inline constexpr uint64_t kFirstLaunchBlocksPerSm = 8;
inline constexpr uint64_t kFirstLaunchMaxBytes = 256ull << 20;

// Whether a branch goes back to its own instruction or an earlier one. `branches` holds (pc, target) of every branch of the kernel's code.
inline bool has_back_edge(const std::vector<std::pair<uint64_t, uint64_t>>& branches) {
  for (const auto& [pc, target] : branches)
    if (target <= pc) return true;
  return false;
}

struct FirstLaunch {
  uint64_t blocks = 0;
  uint32_t threads_per_block = 0;
  uint64_t static_instrs = 0;
  bool back_edge = false;
  uint32_t multiprocessors = 0;
  uint64_t live_bytes = 0;
};

inline bool first_launch_offload(const FirstLaunch& f, const Costs& c) {
  if (!f.back_edge || f.multiprocessors == 0 || f.blocks < kFirstLaunchBlocksPerSm * f.multiprocessors || f.live_bytes > kFirstLaunchMaxBytes) return false;
  Estimate e;
  e.sim_s = static_sim_seconds(f.blocks, f.threads_per_block, f.static_instrs, c);
  e.push_bytes = f.live_bytes;
  e.pull_bytes = f.live_bytes;
  return should_offload(e, c);
}

// A safety net for any estimate that turns out wrong: an offloaded launch that took longer on the card path (copies, launch, copy back)
// than the simulator was expected to need ends the offloading of that kernel. One bad launch is the most it can cost.
inline bool offload_paid_off(double card_total_s, double sim_estimate_s) { return card_total_s <= sim_estimate_s; }

// A second safety net, by traffic (docs/perf-design/09-host-gpu-spike.md 7.3; docs/performance.md "Round sp14"). With a persistent card copy, a kernel that
// alternates with simulator launches on the same buffers gives the copy up and gets it pushed again every time. A kernel is banned when kTrafficLaunches
// offloaded launches IN A ROW each pushed more than kTrafficSharePercent percent of the live bytes AND more than kTrafficMinBytes, and each one's card path
// took more than kTrafficSlowShare of the simulator's estimate. All three conditions are needed: replayed over the 64 saved traces (residency_model.py), the share
// and the size alone also catch GRAMSCHM, whose offloaded launches are worth 3x their traffic (banning it loses 29%), and the share keeps base64e (8.4% of live bytes
// per push, 8.06 MiB) out. The kernel stays in the simulator for the rest of the process; a ban only moves a launch back, so results do not change.
// Measured on the traces the ban fires on LU only, and late: it is a net, not the fix for LU's 18% against write-through.
inline constexpr unsigned kTrafficLaunches = 2;
inline constexpr unsigned kTrafficSharePercent = 25;
inline constexpr uint64_t kTrafficMinBytes = 12ull << 20;
inline constexpr double kTrafficSlowShare = 0.5;

class TrafficTable {
 public:
  // One offloaded launch of `kernel`: it pushed `push_bytes` of `live_bytes`, its card path took `card_total_s`, the simulator was expected to need
  // `sim_estimate_s`. `timing_valid` is false for a kernel's first card launch, whose path includes the module load: then only the traffic counts.
  // Returns true when the kernel must go back to the simulator for good.
  bool record(const std::string& kernel, uint64_t push_bytes, uint64_t live_bytes, double card_total_s, double sim_estimate_s, bool timing_valid = true) {
    unsigned& heavy = heavy_[kernel];
    const bool big = push_bytes > kTrafficMinBytes && live_bytes > 0 && push_bytes * 100 > live_bytes * kTrafficSharePercent;
    const bool slow = !timing_valid || card_total_s > kTrafficSlowShare * sim_estimate_s;
    heavy = (big && slow) ? heavy + 1 : 0;
    return heavy >= kTrafficLaunches;
  }

 private:
  std::map<std::string, unsigned> heavy_;   // per kernel: offloaded launches in a row that were heavy
};

}  // namespace vgpu::hostgpu
