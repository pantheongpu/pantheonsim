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

// An estimate for a launch the simulator has not run yet: every warp of the grid executes the kernel's static instruction count, at the
// simulator's rate. It is NOT a lower bound: a thread that leaves at a bounds check runs a few instructions (PolyBench 3DCONV, a stencil
// whose grid covers the array and whose threads mostly exit, simulated in 3 ms and estimated at 60 ms), and a loop adds to the count.
// The runtime uses it only when asked (VGPU_HOST_GPU_STATIC=1); by default a kernel's first launch is the simulator's.
inline double static_sim_seconds(uint64_t blocks, uint32_t threads_per_block, uint64_t static_instrs, const Costs& c) {
  const uint64_t warps_per_block = (threads_per_block + 31) / 32;
  return static_cast<double>(blocks * warps_per_block * static_instrs) / c.sim_warp_instr_per_s;
}

// What the simulator took for a kernel before, by kernel, per block of the grid. The first two launches of a kernel always run in the
// simulator: the first one pays decoding and set-up that the later ones do not (PolyBench 3DCONV's first launch took far longer than the
// 252 after it, and an estimate from it sent 253 launches of 3 ms to a card path of 100 ms: 38x slower), so the estimate is the
// smallest per-block time seen, which errs towards the simulator, as the rule should. It is available after two samples.
class SimTimeTable {
 public:
  void record(const std::string& kernel, uint64_t blocks, double seconds) {
    Entry& e = table_[kernel];
    const double per_block = blocks ? seconds / static_cast<double>(blocks) : seconds;
    e.per_block_s = e.samples ? std::min(e.per_block_s, per_block) : per_block;
    ++e.samples;
  }
  std::optional<double> predict(const std::string& kernel, uint64_t blocks) const {
    const auto it = table_.find(kernel);
    if (it == table_.end() || it->second.samples < kSamplesNeeded) return std::nullopt;
    return it->second.per_block_s * static_cast<double>(std::max<uint64_t>(blocks, 1));
  }
  static constexpr unsigned kSamplesNeeded = 2;

 private:
  struct Entry {
    double per_block_s = 0;
    unsigned samples = 0;
  };
  std::map<std::string, Entry> table_;
};

// A safety net for any estimate that turns out wrong: an offloaded launch that took longer on the card path (copies, launch, copy back)
// than the simulator was expected to need ends the offloading of that kernel. One bad launch is the most it can cost.
inline bool offload_paid_off(double card_total_s, double sim_estimate_s) { return card_total_s <= sim_estimate_s; }

}  // namespace vgpu::hostgpu
