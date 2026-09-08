// Kernel launch: grid/block iteration + the SIMT warp interpreter.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <functional>
#include <map>

#include "vgpu/exec/scheduler.hpp"
#include "vgpu/exec/texture.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"
#include "vgpu/ptx/regalloc.hpp"

namespace vgpu::exec {

struct LaunchConfig {
  std::array<uint32_t, 3> grid{1, 1, 1};
  std::array<uint32_t, 3> block{1, 1, 1};
  uint32_t shared_bytes = 0;
  SchedulerKind scheduler = SchedulerKind::Deterministic;
  // Seed for the random and adversarial schedulers; ignored by the
  // deterministic one. The same seed replays the same execution.
  uint64_t scheduler_seed = 0;
  // Safety net against infinite loops; counts executed instructions per launch.
  uint64_t max_steps = 1ull << 30;
  // A cooperative launch: every block is resident at once and may wait on the
  // others. Ordinary launches make the opposite promise -- blocks are
  // independent and may run in any order, one at a time -- and running them
  // that way is both faster and a stricter check of that promise, so this
  // changes the scheduler rather than being the default.
  bool cooperative = false;
  // Device address of the grid-barrier workspace a cooperative launch needs.
  // cg::this_grid().sync() reads it out of %envreg1/%envreg2 and traps when it
  // is null, so this is what makes the barrier reachable rather than a crash.
  uint64_t coop_workspace = 0;
  // Texture and surface objects visible to this launch. The handle a kernel
  // receives is only a number; this is what it means. Null when the kernel uses
  // no textures, which is the overwhelming majority.
  const TextureTable* textures = nullptr;
};

// Invoked periodically during a launch so long-running kernels can still
// publish live telemetry. Receives seconds elapsed since the previous call.
using ProgressFn = std::function<void(double seconds)>;

// Register footprint and residency for a kernel on a device, as the launch
// path computes it. Exposed so tools can report what hardware would.
struct KernelResources {
  ptx::RegisterUsage usage;
  ptx::Occupancy occupancy;
};

// Computes a kernel's register footprint and its occupancy at `block_threads`
// on `profile`. Cached per kernel; deterministic.
KernelResources kernel_resources(const ptx::EntryFn& fn, const DeviceProfile& profile,
                                 uint32_t block_threads, uint32_t dynamic_shared = 0);

// Counters for a launch. Everything here is counted exactly as it happens
// rather than sampled, which is the one thing a simulator can offer that
// hardware counters cannot: the numbers are complete and identical run to run.
//
// What is deliberately absent is as important as what is here. There is no
// timing or memory-hierarchy model, so cache hit rates, DRAM throughput, warp
// stall reasons and achieved occupancy are not derivable -- reporting them
// would mean inventing them.
// What a thread-instruction was, in the categories a profiler reports. The
// classes partition every instruction, so they sum to thread_instructions --
// which is a property worth testing, because a classifier that silently drops
// a case looks exactly like one that works.
enum class InstClass : uint8_t {
  Fp16, Fp32, Fp64,   // arithmetic, by operand width
  Integer,            // integer arithmetic and bit manipulation
  BitConvert,         // cvt, cvta, and the pack/unpack moves
  Control,            // branches, returns, barriers, trap
  Memory,             // loads, stores, atomics, async copies
  Tensor,             // mma, wmma, movmatrix -- the tensor-core pipe
  Misc,               // mov, setp, selp, shuffles, votes
  Count
};

struct LaunchStats {
  uint64_t blocks = 0;
  uint64_t warps = 0;
  // Warp-level instruction issues, the same quantity a profiler calls
  // inst_executed: one per instruction executed by a warp regardless of how
  // many lanes were active.
  uint64_t instructions = 0;
  // Summed over active lanes -- thread_inst_executed. The ratio against
  // `instructions` is the average number of lanes doing useful work, so it
  // measures divergence directly.
  uint64_t thread_instructions = 0;
  // Branches where the active mask actually split. A branch all lanes agree on
  // is not divergence and is not counted.
  uint64_t divergent_branches = 0;
  // Memory operations by space, counted per active lane.
  uint64_t global_loads = 0, global_stores = 0;
  uint64_t shared_loads = 0, shared_stores = 0;
  uint64_t local_loads = 0, local_stores = 0;
  uint64_t global_bytes_read = 0, global_bytes_written = 0;
  uint64_t shared_bytes_read = 0, shared_bytes_written = 0;
  // Local memory is the one space that had operation counts and sectors but no
  // byte totals, so a spill-heavy kernel could not be compared against a
  // global-memory-heavy one in the same units. The asymmetry was an oversight,
  // not a decision.
  uint64_t local_bytes_read = 0, local_bytes_written = 0;
  // Atomic read-modify-writes, per active lane, and the bytes they moved. An
  // atomic is a read and a write, so its traffic is counted once here rather
  // than twice in the load and store totals.
  uint64_t atomics = 0;
  uint64_t atomic_bytes = 0;
  // bar.sync executions, per warp.
  uint64_t barriers = 0;

  // ---- counted from the addresses themselves ----
  //
  // These are the two numbers a profiler gives you that actually change how a
  // kernel is written, and both are *derived* on hardware: a device counts
  // them by sampling, so the answer moves between runs. Here every lane's
  // address is in hand at the moment of the access, so they are exact and
  // reproducible.

  // 32-byte sectors touched, which is the granularity memory is moved in. A
  // fully coalesced 32-lane load of 4-byte values touches 4 sectors; the same
  // load with a stride touches up to 32. The ratio against the request count
  // is the coalescing efficiency, and `*_requests` is that denominator: one
  // per warp-level instruction rather than per lane.
  uint64_t global_sectors = 0, local_sectors = 0;
  uint64_t global_requests = 0, shared_requests = 0, local_requests = 0;

  // Shared memory is 32 banks of 4 bytes. Lanes hitting different words in one
  // bank serialize; lanes hitting the *same* word are broadcast and cost
  // nothing. This counts the extra passes that serialization forces -- zero
  // for a conflict-free access, 31 for a 32-way conflict.
  uint64_t shared_bank_conflicts = 0;

  // Instruction mix, per active lane, so these sum to thread_instructions.
  uint64_t inst_by_class[static_cast<size_t>(InstClass::Count)] = {};
  // Tensor-core issues counted per warp rather than per lane: an mma is one
  // instruction the whole warp executes together, and a per-lane figure would
  // say 32 for something that happened once.
  uint64_t tensor_instructions = 0;

  // Per-opcode issue counts, indexed by the interned opcode id (see
  // ptx::intern_opcode). Nine classes tell you a kernel is memory-heavy; this
  // tells you it is memory-heavy because of ld.global.nc, which is the
  // difference between a number and a lead. Counted per warp-level issue, like
  // `instructions`.
  //
  // Not a uint64_t member, so it sits outside the block the merge below walks
  // and is folded in by hand -- see add().
  std::vector<uint64_t> inst_by_opcode;

  // Adding a counter used to mean remembering to add it to the merge that
  // folds each host thread's totals together, and forgetting left the new one
  // reading zero however carefully it was collected. Every field is a uint64_t
  // count, so the merge walks them rather than naming them, and a field added
  // above is summed without anyone having to remember.
  // Every member is a uint64_t count, which is what makes walking them safe.
  // A member of any other type must not be folded in by reinterpretation --
  // give it its own handling instead of letting this reach it.
  // Defined out of line: it needs offsetof, which needs the complete type.
  void add(const LaunchStats& other);
};

// Where the plain-counter block ends. Everything before `inst_by_opcode` is a
// uint64_t count and is summed by walking the block; the vector after it is
// not, and reinterpreting it as counters would corrupt it.
inline constexpr size_t kCounterWords =
    offsetof(LaunchStats, inst_by_opcode) / sizeof(uint64_t);

static_assert(offsetof(LaunchStats, inst_by_opcode) % sizeof(uint64_t) == 0,
              "the counter block before inst_by_opcode must be whole uint64_t words; see add()");

inline void LaunchStats::add(const LaunchStats& other) {
  // Fold the vector first, by hand, then walk only the block that precedes it.
  if (other.inst_by_opcode.size() > inst_by_opcode.size())
    inst_by_opcode.resize(other.inst_by_opcode.size(), 0);
  for (size_t i = 0; i < other.inst_by_opcode.size(); ++i)
    inst_by_opcode[i] += other.inst_by_opcode[i];

  auto* dst = reinterpret_cast<uint64_t*>(this);
  const auto* src = reinterpret_cast<const uint64_t*>(&other);
  for (size_t i = 0; i < kCounterWords; ++i) dst[i] += src[i];
}

// Module global-variable addresses (name -> device VA), materialized by the
// runtime at module load. Kernels referencing globals need this at launch.
using SymbolTable = std::map<std::string, uint64_t>;

// Runs `fn` across the whole grid on the CPU. `args` are the raw kernel
// parameter values, one byte-vector per .param (sizes must match).
//
// The grid is split across host threads -- CUDA blocks are independent, which
// is the programming model's central promise -- so a race-free kernel gives the
// same result every time, exactly as it does on hardware. A kernel that races
// no longer has one blessed answer, which is also true on hardware; set
// VGPU_THREADS=1 to get back a single strictly ordered block sequence, which
// makes even a racy kernel reproducible.
//
// Warp scheduling inside a block stays deterministic regardless.
LaunchStats launch(const ptx::EntryFn& fn, const LaunchConfig& cfg,
                   const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                   const DeviceProfile& profile, const SymbolTable* symbols = nullptr,
                   const ProgressFn& progress = nullptr);

}  // namespace vgpu::exec
