// Warp scheduling.
//
// The execution engine asks a Scheduler which runnable warp to advance next,
// and for how long. Two decisions, and the second one matters more than it
// looks:
//
//   deterministic  fixed round-robin, and each warp runs until it yields at a
//                  barrier or retires. Bit-for-bit reproducible, and the
//                  fastest, so it is the default.
//   random         seeded uniform choice with a short slice, so warps
//                  interleave at instruction granularity. Reproducible given
//                  the seed; a different seed is a different legal execution.
//   adversarial    switches warp as often as the model permits, preferring
//                  whichever has run least recently. The maximum-interleaving
//                  order, which is where a race is most likely to produce a
//                  different answer.
//
// Why the slice is the important half: under the deterministic scheduler a
// warp runs from one barrier to the next without interruption, so two warps in
// the same barrier epoch never interleave at all. The shared-memory race
// detector still finds their conflict, because it reasons about epochs rather
// than orderings -- but a race through *global* memory produces no detector
// report, and with no interleaving it also produces no wrong answer. It simply
// does not show up. Preempting mid-warp is what makes those visible.
//
// Keeping every legal execution order expressible here is a hard architectural
// requirement (see ARCHITECTURE.md: race detection roadmap).
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace vgpu::exec {

class Scheduler {
 public:
  virtual ~Scheduler() = default;
  // Picks one element of `runnable` (indices of ready warps, never empty).
  virtual size_t pick(const std::vector<size_t>& runnable) = 0;
  // How many instructions the picked warp may run before this is consulted
  // again. Zero means "until it yields", which is what the deterministic
  // scheduler wants: no preemption, and no per-instruction bookkeeping.
  virtual uint64_t slice() { return 0; }
};

enum class SchedulerKind { Deterministic, Random, Adversarial };

// `seed` is ignored by the deterministic scheduler and is what makes the other
// two reproducible: the same seed replays the same execution exactly.
std::unique_ptr<Scheduler> make_scheduler(SchedulerKind kind, uint64_t seed = 0);

// Reads VGPU_SCHEDULER (deterministic|random|adversarial) and
// VGPU_SCHEDULER_SEED. Returns false and leaves both untouched when the
// variable is unset; throws on a name it does not know, rather than silently
// running the default -- a mode nobody selected is worse than an error, since
// the whole point is knowing which order produced a result.
bool scheduler_from_env(SchedulerKind* kind, uint64_t* seed);

const char* scheduler_name(SchedulerKind kind);

}  // namespace vgpu::exec
