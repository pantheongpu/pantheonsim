// Warp scheduling abstraction.
//
// The execution engine asks a Scheduler which runnable warp to advance next.
// A warp runs until it yields (barrier) or retires, so scheduling decisions
// happen at yield points. This interface is where future modes plug in:
//   deterministic  — fixed round-robin order (default; bit-for-bit reproducible)
//   random         — seeded shuffle of legal orders, for flushing out races
//   adversarial    — searches legal orders that maximize interleaving damage
// Keeping every legal execution order expressible here is a hard architectural
// requirement (see ARCHITECTURE.md: race detection roadmap).
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace vgpu::exec {

class Scheduler {
 public:
  virtual ~Scheduler() = default;
  // Picks one element of `runnable` (indices of ready warps, never empty).
  virtual size_t pick(const std::vector<size_t>& runnable) = 0;
};

enum class SchedulerKind { Deterministic };

std::unique_ptr<Scheduler> make_scheduler(SchedulerKind kind);

}  // namespace vgpu::exec
