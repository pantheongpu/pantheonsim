// Kernel launch: grid/block iteration + the SIMT warp interpreter.
#pragma once

#include <array>
#include <cstdint>
#include <vector>

#include <functional>
#include <map>

#include "vgpu/exec/scheduler.hpp"
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
  // Safety net against infinite loops; counts executed instructions per launch.
  uint64_t max_steps = 1ull << 30;
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

struct LaunchStats {
  uint64_t blocks = 0;
  uint64_t warps = 0;
  uint64_t instructions = 0;
};

// Module global-variable addresses (name -> device VA), materialized by the
// runtime at module load. Kernels referencing globals need this at launch.
using SymbolTable = std::map<std::string, uint64_t>;

// Runs `fn` across the whole grid on the CPU. `args` are the raw kernel
// parameter values, one byte-vector per .param (sizes must match).
// Deterministic: same inputs -> same result, always.
LaunchStats launch(const ptx::EntryFn& fn, const LaunchConfig& cfg,
                   const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                   const DeviceProfile& profile, const SymbolTable* symbols = nullptr,
                   const ProgressFn& progress = nullptr);

}  // namespace vgpu::exec
