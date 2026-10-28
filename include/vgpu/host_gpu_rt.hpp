// Host-GPU mode, the part that talks to the machine's real driver (docs/perf-design/09-host-gpu-spike.md). Off unless the program
// runs with VGPU_HOST_GPU=exact; the decisions are in vgpu/host_gpu.hpp.
#pragma once

#include <cstdint>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::hostgpu {

struct Stats {
  uint64_t offloaded = 0, simulated = 0;
  uint64_t refused_kernel = 0, refused_size = 0, refused_launch = 0, failed = 0;
  // VGPU_HOST_GPU_VERIFY: launches run on both, by whether the kernel passed the exact scan and whether the bytes were equal.
  uint64_t same_exact_set = 0, differ_exact_set = 0, same_other = 0, differ_other = 0;
};

class Runtime {
 public:
  // The runtime for a device's launches, or null when the mode is off or could not be started (said once on stderr unless
  // VGPU_QUIET). Only device 0 of the simulated machine is served: its window is the one reserved on the card.
  static Runtime* for_device(const DeviceProfile& profile, int ordinal);

  // Runs the launch on the card if it is eligible and worth it, and returns true; false leaves the launch to the simulator, with
  // the simulator's memory untouched. `sim_seconds`: how long the simulator took for a launch it did run (feeds the estimate).
  bool try_launch(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args,
                  MemoryManager& mem);
  // `mem`, when the mode is verifying, is the simulator's memory after the launch it just ran.
  void note_simulated(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, double sim_seconds, const MemoryManager* mem = nullptr,
                      const std::vector<std::vector<uint8_t>>* args = nullptr);

  struct Impl;
  ~Runtime();

 private:
  Runtime();
  Impl* impl_;
};

}  // namespace vgpu::hostgpu
