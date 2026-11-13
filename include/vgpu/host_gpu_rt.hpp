// Host-GPU mode, the part that talks to the machine's real driver (docs/perf-design/09-host-gpu-spike.md). Off unless the program
// runs with VGPU_HOST_GPU=exact or =turbo (docs/perf-design/12-host-gpu-turbo.md); the decisions are in vgpu/host_gpu.hpp.
#pragma once

#include <cstdint>
#include <vector>

#include "vgpu/exec/launch.hpp"
#include "vgpu/host_gpu.hpp"
#include "vgpu/host_gpu_tol.hpp"
#include "vgpu/memory.hpp"
#include "vgpu/profile.hpp"
#include "vgpu/ptx/ast.hpp"

namespace vgpu::hostgpu {

struct Stats {
  uint64_t offloaded = 0, simulated = 0;
  uint64_t refused_kernel = 0, refused_size = 0, refused_launch = 0, failed = 0;
  // VGPU_HOST_GPU_VERIFY: launches run on both, by whether the kernel passed the exact scan and whether the bytes were equal.
  uint64_t pushed_bytes = 0, pulled_bytes = 0;   // moved to and from the card
  uint64_t same_exact_set = 0, differ_exact_set = 0, same_other = 0, differ_other = 0;
  // The machine-readable lines (host_gpu_turbo:, host_gpu_refusals:). `card_runs` counts every launch that ran on the card, the ones a verify run
  // repeats in the simulator included; the seconds are wall time of the card path (copies, launch, wait) and of the simulator's launches.
  uint64_t card_runs = 0;
  double card_seconds = 0, sim_seconds = 0;
  // Launches kept in the simulator for their kernel's code, by why (vgpu/host_gpu.hpp TurboRefusal); `refused_exact` is exact mode's count of
  // launches whose kernel turbo mode would admit with its default switches.
  uint64_t refused_never = 0, refused_atomics = 0, refused_off = 0, refused_exact = 0;
  TolReport tol;   // turbo mode's verify: every compared launch
  // Sampled launches (vgpu/host_gpu.hpp "The first blocks of a launch, timed"): how many, how many of them the card then ran (`sample_sent`, also
  // in `offloaded`), how many copies did not answer in time, and what the samples cost the process in wall seconds (the fork and the wait).
  uint64_t sampled = 0, sample_sent = 0, sample_timeouts = 0;
  uint64_t sample_after_driver = 0;   // samples made although the driver's start had begun: always 0 (sample_plan refuses), printed to show it
  double sample_seconds = 0;
};

// The forked copies of sampled launches (runtime.cpp): reaps the ones that have ended, waits for the rest when asked. Returns how many were
// still running.
size_t sample_copies_reap(bool wait);

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
  // Called before the simulator runs a launch: whatever only the card holds comes back, and the card's copy is no longer valid (the launch may write anything).
  void before_simulated(MemoryManager& mem);
  void note_simulated(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, double sim_seconds, const MemoryManager* mem = nullptr,
                      const std::vector<std::vector<uint8_t>>* args = nullptr);

  // The first blocks of a launch, timed (vgpu/host_gpu.hpp). After try_launch left a launch to the simulator: whether the caller should sample it
  // first, and how (the blocks a forked copy of the process runs, its budget, how long to wait for it).
  struct SamplePlan {
    uint64_t limit = 0, stride = 1;
    double budget_s = 0, timeout_s = 0;
    const exec::BlockProbe* probe = nullptr;   // not null: progress inside the blocks (Round sp61), good until sample_decide
  };
  bool sample_plan(const ptx::EntryFn& fn, SamplePlan* plan);
  bool driver_start_begun() const;
  // What the copy reported (`answered` false: it did not, in time, and was ended), and what the sample cost the process in wall seconds.
  struct SampleRun {
    bool answered = false;
    bool driver_started_at_fork = false;   // whether the driver's start had begun when the process was forked (never: sample_plan refuses)
    exec::LaunchPart part;
    double wall_s = 0, fork_s = 0;
  };
  // Decides on the sample. True: the card ran the whole launch (nothing of the sample exists in this process); the launch is done. False: the
  // simulator runs the launch, as if it had not been sampled.
  bool sample_decide(const ptx::EntryFn& fn, const exec::LaunchConfig& cfg, const std::vector<std::vector<uint8_t>>& args, MemoryManager& mem,
                     const SampleRun& run);

  struct Impl;
  ~Runtime();

 private:
  Runtime();
  Impl* impl_;
};

}  // namespace vgpu::hostgpu
