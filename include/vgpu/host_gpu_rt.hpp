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
  uint64_t pushed_bytes = 0, pulled_bytes = 0;   // moved to and from the card
  uint64_t same_exact_set = 0, differ_exact_set = 0, same_other = 0, differ_other = 0;
  // The machine-readable lines (host_gpu_summary:, host_gpu_refusals:). `card_runs` counts every launch that ran on the card, the ones a verify run
  // repeats in the simulator included; the seconds are wall time of the card path (copies, launch, wait) and of the simulator's launches.
  uint64_t card_runs = 0;
  double card_seconds = 0, sim_seconds = 0;
  // Sampled launches (vgpu/host_gpu.hpp "The first blocks of a launch, timed"): how many, how many of them the card then ran (`sample_sent`, also
  // in `offloaded`), how many copies did not answer in time, and what the samples cost the process in wall seconds (the fork and the wait).
  uint64_t sampled = 0, sample_sent = 0, sample_timeouts = 0;
  uint64_t sample_after_driver = 0;   // samples made with the driver up (Round sp76; 0 with VGPU_HOST_GPU_SAMPLE_STARTED=0, where sample_plan refuses them)
  double sample_seconds = 0;
  // Launches the card ran because the card held the live memory (vgpu/host_gpu.hpp streak_offload, Round sp65; also in `offloaded`), and the bytes
  // only the card held at those launches: what the simulator would have pulled back first had it run them.
  uint64_t streak_sent = 0, streak_card_only_bytes = 0;
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
    // Bounded blocks (Round sp75; exec::LaunchPart::block_limit_s, block_cap_s, need_s): all 0 when the rule is off (VGPU_HOST_GPU_SAMPLE_BLOCK=0).
    double block_limit_s = 0, block_cap_s = 0, need_s = 0;
  };
  bool sample_plan(const ptx::EntryFn& fn, SamplePlan* plan);
  bool driver_start_begun() const;
  // In a sample's copy of a process whose driver is up (Round sp76), before it runs a block: every chunk becomes the simulator's in the copy's
  // own memory, so that no access of its blocks reaches the gate. False: a chunk is the card's alone, or the gate's table was being changed
  // when the process was forked (the copy must end without an answer). Touches the copy's memory only, never the driver.
  bool sample_copy_begin();
  // What the copy reported (`answered` false: it did not, in time, and was ended), and what the sample cost the process in wall seconds.
  struct SampleRun {
    bool answered = false;
    bool driver_started_at_fork = false;   // whether the driver was up when the process was forked (Round sp76; before it: never)
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
