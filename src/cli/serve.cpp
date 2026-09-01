// `vgpu serve` — present a virtual GPU rack so discovery and monitoring tools
// have something to see.
//
// Creating the devices publishes telemetry; this command then simply holds
// them until interrupted. It is what you run to exercise nvidia-smi / rocm-smi
// / rocm_agent_enumerator / lspci paths, and it is the only way to present AMD
// devices today: AMD *discovery* works, AMD kernel *execution* does not (see
// TODO.md), so nothing else keeps an AMD device alive.
//
// Optionally simulates load so the synthetic power/thermal columns move.
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"

namespace {
std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }
}  // namespace

int cmd_serve(const std::vector<std::string>& args) {
  std::string gpu = "nvidia/h100";
  int count = 1;
  double load = 0.0;      // fraction of time to appear busy
  long long alloc_mb = 0; // device memory to hold, to show a memory footprint
  for (size_t i = 0; i < args.size(); ++i) {
    if (args[i] == "--gpu" && i + 1 < args.size())
      gpu = args[++i];
    else if (args[i] == "--count" && i + 1 < args.size())
      count = std::atoi(args[++i].c_str());
    else if (args[i] == "--load" && i + 1 < args.size())
      load = std::atof(args[++i].c_str());
    else if (args[i] == "--alloc-mb" && i + 1 < args.size())
      alloc_mb = std::atoll(args[++i].c_str());
    else {
      std::fprintf(stderr, "vgpu serve: unknown argument '%s'\n", args[i].c_str());
      return 2;
    }
  }
  if (count < 1) {
    std::fprintf(stderr, "vgpu serve: --count must be >= 1\n");
    return 2;
  }

  vgpu::DeviceProfile profile = vgpu::load_gpu(gpu);
  vgpu::runtime::Runtime rt(profile, count);
  std::vector<uint64_t> held;
  if (alloc_mb > 0) {
    for (int i = 0; i < count; ++i)
      held.push_back(rt.device(i).memory().alloc(static_cast<uint64_t>(alloc_mb) * 1024 * 1024));
  }

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::printf("Serving %d x %s (%s). Telemetry is live; press Ctrl-C to stop.\n", count,
              profile.id.c_str(), profile.model.c_str());
  std::printf("  view with:  vgpu smi | vgpu smi --rocm | build/bin/nvidia-smi\n");
  if (load > 0)
    std::printf("  simulating %.0f%% load so the synthetic power/thermal columns move\n",
                load * 100.0);
  std::fflush(stdout);

  // Report activity in slices so utilization (a REAL measure of time spent
  // "executing") reflects the requested load.
  const auto slice = std::chrono::milliseconds(100);
  while (!g_stop) {
    auto busy = std::chrono::duration<double>(slice).count() * load;
    for (int i = 0; i < count; ++i) rt.device(i).note_busy(busy);
    std::this_thread::sleep_for(slice);
  }
  for (size_t i = 0; i < held.size(); ++i) rt.device(static_cast<int>(i)).memory().free(held[i]);
  std::printf("\nstopped\n");
  return 0;
}
