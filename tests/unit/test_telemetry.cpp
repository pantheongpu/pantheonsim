// Tests for live device telemetry: the shared segment, the real counters, and
// the synthetic power/thermal model.
#include "vgpu/telemetry.hpp"

#include <cstdlib>
#include <string>
#include <memory>
#include <thread>

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vgpu/runtime/runtime.hpp"
#include "vtest.hpp"

using namespace vgpu;

namespace {
// Each test gets its own segment so they cannot interfere.
struct TempSegment {
  std::string path;
  explicit TempSegment(const char* tag) {
    path = std::string("/tmp/vgpu-telemetry-test-") + tag + "-" + std::to_string(getpid()) + ".d";
    setenv("VGPU_TELEMETRY_PATH", path.c_str(), 1);
  }
  ~TempSegment() {
    unsetenv("VGPU_TELEMETRY_PATH");
    // The publisher removes its own file; drop the directory.
    ::rmdir(path.c_str());
  }
};
}  // namespace

VTEST(publishes_device_identity) {
  TempSegment seg("identity");
  runtime::Runtime rt(load_gpu("nvidia/h100"), 3);
  telemetry::Shared snap{};
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  VCHECK_EQ(snap.device_count, 3u);
  VCHECK_EQ(std::string(snap.devices[0].name), "NVIDIA H100 80GB HBM3");
  VCHECK_EQ(std::string(snap.devices[0].vendor), "nvidia");
  VCHECK_EQ(std::string(snap.devices[0].architecture), "hopper");
  VCHECK_EQ(snap.devices[0].vram_total_bytes, 85028896768ull);  // measured on hardware
  // Each virtual device gets its own PCI slot and a distinct UUID.
  VCHECK_EQ(std::string(snap.devices[0].bus_id), "00000000:01:00.0");
  VCHECK_EQ(std::string(snap.devices[2].bus_id), "00000000:03:00.0");
  VCHECK(std::string(snap.devices[0].uuid) != std::string(snap.devices[1].uuid));
  // PCI ids are packed as (device << 16) | vendor, as NVML reports them.
  VCHECK_EQ(snap.devices[0].pci_device_id & 0xFFFFu, 0x10DEu);          // NVIDIA
  VCHECK_EQ((snap.devices[0].pci_device_id >> 16) & 0xFFFFu, 0x2330u);  // H100
}

VTEST(memory_usage_is_real_and_live) {
  TempSegment seg("memory");
  runtime::Runtime rt(load_gpu("nvidia/a10"), 1);
  telemetry::Shared snap{};
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  VCHECK_EQ(snap.devices[0].vram_used_bytes, 0ull);

  uint64_t p = rt.device(0).memory().alloc(64 * 1024 * 1024);
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  VCHECK_EQ(snap.devices[0].vram_used_bytes, 64ull * 1024 * 1024);

  uint64_t q = rt.device(0).memory().alloc(16 * 1024 * 1024);
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  VCHECK_EQ(snap.devices[0].vram_used_bytes, 80ull * 1024 * 1024);

  rt.device(0).memory().free(p);
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  VCHECK_EQ(snap.devices[0].vram_used_bytes, 16ull * 1024 * 1024);
  rt.device(0).memory().free(q);
}

VTEST(idle_device_reports_idle_not_zero) {
  // A device that has done no work must look idle (ambient temperature, a
  // small idle draw), not powered off.
  TempSegment seg("idle");
  runtime::Runtime rt(load_gpu("nvidia/h100"), 1);
  telemetry::Shared snap{};
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  const auto& d = snap.devices[0];
  VCHECK_EQ(d.utilization_gpu, 0u);
  VCHECK(d.temperature_c >= 25 && d.temperature_c <= 45);
  VCHECK(d.power_mw > 0);                       // idle draw, not zero
  VCHECK(d.power_mw < d.power_limit_mw / 2);    // but well under the cap
  VCHECK_EQ(d.perf_state, 8u);                  // P8 = idle
  VCHECK(d.sm_clock_mhz < d.sm_clock_max_mhz);  // clocked down
}

VTEST(load_raises_utilization_power_and_temperature) {
  TempSegment seg("load");
  runtime::Runtime rt(load_gpu("nvidia/h100"), 1);
  telemetry::Shared before{};
  VCHECK(telemetry::read_snapshot(&before, seg.path));

  // Report sustained busy time, as a run of kernels would.
  for (int i = 0; i < 40; ++i) {
    rt.device(0).note_busy(0.05);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
  }
  telemetry::Shared after{};
  VCHECK(telemetry::read_snapshot(&after, seg.path));
  VCHECK(after.devices[0].utilization_gpu > 50);
  VCHECK(after.devices[0].power_mw > before.devices[0].power_mw);
  VCHECK(after.devices[0].temperature_c > before.devices[0].temperature_c);
  VCHECK(after.devices[0].power_mw <= after.devices[0].power_limit_mw);
  VCHECK_EQ(after.devices[0].perf_state, 0u);  // P0 under load
  VCHECK(after.devices[0].kernels_launched > 0);
}

VTEST(amd_profiles_are_discoverable) {
  // AMD execution is unimplemented, but discovery must work so rocm-smi and
  // rocm_agent_enumerator have something to report.
  TempSegment seg("amd");
  runtime::Runtime rt(load_gpu("amd/mi300x"), 2);
  telemetry::Shared snap{};
  VCHECK(telemetry::read_snapshot(&snap, seg.path));
  VCHECK_EQ(snap.device_count, 2u);
  VCHECK_EQ(std::string(snap.devices[0].vendor), "amd");
  VCHECK_EQ(std::string(snap.devices[0].architecture), "cdna3");
  VCHECK_EQ(snap.devices[0].pci_device_id & 0xFFFFu, 0x1002u);          // AMD
  VCHECK_EQ((snap.devices[0].pci_device_id >> 16) & 0xFFFFu, 0x74A1u);  // MI300X
}

VTEST(no_publisher_means_no_snapshot) {
  telemetry::Shared snap{};
  VCHECK(!telemetry::read_snapshot(&snap, "/tmp/vgpu-telemetry-does-not-exist.d"));
}

VTEST(publishers_merge_like_processes_sharing_a_gpu) {
  // A session and a workload are separate processes on the same virtual
  // machine: memory adds up and both appear in the per-device process list,
  // and neither one exiting erases the other.
  TempSegment seg("merge");
  auto session = std::make_unique<runtime::Runtime>(load_gpu("nvidia/h100"), 2);
  uint64_t a = session->device(0).memory().alloc(32ull * 1024 * 1024);
  {
    runtime::Runtime workload(load_gpu("nvidia/h100"), 2);
    uint64_t b = workload.device(0).memory().alloc(8ull * 1024 * 1024);
    telemetry::Shared snap{};
    VCHECK(telemetry::read_snapshot(&snap, seg.path));
    VCHECK_EQ(snap.device_count, 2u);
    // Both processes' allocations are visible on device 0.
    VCHECK_EQ(snap.devices[0].vram_used_bytes, 40ull * 1024 * 1024);
    workload.device(0).memory().free(b);
  }
  // The workload is gone; the session's own memory is still reported.
  telemetry::Shared after{};
  VCHECK(telemetry::read_snapshot(&after, seg.path));
  VCHECK_EQ(after.device_count, 2u);
  VCHECK_EQ(after.devices[0].vram_used_bytes, 32ull * 1024 * 1024);
  session->device(0).memory().free(a);
}

VTEST(amd_execution_fails_loudly) {
  // Discovery works; launching a kernel must not silently do the wrong thing.
  TempSegment seg("amdexec");
  runtime::Runtime rt(load_gpu("amd/mi300x"), 1);
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(
      ".version 8.3\n.target sm_90\n.address_size 64\n.visible .entry k() { ret; }\n");
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  auto err = VCAPTURE(Error, dev.launch(*fn, exec::LaunchConfig{}, {}, nullptr));
  VCHECK(err.code() == Err::Unsupported);
  VCHECK_CONTAINS(err.what(), "warp size 64");
}

VTEST_MAIN
