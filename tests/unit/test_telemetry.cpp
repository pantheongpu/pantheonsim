// Tests for live device telemetry: the shared segment, the real counters, and
// the synthetic power/thermal model.
#include "vgpu/telemetry.hpp"

#include <cstdlib>
#include <cstring>
#include <vector>
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
  // Two memory totals, both measured on a real H100, and they are different
  // quantities. The monitor reports the framebuffer (nvidia-smi and NVML both
  // show 81559 MiB); CUDA reports totalGlobalMem (81089 MiB), which excludes the
  // driver's reserve. The monitor used to report the CUDA number, which left
  // nvidia-smi 470 MiB short of any real H100.
  VCHECK_EQ(snap.devices[0].vram_total_bytes, 81559ull * 1024 * 1024);   // framebuffer
  VCHECK_EQ(load_gpu("nvidia/h100").vram_bytes, 85028896768ull);        // totalGlobalMem
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

VTEST(a_64_lane_profile_launches) {
  // A 64-lane wavefront used to be refused at the door: "only 32 is
  // implemented". The interpreter is warp-width parametric now, so a launch on
  // an AMD profile runs. This does not mean AMD software runs -- HIP compiles
  // to a GCN code object, not to PTX -- it means the width is no longer what
  // stops it.
  TempSegment seg("amdexec");
  runtime::Runtime rt(load_gpu("amd/mi300x"), 1);
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(
      ".version 8.3\n.target sm_90\n.address_size 64\n.visible .entry k() { ret; }\n");
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  dev.launch(*fn, exec::LaunchConfig{}, {}, nullptr);  // must not throw
}

VTEST(ptx_warp_primitives_are_refused_at_64_lanes) {
  // The other half of the same decision. PTX defines activemask, vote, shfl
  // and the %lanemask_* registers over 32 lanes with 32-bit masks, so there is
  // no PTX answer for them on a 64-lane wavefront. Widening invents semantics;
  // truncating drops half a wavefront and still returns a number. Both are
  // worse than a named refusal, and this pins that.
  TempSegment seg("amdwarp");
  runtime::Runtime rt(load_gpu("amd/mi300x"), 1);
  auto& dev = rt.device(0);
  uint64_t mod = dev.load_module(
      ".version 8.3\n.target sm_90\n.address_size 64\n"
      ".visible .entry k() { .reg .b32 %r<2>; activemask.b32 %r0; ret; }\n");
  const ptx::EntryFn* fn = dev.get_function(mod, "k");
  auto err = VCAPTURE(Error, dev.launch(*fn, exec::LaunchConfig{}, {}, nullptr));
  VCHECK(err.code() == Err::UnsupportedPtx);
  VCHECK_CONTAINS(err.what(), "activemask");
  VCHECK_CONTAINS(err.what(), "64 lanes wide");
}

VTEST(the_same_kernel_gives_the_same_answer_at_32_and_64_lanes) {
  // The claim the refactor has to earn. Arithmetic, memory and control flow do
  // not depend on how many lanes a warp has; only the warp-level primitives
  // do, and those are refused above. So a kernel using none of them must
  // produce identical results on a 32-lane profile and a 64-lane one -- and if
  // a lane-indexing bug crept into the widening, this is where it shows.
  const char* kSrc =
      ".version 8.3\n.target sm_90\n.address_size 64\n"
      ".visible .entry k(.param .u64 p) {\n"
      "  .reg .b64 %rd<4>; .reg .b32 %r<3>;\n"
      "  ld.param.u64 %rd0, [p];\n"
      "  cvta.to.global.u64 %rd1, %rd0;\n"
      "  mov.u32 %r0, %tid.x;\n"
      "  mul.wide.u32 %rd2, %r0, 4;\n"
      "  add.s64 %rd3, %rd1, %rd2;\n"
      "  mul.lo.s32 %r1, %r0, %r0;\n"
      "  st.global.u32 [%rd3], %r1;\n"
      "  ret;\n"
      "}\n";
  constexpr uint32_t kN = 128;
  auto run = [&](const char* gpu) {
    runtime::Runtime rt(load_gpu(gpu), 1);
    auto& dev = rt.device(0);
    const uint64_t buf = dev.memory().alloc(kN * 4);
    uint64_t mod = dev.load_module(kSrc);
    const ptx::EntryFn* fn = dev.get_function(mod, "k");
    exec::LaunchConfig cfg{};
    cfg.grid[0] = 1; cfg.grid[1] = 1; cfg.grid[2] = 1;
    cfg.block[0] = kN; cfg.block[1] = 1; cfg.block[2] = 1;
    std::vector<uint8_t> arg(sizeof(uint64_t));
    std::memcpy(arg.data(), &buf, sizeof buf);
    dev.launch(*fn, cfg, {arg}, nullptr);
    std::vector<uint32_t> out(kN);
    dev.memory().read(buf, out.data(), kN * 4);
    return out;
  };
  const std::vector<uint32_t> at32 = run("nvidia/h100");
  const std::vector<uint32_t> at64 = run("amd/mi300x");
  VCHECK_EQ(at32.size(), at64.size());
  for (uint32_t i = 0; i < kN; ++i) {
    VCHECK_EQ(at32[i], i * i);   // the kernel is right at 32 lanes
    VCHECK_EQ(at64[i], at32[i]); // and identical at 64
  }
}

VTEST_MAIN
