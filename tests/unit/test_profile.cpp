// Unit tests for device profiles and the built-in registry.
#include <algorithm>
#include <string>
#include <utility>
#include "vgpu/profile.hpp"

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using vgpu::DeviceProfile;
using vgpu::Err;
using vgpu::Error;

VTEST(registry_lists_all_gpus) {
  // By membership rather than position: adding a profile used to shift every
  // index after it and fail this test for the wrong reason.
  auto ids = vgpu::available_gpus();
  const std::vector<std::string> expected = {
      "nvidia/a10",   "nvidia/a100", "nvidia/h100",   "nvidia/h200",
      "nvidia/b200",  "nvidia/rtx3060", "nvidia/a100-sxm4-40gb",
      "nvidia/gh200-480gb", "nvidia/h100-pcie", "nvidia/t4", "nvidia/a10g",
      "nvidia/l4", "nvidia/l40s",
      "amd/mi300x", "amd/mi325x", "amd/mi350x"};
  VCHECK_EQ(ids.size(), expected.size());
  for (const auto& want : expected)
    VCHECK(std::find(ids.begin(), ids.end(), want) != ids.end());
}

VTEST(all_builtin_profiles_parse) {
  for (const auto& id : vgpu::available_gpus()) {
    DeviceProfile p = vgpu::load_gpu(id);
    VCHECK_EQ(p.id, id);
    VCHECK(p.vendor == "nvidia" || p.vendor == "amd");
    VCHECK(p.warp_size == 32u || p.warp_size == 64u);
    VCHECK(p.vram_bytes > 0);
    VCHECK_EQ(p.limits.max_threads_per_block, 1024u);
    VCHECK(p.telemetry.power_limit_w > 0);   // monitoring tools need a scale
    VCHECK(p.telemetry.pci_vendor_id != 0);
    // These were read from physical devices; the rest are still placeholders
    // from public documentation, and the flag has to say which is which.
    VCHECK_EQ(p.verified, p.id == "nvidia/rtx3060" || p.id == "nvidia/a10" ||
                          p.id == "nvidia/a100-sxm4-40gb" || p.id == "nvidia/a100" ||
                          p.id == "nvidia/h100" || p.id == "nvidia/gh200-480gb" ||
                          p.id == "nvidia/h100-pcie" || p.id == "nvidia/t4" ||
                          p.id == "nvidia/a10g" || p.id == "nvidia/l4" ||
                          p.id == "nvidia/l40s" ||
                          p.id == "amd/mi325x");
    // AMD parts have no compute capability, and the profile that carried a
    // plausible "9.4" was inventing one. Each vendor is asked for the thing it
    // actually has.
    if (p.vendor == "amd") {
      VCHECK(!p.gcn_arch.empty());
      VCHECK_EQ(p.cc_major, 0);
    } else {
      VCHECK(p.cc_major > 0);
    }
  }
}

VTEST(h100_profile_values) {
  DeviceProfile p = vgpu::load_gpu("nvidia/h100");
  VCHECK_EQ(p.architecture, "hopper");
  VCHECK_EQ(p.cc_major, 9);
  VCHECK_EQ(p.cc_minor, 0);
  VCHECK_EQ(p.vram_bytes, 85028896768ull);  // measured on hardware
  VCHECK_EQ(p.limits.multiprocessors, 132u);
  VCHECK(p.features.at("bf16"));
}

VTEST(gpu_id_is_case_insensitive) {
  DeviceProfile p = vgpu::load_gpu("NVIDIA/H200");
  VCHECK_EQ(p.id, "nvidia/h200");
}

VTEST(unknown_gpu_lists_available) {
  auto err = VCAPTURE(Error, vgpu::load_gpu("nvidia/h300"));
  VCHECK(err.code() == Err::UnknownGpu);
  VCHECK_CONTAINS(err.what(), "nvidia/h300");
  VCHECK_CONTAINS(err.what(), "nvidia/h100");  // the error suggests what IS available
}

VTEST(profile_missing_key_is_an_error) {
  auto err = VCAPTURE(Error, DeviceProfile::from_yaml("id: x/y\nvendor: x\n", "test-origin"));
  VCHECK(err.code() == Err::ProfileParse);
  VCHECK_CONTAINS(err.what(), "test-origin");
  VCHECK_CONTAINS(err.what(), "missing required key");
}

VTEST(a_driver_that_reports_no_thermal_threshold_still_loads) {
  // Telemetry is presentation-only and a device that does not report a value is
  // a fact about the device, not a broken profile. A real GH200's driver
  // reports no thermal threshold, so nvidia/tools/characterize-telemetry.sh comments
  // the key out -- and requiring it made every profile characterized from one
  // fail to load at all, which surfaced as the device having no memory.
  const char* yaml = R"(
id: nvidia/testcard
vendor: nvidia
model: "Test Card"
architecture: hopper
compute_capability: "9.0"
warp_size: 32
vram_bytes: 101468602368
verified: true
limits:
  max_threads_per_block: 1024
  max_block_dim: [1024, 1024, 64]
  max_grid_dim: [2147483647, 65535, 65535]
  shared_mem_per_block_bytes: 49152
  shared_mem_per_block_optin_bytes: 232448
  registers_per_block: 65536
  multiprocessors: 132
  registers_per_sm: 65536
  max_threads_per_sm: 2048
  max_blocks_per_sm: 32
telemetry:
  power_limit_w: 900
  sm_clock_max_mhz: 1980
)";
  DeviceProfile p = DeviceProfile::from_yaml(yaml, "test");
  VCHECK_EQ(p.telemetry.power_limit_w, 900u);
  VCHECK_EQ(p.telemetry.sm_clock_max_mhz, 1980u);
  // Absent means unknown, which is zero, which callers render as unavailable.
  VCHECK_EQ(p.telemetry.temperature_max_c, 0u);
  VCHECK_EQ(p.telemetry.mem_clock_max_mhz, 0u);
  VCHECK_EQ(p.limits.multiprocessors, 132u);
}

VTEST(the_gh200_profile_loads_and_says_what_hardware_said) {
  // Characterized from a physical GH200 on Lambda; verified: true means the
  // values came off the device rather than a datasheet.
  DeviceProfile p = vgpu::load_gpu("nvidia/gh200-480gb");
  VCHECK(p.verified);
  VCHECK_EQ(p.cc_major, 9);
  VCHECK_EQ(p.cc_minor, 0);
  VCHECK_EQ(p.limits.multiprocessors, 132u);
  VCHECK_EQ(p.warp_size, 32u);
}

namespace {
// A minimal valid profile whose telemetry block the tests below vary.
std::string profile_with_telemetry(const std::string& telemetry, const char* vram = "85028896768") {
  return std::string(R"(
id: nvidia/testcard
vendor: nvidia
model: "Test Card"
architecture: hopper
compute_capability: "9.0"
warp_size: 32
vram_bytes: )") + vram + R"(
verified: true
limits:
  max_threads_per_block: 1024
  max_block_dim: [1024, 1024, 64]
  max_grid_dim: [2147483647, 65535, 65535]
  shared_mem_per_block_bytes: 49152
  shared_mem_per_block_optin_bytes: 232448
  registers_per_block: 65536
  multiprocessors: 132
  registers_per_sm: 65536
  max_threads_per_sm: 2048
  max_blocks_per_sm: 32
telemetry:
  power_limit_w: 700
)" + telemetry;
}
constexpr uint64_t kMiB = 1024ull * 1024ull;
}  // namespace

VTEST(framebuffer_mb_becomes_the_driver_reserve) {
  // An H100: nvidia-smi says 81559 MiB, CUDA says 85028896768 bytes. The
  // profile keeps the CUDA number and carries the difference.
  DeviceProfile p = DeviceProfile::from_yaml(profile_with_telemetry("  framebuffer_mb: 81559\n"), "t");
  VCHECK_EQ(p.vram_bytes, 85028896768ull);
  VCHECK_EQ(p.vram_bytes + p.telemetry.framebuffer_reserve_bytes, 81559 * kMiB);
}

VTEST(framebuffer_mb_is_optional) {
  DeviceProfile p = DeviceProfile::from_yaml(profile_with_telemetry(""), "t");
  VCHECK_EQ(p.telemetry.framebuffer_reserve_bytes, 0ull);
}

VTEST(a_framebuffer_smaller_than_cuda_memory_is_an_impossible_card) {
  // The B200 profile did exactly this: 192 GB read as 192 GiB made its CUDA
  // memory 13 GB larger than the real card's whole framebuffer.
  auto err = VCAPTURE(Error, DeviceProfile::from_yaml(
      profile_with_telemetry("  framebuffer_mb: 183359\n", "206158430208"), "test-origin"));
  VCHECK(err.code() == Err::ProfileParse);
  VCHECK_CONTAINS(err.what(), "framebuffer_mb");
  VCHECK_CONTAINS(err.what(), "smaller than vram_bytes");
}

VTEST(every_builtin_reserve_is_plausible) {
  // A real driver keeps a few hundred MiB at most. A reserve in the gigabytes
  // is the signature of a unit error in either number -- the next B200.
  for (const auto& id : vgpu::available_gpus()) {
    DeviceProfile p = vgpu::load_gpu(id);
    VCHECK(p.telemetry.framebuffer_reserve_bytes < 2048 * kMiB);
  }
}

VTEST(measured_framebuffers_match_real_nvidia_smi) {
  // From nvidia-smi on real cards (pantheongpu_website/database). If one of
  // these moves, the profile has stopped describing the hardware.
  const std::pair<const char*, uint64_t> real[] = {
      {"nvidia/h100", 81559},  {"nvidia/h100-pcie", 81559}, {"nvidia/a100", 81920},
      {"nvidia/a100-sxm4-40gb", 40960}, {"nvidia/a10", 23028}, {"nvidia/a10g", 23028},
      {"nvidia/l4", 23034},    {"nvidia/l40s", 46068},     {"nvidia/t4", 15360},
      {"nvidia/rtx3060", 12288}, {"nvidia/gh200-480gb", 97871}, {"nvidia/b200", 183359}};
  for (const auto& [id, mib] : real) {
    DeviceProfile p = vgpu::load_gpu(id);
    VCHECK_EQ(p.vram_bytes + p.telemetry.framebuffer_reserve_bytes, mib * kMiB);
  }
}

VTEST_MAIN
