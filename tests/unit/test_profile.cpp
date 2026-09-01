// Unit tests for device profiles and the built-in registry.
#include "vgpu/profile.hpp"

#include "vgpu/error.hpp"
#include "vgpu/registry.hpp"
#include "vtest.hpp"

using vgpu::DeviceProfile;
using vgpu::Err;
using vgpu::Error;

VTEST(registry_lists_all_gpus) {
  auto ids = vgpu::available_gpus();
  VCHECK_EQ(ids.size(), size_t{10});
  VCHECK_EQ(ids[0], "nvidia/a10");
  VCHECK_EQ(ids[4], "nvidia/b200");
  VCHECK_EQ(ids[5], "nvidia/rtx3060");
  VCHECK_EQ(ids[6], "nvidia/a100-sxm4-40gb");
  VCHECK_EQ(ids[9], "amd/mi350x");
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
    // rtx3060 is characterized from a physical device; the rest are
    // placeholders from public documentation.
    VCHECK_EQ(p.verified, p.id == "nvidia/rtx3060" || p.id == "nvidia/a10" ||
                          p.id == "nvidia/a100-sxm4-40gb" || p.id == "nvidia/h100");
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

VTEST_MAIN
