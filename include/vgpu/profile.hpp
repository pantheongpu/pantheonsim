// Device profiles: the data-driven description of one virtual GPU model.
//
// A profile holds only API-visible, functionally relevant properties. VirtualGPU
// does not model performance, so clock rates, bandwidths, and cache sizes are
// intentionally absent. Profiles ship as restricted-YAML files under profiles/
// and are embedded into the binaries at build time.
//
// Values with `verified: false` are placeholders taken from public vendor
// documentation and have NOT been confirmed against physical hardware by the
// characterization suite (which does not exist yet).
#pragma once

#include <array>
#include <cstdint>
#include <map>
#include <string>

namespace vgpu {

struct Limits {
  uint32_t max_threads_per_block = 0;
  std::array<uint32_t, 3> max_block_dim{};
  std::array<uint32_t, 3> max_grid_dim{};
  uint32_t shared_mem_per_block = 0;        // default per-block static limit
  uint32_t shared_mem_per_block_optin = 0;  // opt-in dynamic maximum
  uint32_t registers_per_block = 0;
  uint32_t multiprocessors = 0;
  // Residency ceilings. These are functional, not performance: they decide
  // whether a launch fits on the device at all ("too many resources requested
  // for launch") and how many warps can be resident.
  uint32_t registers_per_sm = 0;
  uint32_t max_threads_per_sm = 0;
  uint32_t max_blocks_per_sm = 0;
  uint32_t max_registers_per_thread = 255;
  uint32_t shared_mem_per_sm = 0;
};

// Presentation values for monitoring tools (nvidia-smi, rocm-smi, `vgpu smi`).
//
// These are deliberately NOT part of the functional model and are never
// consulted by the execution engine: VirtualGPU does not predict performance.
// They exist so telemetry has a plausible, per-model scale to render against,
// and every value derived from them is labelled synthetic. See
// ARCHITECTURE.md and docs/telemetry.md.
struct TelemetryClass {
  uint32_t power_limit_w = 0;
  uint32_t sm_clock_max_mhz = 0;
  uint32_t mem_clock_max_mhz = 0;
  uint32_t temperature_max_c = 0;
  uint32_t pci_vendor_id = 0;
  uint32_t pci_device_id = 0;
};

struct DeviceProfile {
  std::string id;            // registry id, e.g. "nvidia/h100"
  std::string vendor;        // "nvidia" | "amd"
  std::string model;         // marketing/device name as APIs report it
  std::string architecture;  // "ampere" | "hopper" | "blackwell" | ...
  int cc_major = 0;          // compute capability (NVIDIA) / ISA generation
  int cc_minor = 0;
  uint32_t warp_size = 0;
  uint64_t vram_bytes = 0;
  bool verified = false;  // true once hardware characterization confirms values
  Limits limits;
  TelemetryClass telemetry;
  std::map<std::string, bool> features;

  // Parses a profile document. `origin` names the source in error messages.
  static DeviceProfile from_yaml(const std::string& src, const std::string& origin);
};

}  // namespace vgpu
