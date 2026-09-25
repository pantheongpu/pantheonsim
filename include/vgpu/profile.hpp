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
#include <cstdlib>
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
  // The L2 a program is told the device has, which some size a buffer
  // against so it does not fit. Zero where the profile does not say.
  uint64_t l2_cache_bytes = 0;
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
  // Memory the driver keeps for itself: the framebuffer nvidia-smi and NVML
  // report as "total", minus the totalGlobalMem a CUDA program sees. On a real
  // H100 those are 81559 and 81089 MiB, and every datacenter card differs by a
  // few hundred. vram_bytes stays the CUDA number -- that is what the verified
  // profiles measured and what programs size their work by -- and the monitor
  // adds this back. Kept as a difference rather than an absolute so that
  // shrinking VRAM for a laptop-scale run shrinks what nvidia-smi says too.
  uint64_t framebuffer_reserve_bytes = 0;
  // Memory reliability and the PCIe link, as monitoring tools read them.
  //   ecc                 the card has ECC and ships with it on (datasheet).
  //   hbm                 HBM memory. HBM parts remap failing rows; GDDR parts
  //                       with ECC retire pages instead.
  //   memory_temperature  the driver reports a memory sensor. Taken from real
  //                       runs, not from the memory type: most HBM cards in
  //                       the benchmark database report none.
  //   pcie_gen/width      the link real cards of the model most often run at,
  //                       which depends on how they are hosted: a T4 is an x16
  //                       card that clouds attach at x8.
  bool ecc = false;
  bool hbm = false;
  bool memory_temperature = false;
  uint32_t pcie_gen = 0;
  uint32_t pcie_width = 0;
};

struct DeviceProfile {
  std::string id;            // registry id, e.g. "nvidia/h100"
  std::string vendor;        // "nvidia" | "amd"
  std::string model;         // marketing/device name as APIs report it
  std::string architecture;  // "ampere" | "hopper" | "blackwell" | "cdna3" | ...
  // AMD's equivalent of a compute capability: the gfx target a binary must be
  // built for. "gfx942" for CDNA3. Empty on NVIDIA.
  std::string gcn_arch;
  // The same target as HIP names the device, with the features it runs with:
  // "gfx942:sramecc+:xnack-". A library that ships code for each setting of a
  // feature picks by it. The bare target where a profile does not say.
  std::string gcn_arch_full;
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

// VGPU_VRAM_MB resizes the card a run sees -- totalGlobalMem directly, and
// nvidia-smi's total through the reserve above. One function so its readers
// cannot disagree: it used to be copied into the runtime and the driver and
// missing from the idle nvidia-smi path, which reported the whole 80 GB card to
// a session whose programs had been given 4. A value that is not a positive
// number is ignored rather than turning the card into one with no memory.
inline void apply_vram_override(DeviceProfile& p) {
  const char* mb = std::getenv("VGPU_VRAM_MB");
  // Digits only. strtoull accepts a leading minus and wraps it -- "-5" parsed
  // as 18446744073709551611 and made a card of about 16 EiB -- and a value
  // that large overflows the multiplication below anyway.
  if (!mb || *mb < '0' || *mb > '9') return;
  char* end = nullptr;
  const unsigned long long v = std::strtoull(mb, &end, 10);
  constexpr unsigned long long kMiB = 1024ull * 1024ull;
  if (*end != '\0' || v == 0 || v > UINT64_MAX / kMiB) return;
  p.vram_bytes = v * kMiB;
}

}  // namespace vgpu
