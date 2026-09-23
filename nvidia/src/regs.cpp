// An NVIDIA GPU's part of the register model (vgpu/regs_vendor.hpp): its
// BARs, its identity in configuration space, the configuration space captured
// from a real card that a GeForce of its family replays
// (nvidia/registers/measurements/), and its BAR0 registers -- offsets and
// fields from NVIDIA's published headers, values from the profile and from
// what one card answered (nvidia/registers/mmio.yaml).
#include <cstring>
#include <string>

#include "vgpu/profiling.hpp"
#include "vgpu/regs_vendor.hpp"

namespace vgpu::regs::vendor {
namespace {

bool is_geforce(const telemetry::DeviceSample& d) {
  return std::strstr(d.name, "GeForce") || std::strstr(d.name, "RTX 30");
}

// BAR0 16 MiB of registers; BAR1 the framebuffer aperture, 64-bit and
// prefetchable -- 256 MiB on a GeForce, the framebuffer rounded up to a power
// of two on data-center cards; BAR3 32 MiB, 64-bit; an I/O BAR on a GeForce.
// Sizes are a model, not measured per card.
std::array<BarLayout, 6> nvidia_bars(const telemetry::DeviceSample& d) {
  std::array<BarLayout, 6> b{};
  const Windows w = windows(d);
  const uint64_t fb = pow2_at_least(d.vram_total_bytes ? d.vram_total_bytes : (1ull << 28));
  b[0] = {16ull << 20, false, false, false, false, w.mmio32};
  b[1] = {is_geforce(d) ? (256ull << 20) : fb, false, true, true, false, w.mmio64};
  b[2].high = true;
  b[3] = {32ull << 20, false, true, true, false, w.mmio64 + 0x8000000000ull};
  b[4].high = true;
  if (is_geforce(d)) b[5] = {128, true, false, false, false, 0x3000ull + w.slot * 0x1000ull};
  return b;
}

// Every NVIDIA GPU has this space: the map's offsets and fields are NVIDIA's
// own published headers', which cover every architecture here, and one card's
// measurement confirms them (nvidia/registers/mmio.yaml). What differs per
// model is identity and which engines are running, both of which come from the
// profile below.
bool has_bar0(const telemetry::DeviceSample& d) { return std::strcmp(d.vendor, "amd") != 0; }

// The architecture id a card reports in PMC_BOOT_0, from nv_ref.h's own table.
// A profile whose architecture is not in it reports 0, which is no
// architecture rather than the wrong one.
uint32_t boot_architecture(const telemetry::DeviceSample& d) {
  const std::string arch = d.architecture;
  if (arch == "turing") return 0x16;       // NV_PMC_BOOT_0_ARCHITECTURE_TU100
  if (arch == "ampere") return 0x17;       // _GA100
  if (arch == "hopper") return 0x18;       // _GH100
  if (arch == "ada") return 0x19;          // _AD100
  if (arch == "blackwell") return 0x1a;    // _GB100
  return 0;
}

// The die within the architecture (PMC_BOOT_0's IMPLEMENTATION). Known only
// where a card was read: the RTX 3080 Ti answered 2, which is GA102. Every
// other model reports 0 rather than a guess at its die.
uint32_t boot_implementation(const telemetry::DeviceSample& d) {
  return (d.pci_device_id >> 16) == 0x2208 ? 2u : 0u;   // RTX 3080 Ti, measured
}

// Which engines a bound driver leaves out of reset: the host and graphics
// engines this simulator actually runs kernels and copies through, the first
// two copy engines, the power and security microcontrollers every modern part
// carries, the performance monitor, and the display engine only on a card with
// display outputs. The bit positions are dev_pmc.h's; which of them are set is
// a model.
//
// The video engines stay clear on purpose. Which decoders, encoders and JPEG
// engines a card has differs by model -- an A100 has decoders and no encoder --
// and the profiles do not record it, so setting those bits would be a guess
// about the card rather than a model of the driver.
uint32_t pmc_enable_value(const telemetry::DeviceSample& d) {
  uint32_t v = (1u << 8) | (1u << 12) | (1u << 6) | (1u << 7) |   // pfifo, pgraph, ce0, ce1
               (1u << 13) | (1u << 14) | (1u << 28);              // pwr, sec, perfmon
  if (is_geforce(d)) v |= (1u << 30);                             // pdisp: display outputs
  return v;
}

// The captured configuration space a device replays, if its family has one:
// an Ampere GeForce, the RTX 3080 Ti's.
const embedded::ConfigImage* captured(const telemetry::DeviceSample& d) {
  if (!is_geforce(d) || std::strcmp(d.architecture, "ampere") != 0) return nullptr;
  for (const auto& img : embedded::kConfigImages)
    if (std::strcmp(img.name, "nvidia-rtx3080ti") == 0) return &img;
  return nullptr;
}

// A GeForce drops its link to Gen1 while idle and retrains when work arrives
// (measured: RTX 3080 Ti, 2.5 GT/s idle of 16 GT/s).
uint32_t link_gen(const telemetry::DeviceSample& d) {
  return is_geforce(d) && d.utilization_gpu == 0 && d.pcie_gen ? 1 : d.pcie_gen;
}

bool nvidia_backed(const std::string& k, const Context& c, uint32_t* out) {
  const telemetry::DeviceSample& d = c.d;
  if (k == "profile.revision") *out = 0xa1;
  // Identity as the card reports it in BAR0: the architecture and die in their
  // published fields, the revision the same 0xa1 configuration space reports,
  // and bits 31:29 as the measured card had them -- the published header does
  // not define them, so they are carried across rather than invented.
  else if (k == "nvidia.pmc_boot_0")
    *out = (0x5u << 29) | (boot_architecture(d) << 24) | (boot_implementation(d) << 20) | 0xa1u;
  else if (k == "nvidia.pmc_enable") *out = pmc_enable_value(d);
  // A counter software polls for elapsed time, in nanoseconds, from the same
  // clock the profiler's timestamps use. It measures simulation, not a device.
  // At power-on it is zero, which is what the model's own file records: a live
  // clock written into the repository would differ on every run.
  else if (k == "nvidia.ptimer_nsec")
    *out = c.derived ? 0u : static_cast<uint32_t>(profiling::now_ns());
  // As a bound driver leaves it: memory and bus mastering on, INTx off for
  // MSI, and I/O on where there is an I/O BAR (measured: RTX 3080 Ti, 0x0407).
  else if (k == "profile.command") *out = is_geforce(d) ? 0x0407 : 0x0406;
  // A GeForce is function 0 of two, the second its HDMI audio controller
  // (measured: RTX 3080 Ti, 0x80).
  else if (k == "profile.header_type") *out = is_geforce(d) ? 0x80 : 0x00;
  // A data-center card is a 3D controller, a GeForce a VGA controller.
  else if (k == "profile.class") *out = is_geforce(d) ? 0x030000 : 0x030200;
  else return false;
  return true;
}

bool no_write(const std::string&, const Context&, uint32_t) { return false; }
// nvidia.ko's own files are in /proc/driver/nvidia, not the PCI device's
// directory, so a device has only the PCI files every device has.
void no_sysfs(const telemetry::DeviceSample&, const std::string&) {}

}  // namespace

const Vendor& nvidia() {
  // An undeclared BAR0 dword reads 0xbadf5040, what the measured card answered
  // with from 0xc on: a protected or absent register.
  static const Vendor v = {
      "nvidia",   Space::NvidiaMmio, "nvidia-mmio", "nvidia/registers/mmio.yaml",
      16u << 20,  0xbadf5040u,       has_bar0,      nvidia_bars,
      captured,   link_gen,          nvidia_backed, no_write,
      no_sysfs,
  };
  return v;
}

}  // namespace vgpu::regs::vendor
