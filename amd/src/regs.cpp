// An AMD Instinct GPU's part of the register model (vgpu/regs_vendor.hpp):
// its BARs, its identity in configuration space, its MMIO space behind BAR5
// (amd/registers/mmio.yaml) -- engine status, the SMU mailbox, NBIO -- and the
// amdgpu files in its sysfs directory.
#include <unistd.h>

#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>

#include "vgpu/amd_metrics.hpp"
#include "vgpu/amd_regs.hpp"
#include "vgpu/ras.hpp"
#include "vgpu/regs_vendor.hpp"

namespace vgpu::regs::vendor {
namespace {

// BAR0 the framebuffer aperture, BAR2 the doorbells, both 64-bit and
// prefetchable; BAR5 the registers. Sizes are a model, not measured per card.
std::array<BarLayout, 6> amd_bars(const telemetry::DeviceSample& d) {
  std::array<BarLayout, 6> b{};
  const Windows w = windows(d);
  const uint64_t fb = pow2_at_least(d.vram_total_bytes ? d.vram_total_bytes : (1ull << 28));
  b[0] = {fb, false, true, true, false, w.mmio64};
  b[1].high = true;
  b[2] = {2ull << 20, false, true, true, false, w.mmio64 + 0x8000000000ull};
  b[3].high = true;
  b[5] = {512ull << 10, false, false, false, false, w.mmio32 + 0x01000000ull};
  return b;
}

bool is_amd(const telemetry::DeviceSample& d) { return std::strcmp(d.vendor, "amd") == 0; }
const embedded::ConfigImage* no_capture(const telemetry::DeviceSample&) { return nullptr; }
uint32_t trained_gen(const telemetry::DeviceSample& d) { return d.pcie_gen; }

// The SMU mailbox in the shared words: message, argument, response, and
// whether a response has been set since reset.
enum Mailbox { kMessage, kArgument, kResponse, kResponseSet };

bool amd_backed(const std::string& k, const Context& c, uint32_t* out) {
  const telemetry::DeviceSample& d = c.d;
  const uint32_t device = d.pci_device_id >> 16;
  // Instinct cards are processing accelerators, a single function, and a
  // bound driver leaves memory and bus mastering on and INTx off for MSI.
  if (k == "profile.revision") *out = 0x00;
  else if (k == "profile.command") *out = 0x0406;
  else if (k == "profile.header_type") *out = 0x00;
  else if (k == "profile.class") *out = 0x120000;
  // NBIO: the device ID and revision strapped, with the function enabled; and
  // the memory partition modes it supports -- NPS1 and NPS4 on CDNA3 (GC
  // 9.4.3 and 9.4.4, which amdgpu assumes when the register is not read,
  // gmc_v9_0.c), NPS1 and NPS2 on CDNA4.
  else if (k == "profile.nbio_strap0") *out = (1u << 28) | device;
  else if (k == "profile.nps_cap") *out = std::strcmp(d.architecture, "cdna4") == 0 ? 0x3u : 0x9u;
  // The VRAM in MiB, as this device has it: VGPU_VRAM_MB changes it for a
  // run, so it follows the device rather than the model.
  else if (k == "vram.memsize_mb") *out = static_cast<uint32_t>(d.vram_total_bytes >> 20);
  // Engine status: the units a busy GPU has working, and the idle state's
  // command FIFOs available and DB and CB clean. Values are a model.
  else if (k == "engine.grbm_status")
    *out = d.utilization_gpu ? 0xE7D84008u   // GUI active, CP, CB, DB, PA, SC, BCI, SPI, SX, IA, TA busy
                             : 0x00003028u;  // FIFOs available, DB and CB clean
  else if (k == "engine.grbm_status2") *out = d.utilization_gpu ? 0x73018008u : 0x00000008u;
  else if (k == "engine.cp_stat") *out = d.utilization_gpu ? 0x80000000u : 0u;
  else if (k == "engine.rlc_stat") *out = d.utilization_gpu ? 0x00000005u : 0u;
  else if (k == "smu.message") *out = static_cast<uint32_t>(__atomic_load_n(&c.words[kMessage], __ATOMIC_RELAXED));
  else if (k == "smu.argument") *out = static_cast<uint32_t>(__atomic_load_n(&c.words[kArgument], __ATOMIC_RELAXED));
  else if (k == "smu.response")
    // Ready after reset, as the SMU is once the driver has loaded.
    *out = __atomic_load_n(&c.words[kResponseSet], __ATOMIC_ACQUIRE)
               ? static_cast<uint32_t>(__atomic_load_n(&c.words[kResponse], __ATOMIC_RELAXED))
               : kSmuResultOk;
  else return false;
  return true;
}

// The SMU mailbox, as the driver drives it: clear the response, write the
// argument, write the message; the SMU answers in the response register, and
// in the argument register when the message returns a value.
bool amd_write(const std::string& k, const Context& c, uint32_t v) {
  uint64_t* s = c.words;
  if (k == "smu.argument") {
    __atomic_store_n(&s[kArgument], uint64_t{v}, __ATOMIC_RELAXED);
  } else if (k == "smu.response") {
    __atomic_store_n(&s[kResponse], uint64_t{v}, __ATOMIC_RELAXED);
    __atomic_store_n(&s[kResponseSet], uint64_t{1}, __ATOMIC_RELEASE);
  } else if (k == "smu.message") {
    __atomic_store_n(&s[kMessage], uint64_t{v}, __ATOMIC_RELAXED);
    const uint32_t arg = static_cast<uint32_t>(__atomic_load_n(&s[kArgument], __ATOMIC_RELAXED));
    uint32_t reply = arg, result = kSmuResultOk;
    switch (v) {
      case kSmuTestMessage: reply = arg + 1; break;
      // A firmware version in the form the driver prints as 85.111.0; a model.
      case kSmuGetSmuVersion: reply = 0x00556F00u; break;
      // The interface version the driver expects (SMU13_0_6_DRIVER_IF_VERSION).
      case kSmuGetDriverIfVersion: reply = 0x08042024u; break;
      case kSmuGetMetricsVersion: reply = 0x11u; break;   // SMU_METRICS_TABLE_VERSION
      default: result = kSmuResultUnknownCmd; break;
    }
    __atomic_store_n(&s[kArgument], uint64_t{reply}, __ATOMIC_RELAXED);
    __atomic_store_n(&s[kResponse], uint64_t{result}, __ATOMIC_RELAXED);
    __atomic_store_n(&s[kResponseSet], uint64_t{1}, __ATOMIC_RELEASE);
  } else {
    return false;
  }
  return true;
}

// Replaces a file whole, so a reader never sees half of one.
void replace_file(const std::string& path, const std::string& bytes) {
  const std::string tmp = path + ".tmp." + std::to_string(::getpid());
  {
    std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
    if (!out) return;
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
}

// The metrics table, amdgpu's device files and hwmon, and the partition modes,
// as amdgpu reads NBIO for them and names them (amdgpu_gfx.c's
// current_compute_partition, amdgpu_gmc.c's current_memory_partition and
// available_memory_partition).
void amd_sysfs(const telemetry::DeviceSample& d, const std::string& dir) {
  ras::Counters c{};
  try {
    c = ras::read(d.uuid).since_load;
  } catch (const std::exception&) {
  }
  replace_file(dir + "/gpu_metrics", amd::gpu_metrics(d, c));
  amd::write_driver_files(d, dir);
  RegisterSpace mmio(Space::AmdMmio, d);
  const auto reg = [&](const char* name) { return mmio.value(*find(Space::AmdMmio, name)); };
  static const char* const kCompute[] = {"SPX", "DPX", "TPX", "QPX", "CPX"};
  const uint32_t px = (reg("nbio_partition_compute_status") >> 4) & 0xF;
  replace_file(dir + "/current_compute_partition", std::string(px < 5 ? kCompute[px] : "UNKNOWN") + "\n");
  const auto nps_known = [](int m) { return m == 1 || m == 2 || m == 3 || m == 4 || m == 6 || m == 8; };
  const int nps = __builtin_ffs(static_cast<int>((reg("nbio_partition_mem_status") >> 4) & 0xFF));
  replace_file(dir + "/current_memory_partition",
               (nps_known(nps) ? "NPS" + std::to_string(nps) : std::string("UNKNOWN")) + "\n");
  std::string avail, sep;
  for (uint32_t cap = reg("nbio_partition_mem_cap"); cap; cap &= cap - 1)
    if (const int m = __builtin_ffs(static_cast<int>(cap)); nps_known(m)) {
      avail += sep + "NPS" + std::to_string(m);
      sep = ", ";
    }
  replace_file(dir + "/available_memory_partition", avail + "\n");
}

}  // namespace

const Vendor& amd() {
  static const Vendor v = {
      "amd",           Space::AmdMmio, "amd-mmio", "amd/registers/mmio.yaml",
      512u << 10,      0u,             is_amd,     amd_bars,
      no_capture,      trained_gen,    amd_backed, amd_write,
      amd_sysfs,
  };
  return v;
}

}  // namespace vgpu::regs::vendor
