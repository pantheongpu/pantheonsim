// An AMD GPU's metrics table, as its driver publishes it in sysfs
// (/sys/class/drm/cardN/device/gpu_metrics): a binary struct the SMU fills and
// amd-smi, rocm-smi and monitoring daemons read. The layout is the Linux
// kernel's gpu_metrics_v1_5 (kgd_pp_interface.h; MIT, notice in
// amd/registers/LICENSES/amdgpu-headers.txt), the one MI300-class GPUs have
// published; later kernels moved to later versions.
#pragma once

#include <time.h>

#include <chrono>
#include <cstdint>
#include <map>
#include <mutex>
#include <string>

#include "vgpu/ras.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::amd {

// The table's bytes for a device read with its faults applied, and its
// reliability counts. Fields the simulator has nothing for read as all ones,
// as the driver leaves fields a GPU does not report.
std::string gpu_metrics(const telemetry::DeviceSample& d, const ras::Counters& c);

inline constexpr size_t kGpuMetricsSize = 360;

// gpu_metrics_v1_5, field for field, with its natural alignment.
struct MetricsV15 {
  uint16_t structure_size;
  uint8_t format_revision;
  uint8_t content_revision;
  uint16_t temperature_hotspot, temperature_mem, temperature_vrsoc;   // Celsius
  uint16_t curr_socket_power;                                         // W
  uint16_t average_gfx_activity, average_umc_activity;                // %
  uint16_t vcn_activity[4];
  uint16_t jpeg_activity[32];
  uint64_t energy_accumulator;   // 15.259 uJ units
  uint64_t system_clock_counter; // ns
  uint32_t throttle_status;
  uint32_t gfxclk_lock_status;
  uint16_t pcie_link_width;      // lanes
  uint16_t pcie_link_speed;      // 0.1 GT/s
  uint16_t xgmi_link_width, xgmi_link_speed;
  uint32_t gfx_activity_acc, mem_activity_acc;
  uint64_t pcie_bandwidth_acc, pcie_bandwidth_inst;
  uint64_t pcie_l0_to_recov_count_acc, pcie_replay_count_acc, pcie_replay_rover_count_acc;
  uint32_t pcie_nak_sent_count_acc, pcie_nak_rcvd_count_acc;
  uint64_t xgmi_read_data_acc[8], xgmi_write_data_acc[8];
  uint64_t firmware_timestamp;   // 10 ns
  uint16_t current_gfxclk[8];    // MHz
  uint16_t current_socclk[4], current_vclk0[4], current_dclk0[4];
  uint16_t current_uclk;
  uint16_t padding;
};

// The energy accumulator an Instinct GPU has (a Radeon's driver exposes none):
// a count of 15.3 uJ ticks (ROCm SMI's counter_resolution, in micro joules),
// integrated from the power model's reading at each call. It starts from an
// offset for each GPU, as a hardware counter does not start at zero, and the
// timestamp is the table's system_clock_counter (ns). The one accumulator
// serves libamd_smi and librocm_smi64, which ask the same question: the
// function is inline with default visibility, so the two libraries, each built
// with hidden symbols, still share the one copy of its state in a process.
inline constexpr float kEnergyTickUj = 15.3f;
struct EnergyReading {
  uint64_t ticks, timestamp_ns;
};
__attribute__((visibility("default"))) inline EnergyReading energy_counter(const telemetry::DeviceSample& d) {
  struct Account {
    double uj = 0;
    uint64_t at_ns = 0;
  };
  static std::mutex mu;
  static std::map<std::string, Account> accounts;
  timespec ts{};
  clock_gettime(CLOCK_BOOTTIME, &ts);
  const uint64_t now = static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
  uint64_t h = 1469598103934665603ull;
  for (const char* c = d.uuid; *c; ++c) h = (h ^ static_cast<unsigned char>(*c)) * 1099511628211ull;
  std::lock_guard<std::mutex> lock(mu);
  Account& a = accounts[d.uuid];
  if (a.at_ns) a.uj += d.power_mw * 1000.0 * (static_cast<double>(now - a.at_ns) / 1e9);   // mW * s = mJ; x1000 = uJ
  a.at_ns = now;
  return {1000000 + (h & 0xFFFF) * 64 + static_cast<uint64_t>(a.uj / kEnergyTickUj), now};
}

// The amdgpu driver's text files for a device, written into `dir` as the
// driver keeps them in the device's sysfs directory -- gpu_busy_percent,
// mem_busy_percent, mem_info_vram_total/_used, mem_info_vis_vram_total/_used
// -- and its hwmon files into `dir`/hwmon, in the hwmon ABI's units: an
// MI300-class GPU's junction (temp2) and memory (temp3) temperatures with
// their critical and emergency limits, power (power1_average, power1_cap*),
// and the graphics and memory clocks (freq1, freq2). What sensors, nvtop and
// exporters read.
void write_driver_files(const telemetry::DeviceSample& d, const std::string& dir);
// The files write_driver_files writes, and the partition files
// regs::write_sysfs_files writes from NBIO's registers, for a session to link to.
extern const char* const kDriverFiles[9];
extern const char* const kHwmonFiles[21];

}  // namespace vgpu::amd
