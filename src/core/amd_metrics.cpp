#include "vgpu/amd_metrics.hpp"

#include <time.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace vgpu::amd {
namespace {

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
// The kernel's own layout, checked field by field where it matters.
static_assert(sizeof(MetricsV15) == kGpuMetricsSize);
static_assert(offsetof(MetricsV15, temperature_hotspot) == 4);
static_assert(offsetof(MetricsV15, curr_socket_power) == 10);
static_assert(offsetof(MetricsV15, energy_accumulator) == 88);
static_assert(offsetof(MetricsV15, throttle_status) == 104);
static_assert(offsetof(MetricsV15, pcie_link_width) == 112);
static_assert(offsetof(MetricsV15, pcie_replay_count_acc) == 152);
static_assert(offsetof(MetricsV15, xgmi_read_data_acc) == 176);
static_assert(offsetof(MetricsV15, current_gfxclk) == 312);
static_assert(offsetof(MetricsV15, current_uclk) == 352);

uint64_t now_ns(clockid_t clock) {
  timespec ts{};
  clock_gettime(clock, &ts);
  return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// A link generation as the table gives its speed, in tenths of GT/s.
uint16_t link_speed(uint32_t gen) {
  static const uint16_t kSpeeds[] = {0, 25, 50, 80, 160, 320, 640};
  return gen < 7 ? kSpeeds[gen] : 0xFFFF;
}

}  // namespace

std::string gpu_metrics(const telemetry::DeviceSample& d, const ras::Counters& c) {
  MetricsV15 m;
  std::memset(&m, 0xFF, sizeof m);   // what the GPU does not report
  m.structure_size = sizeof m;
  m.format_revision = 1;
  m.content_revision = 5;
  m.temperature_hotspot = static_cast<uint16_t>(d.temperature_c);
  m.temperature_mem = static_cast<uint16_t>(d.has_memory_temperature && d.temperature_mem_c ? d.temperature_mem_c
                                                                                             : d.temperature_c);
  m.curr_socket_power = static_cast<uint16_t>(d.power_mw / 1000);
  m.average_gfx_activity = static_cast<uint16_t>(d.utilization_gpu);
  m.average_umc_activity = static_cast<uint16_t>(d.utilization_mem);
  m.system_clock_counter = now_ns(CLOCK_BOOTTIME);
  m.firmware_timestamp = now_ns(CLOCK_MONOTONIC) / 10;
  m.throttle_status = 0;
  m.gfxclk_lock_status = 0;
  m.pcie_link_width = static_cast<uint16_t>(d.pcie_width);
  m.pcie_link_speed = link_speed(d.pcie_gen);
  const auto pcie = [&](ras::Pcie p) { return c.pcie[static_cast<uint32_t>(p)]; };
  m.pcie_l0_to_recov_count_acc = pcie(ras::Pcie::L0ToRecovery);
  m.pcie_replay_count_acc = pcie(ras::Pcie::Replay);
  m.pcie_replay_rover_count_acc = pcie(ras::Pcie::ReplayRollover);
  m.pcie_nak_sent_count_acc = static_cast<uint32_t>(pcie(ras::Pcie::NaksSent));
  m.pcie_nak_rcvd_count_acc = static_cast<uint32_t>(pcie(ras::Pcie::NaksReceived));
  for (auto& x : m.xgmi_read_data_acc) x = 0;
  for (auto& x : m.xgmi_write_data_acc) x = 0;
  // Every XCD at the graphics clock; the memory clock.
  for (auto& clk : m.current_gfxclk) clk = static_cast<uint16_t>(d.sm_clock_mhz);
  m.current_uclk = static_cast<uint16_t>(d.mem_clock_mhz);
  m.padding = 0;
  return std::string(reinterpret_cast<const char*>(&m), sizeof m);
}

const char* const kDriverFiles[9] = {"gpu_busy_percent",        "mem_busy_percent",
                                    "mem_info_vram_total",     "mem_info_vram_used",
                                    "mem_info_vis_vram_total", "mem_info_vis_vram_used",
                                    // Written from NBIO's registers (regs::write_sysfs_files).
                                    "current_compute_partition", "current_memory_partition",
                                    "available_memory_partition"};
const char* const kHwmonFiles[21] = {
    "name",          "temp2_input",   "temp2_label",    "temp2_crit",      "temp2_crit_hyst",
    "temp2_emergency", "temp3_input", "temp3_label",    "temp3_crit",      "temp3_crit_hyst",
    "temp3_emergency", "power1_average", "power1_label", "power1_cap",     "power1_cap_max",
    "power1_cap_min", "power1_cap_default", "freq1_input", "freq1_label", "freq2_input", "freq2_label"};

namespace {
// Replaces a file whole, so a reader never sees half of one.
void put(const std::string& path, const std::string& text) {
  const std::string tmp = path + ".tmp";
  {
    std::ofstream out(tmp, std::ios::trunc);
    if (!out) return;
    out << text << "\n";
  }
  if (std::rename(tmp.c_str(), path.c_str()) != 0) std::remove(tmp.c_str());
}
std::string num(uint64_t v) { return std::to_string(v); }
}  // namespace

void write_driver_files(const telemetry::DeviceSample& d, const std::string& dir) {
  put(dir + "/gpu_busy_percent", num(d.utilization_gpu));
  put(dir + "/mem_busy_percent", num(d.utilization_mem));
  // The framebuffer is all CPU-visible on a large-BAR Instinct card.
  put(dir + "/mem_info_vram_total", num(d.vram_total_bytes));
  put(dir + "/mem_info_vram_used", num(d.vram_used_bytes));
  put(dir + "/mem_info_vis_vram_total", num(d.vram_total_bytes));
  put(dir + "/mem_info_vis_vram_used", num(d.vram_used_bytes));
  const std::string h = dir + "/hwmon";
  std::error_code ec;
  std::filesystem::create_directories(h, ec);
  // hwmon's units: millidegrees Celsius, microwatts, hertz. The critical limit
  // is the device's slowdown threshold; the hysteresis and emergency margins
  // are a model.
  const uint64_t crit = uint64_t{d.temperature_max_c} * 1000;
  const uint64_t mem_c = d.has_memory_temperature && d.temperature_mem_c ? d.temperature_mem_c : d.temperature_c;
  put(h + "/name", "amdgpu");
  put(h + "/temp2_input", num(uint64_t{d.temperature_c} * 1000));
  put(h + "/temp2_label", "junction");
  put(h + "/temp2_crit", num(crit));
  put(h + "/temp2_crit_hyst", num(crit - 5000));
  put(h + "/temp2_emergency", num(crit + 10000));
  put(h + "/temp3_input", num(mem_c * 1000));
  put(h + "/temp3_label", "mem");
  put(h + "/temp3_crit", num(crit));
  put(h + "/temp3_crit_hyst", num(crit - 5000));
  put(h + "/temp3_emergency", num(crit + 10000));
  put(h + "/power1_average", num(uint64_t{d.power_mw} * 1000));
  put(h + "/power1_label", "PPT");
  put(h + "/power1_cap", num(uint64_t{d.power_limit_mw} * 1000));
  put(h + "/power1_cap_max", num(uint64_t{d.power_limit_mw} * 1000));
  put(h + "/power1_cap_min", "0");
  put(h + "/power1_cap_default", num(uint64_t{d.power_limit_mw} * 1000));
  put(h + "/freq1_input", num(uint64_t{d.sm_clock_mhz} * 1000000));
  put(h + "/freq1_label", "sclk");
  put(h + "/freq2_input", num(uint64_t{d.mem_clock_mhz} * 1000000));
  put(h + "/freq2_label", "mclk");
}

}  // namespace vgpu::amd
