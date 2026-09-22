// An AMD GPU's metrics table, as its driver publishes it in sysfs
// (/sys/class/drm/cardN/device/gpu_metrics): a binary struct the SMU fills and
// amd-smi, rocm-smi and monitoring daemons read. The layout is the Linux
// kernel's gpu_metrics_v1_5 (kgd_pp_interface.h; MIT, notice in
// registers/LICENSES/amdgpu-headers.txt), the one MI300-class GPUs have
// published; later kernels moved to later versions.
#pragma once

#include <string>

#include "vgpu/ras.hpp"
#include "vgpu/telemetry.hpp"

namespace vgpu::amd {

// The table's bytes for a device read with its faults applied, and its
// reliability counts. Fields the simulator has nothing for read as all ones,
// as the driver leaves fields a GPU does not report.
std::string gpu_metrics(const telemetry::DeviceSample& d, const ras::Counters& c);

inline constexpr size_t kGpuMetricsSize = 360;

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
