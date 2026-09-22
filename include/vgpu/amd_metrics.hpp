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

}  // namespace vgpu::amd
