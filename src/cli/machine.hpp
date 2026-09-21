// The machine a command describes. Shared by nvidia-smi, rocm-smi and
// `vgpu fault`, so every one of them sees the same devices.
#pragma once

#include "vgpu/telemetry.hpp"

namespace vgpu::cli {

// Live telemetry when anything is publishing, otherwise the rack VGPU_GPU and
// VGPU_DEVICE_COUNT describe, idle. False, with a message on stderr, when there
// is neither.
bool read_machine(vgpu::telemetry::Shared* snap);

}  // namespace vgpu::cli
