// The machine a tool describes: the devices it reports and the registers it
// reads. Shared by nvidia-smi, rocm-smi, amd-smi, `vgpu fault`, `vgpu regs` and
// the register C API, so every one of them sees the same devices.
#pragma once

#include "vgpu/telemetry.hpp"

namespace vgpu {

// Live telemetry when anything is publishing, otherwise the rack VGPU_GPU and
// VGPU_DEVICE_COUNT describe, idle -- with the injected faults applied to every
// reading (clock-event reasons, a degraded link). False, with a message on
// stderr, when there is neither.
bool read_machine(telemetry::Shared* snap);

}  // namespace vgpu
