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

// The machine as AMD's tools see it, which is as amdgpu sees it: an AMD GPU
// that has fallen off the bus (`vgpu fault lose`) was let go by the driver,
// so it is not there to list, and the GPUs after it move up. NVIDIA's tools
// still list a lost GPU, as lost; `vgpu` itself keeps every GPU's index.
void drop_lost_amd(telemetry::Shared* snap);

}  // namespace vgpu
