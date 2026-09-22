// An AMD GPU's register-level device logic (amd/src/regs.cpp).
#pragma once

#include <cstdint>

namespace vgpu::regs {

// The SMU messages the mailbox answers, from smu_v13_0_6_ppsmc.h, and what it
// answers with. Any other message is refused as an unknown command.
inline constexpr uint32_t kSmuTestMessage = 0x1, kSmuGetSmuVersion = 0x2, kSmuGetDriverIfVersion = 0x4,
                          kSmuGetMetricsVersion = 0x8;
inline constexpr uint32_t kSmuResultOk = 0x1, kSmuResultUnknownCmd = 0xFE;

}  // namespace vgpu::regs
