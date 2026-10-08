// An AMD GPU's register-level device logic (amd/src/regs.cpp).
#pragma once

#include <cstdint>

namespace vgpu::regs {

// The SMU messages the mailbox answers, from smu_v13_0_6_ppsmc.h, and what it
// answers with. A message that smu_v13_0_6_ppsmc.h defines but the model has no
// handler for fails (PPSMC_Result_Failed); a number it does not define is refused
// as an unknown command. The model never answers busy or prerequisite-rejected
// (PPSMC_Result_CmdRejectedBusy, _CmdRejectedPrereq): it answers at once, and
// the header publishes no condition for either.
inline constexpr uint32_t kSmuTestMessage = 0x1, kSmuGetSmuVersion = 0x2, kSmuGetDriverIfVersion = 0x4,
                          kSmuGetMetricsVersion = 0x8;
inline constexpr uint32_t kSmuResultOk = 0x1, kSmuResultFailed = 0xFF, kSmuResultUnknownCmd = 0xFE,
                          kSmuResultCmdRejectedPrereq = 0xFD, kSmuResultCmdRejectedBusy = 0xFC;

// Whether smu_v13_0_6_ppsmc.h defines the message (PPSMC_MSG_*).
inline constexpr bool smu_message_defined(uint32_t m) {
  return (m >= 0x1 && m <= 0x3B) || m == 0x40 || m == 0x43 || m == 0x44 || m == 0x4D || m == 0x59 || m == 0x5B;
}

}  // namespace vgpu::regs
