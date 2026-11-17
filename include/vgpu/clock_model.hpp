// What the simulator charges for reading the global timer, shared by the PTX and the SASS engines.
//
// There is no timing model: %clock, %clock64 and %globaltimer count instructions, one per warp instruction. That makes a kernel that busy-waits N ns
// on %globaltimer (a loop of a few instructions that reads it, compares and branches) run about N instructions, and a wait of a couple of seconds ran into
// the step budget (1,073,741,824 per warp) as a possible infinite loop (pantheonsim issue #687; on an RTX 3080 Ti the same kernel simply runs 2 s).
// A read of the timer is slow on a card (a special-register read that goes out to the chip's timer, a hundred cycles or more), so a read here is
// charged kGlobalTimerReadNs of the clock on top of the instruction itself: the timer advances that much each time a warp reads it (once per warp
// instruction, not per lane; %globaltimer and %globaltimer_lo, SR_GLOBALTIMERLO, not the high word read beside it). A wait of N ns then takes
// about N / (kGlobalTimerReadNs + 4) loop passes: a two-second wait is some 125 million instructions, and one has to last about seventeen
// seconds to reach the default budget. The figure is a modelled latency, not a measurement, and only the reads of the timer pay it: %clock and
// %clock64 stay one per instruction.
#pragma once
#include <cstdint>

namespace vgpu {
inline constexpr uint64_t kGlobalTimerReadNs = 64;

// The end of the SASS engine's step-budget error ("... instructions (an infinite loop?" + this + ")"), thrown from two places (the interpreter's run loop and
// the tier's chain) that must say the same: a long wait on a clock reaches the budget too, and how to lift it (#687).
inline constexpr const char* kSassStepBudgetHint =
    " or a long wait on SR_CLOCK, which counts one instruction per cycle, or on SR_GLOBALTIMER, which counts one per read plus 64 ns;"
    " --max-steps 0 lifts the budget";
}  // namespace vgpu
