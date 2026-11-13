// Progress inside a block: how much of a block's work is left, read from the induction register of the loop its warps are in (Round sp61;
// docs/perf-design/12-host-gpu-turbo.md section 8, "Progress inside a block").
//
// The host-GPU mode's sample (vgpu/host_gpu.hpp, "The first blocks of a launch, timed") times whole blocks in a forked copy of the process.
// A grid of a few blocks whose threads loop for seconds (PolyBench COVAR and CORR: 8 blocks of 256 threads, each thread looping over the whole
// matrix) has no block that ends within any budget a sample can afford. What such a block has left is in its registers instead: the loop's
// induction register against the bound its compare reads. This header is the arithmetic of that reading, with no engine in it, so that it
// can be tested alone (tests/unit/test_host_gpu_logic.cpp); the loops themselves are found in the code by vgpu/host_gpu.hpp find_probe_loops,
// and the engine reads the registers (src/sass/exec.cpp, Runner::probe_block).
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace vgpu::exec {

// An operand of a loop's compare as the copy reads it at a pause: a register of the lane, a uniform register of the warp, an immediate, or a
// word of a constant bank (a kernel parameter).
struct ProbeOperand {
  enum Kind : uint8_t { None, Reg, UReg, Imm, CBank };
  uint8_t kind = None;
  uint32_t reg = 0;   // the register, or the bank of a CBank
  int64_t imm = 0;    // the immediate, or the offset of a CBank
};

// One top-level loop of a kernel that the probe follows: its instructions are the code addresses [lo, hi]; each turn adds `step` to the
// induction register `iv` (one add of a constant, the only write of `iv` in the loop), and the compare that decides whether another turn is
// taken reads `iv` + `iv_offset` against `bound` (an immediate, a constant-bank word, or a register the loop does not write).
struct ProbeLoop {
  uint64_t lo = 0, hi = 0;
  ProbeOperand iv;
  int64_t iv_offset = 0;
  int64_t step = 0;
  ProbeOperand bound;
  bool is_unsigned = false;
};

inline constexpr unsigned kProbeLoops = 4;   // top-level loops followed at most (the corpus's kernels have one or two)

// What the copy is asked to do in place of running whole blocks (exec::LaunchPart::probe).
struct BlockProbe {
  ProbeLoop loops[kProbeLoops];
  unsigned count = 0;
  double enter_s = 0;     // per probed block: the most its warps are run, round by round, to reach a followed loop
  double measure_s = 0;   // one warp of the first probed block, alone: how long its steps per turn of the loop are measured
};

// How many turns a lane has left, from what it holds at a pause: the distance from its compared value (iv + iv_offset) to the bound, in
// steps. The compare's direction is not read: the distance is the same whichever way the branch is written (a loop that goes while i < n
// and one that leaves when i >= n), and a lane already past the bound, or with no distance in the step's direction, has none left. The values
// are the 32-bit words the compare reads, signed unless it compares unsigned. A pause may fall between the add and the compare, so the result
// can be one turn off.
inline double lane_turns_left(uint32_t iv_value, uint32_t bound_value, const ProbeLoop& l) {
  if (l.step == 0) return 0;
  const int64_t x = (l.is_unsigned ? static_cast<int64_t>(iv_value) : static_cast<int64_t>(static_cast<int32_t>(iv_value))) + l.iv_offset;
  const int64_t b = l.is_unsigned ? static_cast<int64_t>(bound_value) : static_cast<int64_t>(static_cast<int32_t>(bound_value));
  const int64_t d = b - x;
  if ((d > 0) != (l.step > 0)) return 0;   // (at the bound: none either way)
  const int64_t s = l.step > 0 ? l.step : -l.step;
  const int64_t a = d > 0 ? d : -d;
  return static_cast<double>((a + s - 1) / s);
}

// How many turns of the loop a warp completed between two pauses, from the values of one lane that was in the loop at both: the distance
// its induction register moved, in steps.
inline double turns_between(uint32_t before, uint32_t after, const ProbeLoop& l) {
  if (l.step == 0) return 0;
  const int64_t x0 = l.is_unsigned ? static_cast<int64_t>(before) : static_cast<int64_t>(static_cast<int32_t>(before));
  const int64_t x1 = l.is_unsigned ? static_cast<int64_t>(after) : static_cast<int64_t>(static_cast<int32_t>(after));
  const int64_t d = x1 - x0;
  if ((d > 0) != (l.step > 0)) return 0;
  const int64_t s = l.step > 0 ? l.step : -l.step;
  return static_cast<double>((d > 0 ? d : -d) / s);
}

// One warp's turn of the loop, measured: the warp of the first probed block that has the most turns left is run alone for
// BlockProbe::measure_s, and its steps and the turns its induction register moved give the steps of one turn.
struct TurnMeasure {
  uint64_t steps = 0;    // the warp's steps while it ran alone
  double turns = 0;      // turns of its loop completed meanwhile
  double seconds = 0;    // the wall time of that run
  bool left = false;     // it left the loop (or ended) meanwhile: `turns` is then what it had left when it started
};

// The steps of one turn. With no turn completed, the turn under way has lasted the whole run, so the run's steps are the least a turn takes:
// the estimate then errs low, towards the simulator. None (0) when the warp took no step.
inline double steps_per_turn(const TurnMeasure& m) {
  return m.turns >= 1 ? static_cast<double>(m.steps) / m.turns : static_cast<double>(m.steps);
}

// What the simulator's engine takes for one warp step of this code: the measured run's time over its steps. None (0) without a measure.
inline double seconds_per_step(const TurnMeasure& m) {
  if (m.steps == 0 || !(m.seconds > 0)) return 0;
  return m.seconds / static_cast<double>(m.steps);
}

// A probed block, as its warps stood when they had all reached a followed loop (or ended, or the time ran out): the steps they had taken,
// and the turns they had left (per warp, the most any of its lanes had; warps in no followed loop add none, so the estimate errs low).
struct BlockReading {
  double steps = 0;   // sum over the warps
  double turns = 0;   // sum over the warps
};

// The steps the whole block takes: what it took, and its turns left at the measured cost of a turn. (What follows the loop is not counted.)
inline double block_steps(const BlockReading& b, double per_turn) { return b.steps + b.turns * std::max(per_turn, 0.0); }

}  // namespace vgpu::exec
