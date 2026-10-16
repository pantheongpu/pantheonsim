// Blocks that wait on one another.
//
// A card keeps every block of a grid resident at once when the grid fits the device (the SMs times
// the blocks each holds, from the kernel's registers, shared memory and threads), and starts the rest
// as the first ones finish. A block may therefore spin on a word another block writes: a grid barrier
// made of atomics, a chain of look-back flags, a producer in one block and its consumers in others, a
// lock taken by blocks that all start together.
//
// The engines used to run a grid's blocks one after another on each host thread, each to the end.
// That is a valid order for a program whose blocks are independent, but a block that waits for one
// that has not started then spins for ever, and a program whose answer depends on which blocks are in
// flight together (RXMesh's patch locks: the block with the lowest id wins every lock if all the
// blocks are resident, and any number of them win if they are not) gives a different answer on every
// run.
//
// Lockstep launch: for a kernel with a loop that polls global memory (the engines find them in the
// machine code: a loop with an atomic or a strong load whose address and operands do not change from
// one turn to the next, and no stores), each host thread holds a window of resident blocks and gives
// every one a short turn in order, so a block that waits is passed over while the block it waits for
// goes on. When a block finishes the next of the grid starts, so a grid larger than the device holds
// still runs its first blocks together and the rest as those finish.
//
// A grid of 16 blocks or fewer per host thread runs on one thread, in a fixed order, as a
// cooperative launch does: the answer of a program like the patch locks is then the same on every
// run. Larger grids use every host thread, each with its own window, so a kernel with a polling loop
// (CUB's decoupled look-back scan, say) keeps its speed.
//
// Kernels with no polling loop run as before. VGPU_LOCKSTEP=0 turns the whole thing off (the old
// behaviour, for debugging), VGPU_LOCKSTEP=all runs every launch this way (for checking that nothing
// depends on the order blocks start in), and the default, VGPU_LOCKSTEP=auto, is the polling kernels.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

namespace vgpu::exec {

enum class LockstepMode { Auto, Off, All };

inline LockstepMode lockstep_mode() {
  const char* v = std::getenv("VGPU_LOCKSTEP");
  if (!v || !*v || !std::strcmp(v, "auto")) return LockstepMode::Auto;
  if (!std::strcmp(v, "0") || !std::strcmp(v, "off")) return LockstepMode::Off;
  if (!std::strcmp(v, "all") || !std::strcmp(v, "1")) return LockstepMode::All;
  return LockstepMode::Auto;
}

// Instructions a warp (a wave) runs in its turn. Short enough that every resident block starts before
// the first has done its first lock or flag, long enough that the switch is not the cost.
inline constexpr uint64_t kLockstepSlice = 64;
// Blocks per host thread below which another thread is not worth starting (see above).
inline constexpr uint64_t kLockstepBlocksPerThread = 16;

struct LockstepShape {
  unsigned workers = 1;     // host threads
  uint64_t window = 1;      // blocks each holds resident at once
};

// `units` blocks (or clusters) in the grid, `capacity` of them resident at once on the device, and
// `threads` host threads available.
inline LockstepShape lockstep_shape(uint64_t units, uint64_t capacity, unsigned threads) {
  LockstepShape s;
  const uint64_t by_size = std::max<uint64_t>(1, units / kLockstepBlocksPerThread);
  s.workers = static_cast<unsigned>(std::max<uint64_t>(1, std::min<uint64_t>(threads, by_size)));
  const uint64_t resident = std::max<uint64_t>(1, std::min(units, capacity ? capacity : units));
  s.window = std::max<uint64_t>(1, (resident + s.workers - 1) / s.workers);
  return s;
}

// One host thread's part of a lockstep launch, and the one place that decides how resident blocks
// take turns. It keeps up to `window` blocks resident, taking the grid's next block from `ticket`
// whenever it has room, and gives each a turn in the order they started; a block whose turn says it
// is finished leaves, and its storage is used for the next. Blocks are the engine's own type:
//   admit(Block&, unit, slot)   sets the block up for grid unit `unit`; `slot` numbers it among this
//                               thread's resident blocks
//   turn(Block&) -> bool        runs the block for one turn; true when it has finished
//   retire(Block&)              takes what a finished block counted
//   stop() -> bool              true when the launch has failed elsewhere
// An exception from `turn` or `admit` leaves the worker: the caller records it.
template <class Block, class Admit, class Turn, class Retire, class Stop>
void lockstep_worker(std::atomic<uint64_t>& ticket, uint64_t units, uint64_t window, Admit&& admit, Turn&& turn,
                     Retire&& retire, Stop&& stop) {
  std::vector<std::unique_ptr<Block>> live, spare;
  bool drained = false;
  for (;;) {
    while (!drained && live.size() < window) {
      const uint64_t unit = ticket.fetch_add(1);
      if (unit >= units) {
        drained = true;
        break;
      }
      std::unique_ptr<Block> b;
      if (!spare.empty()) {
        b = std::move(spare.back());
        spare.pop_back();
      } else {
        b = std::make_unique<Block>();
      }
      admit(*b, unit, live.size());
      live.push_back(std::move(b));
    }
    if (live.empty() || stop()) return;
    size_t keep = 0;
    for (size_t k = 0; k < live.size(); ++k) {
      if (turn(*live[k])) {
        retire(*live[k]);
        spare.push_back(std::move(live[k]));
      } else {
        if (keep != k) live[keep] = std::move(live[k]);
        ++keep;
      }
    }
    live.resize(keep);
  }
}

}  // namespace vgpu::exec
