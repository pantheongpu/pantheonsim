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
// A grid of fewer than 16 blocks runs on one host thread with all its resident blocks taking turns
// from the start, in a fixed order, as a cooperative launch does: the answer of a program like the
// patch locks is then the same on every run. A larger grid uses the host threads, two blocks to a thread at least, each
// starting with one block, as before, and holding one more block at a time only after no block of the
// launch, on any thread, has stored or finished for three milliseconds: a chain of look-back flags,
// where each block waits on the one before it, then runs as it did (the blocks of a real kernel store
// to shared or global memory all the time), and a grid barrier gets all its blocks resident within a
// few tens of milliseconds. While a thread's blocks all stand still (their registers and program
// counters unchanged between two looks) it sleeps between passes (to half a millisecond), so blocks
// waiting on the host do not take a core.
//
// Kernels with no polling loop run as before. VGPU_LOCKSTEP=0 turns the whole thing off (the old
// behaviour, for debugging), VGPU_LOCKSTEP=all runs every launch this way (for checking that nothing
// depends on the order blocks start in), and the default, VGPU_LOCKSTEP=auto, is the polling kernels.
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include <cstring>
#include <memory>
#include <thread>
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
// A grid of fewer blocks than this runs on one host thread, all of them resident from the start, in
// a fixed order (see above); a larger one has a host thread for every kLockstepBlocksPerThread blocks.
inline constexpr uint64_t kLockstepOneThread = 16, kLockstepBlocksPerThread = 2;
// How long no block of the launch finishes or stores before a thread holds one more, and how long
// again before it holds one more still.
inline constexpr std::chrono::microseconds kLockstepGrowAfter{3000}, kLockstepGrowAgain{1000};
// How often a thread whose blocks stored nothing looks whether they are standing still.
inline constexpr std::chrono::microseconds kLockstepCheckEvery{250};

struct LockstepShape {
  unsigned workers = 1;     // host threads
  uint64_t window = 1;      // the most blocks each holds resident at once
  uint64_t initial = 1;     // and how many it starts with: all of them for one thread, else one
  uint64_t chunk = 1;       // consecutive blocks it takes from the grid at a time
};

// `units` blocks (or clusters) in the grid, `capacity` of them resident at once on the device, and
// `threads` host threads available.
// `chunk`: blocks a thread takes at a time (the NVIDIA engines take them one by one, as they always did;
// the AMD engine took contiguous ranges, and a chain of work-groups each waiting on the one before it
// then hands over between threads once a range, not at every link).
inline LockstepShape lockstep_shape(uint64_t units, uint64_t capacity, unsigned threads, bool ranges = false) {
  LockstepShape s;
  const uint64_t by_size = units < kLockstepOneThread ? 1 : std::max<uint64_t>(1, units / kLockstepBlocksPerThread);
  s.workers = static_cast<unsigned>(std::max<uint64_t>(1, std::min<uint64_t>(threads, by_size)));
  const uint64_t resident = std::max<uint64_t>(1, std::min(units, capacity ? capacity : units));
  s.window = std::max<uint64_t>(1, (resident + s.workers - 1) / s.workers);
  s.initial = s.workers == 1 ? s.window : 1;
  if (ranges) s.chunk = std::clamp<uint64_t>(units / (uint64_t{s.workers} * 8), 1, 32);
  return s;
}

// Says a launch is running this way: on stderr when VGPU_LOCKSTEP_LOG is 1, and otherwise as a line appended to the
// file VGPU_LOCKSTEP_LOG names (which the tests use to count the kernels that take this path).
inline void lockstep_note(const char* kernel, uint64_t units, const LockstepShape& s) {
  const char* file = std::getenv("VGPU_LOCKSTEP_LOG");
  if (!file || !file[0]) return;
  const bool to_stderr = !std::strcmp(file, "1"), to_file = !to_stderr;
  char line[600];
  std::snprintf(line, sizeof line, "[vgpu] lockstep: kernel %s, %llu blocks, %u host thread(s), up to %llu resident each\n", kernel,
                static_cast<unsigned long long>(units), s.workers, static_cast<unsigned long long>(s.window));
  if (to_stderr) std::fputs(line, stderr);
  if (to_file)
    if (std::FILE* f = std::fopen(file, "a")) {
      std::fputs(line, f);
      std::fclose(f);
    }
}

// Folds `n` bytes into a running fingerprint (the engines fingerprint a block's registers and
// program counters to tell a block that is waiting from one that is computing).
inline uint64_t lockstep_fold(uint64_t h, const void* data, size_t n) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  for (; n >= 8; n -= 8, p += 8) {
    uint64_t w;
    std::memcpy(&w, p, 8);
    h = (h ^ w) * 0x100000001b3ull;
    h ^= h >> 29;
  }
  uint64_t w = 0;
  if (n) std::memcpy(&w, p, n);
  return (h ^ w) * 0x100000001b3ull;
}

// One host thread's part of a lockstep launch, and the one place that decides how resident blocks
// take turns. It keeps blocks resident, taking the grid's next from `ticket` whenever it has room
// (up to `initial` of them to begin with, then doubling, up to `window`, once no block of the launch,
// on any thread, has finished or stored for kLockstepGrowAfter and this thread's are standing still:
// the threads tell each other by bumping `beat` whenever a pass of theirs did either, or a look found
// their blocks changed), and gives each a
// turn in the order they started; a block whose turn says it is finished leaves, and its storage is
// used for the next. Blocks are the engine's own type:
//   admit(Block&, unit, slot)   sets the block up for grid unit `unit`; `slot` numbers it among this
//                               thread's resident blocks
//   turn(Block&) -> bool        runs the block for one turn; true when it has finished
//   stored(const Block&)        how many stores the block has made so far (it only grows)
//   fingerprint(const Block&)   a hash of the block's registers and program counters: equal twice
//                               over a long run of turns, a block is waiting, not computing
//   retire(Block&)              takes what a finished block counted
//   stop() -> bool              true when the launch has failed elsewhere
// While every block this thread holds has the same fingerprint at three checks in a row (and did
// not store or finish in between) the thread sleeps between passes, for longer and longer, to a
// millisecond: a block waiting on the host, or on a block that is running on a thread the host has
// descheduled, does not take a core from it. It starts again at once when anything moves.
// An exception from `turn` or `admit` leaves the worker: the caller records it.
template <class Block, class Admit, class Turn, class Stored, class Fingerprint, class Retire, class Stop>
void lockstep_worker(std::atomic<uint64_t>& ticket, std::atomic<uint64_t>& beat, uint64_t units, uint64_t window,
                     uint64_t initial, uint64_t chunk, Admit&& admit, Turn&& turn, Stored&& stored, Fingerprint&& fingerprint,
                     Retire&& retire, Stop&& stop) {
  std::vector<std::unique_ptr<Block>> live, spare;
  bool drained = false;
  uint64_t target = std::max<uint64_t>(1, std::min(initial, window)), done_stores = 0;
  uint64_t next_unit = 0, chunk_end = 0, last_print = 0;
  unsigned same = 0, nap_us = 0;
  auto moved_at = std::chrono::steady_clock::now(), last_check = moved_at;
  auto grow_wait = kLockstepGrowAfter;
  uint64_t seen = beat.load(std::memory_order_relaxed);
  // This thread's blocks, or some other thread's, got somewhere.
  auto last_bump = moved_at;
  std::chrono::steady_clock::duration check_every = kLockstepCheckEvery;
  const auto moved = [&](std::chrono::steady_clock::time_point t) {
    // (told at most every 20 microseconds: the others look at it every few hundred)
    if (t - last_bump >= std::chrono::microseconds(20)) {
      beat.fetch_add(1, std::memory_order_relaxed);
      last_bump = t;
    }
    seen = beat.load(std::memory_order_relaxed);
    same = nap_us = 0;
    moved_at = last_check = t;
    last_print = 0;
    grow_wait = kLockstepGrowAfter;
  };
  const auto stores_now = [&] {
    uint64_t n = done_stores;
    for (const auto& b : live) n += stored(*b);
    return n;
  };
  for (;;) {
    while (!drained && live.size() < target) {
      if (next_unit == chunk_end) {   // this thread's chunk is used up: the grid's next one
        next_unit = ticket.fetch_add(chunk);
        chunk_end = next_unit + chunk;
      }
      const uint64_t unit = next_unit++;
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
    if (live.empty() || stop()) {
      return;
    }
    const uint64_t before = stores_now();
    bool finished = false;
    size_t keep = 0;
    for (size_t k = 0; k < live.size(); ++k) {
      if (turn(*live[k])) {
        finished = true;
        done_stores += stored(*live[k]);
        retire(*live[k]);
        spare.push_back(std::move(live[k]));
      } else {
        if (keep != k) live[keep] = std::move(live[k]);
        ++keep;
      }
    }
    live.resize(keep);
    const auto t = std::chrono::steady_clock::now();
    // A pass that finished a block or stored something is progress, and the other threads are told so.
    if (finished || stores_now() != before) {
      moved(t);
      continue;
    }
    // Nothing stored or finished. Every so often, look whether the blocks are standing still (their
    // registers and program counters the same as at the last look).
    if (t - last_check >= check_every) {
      uint64_t h = 0x9e3779b97f4a7c15ull;
      for (const auto& b : live) h = (h ^ fingerprint(*b)) * 0x100000001b3ull;
      last_check = std::chrono::steady_clock::now();
      // (looking costs about a twelfth of the time between looks, however big the blocks are)
      check_every = std::max<std::chrono::steady_clock::duration>(kLockstepCheckEvery, 12 * (last_check - t));
      same = h == last_print ? same + 1 : 0;
      last_print = h;
      if (!same) nap_us = 0;
    }
    // Standing still twice running: sleep between passes, for longer and longer. (A wait that counts its
    // polls never stands still, and does not sleep; it gives way to whatever else there is to run.)
    if (same >= 2) {
      nap_us = nap_us ? std::min(nap_us * 2, 500u) : 5u;
      std::this_thread::sleep_for(std::chrono::microseconds(nap_us));
    }
    // And whether to hold more blocks: not while any thread's blocks store or finish (they may be the
    // ones these wait for), but when none have for a while, the block waited for may not be here yet.
    if (target < window && !drained) {
      const uint64_t now = beat.load(std::memory_order_relaxed);
      if (now != seen) {
        seen = now;
        moved_at = t;
        grow_wait = kLockstepGrowAfter;
      } else if (t - moved_at >= grow_wait) {
        ++target;
        moved_at = t;
        grow_wait = kLockstepGrowAgain;
      }
    }
  }
}

}  // namespace vgpu::exec
