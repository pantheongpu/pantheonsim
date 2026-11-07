// The shape of a lockstep launch (vgpu/exec/lockstep.hpp): how many host threads take a grid whose blocks may wait on one another, and
// how many blocks each holds at the start.
//
// A grid of fewer than kLockstepOneThread blocks runs on ONE thread with all its blocks resident, in a fixed order: a kernel that takes
// locks (RXMesh's patch locks) gives a different answer on every run when its blocks start in whatever order the host threads pick. A
// kernel whose waits are only loads (CUB's radix-sort look-back, a flag chain) gives the same answer in any order, so its small grid may
// use the host threads too (CUDA sample-sized radix sorts launch a 9-block onesweep kernel hundreds of times, and one thread ran 69% of
// the instructions of hecbench radixsort2 on its own at any VGPU_THREADS). That launch holds every block from the start, as the one
// thread did, because a block may wait on one that is last in line.
#include "vgpu/exec/lockstep.hpp"
#include "vtest.hpp"

using vgpu::exec::kLockstepOneThread;
using vgpu::exec::lockstep_shape;

VTEST(a_small_grid_stays_on_one_thread_by_default) {
  const auto s = lockstep_shape(9, 9, 8);
  VCHECK_EQ(s.workers, 1u);
  VCHECK_EQ(s.window, 9u);
  VCHECK_EQ(s.initial, 9u);
}

VTEST(a_small_grid_of_waits_on_loads_spreads_and_holds_every_block) {
  const auto s = lockstep_shape(9, 9, 8, false, /*small_grid_spreads=*/true);
  VCHECK_EQ(s.workers, 4u);   // two blocks to a thread at least
  VCHECK_EQ(s.window, 3u);    // 4 threads hold 12 places for 9 blocks
  VCHECK_EQ(s.initial, s.window);   // all resident from the start: the last block may be the one the others wait for
  const auto two = lockstep_shape(9, 9, 2, false, true);
  VCHECK_EQ(two.workers, 2u);
  VCHECK_EQ(two.window, 5u);
  VCHECK_EQ(two.initial, 5u);
}

VTEST(the_threads_the_launch_has_cap_a_spreading_grid) {
  VCHECK_EQ(lockstep_shape(15, 15, 1, false, true).workers, 1u);
  VCHECK_EQ(lockstep_shape(15, 15, 3, false, true).workers, 3u);
  VCHECK_EQ(lockstep_shape(15, 15, 64, false, true).workers, 7u);   // 15 / 2
}

VTEST(a_grid_too_small_for_two_blocks_a_thread_has_one_thread) {
  VCHECK_EQ(lockstep_shape(3, 3, 8, false, true).workers, 1u);
  VCHECK_EQ(lockstep_shape(1, 1, 8, false, true).workers, 1u);
}

VTEST(a_grid_of_the_one_thread_size_or_more_is_unchanged_by_the_flag) {
  for (const uint64_t units : {kLockstepOneThread, kLockstepOneThread + 1, uint64_t{40}, uint64_t{1000}}) {
    const auto a = lockstep_shape(units, 200, 8), b = lockstep_shape(units, 200, 8, false, true);
    VCHECK_EQ(a.workers, b.workers);
    VCHECK_EQ(a.window, b.window);
    VCHECK_EQ(a.initial, 1u);   // (started with one block each, as before)
    VCHECK_EQ(b.initial, 1u);
  }
}

VTEST(fewer_resident_places_than_blocks_limit_the_window) {
  // A device that holds 6 of the 9 blocks at once: 4 threads hold 2 each, and the rest start as those finish.
  const auto s = lockstep_shape(9, 6, 8, false, true);
  VCHECK_EQ(s.workers, 4u);
  VCHECK_EQ(s.window, 2u);
  VCHECK_EQ(s.initial, 2u);
}

VTEST_MAIN
