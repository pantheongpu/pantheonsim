# Cooperative launch and the grid barrier

`cooperative_groups::this_grid().sync()` works on VirtualGPU:

```cpp
#include <cooperative_groups.h>
namespace cg = cooperative_groups;

__global__ void k(float* x, float* y, int n) {
    cg::grid_group grid = cg::this_grid();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += 1.0f;
    grid.sync();                       // every block has finished its store
    if (i < n) y[i] = x[(i + 64) % n]; // ...so reading another block's is safe
}
```

launched with `cudaLaunchCooperativeKernel` (or `cuLaunchCooperativeKernel`),
exactly as on hardware. `cudaDevAttrCooperativeLaunch` reports 1.

## It is not a PTX gap

This was on the "remaining PTX gaps" list, which turned out to be the wrong
description. A grid barrier is not an instruction. Compiled for sm_86 it is:

```
mov.u32 %r7, %envreg1;                 // high half of a driver-supplied address
mov.u32 %r6, %envreg2;                 // low half
bfi.b64 %rd1, %rd7, %rd6, 32, 32;      // reassemble the pointer
setp.ne.s64 %p2, %rd1, 0;
@%p2 bra $L__BB0_4;
trap;                                  // null pointer: not a cooperative launch
$L__BB0_4:
barrier.sync 0;                        // gather this block first
...
atom.add.release.gpu.u32 %r13, [%rd9], %r14;
$L__BB0_6:
ld.acquire.gpu.u32 %r26, [%rd9];
xor.b32 %r27, %r26, %r13;
setp.gt.s32 %p5, %r27, -1;
@%p5 bra $L__BB0_6;                    // spin until the counter's sign flips
```

Every one of those instructions was already implemented. It is a
sense-reversing barrier on an ordinary word of device memory: block (0,0,0)
adds `INT_MIN + 1 - nblocks` and every other block adds 1, so the sum is
exactly `INT_MIN` and the counter's sign flips precisely when the last block
arrives. Each block spins until it sees a value whose sign differs from the one
its own atomic returned.

What the kernel needs from the driver is two things, and neither is an
instruction.

## What actually had to change

**A scheduler that keeps every block resident.** The spin only terminates if
the blocks that have not arrived yet can still run. Ordinary CUDA makes the
opposite promise -- blocks are independent and may run in any order, one at a
time -- and running them that way is both faster and a stricter check of that
promise, so it stays the default. A cooperative launch instead sets every block
up before any of them runs and gives each a bounded turn in round-robin order.

The turn has to be bounded. A warp spinning on the barrier never reaches a
`bar.sync` and never finishes, so without a bound it would hold the grid
forever and never let the block it is waiting for run. It yields after a fixed
number of instructions instead.

The order is fixed and the same every run: a run of this simulator is
reproducible, which rules out handing the blocks to OS threads and letting them
race. A cooperative launch therefore ignores `VGPU_THREADS` and runs on one
worker.

**The barrier's workspace.** `%envreg1` and `%envreg2` are a bank of words the
driver fills in before the launch; a cooperative launch puts the address of a
small zeroed buffer there, and the barrier counter lives a few bytes into it.
`%envreg1` is the *high* half — which is not a guess, it is what the `bfi.b64`
above reassembles. Every other `%envreg` stays zero, because the null check on
that pair is how the generated code detects a `grid.sync()` outside a
cooperative launch.

## What is refused, and why

**A grid too large to be resident.** Every block waits for every other, so a
grid that does not fit does not run slowly — it hangs. Hardware refuses it and
so does this, with `cudaErrorCooperativeLaunchTooLarge` and the arithmetic:

```
[vgpu] cudaLaunchCooperativeKernel: grid of 1048576 blocks exceeds what
       nvidia/a10 can hold resident (16 blocks/SM x 72 SMs = 1152).
```

**An ordinary launch of a grid-sync kernel.** With no cooperative launch there
is no workspace, the `%envreg` pair is zero, and the generated code executes
`trap`. VirtualGPU reports that as `cudaErrorIllegalInstruction` with the kernel
and PTX line, rather than letting it look like a wrong answer.

**`cudaLaunchCooperativeKernelMultiDevice`.** That needs grids on separate
devices waiting on each other. Refused rather than run as if it were
single-device.

## Related

`trap` is now implemented generally, not just for this: it is what a failed
device `assert()` and a compiler-inserted unreachable path lower to, and
reaching one is a fact about the program worth reporting with its line.

<a id="lockstep"></a>

## Blocks that wait on one another

Nothing in CUDA or HIP promises that the blocks of an ordinary launch are resident
together, but every card does keep them so when the grid fits, and programs rely on
it: hand-written grid barriers, a chain of flags run from the last block to the first,
a producer in one block with its consumers in others, RXMesh's patch locks (the
block with the lowest id ends up with every lock, but only if all the blocks are in
flight together; run one after another every block wins them all).

`exec/lockstep.hpp` is the one place that decides how such blocks take turns.

- **Which kernels.** A kernel is searched for a polling loop when it first launches:
  a backward branch whose loop holds an atomic that returns the old value, or a
  volatile, relaxed or acquire load (SASS `STRONG` loads, PTX `.volatile` and
  `.relaxed`, AMD loads with cache bits), whose address and operands are the same on
  every turn of the loop, and no store to global memory and no call. A register counts
  as the same when the loop only reads it, or sets it from constants and such registers
  before it reads it; a counter, or anything built from what a load returned, is not.
  A compare-and-swap retry loop (`old = atomicCAS(p, assumed, f(assumed))`) is
  therefore not a poll. The test errs towards "no": a kernel it misses runs as it did,
  and one it takes for a poller only loses some speed. The search is in
  `kernel_polls` (SASS, `src/sass/exec.cpp`), `ptx_kernel_polls`
  (`src/exec/interpreter.cpp`) and `amd_kernel_polls` (`amd/src/gcn_exec.cpp`); for
  AMD it assumes, and the ISA guides do not say, that a compiler marks the loads of a
  polling loop with cache-control bits.
- **How many are resident.** The runtime works out how many blocks of the kernel the
  device holds at once, as `cudaOccupancyMaxActiveBlocksPerMultiprocessor` does,
  times the multiprocessors (`LaunchConfig::resident_blocks`, `Dispatch::resident_groups`).
- **The turns.** Each host thread holds a window of resident blocks (the capacity
  divided between the threads) and runs each for one turn of `kLockstepSlice` (64)
  instructions per warp, in the order the blocks started. When a block finishes the
  grid's next block takes its place, so blocks beyond the capacity start as earlier
  ones end. A grid of fewer than 16 blocks uses one thread and starts them all, so the
  order is the same on every run. A larger one uses every host thread it has two blocks for; a thread starts
  with one block, as the old path did (the AMD engine takes ranges of consecutive
  groups, as it did), and holds one more at a time only after no block on any thread has
  stored (to shared or global memory) or finished for three milliseconds, then one more
  every millisecond. So a chain of look-back flags (CUB's scan and radix sort) runs as
  before, since real blocks store all the time, and a grid barrier, or a lock every
  block contends for, has all its blocks resident within a few tens of milliseconds. A
  thread whose blocks all stand still sleeps between passes (5 microseconds, doubling to
  half a millisecond), so blocks waiting on the host do not take a core. "Standing still"
  is a hash of the blocks' registers, program counters and shared memory, equal at two
  looks a quarter of a millisecond or more apart; a wait that counts its polls never
  stands still, and just gives way to whatever else there is to run. The warps of one
  block keep the turns they had.
- **Left alone.** Cooperative launches (they already run this way), clusters, a
  non-default `VGPU_SCHEDULER`, device-side launches and the debugger.
- **Switches.** `VGPU_LOCKSTEP=0` is the old behaviour. `VGPU_LOCKSTEP=all` makes every
  launch this way, which finds code that depends on the order blocks start in.
  `VGPU_LOCKSTEP_LOG=1` prints a line for each launch that used it (a file name there appends the lines to that file).

A kernel that waits on something no resident block will ever do (a grid that does not
fit the device, a block that never writes the flag) still spins for ever, as it does
on a card; a kernel waiting for the host spins one core, as it did.
