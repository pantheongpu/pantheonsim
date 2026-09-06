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
