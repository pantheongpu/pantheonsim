# Diagnostics: counters, races, strict mode and errors

PantheonSim aims to be an ASan/TSan-analogue for GPU software. This page covers the checks and counters you switch on with `--counters`, `--race` and `--strict` (or `VGPU_COUNTERS`, `VGPU_RACE`, `VGPU_STRICT`), and what an error looks like.

`VGPU_COUNTERS` reports what a profiler reports, except that every number is
counted rather than sampled. Instructions and thread-instructions (their ratio
is the average number of lanes doing useful work), divergent branches, memory
by space in operations *and* bytes, atomics with the bytes they moved, and
barriers -- and two that are worth the simulator on their own:

- **Sectors and coalescing.** Memory moves in 32-byte sectors. Every lane's
  address is in hand at the moment of the access, so the sectors a warp touches
  are counted exactly: four for a fully coalesced 32-lane load of 4-byte
  values, up to thirty-two when every lane lands in its own sector. The
  percentage is how much of the traffic was asked for rather than rounded up to.
- **Shared-memory bank conflicts.** Thirty-two banks of four bytes; lanes
  reaching different words in one bank serialize, lanes reaching the same word
  are broadcast and free. The count is the extra passes serialization forces --
  zero for a conflict-free access, thirty-one for a 32-way conflict.

A device derives both by sampling, so its answer moves between runs. These do
not move.

The instruction mix is reported alongside them, in the categories a profiler
uses: fp16/fp32/fp64 by operand width, integer, conversion, control, memory,
tensor and everything else. The classes partition the instructions, so they sum
to the thread-instruction total -- which is checked by a test, because a
classifier that silently drops a case looks exactly like one that works.
Tensor-core work is reported twice: per lane with the rest of the mix, and per
warp as `tensor issues`, since an mma is one instruction the whole warp
executes together and a per-lane figure would say thirty-two for something that
happened once.

An atomic is a read and a write, so its traffic is counted once as
`atomic_bytes` rather than twice in the load and store totals.

There is no timing model and no cache model here, so there are no cycles, no
stall reasons and no hit rates. Those are the numbers a profiler is mostly
made of, and inventing them would be worse than not having them.

`VGPU_RACE` (and `--race`) checks the rule a CUDA block promises: two warps may touch the same
shared word without a `bar.sync` between them only if both are reading.
Anything else is a race, and which warp wins is not something the program
decided. Hardware usually hides this -- warps advance together and the window
is small -- which is exactly why it is worth checking somewhere that does not.

One exception, and it is not a softening of the rule: a store that leaves the
bytes exactly as it found them cannot be observed by anyone, because no reader
and no other writer can tell whether it happened before or after. Real kernels
do this deliberately -- llama.cpp's `mul_mat_q` clamps out-of-range tile rows
with `i = min(i, i_max)`, so several warps recompute the same pointer and store
the same value to the same word. The write is still recorded, so a later store
of a *different* value is caught against it; only the report is suppressed.
`VGPU_RACE=2` reports these too.

Atomics (`atom`, `red`) from different warps on one word between two barriers
are not a race: the hardware orders them. An atomic beside another warp's plain
load or store of that word is still reported.

The check stops the launch at the first race. `VGPU_RACE_CONTINUE=1` prints each
one (once per instruction and kind, up to 50) and lets the launch finish, so a
false positive early on does not hide the later kernels.

It is off by default because it costs a shadow word per shared word, and on
because you are looking for something. It found the flash-attention race in
llama.cpp in one run, naming the kernel, the PTX line and both warps; that same
bug took a day to corner by bisection.

It runs in both engines. The SASS engine, which is what an `nvcc` binary executes, used to ignore `VGPU_RACE`: only the PTX engine
(`VGPU_SASS=0`) had the detector, so a racy kernel in a compiled program ran clean. The SASS engine now applies the same rule (a
completed `bar.sync` or mbarrier phase ends the epoch, atomics are ordered against each other, a store of the bytes already there is
not reported) and runs without its decoded-instruction tier and shared-memory fast paths while the detector is on. It names the
kernel, the SASS instruction and both warps, with the same three kinds, `write-read`, `read-write` and `write-write`.

## What compute-sanitizer reports, and what this reports

`e2e_sanitizer_kernels` runs small kernels with one known defect each, on the SASS and the PTX engine, for an sm_86 and an sm_75
profile, and checks the error code the runtime returns afterwards and the diagnostic:

| compute-sanitizer | the kernel | here |
| --- | --- | --- |
| memcheck: invalid `__global__` read/write | one element past a buffer, one byte past the requested size, an 8-byte read that straddles the end, a freed buffer, null, an address nothing maps | `[out-of-bounds]`, `[use-after-free]`, `[invalid-pointer]`, naming the direction, the address and the distance from the allocation; 700 (`cudaErrorIllegalAddress`) |
| memcheck: misaligned `__global__` / `__shared__` access | a 4-byte load at +1, an 8-byte store at +4, an atomic at +2, a shared load at +2 | `[misaligned-access]`; 716 (`cudaErrorMisalignedAddress`) |
| memcheck: invalid `__shared__` read/write | far past the array, static and dynamic | `[out-of-bounds]` on shared memory; 700 |
| racecheck: hazard between warps | write then read, write then write, the reduction with a missing `__syncthreads()` | `[data-race]` with `VGPU_RACE=1`; 719, which a card never returns |
| (none: must be quiet) | a barrier between the accesses, atomics from every warp, every warp reading one word, in-bounds accesses | no error, the right value |

What differs from compute-sanitizer, and is not a defect of either:

- A shared-memory overrun of one word on sm_80 and later, in the SASS engine, runs. The driver keeps 1 KiB of shared memory behind
  each block's own and the engine models it, so the word is in memory that exists, as on a card without the sanitizer. The PTX engine
  has no such region and reports it; on sm_75 both do.
- initcheck (a read of device memory nothing wrote) is not done: the simulator hands out zeroed memory and says nothing.
  `VGPU_STRICT` catches the register version of it. synccheck (a barrier some threads of the block do not reach) is not reported as
  such; a block that cannot finish is the deadlock error.
- Races are found between warps of a block, on shared memory. Global-memory races are not (`racecheck` does not do them either).

Two defects came out of this comparison and are fixed: the SASS engine ran a misaligned global, shared, local or atomic access that a
card (and the PTX engine) faults on with 716, and ignored `VGPU_RACE` altogether.

## Resource check

`VGPU_LEAK_CHECK=1` makes the process write one line when it exits, with the machine still there:

```
vgpu-leak-check: device_bytes=0 device_allocs=0 pinned_allocs=0 pinned_bytes=0 fds=+0 threads=+0 maps=+6 rss_kb=14812 exe=my_app -> clean
```

`device_bytes` and `device_allocs` are the allocator's own counts, what `cudaMemGetInfo` shows as missing; `pinned_*` are
`cudaMallocHost` and `cudaHostAlloc` blocks; `fds` and `threads` are growth since the machine was made, without the worker pool's
threads. `maps` and `rss_kb` are shown, not judged. The line ends `LEAK` and the names of what grew, and with `strict` the process
exits with status 70. A program that frees everything it allocates must read clean, which makes a library handle that is destroyed
with memory still allocated for it, or a stream whose thread stays, visible.

In CI every `e2e_` test runs its programs with the check and a log of its own (`build/leak-check/<test>.log`), and the step "Leak
sweep" (`tests/lint/check_leak_sweep.py`) fails on a leak that `tests/lint/leak_known.txt` does not name or on a listed entry that no
longer leaks, so the list only shrinks. Growth over many repeats, which a line at exit cannot show, is `e2e_leak_soak`: a workload
through the runtime, the driver and cuBLAS repeated 400 times, and device memory, descriptors, threads, resident memory and
mappings must not have grown. Both checks are shown to bite: `e2e_leak_probe` and `e2e_leak_soak` keep a device allocation, pinned
memory, a descriptor or a thread on purpose, and must name it. The `e2e_ret_condition_predicate` leak was found only by AddressSanitizer
before these existed.

`VGPU_STRICT` is off by default for a reason worth knowing. Its two checks,
integer division by zero and storing a register nothing has written, find real bugs, and both fire on code that is perfectly correct: ptxas
emits arithmetic whose result is dead, ggml divides by a stride its
configuration does not use, and CUB stores an undefined register into the
unused part of a shared tile. A check that also rejects working programs
cannot be the default, so it is a mode you turn on when you are hunting.

Errors return documented `CUresult` codes **and** print a rich diagnostic:

```
[vgpu] cuLaunchKernel: VirtualGPU error [out-of-bounds]: device memory read at
0x200000003140 is 0 bytes past the end of the 64-byte allocation at 0x200000003100
  in kernel 'vecAdd', PTX line 30
  lane 16
  instruction: ld.global.f32 %f1,[%rd8]
  GPU profile: nvidia/h100
```

These diagnostics are a core product feature: VirtualGPU is built to be an
ASan/TSan-analogue for GPU software (races, invalid memory, injected errors and
hangs with `vgpu fault`), which real GPUs cannot give you cheaply.
