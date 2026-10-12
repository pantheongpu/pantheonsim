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

`VGPU_RACE` checks the rule a CUDA block promises: two warps may touch the same
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

It is off by default because it costs a shadow word per shared word, and on
because you are looking for something. It found the flash-attention race in
llama.cpp in one run, naming the kernel, the PTX line and both warps; that same
bug took a day to corner by bisection.

`VGPU_STRICT` is off by default for a reason worth knowing. Both of those
checks find real bugs, and both fire on code that is perfectly correct: ptxas
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

These diagnostics are a core product feature: VirtualGPU aims to grow into an
ASan/TSan-analogue for GPU software (races, invalid memory, OOM injection),
which real GPUs cannot give you cheaply.
