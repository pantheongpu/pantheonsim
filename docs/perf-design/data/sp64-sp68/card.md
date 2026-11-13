# Integration of sp64 to sp68 on the card (one A10G, g5.xlarge, driver 595.112.07, nvcc 13.0, 4 vCPUs)

Branch head of the run: 864b88c4 for the card CI's first pass, the four programs (`card/four.md`), the keep list (`card/keep.md`) and the rounds'
headlines (`card/sp65.md` to `card/sp68.md`, `card/sp64.txt`, `card/sp68-extra.txt`); 332606d8 (plus the harness forms) for the second card CI pass
(`card/card-ci2.txt`), `card/four2.md` and the reruns (`card/re10.md`, `card/re3.md`). Harness `tools/bench/hostgpu/run.py`, corpora of
`tools/bench/real/pack_corpora.sh`; `exactl0`/`turbol0` are the same build with `VGPU_HOST_GPU_START_LAUNCHES=0` (the account alone), interleaved.

## Card CI (`tools/card-ci/run-card-ci.sh`)

First pass: e2e_host_gpu passed (card steps 1 to 12), e2e_host_gpu_stub failed at step 27 only: with the mode off the zero grid gives
`cudaErrorInvalidValue`, as NVIDIA's own runtime 13.0 does on this card (the step expected `cudaErrorInvalidConfiguration`, a CUDA 12 answer; fixed
in 864b88c4). Second pass on 332606d8: **PASS** (both e2e tests, steps 1 to 36 of the stand-in test, unit tests, verify 31 identical / 0 different,
turbo verify words_outside=0, division exhaustive differ=0).

## The four short programs (10 runs, medians, s)

| program | sim | exact | exactl0 (old rule) | turbo | turbol0 (old rule) |
|---|---:|---:|---:|---:|---:|
| MC_EstimatePiInlineP | 3.28 | 3.28 | 3.28 | 3.28 | 3.52 |
| dct8x8 | 1.08 | 1.08 | 1.21 | 1.08 | 1.19 |
| threadFenceReduction | 0.38 | 0.38 | 0.38 | 0.38 | 0.53 |
| LargeKernelParameter | 0.49 | 0.51 | 0.71 | 0.51 | 0.74 |

Under the new rule none of the four starts the driver (0 card launches). LargeKernelParameter stays about 3 % over the plain simulator with
no launch sent (24 interleaved runs: sim 0.492 s, exact 0.507 s, the same with `VGPU_HOST_GPU_SAMPLE=0` or `VGPU_HOST_GPU_NO_FIRST_LAUNCH=1`):
`perf` puts 1.0 % in `kernel_eligible` (the argument scan, 32 KiB a launch, 2,002 launches, after 332606d8's inline check) and 0.3 % in the rest
of the mode's code; the remainder was not attributed. It was 0.52 s (6 %) before 332606d8.

## The keep list (3 runs, programs over 20 s one; medians, s)

| program | sim | exact | exactl0 | turbo | turbol0 |
|---|---:|---:|---:|---:|---:|
| 2MM (rerun, 3 runs) | 150.55 | 97.63 | 97.29 | | |
| 3MM | 2.47 | 1.73 | 1.79 | 1.76 | 1.75 |
| CORR (rerun, 3 runs) | 58.29 | 37.20 | 36.84 | 37.65 | 38.62 |
| COVAR | 58.06 | 36.92 | 36.66 | 37.12 | 36.38 |
| FDTD-2D | 103.87 | 4.73 | 4.63 | 4.76 | 4.75 |
| LU | 30.62 | 10.62 | 10.67 | 1.28 | 1.28 |
| SYR2K | 9.69 | 2.32 | 2.32 | 2.32 | 2.32 |
| SYRK | 5.12 | 1.29 | 1.29 | 1.29 | 1.29 |
| gaussian (turbo: rerun, 10 runs) | 4.09 | 0.92 | 0.91 | 0.52 | 0.52 |
| lud (turbo: rerun, 10 runs) | 3.60 | 2.97 | 2.94 | 2.68 | 2.76 |
| particlefilter_float | 3.16 | 2.24 | 2.27 | 1.04 | 1.05 |
| asyncAllocation | 3.95 | 3.94 | 3.95 | 1.01 | 0.99 |
| atomicAggregate | 3.58 | 3.58 | 3.58 | 1.46 | 1.46 |
| atomicPerf | 1.17 | 1.17 | 1.17 | 0.80 | 0.80 |
| black-scholes | 25.30 | 25.32 | 25.30 | 6.06 | 6.02 |

No program is more than 3 % slower than with the old rule: the first pass's outliers were reruns' noise (2MM exact 133.4 s in one run against
97.6 s in three, a program whose wall is mostly its own CPU reference; CORR exact 38.1 against 37.2; gaussian turbo 0.58 against 0.52 in ten).
The largest left are FDTD-2D exact (+2.2 %) and COVAR turbo (+2.0 %), one run each. Every exact-mode output is identical to the simulator's except
particlefilter_float's, whose simulator runs do not repeat themselves (as in Round sp66).

## The rounds' headlines

| round | what | measured here | the round's |
|---|---|---|---|
| sp64 | GEMVER, JACOBI1D, `VGPU_JIT=1` against 0 (3 runs) | 0.65 / 0.65 s, 2.49 / 2.45 s: no gain on 4 vCPUs | measured at 1 and 16 threads on server3; not a card result |
| sp65 | LU turbo against `turbon` (`STREAK=0`) | 1.29 s against 10.73 s | 11.6 s to 1.5 s |
| sp65 | gaussian turbo against `turbon` | 0.52 s against 0.93 s | 0.99 s to 0.53 s |
| sp66 | turbo against `turbow0` (`WIDER=0`): radixSortThrust, segmentationTreeThrust, asyncAllocation, atomicPerf | 1.02 / 34.22, 6.52 / 15.76, 1.01 / 3.95, 0.81 / 1.40 s | 34.1 to 1.84, 15.5 to 8.7, 6.6x, 1.43x |
| sp66 | segmentationTreeThrust exact against `exactw0` | 12.08 s against 15.76 s | 11.9 s |
| sp67 | 3DCONV under `VGPU_HOST_GPU_VERIFY=1` (exact) | 8.53 s | 8.5 s (46.5 s before) |
| sp68 | GRAMSCHM, exact with `VGPU_HOST_GPU_SQRT=1 VGPU_HOST_GPU_DIV=1`, one run | 125.2 s against 302.6 s simulated (turbo 135.6 s) | device part 44.6 s to 3.1 s |
| sp68 | GRAMSCHM the same with `VGPU_HOST_GPU_VERIFY=1` | `in_set_identical=6144 in_set_different=0`, 236.0 s | identical |
| sp68 | hellinger (HeCBench), exact, `DIV=1` with and without `SQRT=1` (3 runs) | 6.75 s against 45.4 s, output the simulator's | 43.0 to 6.76 s |
