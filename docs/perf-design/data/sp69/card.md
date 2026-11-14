# Round sp69 on the card (one A10G, g5.xlarge, driver 595.112.07, nvcc 13.0, 4 vCPUs)

Branch head 6c64b703 (`sp69-hotspot-start` with origin/main merged), built on the rig as in `../summary-2026-11-13/README.md`; harness
`tools/bench/hostgpu/run.py`, corpora of `tools/bench/real/pack_corpora.sh`. Median wall seconds; in brackets the launches on the card of all
launches. `exactr0` / `turbor0` are the same build with `VGPU_HOST_GPU_START_REPEATS=0` (the rule before the round), `turbol0` with
`VGPU_HOST_GPU_START_LAUNCHES=0` (the account alone, the rule of Round sp66), interleaved. "outputs": every form's output against the simulator's.

## 1. hotspot, sc_gpu and the four Samples (10 runs)

| program | sim s (card/all) | exact s (card/all) | exactr0 s (card/all) | turbo s (card/all) | turbor0 s (card/all) | turbol0 s (card/all) | outputs |
|---|---:|---:|---:|---:|---:|---:|---|
| rodinia/sc_gpu | 0.97 | 0.98 (0/1432) | 0.98 (0/1432) | 0.98 (0/1432) | 0.98 (0/1432) | 0.79 (1029/1432) | all identical |
| rodinia/hotspot | 1.05 | 1.06 (0/1000) | 1.06 (0/1000) | 0.57 (712/1000) | 1.06 (0/1000) | 0.51 (760/1000) | all identical |
| samples/MC_EstimatePiInlineP | 3.06 | 3.07 (0/2) | 3.06 (0/2) | 3.07 (0/2) | 3.07 (0/2) | 3.30 (1/2) | all identical |
| samples/dct8x8 | 0.94 | 0.95 (0/118) | 0.95 (0/118) | 0.95 (0/118) | 0.95 (0/118) | 1.12 (20/118) | all identical |
| samples/threadFenceReduction | 0.35 | 0.35 (0/101) | 0.35 (0/101) | 0.35 (0/101) | 0.35 (0/101) | 0.52 (21/101) | all identical |
| samples/LargeKernelParameter | 0.40 | 0.42 (0/2002) | 0.42 (0/2002) | 0.42 (0/2002) | 0.42 (0/2002) | 0.42 (0/2002) | all identical |

hotspot in turbo mode: 1.85x over the simulator (1.00x before the round, 2.05x under the account alone on this rig: the start is bought after
288 simulated launches instead of 240). The four Samples send nothing in any form of the new rule and run at the simulator's time
(LargeKernelParameter's 0.42 s against 0.40 s is the mode's per-launch work, as in the integration). sc_gpu is not helped: its launch saves
under the 0.5 ms floor (0.6 ms in the simulator), and it stays at the simulator's time where the account alone made it 1.24x.

## 2. The other programs with 200 launches or more that did not start the driver, and asyncAllocation (5 runs)

| program | sim s (card/all) | exact s (card/all) | exactr0 s (card/all) | turbo s (card/all) | turbor0 s (card/all) | outputs |
|---|---:|---:|---:|---:|---:|---|
| polybench/3DCONV | 0.73 | 0.74 (0/254) | 0.74 (0/254) | 0.74 (0/254) | 0.73 (0/254) | all identical |
| polybench/ADI | 0.35 | 0.36 (0/2049) | 0.36 (0/2049) | 0.36 (0/2049) | 0.36 (0/2049) | all identical |
| polybench/JACOBI1D | 2.26 | 2.29 (0/20000) | 2.29 (0/20000) | 2.30 (0/20000) | 2.32 (0/20000) | all identical |
| rodinia/needle | 0.32 | 0.33 (0/255) | 0.33 (0/255) | 0.33 (0/255) | 0.33 (0/255) | all identical |
| hecbench/asyncAllocation-cuda | 3.44 | 3.44 (0/800) | 3.43 (0/800) | 1.09 (724/800) | 1.08 (724/800) | all identical |
| samples/cudaGraphsPerfScaling | 0.01 | 0.01 (0/213) | 0.01 (0/213) | 0.01 (0/213) | 0.01 (0/213) | exact:different, exactr0:different, turbo:different, turbor0:different |

None starts the driver under either rule; asyncAllocation is as before. cudaGraphsPerfScaling prints timings: its simulator runs differ from
each other too (`../summary-2026-11-13/README.md`).

## 3. The keep list of the integration (3 runs; FDTD-2D, LU and black-scholes, over 20 s in the simulator, one)

| program | sim s (card/all) | exact s (card/all) | exactr0 s (card/all) | turbo s (card/all) | turbor0 s (card/all) | outputs |
|---|---:|---:|---:|---:|---:|---|
| polybench/3MM | 2.25 | 1.67 (2/3) | 1.66 (2/3) | 1.66 (2/3) | 1.71 (2/3) | all identical |
| polybench/FDTD-2D | 99.16 | 4.67 (1494/1500) | 4.70 (1494/1500) | 4.65 (1494/1500) | 4.63 (1494/1500) | all identical |
| polybench/JACOBI2D | 0.60 | 0.56 (21/40) | 0.56 (21/40) | 0.56 (21/40) | 0.56 (21/40) | all identical |
| polybench/LU | 28.93 | 10.51 (1394/4096) | 10.52 (1398/4096) | 1.32 (4083/4096) | 1.36 (4083/4096) | all identical |
| polybench/SYR2K | 9.83 | 2.32 (1/1) | 2.32 (1/1) | 2.31 (1/1) | 2.32 (1/1) | all identical |
| polybench/SYRK | 5.37 | 1.29 (1/1) | 1.29 (1/1) | 1.29 (1/1) | 1.29 (1/1) | all identical |
| rodinia/gaussian | 3.82 | 0.91 (494/1022) | 0.91 (494/1022) | 0.52 (987/1022) | 0.52 (987/1022) | all identical |
| rodinia/lud | 3.34 | 2.92 (47/190) | 3.00 (47/190) | 2.72 (170/190) | 2.74 (170/190) | all identical |
| rodinia/particlefilter_float | 2.94 | 2.12 (7/36) | 2.14 (5/36) | 0.83 (28/36) | 0.85 (28/36) | exact:different, exactr0:different, turbo:different, turbor0:different |
| hecbench/atomicAggregate-cuda | 3.31 | 3.30 (0/6) | 3.30 (0/6) | 1.37 (4/6) | 1.42 (4/6) | all identical |
| hecbench/atomicPerf-cuda | 0.98 | 0.99 (0/18) | 0.99 (0/18) | 0.72 (12/18) | 0.71 (12/18) | all identical |
| hecbench/black-scholes-cuda | 21.52 | 21.53 (0/1) | 21.59 (0/1) | 5.99 (1/1) | 6.04 (1/1) | all identical |

No program is more than 3 % slower than its `r0` form (the largest: atomicPerf turbo +1.0 %, 3MM exact +0.9 %). particlefilter_float seeds
its random numbers from the clock: its simulator runs differ from each other too. Every other exact and turbo output of the three sets is
identical to the simulator's.

## 4. Card CI (`tools/card-ci/run-card-ci.sh`)

PASS: e2e_host_gpu and e2e_host_gpu_stub (steps 1 to 37) pass, neither skipped; unit tests pass; `host_gpu_verify: in_set_identical=31
in_set_different=0`; turbo verify `words_outside=0 kernels_outside=0`; division exhaustive (stride 257) `differ=0` for all twelve dividends.
