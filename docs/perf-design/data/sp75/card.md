# Round sp75 on the card (one A10G, g5.xlarge, driver 595.112.07, nvcc 13.0, 4 vCPUs)

2026-11-15, 13:14 to 15:51 UTC. Branch head f1cf3e73, shipped with `git archive`; `card-run.sh` stage by stage, `build` in the card CI's build
(`RelWithDebInfo`, no LTO), the other stages with `SP75_BUILD` set to the summary run's build (`Release`, LTO, targets `vgpucudart` and
`vgpucuda`: `../summary-2026-11-13/README.md`), so the times compare with that run's. Median wall seconds; in brackets the launches on the card
of all launches; `exactb0` / `turbob0` are the same build with `VGPU_HOST_GPU_SAMPLE_BLOCK=0`, interleaved.

## 1. The programs the rule is for (`changed`; one run a form)

| program | sim s | exact s | exactb0 s | turbo s | turbob0 s | turbo output |
|---|---:|---:|---:|---:|---:|---|
| samples/tf32TensorCoreGemm | 208.97 | 209.42 (0/1) | 209.54 (0/1) | 2.97 (1/1) | 209.35 (0/1) | TFLOPS line; verify: identical |
| samples/bf16TensorCoreGemm | 157.15 | 157.74 (0/1) | 156.91 (0/1) | 3.80 (1/1) | 156.68 (0/1) | TFLOPS line; verify: identical |
| samples/cudaTensorCoreGemm | 16.05 | 16.03 (0/1) | 16.00 (0/1) | 1.30 (1/1) | 16.09 (0/1) | TFLOPS line; verify: identical |
| samples/immaTensorCoreGemm | 3.52 | 3.53 (0/1) | 3.53 (0/1) | 1.19 (1/1) | 3.53 (0/1) | TFLOPS line; verify: identical |
| hecbench/aidw-cuda | over 600 | - | - | 193.23 (3/4) | 359.63 (2/4) | not compared (the simulator's run was killed at 600 s) |

Turbo mode is 70x, 41x, 12x and 3.0x the simulator on the four GEMMs (1.00x with the switch off and in exact mode, whose set does not hold
their kernels), and aidw is 1.86x its time with the switch off. What is left of a GEMM is its host code, the sample, the
driver start and the push of its memory (section 2).

**Outputs.** Every exact, `exactb0` and `turbob0` output is identical to the simulator's. The harness calls the four turbo outputs
`different`, by one number each: the programs' `TFLOPS:` line, a timing the filter does not drop (2.54 against 0.01 for bf16). The programs
print nothing of their result (the Samples' own check is compiled out, `CPU_DEBUG 0`), so what the card computed is the verify runs'
(`VGPU_HOST_GPU_VERIFY=all`: the launch on both sides, every word the kernel wrote compared, tolerance 1e-8 + 1e-5 |sim|).
Words compared: cudaTensorCoreGemm 50,331,648, immaTensorCoreGemm 41,943,040, bf16 and tf32 201,326,592 each; different: 0 in all four.
The card's four tensor GEMMs are bit for bit the simulator's (`nvidia/a10g`'s tensor-core model): no word differs, so none is outside the
tolerance. aidw's outputs were not compared: its simulator run was killed at 600 s, and turbo against `turbob0` was not set side by side.

## 2. What the mode said (`decisions`; one logged turbo run each)

| launch | block limit | estimate (at least) | needed with the start | on the card |
|---|---:|---:|---:|---|
| cudaTensorCoreGemm, 80 blocks | 53.6 ms | 0.986 s | 0.410 s | kernel 3.0 ms, 55 ms in all, 192 MiB pushed |
| immaTensorCoreGemm, 80 blocks | 52.4 ms | 0.958 s | 0.401 s | kernel 2.0 ms, 57 ms, 160 MiB |
| tf32TensorCoreGemm, 160 blocks | 37.1 ms | 1.357 s | 0.567 s | kernel 182 ms, 342 ms, 768 MiB |
| bf16TensorCoreGemm, 160 blocks | 37.1 ms | 1.400 s | 0.573 s | kernel 22 ms, 180 ms, 768 MiB |
| aidw `AIDW_Kernel`, 400 blocks | 15 ms | 1.269 s | 0.409 s | kernel 17 ms, 18 ms, 2.7 MiB |

Each is `sampled: ... (the last block stopped at the limit: at least) ... sent`, then `starts the driver: sampled`; the samples
cost 0.091 to 0.110 s for the GEMMs (the fork 5 to 19 ms) and 0.029 s for aidw. Every estimate is 2.4 to 3.1 times what the card path needs:
no GEMM is near the bar on the rig, immaTensorCoreGemm included. aidw's `AIDW_Kernel_Tiled`: `host_gpu_nosample: ... blocks=400 ...
reason=the driver has been started`, simulated once (149 s of the 193 s); the two launches after it are sent.

**#662.** `host_gpu_overhead` of the logged runs (the log's own printing is in the numbers):

```
silu-cuda: host_gpu_overhead: kept_launches=30 decide_s=0.001496 scan_s=0.001088 after_s=0.000044 kernels=5
s3d-cuda:  host_gpu_overhead: kept_launches=54 decide_s=0.078120 scan_s=0.077414 after_s=0.000124 kernels=54
```

silu: `silu_forward_kernel2 starts the driver: size rule (sim 7.443 ms, card path with the start 250.457 ms, 0.276 s of eligible launches
simulated so far, 5 of this kernel)`, then 20 of 50 launches on the card: 0.470 s in the simulator, 0.541 s in turbo mode, 0.540 s with the
switch off (`rest`). s3d: 0/54 on the card, no sample, 0.327 s in the simulator, 0.357 s in turbo mode and 0.356 s with the switch off
(`small`, 10 runs). `scan_s` is all of `decide_s`, 1.4 ms a kernel, but so it was on server3 (0.070 s with the log on, where the program
loses 2 ms): the line does not part the scan from the log's printing, so the rig's 30 ms is not placed yet. It is not this round's rule.

## 3. The keep list (`keep`; 3 runs, one for a program over 20 s in the simulator)

| program | sim s | exact s | exactb0 s | turbo s | turbob0 s | exact / b0 | turbo / b0 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2MM | 171.40 | 99.83 (2/2) | 98.77 (2/2) | 98.11 (2/2) | 97.15 (2/2) | +1.1 % | +1.0 % |
| 3MM | 2.19 | 1.64 (2/3) | 1.61 (2/3) | 1.60 (2/3) | 1.64 (2/3) | +2.4 % | -2.0 % |
| CORR | 58.40 | 37.31 (1/4) | 37.33 (1/4) | 37.69 (1/4) | 36.66 (1/4) | -0.0 % | +2.8 % |
| COVAR | 58.83 | 35.39 (1/3) | 35.26 (1/3) | 35.54 (1/3) | 34.65 (1/3) | +0.4 % | +2.6 % |
| FDTD-2D | 93.32 | 4.66 (1494/1500) | 4.66 (1494/1500) | 4.63 (1494/1500) | 4.65 (1494/1500) | +0.1 % | -0.2 % |
| JACOBI2D | 0.57 | 0.55 (21/40) | 0.59 (21/40) | 0.54 (21/40) | 0.55 (21/40) | -7.7 % | -0.4 % |
| LU | 28.07 | 10.59 (1371/4096) | 10.60 (1380/4096) | 1.32 (4083/4096) | 1.37 (4083/4096) | -0.1 % | -3.5 % |
| SYR2K | 9.54 | 2.32 (1/1) | 2.31 (1/1) | 2.32 (1/1) | 2.32 (1/1) | +0.2 % | -0.0 % |
| SYRK | 5.11 | 1.29 (1/1) | 1.29 (1/1) | 1.34 (1/1) | 1.34 (1/1) | -0.1 % | -0.2 % |
| gaussian | 3.71 | 0.92 (493/1022) | 0.92 (493/1022) | 0.53 (985/1022) | 0.53 (985/1022) | +0.2 % | +0.9 % |
| lud | 3.21 | 2.99 (47/190) | 2.87 (47/190) | 2.74 (170/190) | 2.66 (170/190) | +4.3 % | +3.1 % |
| particlefilter_float | 2.65 | 2.07 (6/36) | 2.06 (7/36) | 0.84 (28/36) | 0.79 (28/36) | +0.6 % | +5.9 % |
| atomicAggregate | 3.24 | 3.24 (0/6) | 3.24 (0/6) | 1.35 (4/6) | 1.35 (4/6) | -0.1 % | +0.5 % |
| atomicPerf | 0.97 | 0.98 (0/18) | 0.98 (0/18) | 0.71 (12/18) | 0.71 (12/18) | -0.1 % | +0.3 % |
| black-scholes | 21.52 | 21.38 (0/1) | 21.40 (0/1) | 6.03 (1/1) | 6.03 (1/1) | -0.1 % | +0.1 % |
| shmembench | 4.23 | 4.22 (0/1) | 4.22 (0/1) | 0.29 (1/1) | 0.29 (1/1) | -0.1 % | +0.8 % |
| binomial | over 90 | over 90 | over 90 | 1.41 (1000/1000) | 1.35 (1000/1000) | - | +4.5 % |

Every output is identical to the simulator's but particlefilter_float's, which seeds its random numbers from the clock (its simulator runs
differ from each other; its exact launches, 6 and 7, with them). LU's exact launches differ between any two runs (`census.md`). binomial ran
apart with its runs killed at 90 s (simulator and exact mode: over 600 s in the summary run), so its outputs have no reference.

Over 3 % in this table: lud (exact +4.3 %, turbo +3.1 %), particlefilter_float (turbo +5.9 %), binomial (turbo +4.5 %, one run). None takes
a different decision or a sample more with the switch on (lud: no sample in any form, the same launches, 0.274 s against 0.273 s of
simulator time in turbo mode). Run again after the stages: lud, 10 runs, exact +1.5 %, turbo -1.9 %; binomial, 7 runs, 1.356 s against
1.357 s. **particlefilter_float is not settled**: 10 runs, turbo 0.96 s against 0.89 s; 30 runs, median 0.947 s against 0.818 s, mean 0.895 s
against 0.849 s. Its runs fall in two groups (0.67 to 0.89 s and 0.93 to 1.03 s: `find_index_kernel`'s first launch takes 34 or 178 ms
by the seed), 12 of 30 turbo runs and 19 of 30 `turbob0` runs in the fast one. The logged runs do the same in both forms (one sample, inside
the blocks, `block_limit_s=0`, 8.0 and 8.7 ms; the start bought at the same launch; 28 of 36 on the card), so no cause in the rule was
found, but four measurements lean the same way.

## 4. The small grids that are not sampled (`small`; 10 runs)

| program | sim s | turbo s | turbob0 s | turbo / b0 |
|---|---:|---:|---:|---:|
| lud | 3.35 | 2.74 (170/190) | 2.72 (170/190) | +0.7 % |
| lavaMD | 1.52 | 1.52 (0/1) | 1.52 (0/1) | +0.0 % |
| radixsort2 | 2.95 | 0.59 (455/600) | 0.58 (455/600) | +0.8 % |
| s3d | 0.33 | 0.36 (0/54) | 0.36 (0/54) | +0.3 % |
| mergeSort | 6.88 | 2.82 (34/49) | 2.84 (34/49) | -0.8 % |
| segmentationTreeThrust | 13.94 | 5.66 (282/514) | 5.65 (282/514) | +0.1 % |

No sample but segmentationTreeThrust's one (taken in both forms); every output identical; the same launches in both forms.

## 5. The rest of the corpus (`rest`; 115 programs, one run a form: `sim`, `turbo`, `turbob0`)

The sum of the turbo times is 297.5 s, 297.7 s with the switch off. With one run a form, 10 programs are over 3 % slower and 6 over 3 %
faster than their `turbob0` run; 8 of the 10 run under 0.04 s (3 ms at most between the forms). The others:

* rodinia/particlefilter_float: 0.97 s (28/36) against 0.92 s (28/36), +5.6 %; the clock-seeded program of section 3.
* samples/radixSortThrust (30.62 s in the simulator): 1.11 s (592/600) against 1.06 s (592/600), +5.0 %. It is the one program whose
  sampling the rule changes without sending: one sample more, 1.7 ms. Again, 10 runs: 1.103 s against 1.089 s, +1.4 %.

Faster by as much: SYRK, aop, interval, reduction (-3.7 to -6.4 %). And:

* **Launches on the card**: four programs differ between the two forms, by launches at the edge of the size rule, as between any two runs:
  hotspot 706 / 709 of 1000, asyncAllocation 724 / 729 of 800, jacobiCudaGraphs 2910 / 2911, transpose 753 / 754.
* **Outputs**: no output differs between turbo and `turbob0`. Against the simulator four are `different` in both forms, the known ones
  (particlefilter_float, simplePrintf, simpleIPC's segment name, cudaGraphsPerfScaling's timings); four have no reference
  (aligned-types, babelstream, base64e, binomial: apart, killed at 90 s in the simulator; turbo 8.10, 12.92, 8.95 and 1.36 s against 8.11,
  12.85, 8.98 and 1.35 s).
* **Exit codes**: MC_EstimatePiP exits 1 in every form, the simulator's too. (`Could not create pseudo-random number generator`: the rig's build had no cuRAND shim.)
* **Left over**: 0 zombies, 0 processes of the corpus, no sample's copy left at exit in any record.

## 6. Card CI (`build`: `tools/card-ci/run-card-ci.sh`)

PASS: unit tests; e2e_host_gpu (54 s) and e2e_host_gpu_stub (184 s), neither skipped: the stand-in test's steps 1 to 40, steps 38 to 40
for the first time with a real nvcc; `host_gpu_verify: in_set_identical=31 in_set_different=0`; turbo verify `words_outside=0
kernels_outside=0`; division exhaustive (stride 257) `differ=0` for all twelve dividends; the harness self-test.

## 7. What the script needed, and what was not run

* `card-run.sh` ran as written but for time: the four programs over 300 s in the simulator (binomial in `keep`; aligned-types, babelstream,
  base64e, binomial in `rest`) run apart with a 90 s timeout (`SP75_SLOW_TIMEOUT`), and the verify run, which held cudaTensorCoreGemm only,
  has a second line for the other three GEMMs. No stage was cut.
* By hand after the stages, same harness: the reruns of sections 3 and 5 (the simulator's run killed after 2 to 20 s where it is not needed).
* Not run: aidw's outputs against a reference; another card or host thread count.
