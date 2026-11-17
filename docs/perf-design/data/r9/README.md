# Round r9: what keeps the launches of mergeSort, histogram, segmentationTreeThrust, silu and s3d in the simulator (card run, A10G; #657, #659, #662)

A card run of 2026-11-17 on one A10G (AWS g5.xlarge, driver 595.112.07, 4 vCPUs), `main` at 37c4cc9d, `Release` build, the sm_86 corpus of
`tools/bench/real/pack_corpora.sh` (nvcc 13.0, `-cudart shared`). Nothing in `include/` or `src/` was changed: this is a measurement, and the
decisions below are for whoever changes a rule next. Tags: **measured** is this run; **inferred** is anything not run.

`card-run.sh` is the script as run (the stages `decisions`, `samples`, `others`, `silu`, `gram`; the first run of the last three failed on the
packed corpus (aidw was listed without its directory) and on a missing `sim` form, and was run again, which is what the tables are).
Tables: `samples.md` (10 runs, sim / exact / turbo / `turbod0` = the rule before sp76), `others.md` (5 runs), `silu-s3d.md` (15 runs),
`gram.md` (1 run), `kept.md` (what stayed in the simulator, by the reason the mode logs).

## #657 aidw and #659 the partial programs (measured)

| program | sim s | turbo s | `turbod0` s (the rule before sp76) | turbo launches on the card |
|---|---:|---:|---:|---|
| samples/mergeSort | 6.87 | 1.89 (3.64x) | 2.80 | 42 / 49 |
| samples/histogram | 11.49 | 2.12 (5.41x) | 2.72 | 62 / 68 |
| samples/segmentationTreeThrust | 13.95 | 4.50 (3.10x) | 5.62 | 284 / 514 |
| hecbench/radixsort2-cuda | 2.92 | 0.58 (5.04x) | 0.58 | 455 / 600 |
| rodinia/lud | 3.37 | 2.80 (1.20x) | 2.69 | 170 / 190 |
| polybench/3MM | 2.18 | 1.64 (1.33x) | 1.64 | 2 / 3 |
| polybench/GRAMSCHM | 164.04 | 121.06 (1.35x) | not run | 6118 / 6144 |

Every output identical to the simulator's. The sp76 rule is a gain on all three CUDA Samples (turbo at 0.67 to 0.80 of `turbod0`) and neutral on
the others. aidw was not run again (its card result is in `../sp76/card.md`: 4 of 4 launches on the card, 44 s).

**What is left in the simulator** (`kept.md`, one logged turbo run of each program):

* mergeSort 1.39 s of the 1.89 s: `mergeSortSharedKernel<1>`, **one launch of 4096 blocks x 512 threads, 1.38 s in the simulator**, kept as
  "static estimate under the floor and the process too young". Its count is 120 instructions and 0.079 s against a floor of 0.18 s; the launch
  took 1.38 s (17 times the estimate). It is the kernel's first launch, before the driver is up, so no sample can be taken.
* histogram 0.62 s of 2.12 s: the first launches of `histogram64Kernel` (4370 x 64; counted 0.025 s, took 0.318 s, 13 times) and
  `histogram256Kernel` (240 x 192; counted 0.003 s, took 0.292 s, 100 times), the same reason. Each kernel's second launch was sampled with the
  driver up and sent (estimates 201 ms and 185 ms against a card path of about 7 ms), as sp76 made it.
* segmentationTreeThrust 2.47 s of 4.50 s, in five reasons; the largest, 0.78 s, is the same one.
* radixsort2 0.28 s, silu 0.31 s, s3d 0.29 s over 145, 25 and 54 launches: nothing a rule would recover (launches of 0.1 to 6 ms each).

So **the one remaining cost of #659 is a first launch, before the driver is up, whose static count is 13 to 100 times under what it takes**
(mergeSort 1.38 s, histogram 0.61 s, segmentationTreeThrust 0.78 s: 2.8 s of the 8.5 s these three take in turbo). Nothing in the launch's
static facts says it: the loop-weighted count of #689 (`../sp76/census-weighted.md`) does not reach these kernels (not among its five), because
their loops' bounds are not immediates. What would is a measurement of the launch itself, which before the start only the simulator can give,
by running it. A rule that starts the driver on a guess about the kernel's size would trade the 0.5 s start against launches this size; it is not
made here, and its cost on the programs whose big first launch is a loser is not measured. 14 of lud's 190 launches and 20 of GRAMSCHM's 6,144
stay in the simulator too ("small": lud's are 0.27 s of simulator time in a 2.80 s run; GRAMSCHM's 26 are 0.28 s in a 121 s run).

## #662 silu and s3d (measured)

| program | sim s | exact s | turbo s | turbo launches on the card | `host_gpu_overhead` of a logged turbo run |
|---|---:|---:|---:|---|---|
| hecbench/silu-cuda | 0.65 | 0.65 (1.00x) | 0.59 (1.09x) | 25 / 50 | 25 launches, 5 kernels, 1.4 ms deciding |
| hecbench/s3d-cuda | 0.36 | 0.39 (0.93x) | 0.39 (0.93x) | 0 / 54 | 54 launches, **54 kernels, 76 ms deciding, 76 ms of it in the scan of each kernel's code** (`kernel_eligible`) |

* **silu is not slower than the simulator any more**: 1.09x in turbo (the issue's 0.87x was the run of 2026-11-13, before sp75/sp76), 25 of 50 launches
  on the card. What it keeps costs 0.31 s of simulator time over 25 launches (0.1 to 6 ms each).
* **s3d pays the scan of the code of 54 kernels, each launched once**, and sends nothing: the same 0.93x in exact, turbo and turbo with no
  sampling, so it is not the sample, the driver start or a card launch that loses. The scan (`kernel_eligible`: every reachable instruction of the
  kernel decoded and classified) is 1.4 ms of a logged run per kernel; a program's own simulation decodes only the instructions it runs. It is
  about 30 ms of a run without the log (0.36 s to 0.39 s, 3 runs each, identical to a millisecond). The other programs here pay 1 to 5 ms in all
  (a few kernels, launched many times; the scan is once a kernel). A trivial one-kernel program costs the same with the mode on and off (4 ms).
  A way to recover it: do not scan the kernel of a launch that cannot be worth sending (a grid too small for the launch to reach the card path
  even if every thread loops for its static count 100 times over), or cache the decode the simulator then reuses. Neither is made here; the first
  is a rule whose false-keep side has the same measure-first problem as the one above.

## Reproducing

```
card-run.sh decisions|samples|... [out dir]   # from the root of the tree on the rig; corpora in ~/corp; build in ~/build (Release, vgpucudart vgpucuda)
```
