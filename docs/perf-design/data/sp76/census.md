# Round sp76: the census (server3, the stand-in driver, turbo mode)

Every program once in each form, `VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1`, profile `nvidia/a10g`, `VGPU_THREADS=3`, against the stand-in
driver, which refuses every module: a launch the mode sends "fails on the card" and the simulator runs it, and no memory is ever the
card's alone (nothing is pulled before a sample here; aidw's run in `launches.md` and the stand-in test have that). `d0`: the round's
build with `VGPU_HOST_GPU_SAMPLE_STARTED=0`, the rule before. `new`: the same build, the rule. A run is cut at 40 s.

This is the second pass, on the round's final build: the 29 programs that start the driver in a form and the 14 the first pass had not
run, 43 in all (2 cut at the time limit). The first pass (89 programs, the rule as first built, without the measured launch in the
count's place; not kept) found 24 programs that start the driver and no difference in the 65 that do not.

## The launches the rule touches

From the records of `census.py`: per program, the `host_gpu_nosample` lines of `d0` whose reason is the driver's start, and the samples
`new` took with the driver up.

| program | `d0`: lines `not sampled: the driver has been started` (a kernel's first two launches are said) | `new`: sampled with the driver up and sent | ... and kept | ... the copy did not answer | estimates of the ones sent, ms | samples cost, `d0` s | `new` s | wall `d0` s | `new` s |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| hecbench/atomicReduction-cuda | 4 | 0 | 2 | 0 | - | 0.0000 | 0.0000 | 40.02 | 40.02 |
| samples/alignedTypes | 22 | 10 | 1 | 0 | 592, 427, 402, 389, 101, 94 | 0.0000 | 0.0642 | 20.46 | 20.61 |
| samples/histogram | 4 | 1 | 1 | 0 | 113 | 0.0000 | 0.0115 | 5.39 | 5.45 |
| samples/jacobiCudaGraphs | 1 | 0 | 1 | 0 | - | 0.0000 | 0.0017 | 8.49 | 8.55 |
| samples/mergeSort | 2 | 1 | 0 | 0 | 85 | 0.0000 | 0.0029 | 3.33 | 3.29 |
| samples/segmentationTreeThrust | 12 | 0 | 0 | 0 | - | 0.0055 | 0.0056 | 28.34 | 22.90 |
| samples/simpleHyperQ | 1 | 0 | 0 | 0 | - | 0.0000 | 0.0000 | 11.40 | 11.35 |
| samples/sortingNetworks | 3 | 0 | 0 | 0 | - | 0.0000 | 0.0000 | 5.71 | 5.74 |

43 programs; 29 start the driver in a form; 2 cut at the time limit in a form; 35 with no launch the rule touches. Over the others: 49 lines of `d0`, 12 samples sent, 5 kept, 0 copies that did not answer.

Not sampled with the rule on, of the lines of `d0`: 11 second launches of alignedTypes (the first took 71 ms: under the floor of 0.18 s
that 100 MiB live ask), `histogram256Kernel`'s first (2.8 ms counted, under the floor), one of mergeSort's (its earlier whole launch was
short), and all of segmentationTreeThrust's, sortingNetworks' and simpleHyperQ's (the kernel has no loop: 15; under the floor of 0.52 s
that 290 MiB live ask, or short before: the rest).

## Programs whose decisions differ between the two forms (`../sp75/table.py --old d0 --new new`)

| program | `d0`: kept / too small / sent / sampled / sent on a sample / copies that did not answer | `new` | `new`: the sampled launches |
|---|---|---|---|
| hecbench/asyncAllocation-cuda | 800 / 730 / 68 / 0 / 0 / 0 | 800 / 737 / 61 / 0 / 0 / 0 |  |
| hecbench/atomicPerf-cuda | 18 / 0 / 11 / 1 / 0 / 0 | 18 / 0 / 10 / 2 / 0 / 0 | 2 before the start, kept |
| hecbench/atomicReduction-cuda | - / - / - / - / - / - (cut) | - / - / - / - / - / - (cut) | with the driver up: 101.507 ms against 51.143 ms: kept; 57.928 ms against 45.696 ms: kept |
| rodinia/gaussian | 1022 / 546 / 472 / 0 / 0 / 0 | 1022 / 548 / 470 / 0 / 0 / 0 |  |
| rodinia/lud | 190 / 132 / 52 / 0 / 0 / 0 | 190 / 133 / 51 / 0 / 0 / 0 |  |
| rodinia/particlefilter_float | 36 / 9 / 19 / 1 / 0 / 0 | 36 / 7 / 21 / 1 / 0 / 0 | 1 before the start, kept |
| samples/alignedTypes | 384 / 60 / 300 / 0 / 0 / 0 | 384 / 60 / 310 / 11 / 0 / 0 | with the driver up: 592.062 ms against 92.348 ms: sent; 401.681 ms against 68.550 ms: sent; 93.625 ms against 30.043 ms: sent; 92.497 ms against 29.902 ms: sent; 101.483 ms against 31.026 ms: sent; 426.691 ms against 71.677 ms: sent; 389.473 ms against 67.024 ms: sent; 92.039 ms against 29.845 ms: sent; 90.493 ms against 29.652 ms: sent; 90.171 ms against 29.612 ms: sent; 0.000 ms against 18.440 ms: kept |
| samples/histogram | 68 / 30 / 30 / 0 / 0 / 0 | 68 / 30 / 31 / 2 / 0 / 0 | with the driver up: 1.408 ms against 12.532 ms: kept; 113.007 ms against 26.481 ms: sent |
| samples/jacobiCudaGraphs | 2955 / 198 / 2754 / 0 / 0 / 0 | 2955 / 198 / 2754 / 1 / 0 / 0 | with the driver up: 0.037 ms against 0.301 ms: kept |
| samples/mergeSort | 49 / 32 / 11 / 0 / 0 / 0 | 49 / 32 / 12 / 1 / 0 / 0 | with the driver up: 85.341 ms against 29.226 ms: sent |

The rows with no sample taken with the driver up (asyncAllocation, atomicPerf, gaussian, lud, particlefilter) differ between two runs of
one rule: when the account of simulated time fills, whether the process is old enough to afford a sample, and particlefilter's race are
decided by timing.
