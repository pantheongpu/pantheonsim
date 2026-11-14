# Host-GPU modes against the simulator, one A10G (2026-11-13)

Speed-up = simulator wall time / mode wall time, median of the runs (README.md has the method). Geometric means are over the programs where the simulator and that mode both finished; programs whose simulator run passed 600 s are counted separately (a lower bound).

## Headline

| suite | programs | geomean exact | geomean exact + idioms | geomean turbo | total s simulator | total s exact | total s exact + idioms | total s turbo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench | 20 | 1.58x (20) | 1.79x (20) | 1.78x (20) | 739 | 363 | 303 | 320 |
| rodinia | 12 | 1.16x (12) | 1.22x (12) | 1.29x (12) | 15 | 11 | 11 | 10 |
| hecbench | 32 | 1.27x (29) | 1.31x (29) | 1.98x (29) | 2825 (3 at > 600) | 1583 (2 at > 600) | 1579 (2 at > 600) | 437 |
| samples | 36 | 1.16x (36) | 1.16x (36) | 1.65x (36) | 545 | 501 | 501 | 420 |
| all | 100 | 1.27x (97) | 1.33x (97) | 1.72x (97) | 4125 (3 at > 600) | 2459 (2 at > 600) | 2394 (2 at > 600) | 1187 |

Totals sum the median wall time of every program in the suite; a timed-out run counts as 600 s.

The same geometric means over the programs whose simulator run took 1 s or more (the rest mostly time process start-up):

| suite | programs | geomean exact | geomean exact + idioms | geomean turbo |
|---|---:|---:|---:|---:|
| polybench | 12 | 2.14x (12) | 2.64x (12) | 2.62x (12) |
| rodinia | 6 | 1.36x (6) | 1.50x (6) | 1.69x (6) |
| hecbench | 11 | 1.88x (11) | 2.10x (11) | 5.96x (11) |
| samples | 17 | 1.40x (17) | 1.40x (17) | 2.98x (17) |
| all | 46 | 1.67x (46) | 1.84x (46) | 3.16x (46) |

## Gains and losses (programs where the simulator finished)

| suite | mode | > 2x faster | 1.03x to 2x | within 3 % | > 3 % slower | of those, simulator run >= 0.5 s |
|---|---|---:|---:|---:|---:|---:|
| polybench | exact | 4 | 6 | 10 | 0 | 0 |
| polybench | exact + idioms | 5 | 5 | 10 | 0 | 0 |
| polybench | turbo | 5 | 4 | 11 | 0 | 0 |
| rodinia | exact | 1 | 2 | 8 | 1 | 0 |
| rodinia | exact + idioms | 1 | 2 | 9 | 0 | 0 |
| rodinia | turbo | 2 | 1 | 8 | 1 | 0 |
| hecbench | exact | 2 | 1 | 24 | 2 | 0 |
| hecbench | exact + idioms | 3 | 1 | 21 | 4 | 0 |
| hecbench | turbo | 9 | 4 | 12 | 4 | 0 |
| samples | exact | 3 | 3 | 25 | 5 | 0 |
| samples | exact + idioms | 3 | 3 | 26 | 4 | 0 |
| samples | turbo | 10 | 1 | 20 | 5 | 0 |
| all | exact | 10 | 12 | 67 | 8 | 0 |
| all | exact + idioms | 12 | 11 | 66 | 8 | 0 |
| all | turbo | 26 | 10 | 51 | 10 | 0 |

A program that runs for a few hundredths of a second times process start-up: most losses are of those (listed below).

Slower by more than 3 %: hecbench/adam-cuda exact + idioms 0.90x (0.03 s against 0.03 s); hecbench/bgmv-cuda exact 0.92x (0.02 s against 0.01 s); hecbench/bgmv-cuda exact + idioms 0.95x (0.02 s against 0.01 s); hecbench/bgmv-cuda turbo 0.91x (0.02 s against 0.01 s); hecbench/blockAccess-cuda exact + idioms 0.89x (0.02 s against 0.02 s); hecbench/blockAccess-cuda turbo 0.96x (0.02 s against 0.02 s); hecbench/s3d-cuda exact 0.92x (0.35 s against 0.33 s); hecbench/s3d-cuda exact + idioms 0.85x (0.39 s against 0.33 s); hecbench/s3d-cuda turbo 0.92x (0.36 s against 0.33 s); hecbench/silu-cuda turbo 0.87x (0.55 s against 0.48 s); rodinia/bfs exact 0.97x (0.01 s against 0.01 s); rodinia/srad_v2 turbo 0.95x (0.07 s against 0.06 s); samples/binaryPartitionCG exact 0.96x (0.01 s against 0.01 s); samples/cdpBezierTessellation exact 0.95x (0.01 s against 0.01 s); samples/cdpQuadtree exact 0.96x (0.01 s against 0.01 s); samples/cdpQuadtree exact + idioms 0.95x (0.01 s against 0.01 s); samples/cdpQuadtree turbo 0.95x (0.01 s against 0.01 s); samples/cdpSimplePrint exact 0.93x (0.01 s against 0.01 s); samples/cdpSimplePrint exact + idioms 0.93x (0.01 s against 0.01 s); samples/cdpSimplePrint turbo 0.93x (0.01 s against 0.01 s); samples/cdpSimpleQuicksort turbo 0.95x (0.01 s against 0.01 s); samples/newdelete exact + idioms 0.96x (0.01 s against 0.01 s); samples/newdelete turbo 0.92x (0.01 s against 0.01 s); samples/simpleAtomicIntrinsics exact 0.96x (0.01 s against 0.01 s); samples/simpleAtomicIntrinsics exact + idioms 0.96x (0.01 s against 0.01 s); samples/simpleAtomicIntrinsics turbo 0.94x (0.01 s against 0.01 s).

## Top 15 by turbo speed-up

| program | runs | simulator s | exact s | x | exact + idioms s | x | turbo s | x | turbo launches on card |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/atomicReduction-cuda | 1 | 288.01 | 288.61 | 1.00x | 287.21 | 1.00x | 3.77 | 76.36x | 2090 / 2100 |
| hecbench/base64e-cuda | 1 | 307.95 | 8.74 | 35.22x | 8.72 | 35.30x | 8.71 | 35.35x | 514 / 516 |
| samples/radixSortThrust | 1 | 31.29 | 31.28 | 1.00x | 31.19 | 1.00x | 1.05 | 29.76x | 592 / 600 |
| hecbench/babelstream-cuda | 1 | 364.20 | 12.84 | 28.37x | 12.75 | 28.56x | 12.78 | 28.49x | 1188 / 1214 |
| polybench/LU | 3 | 27.68 | 10.30 | 2.69x | 1.28 | 21.67x | 1.30 | 21.27x | 4083 / 4096 |
| polybench/FDTD-2D | 1 | 93.49 | 4.58 | 20.42x | 4.64 | 20.14x | 4.57 | 20.44x | 1494 / 1500 |
| hecbench/shmembench-cuda | 3 | 4.33 | 4.33 | 1.00x | 4.33 | 1.00x | 0.29 | 15.10x | 1 / 1 |
| samples/jacobiCudaGraphs | 1 | 17.75 | 17.55 | 1.01x | 17.51 | 1.01x | 1.37 | 12.92x | 2910 / 2955 |
| samples/simpleHyperQ | 1 | 22.61 | 2.24 | 10.09x | 2.24 | 10.10x | 2.24 | 10.08x | 60 / 65 |
| rodinia/gaussian | 3 | 3.63 | 0.91 | 4.01x | 0.52 | 6.94x | 0.53 | 6.83x | 985 / 1022 |
| samples/reduction | 1 | 5.74 | 5.64 | 1.02x | 5.66 | 1.01x | 0.85 | 6.74x | 192 / 201 |
| samples/interval | 1 | 13.88 | 13.31 | 1.04x | 13.45 | 1.03x | 2.81 | 4.93x | 97 / 100 |
| hecbench/radixsort2-cuda | 3 | 3.02 | 2.99 | 1.01x | 2.97 | 1.02x | 0.64 | 4.74x | 455 / 600 |
| samples/sortingNetworks | 1 | 11.43 | 3.50 | 3.26x | 3.52 | 3.25x | 2.46 | 4.65x | 282 / 290 |
| samples/convolutionSeparable | 1 | 10.04 | 2.35 | 4.27x | 2.39 | 4.20x | 2.35 | 4.28x | 30 / 34 |

## Simulator past 600 s (speed-up is a lower bound)

| program | exact s | exact + idioms s | turbo s | turbo at least |
|---|---:|---:|---:|---:|
| hecbench/aidw-cuda | > 600 | > 600 | 365.33 | 2x |
| hecbench/aligned-types-cuda | 8.01 | 7.98 | 8.11 | 74x |
| hecbench/binomial-cuda | > 600 | > 600 | 1.36 | 441x |

## Outputs against the simulator's first run

| mode | identical | within tolerance | different | different, timing-dependent output (the simulator's own runs differ, or the output carries timings) | not compared |
|---|---:|---:|---:|---:|---:|
| exact | 92 | 0 | 0 | 5 | 3 |
| exact + idioms | 92 | 0 | 0 | 5 | 3 |
| turbo | 92 | 0 | 0 | 5 | 3 |

- hecbench/babelstream-cuda, exact: different (479195.002 against 1146.683; largest abs 5.02e+05, rel 442): the bandwidth table's rows (MBytes/sec, min, max, average seconds) are timings; only one simulator run (> 300 s), so no repeat check
- hecbench/babelstream-cuda, exact + idioms: different (479263.446 against 1146.683; largest abs 5e+05, rel 442): the bandwidth table's rows (MBytes/sec, min, max, average seconds) are timings; only one simulator run (> 300 s), so no repeat check
- hecbench/babelstream-cuda, turbo: different (469210.725 against 1146.683; largest abs 4.95e+05, rel 433): the bandwidth table's rows (MBytes/sec, min, max, average seconds) are timings; only one simulator run (> 300 s), so no repeat check
- hecbench/base64e-cuda, exact: different (161 words against 142; largest abs 0, rel 0): it repeats the whole test when the measured overlap of two streams is not acceptable (main.cu, RETRIES_ON_FAILURE): the number of lines depends on timing
- hecbench/base64e-cuda, exact + idioms: different (161 words against 142; largest abs 0, rel 0): it repeats the whole test when the measured overlap of two streams is not acceptable (main.cu, RETRIES_ON_FAILURE): the number of lines depends on timing
- hecbench/base64e-cuda, turbo: different (161 words against 142; largest abs 0, rel 0): it repeats the whole test when the measured overlap of two streams is not acceptable (main.cu, RETRIES_ON_FAILURE): the number of lines depends on timing
- rodinia/particlefilter_float, exact: different (0.008513 against 0.008469; largest abs 0.648, rel 6.14): the simulator's own runs differ too
- rodinia/particlefilter_float, exact + idioms: different (0.008465 against 0.008469; largest abs 0.567, rel 6.14): the simulator's own runs differ too
- rodinia/particlefilter_float, turbo: different (0.008441 against 0.008469; largest abs 1.68, rel 6.29): the simulator's own runs differ too
- samples/cudaGraphsPerfScaling, exact: different (19.001 against 18.001; largest abs 31.8, rel 0.684): the simulator's own runs differ too
- samples/cudaGraphsPerfScaling, exact + idioms: different (40.420 against 18.001; largest abs 22.4, rel 1.25): the simulator's own runs differ too
- samples/cudaGraphsPerfScaling, turbo: different (18.500 against 18.001; largest abs 17.6, rel 0.379): the simulator's own runs differ too
- samples/simplePrintf, exact: different ('1' against '3'; largest abs 0, rel 0): the simulator's own runs differ too
- samples/simplePrintf, exact + idioms: different ('0' against '1'; largest abs 0, rel 0): the simulator's own runs differ too
- samples/simplePrintf, turbo: different ('1' against '3'; largest abs 0, rel 0): the simulator's own runs differ too
- the simulator does not repeat itself run to run: rodinia/particlefilter_float, samples/cudaGraphsPerfScaling, samples/simplePrintf

Non-zero exit status: hecbench/atomicCost-cuda simulator (rc [-9, -9, -9]), hecbench/atomicCost-cuda exact (rc [-9, -9, -9]), hecbench/atomicCost-cuda exact + idioms (rc [-9, -9, -9]), hecbench/atomicCost-cuda turbo (rc [-9, -9, -9]).

## Every program

| program | runs | simulator s | exact s | x | exact + idioms s | x | turbo s | x | output exact / idioms / turbo |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| polybench/2DCONV | 3 | 0.94 | 0.93 | 1.01x | 0.93 | 1.01x | 0.93 | 1.01x | = / = / = |
| polybench/2MM | 1 | 181.22 | 98.10 | 1.85x | 96.60 | 1.88x | 97.13 | 1.87x | = / = / = |
| polybench/3DCONV | 3 | 0.72 | 0.73 | 0.99x | 0.72 | 1.01x | 0.72 | 1.01x | = / = / = |
| polybench/3MM | 3 | 2.12 | 1.69 | 1.25x | 1.62 | 1.31x | 1.61 | 1.32x | = / = / = |
| polybench/ADI | 3 | 0.35 | 0.36 | 0.97x | 0.36 | 0.97x | 0.36 | 0.97x | = / = / = |
| polybench/ATAX | 3 | 1.05 | 1.05 | 1.00x | 1.05 | 1.00x | 1.05 | 1.00x | = / = / = |
| polybench/BICG | 3 | 0.26 | 0.26 | 1.00x | 0.26 | 1.00x | 0.26 | 1.00x | = / = / = |
| polybench/CORR | 1 | 59.43 | 37.30 | 1.59x | 37.22 | 1.60x | 36.74 | 1.62x | = / = / = |
| polybench/COVAR | 1 | 59.11 | 37.02 | 1.60x | 37.22 | 1.59x | 35.67 | 1.66x | = / = / = |
| polybench/FDTD-2D | 1 | 93.49 | 4.58 | 20.42x | 4.64 | 20.14x | 4.57 | 20.44x | = / = / = |
| polybench/GEMM | 3 | 0.71 | 0.70 | 1.01x | 0.70 | 1.01x | 0.70 | 1.01x | = / = / = |
| polybench/GEMVER | 3 | 0.62 | 0.62 | 1.00x | 0.62 | 0.99x | 0.62 | 1.00x | = / = / = |
| polybench/GESUMMV | 3 | 0.38 | 0.38 | 1.00x | 0.38 | 1.00x | 0.39 | 0.98x | = / = / = |
| polybench/GRAMSCHM | 1 | 292.69 | 161.54 | 1.81x | 112.18 | 2.61x | 130.70 | 2.24x | = / = / = |
| polybench/JACOBI1D | 3 | 2.21 | 2.25 | 0.98x | 2.25 | 0.98x | 2.25 | 0.98x | = / = / = |
| polybench/JACOBI2D | 3 | 0.57 | 0.55 | 1.04x | 0.54 | 1.05x | 0.56 | 1.02x | = / = / = |
| polybench/LU | 3 | 27.68 | 10.30 | 2.69x | 1.28 | 21.67x | 1.30 | 21.27x | = / = / = |
| polybench/MVT | 3 | 1.20 | 1.19 | 1.00x | 1.19 | 1.01x | 1.19 | 1.01x | = / = / = |
| polybench/SYR2K | 3 | 9.56 | 2.33 | 4.11x | 2.32 | 4.12x | 2.32 | 4.12x | = / = / = |
| polybench/SYRK | 3 | 5.14 | 1.34 | 3.83x | 1.29 | 3.98x | 1.29 | 3.97x | = / = / = |
| rodinia/backprop | 3 | 0.57 | 0.58 | 1.00x | 0.58 | 1.00x | 0.57 | 1.00x | = / = / = |
| rodinia/bfs | 3 | 0.01 | 0.01 | 0.97x | 0.01 | 0.98x | 0.01 | 0.98x | = / = / = |
| rodinia/gaussian | 3 | 3.63 | 0.91 | 4.01x | 0.52 | 6.94x | 0.53 | 6.83x | = / = / = |
| rodinia/hotspot | 3 | 1.04 | 1.05 | 0.99x | 1.04 | 1.00x | 1.04 | 0.99x | = / = / = |
| rodinia/lavaMD | 3 | 1.51 | 1.51 | 1.00x | 1.51 | 1.00x | 1.54 | 0.98x | = / = / = |
| rodinia/lud | 3 | 3.18 | 2.91 | 1.09x | 2.75 | 1.16x | 2.72 | 1.17x | = / = / = |
| rodinia/needle | 3 | 0.32 | 0.32 | 0.99x | 0.32 | 0.99x | 0.32 | 1.00x | = / = / = |
| rodinia/nn | 3 | 0.02 | 0.02 | 0.99x | 0.02 | 0.99x | 0.02 | 1.00x | = / = / = |
| rodinia/particlefilter_float | 3 | 2.90 | 2.04 | 1.42x | 2.05 | 1.41x | 0.96 | 3.03x | DIFF / DIFF / DIFF |
| rodinia/pathfinder | 3 | 1.02 | 1.01 | 1.01x | 1.01 | 1.02x | 1.02 | 1.00x | = / = / = |
| rodinia/sc_gpu | 3 | 0.94 | 0.94 | 0.99x | 0.94 | 0.99x | 0.94 | 0.99x | = / = / = |
| rodinia/srad_v2 | 3 | 0.06 | 0.06 | 0.97x | 0.06 | 0.99x | 0.07 | 0.95x | = / = / = |
| hecbench/adam-cuda | 3 | 0.03 | 0.03 | 0.98x | 0.03 | 0.90x | 0.03 | 0.98x | = / = / = |
| hecbench/aidw-cuda | 1 | > 600 | > 600 | - | > 600 | - | 365.33 | - | - / - / - |
| hecbench/aligned-types-cuda | 1 | > 600 | 8.01 | - | 7.98 | - | 8.11 | - | - / - / - |
| hecbench/aobench-cuda | 3 | 0.50 | 0.50 | 1.00x | 0.50 | 1.00x | 0.50 | 1.00x | = / = / = |
| hecbench/aop-cuda | 3 | 1.68 | 1.66 | 1.01x | 1.66 | 1.01x | 0.79 | 2.12x | = / = / = |
| hecbench/asmooth-cuda | 3 | 0.22 | 0.22 | 1.00x | 0.22 | 1.00x | 0.22 | 1.00x | = / = / = |
| hecbench/asyncAllocation-cuda | 3 | 3.35 | 3.34 | 1.00x | 1.01 | 3.32x | 1.02 | 3.29x | = / = / = |
| hecbench/atan2-cuda | 3 | 0.03 | 0.03 | 1.03x | 0.03 | 1.01x | 0.03 | 1.03x | = / = / = |
| hecbench/atomicAggregate-cuda | 3 | 3.22 | 3.25 | 0.99x | 3.22 | 1.00x | 1.34 | 2.41x | = / = / = |
| hecbench/atomicCost-cuda | 3 | 21.55 rc-9 | 21.72 rc-9 | 0.99x | 21.57 rc-9 | 1.00x | 21.58 rc-9 | 1.00x | = / = / = |
| hecbench/atomicIntrinsics-cuda | 3 | 0.17 | 0.17 | 1.00x | 0.17 | 1.00x | 0.17 | 1.00x | = / = / = |
| hecbench/atomicPerf-cuda | 3 | 0.99 | 0.98 | 1.02x | 0.98 | 1.02x | 0.76 | 1.31x | = / = / = |
| hecbench/atomicReduction-cuda | 1 | 288.01 | 288.61 | 1.00x | 287.21 | 1.00x | 3.77 | 76.36x | = / = / = |
| hecbench/attention-cuda | 3 | 0.03 | 0.03 | 1.00x | 0.03 | 0.99x | 0.03 | 1.01x | = / = / = |
| hecbench/axpby-cuda | 3 | 0.39 | 0.39 | 1.02x | 0.38 | 1.02x | 0.39 | 0.99x | = / = / = |
| hecbench/babelstream-cuda | 1 | 364.20 | 12.84 | 28.37x | 12.75 | 28.56x | 12.78 | 28.49x | DIFF / DIFF / DIFF |
| hecbench/background-subtract-cuda | 3 | 0.00 | 0.00 | 1.13x | 0.00 | 1.09x | 0.00 | 1.09x | = / = / = |
| hecbench/backprop-cuda | 3 | 0.01 | 0.01 | 1.00x | 0.01 | 1.00x | 0.01 | 1.00x | = / = / = |
| hecbench/base64e-cuda | 1 | 307.95 | 8.74 | 35.22x | 8.72 | 35.30x | 8.71 | 35.35x | DIFF / DIFF / DIFF |
| hecbench/bgmv-cuda | 3 | 0.01 | 0.02 | 0.92x | 0.02 | 0.95x | 0.02 | 0.91x | = / = / = |
| hecbench/bh-cuda | 3 | 0.75 | 0.74 | 1.01x | 0.74 | 1.00x | 0.63 | 1.18x | = / = / = |
| hecbench/bilateral-cuda | 3 | 1.11 | 1.10 | 1.01x | 1.10 | 1.01x | 1.10 | 1.00x | = / = / = |
| hecbench/binomial-cuda | 1 | > 600 | > 600 | - | > 600 | - | 1.36 | - | - / - / - |
| hecbench/bitonic-sort-cuda | 3 | 0.29 | 0.29 | 1.00x | 0.29 | 1.00x | 0.29 | 1.00x | = / = / = |
| hecbench/black-scholes-cuda | 3 | 21.87 | 21.82 | 1.00x | 21.78 | 1.00x | 6.04 | 3.62x | = / = / = |
| hecbench/blockAccess-cuda | 3 | 0.02 | 0.02 | 0.99x | 0.02 | 0.89x | 0.02 | 0.96x | = / = / = |
| hecbench/perplexity-cuda | 3 | 0.36 | 0.36 | 1.00x | 0.37 | 0.99x | 0.36 | 1.00x | = / = / = |
| hecbench/radixsort2-cuda | 3 | 3.02 | 2.99 | 1.01x | 2.97 | 1.02x | 0.64 | 4.74x | = / = / = |
| hecbench/s3d-cuda | 3 | 0.33 | 0.35 | 0.92x | 0.39 | 0.85x | 0.36 | 0.92x | = / = / = |
| hecbench/shmembench-cuda | 3 | 4.33 | 4.33 | 1.00x | 4.33 | 1.00x | 0.29 | 15.10x | = / = / = |
| hecbench/silu-cuda | 3 | 0.48 | 0.48 | 1.00x | 0.48 | 1.00x | 0.55 | 0.87x | = / = / = |
| hecbench/sortKV-cuda | 3 | 0.25 | 0.26 | 0.97x | 0.25 | 0.98x | 0.25 | 0.98x | = / = / = |
| samples/bf16TensorCoreGemm | 1 | 160.06 | 158.90 | 1.01x | 159.23 | 1.00x | 159.11 | 1.01x | = / = / = |
| samples/binaryPartitionCG | 3 | 0.01 | 0.01 | 0.96x | 0.01 | 0.99x | 0.01 | 0.98x | = / = / = |
| samples/cdpBezierTessellation | 3 | 0.01 | 0.01 | 0.95x | 0.01 | 0.98x | 0.01 | 0.98x | = / = / = |
| samples/cdpQuadtree | 3 | 0.01 | 0.01 | 0.96x | 0.01 | 0.95x | 0.01 | 0.95x | = / = / = |
| samples/cdpSimplePrint | 3 | 0.01 | 0.01 | 0.93x | 0.01 | 0.93x | 0.01 | 0.93x | = / = / = |
| samples/cdpSimpleQuicksort | 3 | 0.01 | 0.01 | 1.00x | 0.01 | 1.00x | 0.01 | 0.95x | = / = / = |
| samples/convolutionSeparable | 1 | 10.04 | 2.35 | 4.27x | 2.39 | 4.20x | 2.35 | 4.28x | = / = / = |
| samples/cudaGraphsPerfScaling | 3 | 0.01 | 0.01 | 0.98x | 0.01 | 0.98x | 0.01 | 0.98x | DIFF / DIFF / DIFF |
| samples/cudaTensorCoreGemm | 1 | 16.36 | 16.43 | 1.00x | 16.39 | 1.00x | 16.31 | 1.00x | = / = / = |
| samples/dct8x8 | 3 | 0.95 | 0.95 | 1.00x | 0.95 | 1.00x | 0.95 | 1.00x | = / = / = |
| samples/histogram | 1 | 11.76 | 11.53 | 1.02x | 11.57 | 1.02x | 2.79 | 4.21x | = / = / = |
| samples/interval | 1 | 13.88 | 13.31 | 1.04x | 13.45 | 1.03x | 2.81 | 4.93x | = / = / = |
| samples/jacobiCudaGraphs | 1 | 17.75 | 17.55 | 1.01x | 17.51 | 1.01x | 1.37 | 12.92x | = / = / = |
| samples/LargeKernelParameter | 3 | 0.40 | 0.41 | 0.98x | 0.41 | 0.97x | 0.41 | 0.97x | = / = / = |
| samples/MC_EstimatePiInlineP | 3 | 3.13 | 3.10 | 1.01x | 3.10 | 1.01x | 3.10 | 1.01x | = / = / = |
| samples/MC_SingleAsianOptionP | 3 | 3.41 | 3.41 | 1.00x | 3.41 | 1.00x | 3.42 | 1.00x | = / = / = |
| samples/mergeSort | 1 | 7.10 | 6.96 | 1.02x | 6.96 | 1.02x | 2.84 | 2.50x | = / = / = |
| samples/newdelete | 3 | 0.01 | 0.01 | 0.97x | 0.01 | 0.96x | 0.01 | 0.92x | = / = / = |
| samples/ptxjit | 3 | 0.01 | 0.01 | 1.00x | 0.01 | 1.00x | 0.01 | 1.00x | = / = / = |
| samples/radixSortThrust | 1 | 31.29 | 31.28 | 1.00x | 31.19 | 1.00x | 1.05 | 29.76x | = / = / = |
| samples/reduction | 1 | 5.74 | 5.64 | 1.02x | 5.66 | 1.01x | 0.85 | 6.74x | = / = / = |
| samples/scalarProd | 3 | 0.05 | 0.05 | 0.99x | 0.05 | 0.98x | 0.05 | 0.98x | = / = / = |
| samples/segmentationTreeThrust | 1 | 16.41 | 10.71 | 1.53x | 10.84 | 1.51x | 5.60 | 2.93x | = / = / = |
| samples/shfl_scan | 3 | 0.10 | 0.09 | 1.01x | 0.09 | 1.02x | 0.09 | 1.01x | = / = / = |
| samples/simpleAtomicIntrinsics | 3 | 0.01 | 0.01 | 0.96x | 0.01 | 0.96x | 0.01 | 0.94x | = / = / = |
| samples/simpleCooperativeGroups | 3 | 0.00 | 0.00 | 1.03x | 0.00 | 1.03x | 0.00 | 0.97x | = / = / = |
| samples/simpleCubemapTexture | 3 | 0.01 | 0.01 | 0.99x | 0.01 | 0.98x | 0.01 | 1.01x | = / = / = |
| samples/simpleCudaGraphs | 3 | 1.53 | 1.20 | 1.28x | 1.20 | 1.28x | 1.05 | 1.46x | = / = / = |
| samples/simpleHyperQ | 1 | 22.61 | 2.24 | 10.09x | 2.24 | 10.10x | 2.24 | 10.08x | = / = / = |
| samples/simpleLayeredTexture | 3 | 0.20 | 0.20 | 1.00x | 0.20 | 1.00x | 0.20 | 1.00x | = / = / = |
| samples/simplePrintf | 3 | 0.00 | 0.00 | 1.00x | 0.00 | 1.02x | 0.00 | 1.00x | DIFF / DIFF / DIFF |
| samples/simpleSurfaceWrite | 3 | 0.10 | 0.10 | 0.99x | 0.10 | 0.98x | 0.10 | 0.98x | = / = / = |
| samples/sortingNetworks | 1 | 11.43 | 3.50 | 3.26x | 3.52 | 3.25x | 2.46 | 4.65x | = / = / = |
| samples/tf32TensorCoreGemm | 1 | 208.22 | 208.40 | 1.00x | 208.01 | 1.00x | 207.92 | 1.00x | = / = / = |
| samples/threadFenceReduction | 3 | 0.34 | 0.34 | 1.00x | 0.34 | 1.00x | 0.34 | 1.00x | = / = / = |
| samples/warpAggregatedAtomicsCG | 3 | 2.01 | 2.01 | 1.00x | 2.04 | 0.99x | 2.02 | 1.00x | = / = / = |

`=` identical, `~` within tolerance (atol 1e-8 + rtol 1e-5), `DIFF` different, `-` not compared (the simulator timed out).
