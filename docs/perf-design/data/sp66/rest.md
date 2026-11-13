Host-GPU benchmark: nvidia/a10g, 1 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | 0.96 | 0.96 | 1.00x | 0 / 1 | 0.00 / 0.38 | identical | 0.96 | 1.00x | 0 / 1 | 0.00 / 0.38 | identical | 0/0/0/0/0/0 |
| polybench/3DCONV | 0.77 | 0.82 | 0.94x | 0 / 254 | 0.00 / 0.58 | identical | 0.76 | 1.01x | 0 / 254 | 0.00 / 0.50 | identical | 0/0/0/0/0/252 |
| polybench/3MM | 2.36 | 1.88 | 1.25x | 2 / 3 | 0.00 / 0.40 | identical | 1.84 | 1.28x | 2 / 3 | 0.00 / 0.40 | identical | 0/0/0/0/0/0 |
| polybench/ADI | 0.40 | 0.40 | 0.99x | 0 / 2049 | 0.00 / 0.33 | identical | 0.41 | 0.98x | 0 / 2049 | 0.00 / 0.32 | identical | 0/0/0/0/0/2041 |
| polybench/ATAX | 1.08 | 1.08 | 1.00x | 0 / 2 | 0.00 / 0.99 | identical | 1.10 | 0.98x | 0 / 2 | 0.00 / 1.00 | identical | 0/0/0/0/0/0 |
| polybench/BICG | 0.26 | 0.26 | 1.00x | 0 / 2 | 0.00 / 0.12 | identical | 0.26 | 1.00x | 0 / 2 | 0.00 / 0.12 | identical | 0/0/0/0/0/0 |
| polybench/CORR | 63.83 | 41.81 | 1.53x | 1 / 4 | 1.67 / 0.18 | identical | 46.54 | 1.37x | 1 / 4 | 1.67 / 0.18 | identical | 0/0/0/0/0/0 |
| polybench/COVAR | 63.92 | 41.80 | 1.53x | 1 / 3 | 1.67 / 0.03 | identical | 45.29 | 1.41x | 1 / 3 | 1.67 / 0.03 | identical | 0/0/0/0/0/0 |
| polybench/GEMM | 0.88 | 0.88 | 1.01x | 0 / 1 | 0.00 / 0.44 | identical | 0.86 | 1.02x | 0 / 1 | 0.00 / 0.44 | identical | 0/0/0/0/0/0 |
| polybench/GEMVER | 0.65 | 0.67 | 0.98x | 0 / 3 | 0.00 / 0.42 | identical | 0.68 | 0.96x | 0 / 3 | 0.00 / 0.45 | identical | 0/0/0/0/0/0 |
| polybench/GESUMMV | 0.38 | 0.38 | 0.99x | 0 / 1 | 0.00 / 0.18 | identical | 0.38 | 1.00x | 0 / 1 | 0.00 / 0.18 | identical | 0/0/0/0/0/0 |
| polybench/JACOBI1D | 2.41 | 2.48 | 0.97x | 0 / 20000 | 0.00 / 2.47 | identical | 2.47 | 0.97x | 0 / 20000 | 0.00 / 2.32 | identical | 0/0/0/0/0/19996 |
| polybench/JACOBI2D | 0.62 | 0.55 | 1.12x | 22 / 40 | 0.00 / 0.26 | identical | 0.55 | 1.12x | 22 / 40 | 0.00 / 0.26 | identical | 0/0/0/0/0/14 |
| polybench/MVT | 1.22 | 1.26 | 0.97x | 0 / 2 | 0.00 / 0.98 | identical | 1.21 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 0/0/0/0/0/0 |
| polybench/SYR2K | 9.83 | 2.33 | 4.22x | 1 / 1 | 0.02 / 0.00 | identical | 2.37 | 4.14x | 1 / 1 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/SYRK | 5.22 | 1.29 | 4.03x | 1 / 1 | 0.01 / 0.00 | identical | 1.29 | 4.03x | 1 / 1 | 0.01 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/backprop | 0.61 | 0.61 | 1.00x | 0 / 2 | 0.00 / 0.44 | identical | 0.71 | 0.86x | 1 / 2 | 0.02 / 0.26 | identical | 0/0/0/0/0/0 |
| rodinia/gaussian | 4.02 | 0.93 | 4.33x | 494 / 1022 | 0.17 / 0.28 | identical | 0.94 | 4.29x | 494 / 1022 | 0.18 / 0.28 | identical | 0/0/0/0/0/524 |
| rodinia/lud | 3.51 | 2.93 | 1.20x | 50 / 190 | 0.04 / 0.43 | identical | 2.85 | 1.23x | 94 / 190 | 0.03 / 0.30 | identical | 0/0/0/0/0/90 |
| rodinia/needle | 0.36 | 0.36 | 0.99x | 0 / 255 | 0.00 / 0.48 | identical | 0.36 | 0.98x | 0 / 255 | 0.00 / 0.32 | identical | 0/0/0/0/0/251 |
| rodinia/pathfinder | 1.05 | 1.07 | 0.98x | 0 / 5 | 0.00 / 0.22 | identical | 1.08 | 0.97x | 0 / 5 | 0.00 / 0.23 | identical | 0/0/0/0/0/3 |
| rodinia/srad_v2 | 0.07 | 0.06 | 1.07x | 0 / 1 | 0.00 / 0.00 | identical | 0.06 | 1.05x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/sc_gpu | 1.03 | 0.78 | 1.32x | 1076 / 1432 | 0.11 / 0.25 | identical | 0.79 | 1.30x | 1059 / 1432 | 0.10 / 0.25 | identical | 0/0/0/0/0/371 |
| rodinia/hotspot | 1.18 | 1.18 | 1.00x | 0 / 1000 | 0.00 / 1.61 | identical | 0.52 | 2.27x | 789 / 1000 | 0.02 / 0.25 | identical | 0/0/0/0/0/209 |
| rodinia/bfs | 0.01 | 0.01 | 1.01x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 1.02x | 0 / 26 | 0.00 / 0.01 | identical | 0/0/0/0/0/22 |
| rodinia/nn | 0.02 | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/lavaMD | 1.75 | 1.76 | 0.99x | 0 / 1 | 0.00 / 1.72 | identical | 1.75 | 1.00x | 0 / 1 | 0.00 / 1.72 | identical | 0/0/0/0/0/0 |
| hecbench/adam-cuda | 0.03 | 0.03 | 1.01x | 0 / 1 | 0.00 / 0.02 | identical | 0.03 | 1.01x | 0 / 1 | 0.00 / 0.02 | identical | 0/0/0/0/0/0 |
| hecbench/aligned-types-cuda | TIMEOUT | 8.30 | - | 12974 / 13000 | 3.46 / 3.52 | - | 8.49 | - | 12974 / 13000 | 3.45 / 3.52 | - | 0/0/0/0/0/0 |
| hecbench/aobench-cuda | 0.54 | 0.54 | 1.00x | 0 / 1 | 0.00 / 0.53 | identical | 0.54 | 1.00x | 0 / 1 | 0.00 / 0.53 | identical | 0/0/0/0/0/0 |
| hecbench/asmooth-cuda | 0.26 | 0.26 | 1.00x | 0 / 3 | 0.00 / 0.22 | identical | 0.26 | 1.00x | 0 / 3 | 0.00 / 0.22 | identical | 0/0/0/0/0/0 |
| hecbench/atan2-cuda | 0.03 | 0.03 | 0.99x | 0 / 3 | 0.00 / 0.02 | identical | 0.03 | 0.94x | 0 / 3 | 0.00 / 0.02 | identical | 0/0/0/0/0/0 |
| hecbench/atomicAggregate-cuda | 3.55 | 3.55 | 1.00x | 0 / 6 | 0.00 / 3.55 | identical | 1.45 | 2.44x | 4 / 6 | 0.00 / 1.21 | identical | 0/0/0/0/0/0 |
| hecbench/atomicIntrinsics-cuda | 0.18 | 0.18 | 0.99x | 0 / 4 | 0.00 / 0.17 | identical | 0.18 | 1.00x | 0 / 4 | 0.00 / 0.17 | identical | 0/0/0/0/0/0 |
| hecbench/atomicReduction-cuda | 294.53 | 294.26 | 1.00x | 0 / 2100 | 0.00 / 292.76 | identical | 3.85 | 76.50x | 2090 / 2100 | 1.15 / 1.35 | identical | 0/0/0/0/0/0 |
| hecbench/axpby-cuda | 0.46 | 0.65 | 0.71x | 5 / 35 | 0.00 / 0.33 | identical | 0.65 | 0.72x | 5 / 35 | 0.00 / 0.33 | identical | 0/0/0/0/0/28 |
| hecbench/babelstream-cuda | TIMEOUT | 13.55 | - | 1188 / 1214 | 2.24 / 8.67 | - | 13.63 | - | 1188 / 1214 | 2.24 / 8.74 | - | 0/0/0/0/0/0 |
| hecbench/background-subtract-cuda | 0.00 | 0.00 | 1.28x | - | - | identical | 0.00 | 1.28x | - | - | identical | - |
| hecbench/backprop-cuda | 0.01 | 0.01 | 1.01x | 0 / 2 | 0.00 / 0.01 | identical | 0.01 | 0.92x | 0 / 2 | 0.00 / 0.01 | identical | 0/0/0/0/0/0 |
| hecbench/base64e-cuda | TIMEOUT | 8.81 | - | 514 / 516 | 2.04 / 0.81 | - | 8.84 | - | 514 / 516 | 2.04 / 0.78 | - | 0/0/0/0/0/0 |
| hecbench/bilateral-cuda | 1.38 | 1.28 | 1.08x | 0 / 3 | 0.00 / 1.17 | identical | 1.27 | 1.08x | 0 / 3 | 0.00 / 1.17 | identical | 0/0/0/0/0/0 |
| hecbench/binomial-cuda | TIMEOUT | TIMEOUT | - | - | - | - | 1.36 | - | 1000 / 1000 | 0.76 / 0.00 | - | 0/0/0/0/0/0 |
| hecbench/bitonic-sort-cuda | 0.32 | 0.32 | 0.99x | 0 / 136 | 0.00 / 0.35 | identical | 0.52 | 0.61x | 20 / 136 | 0.00 / 0.25 | identical | 0/0/0/0/0/114 |
| hecbench/black-scholes-cuda | 24.98 | 24.96 | 1.00x | 0 / 1 | 0.00 / 19.82 | identical | 5.99 | 4.17x | 1 / 1 | 0.39 / 0.00 | identical | 0/0/0/0/0/0 |
| hecbench/blockAccess-cuda | 0.02 | 0.02 | 1.00x | 0 / 4 | 0.00 / 0.01 | identical | 0.02 | 1.01x | 0 / 4 | 0.00 / 0.01 | identical | 0/0/0/0/0/0 |
| hecbench/perplexity-cuda | 0.33 | 0.33 | 1.00x | 0 / 1 | 0.00 / 0.31 | identical | 0.33 | 1.00x | 0 / 1 | 0.00 / 0.30 | identical | 0/0/0/0/0/0 |
| hecbench/s3d-cuda | 0.37 | 0.39 | 0.96x | 0 / 54 | 0.00 / 0.28 | identical | 0.39 | 0.97x | 0 / 54 | 0.00 / 0.28 | identical | 0/0/0/0/0/0 |
| hecbench/shmembench-cuda | 6.12 | 6.16 | 0.99x | 0 / 1 | 0.00 / 6.13 | identical | 0.30 | 20.10x | 1 / 1 | 0.01 / 0.00 | identical | 0/0/0/0/0/0 |
| hecbench/silu-cuda | 0.52 | 0.52 | 1.00x | 0 / 50 | 0.00 / 0.48 | identical | 0.54 | 0.97x | 24 / 50 | 0.00 / 0.25 | identical | 0/0/0/0/0/16 |

exact: geometric mean speed-up over sim 1.14x (45 programs).
turbo: geometric mean speed-up over sim 1.41x (45 programs).
