Host-GPU benchmark: nvidia/a10g, 5 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbos0 s | x sim | turbos0 card / all launches | turbos0 card s / sim s | turbos0 output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | 0.94 | 0.94 | 1.00x | 0 / 1 | 0.00 / 0.36 | identical | 0.94 | 1.00x | 0 / 1 | 0.00 / 0.36 | identical | 0/0/0/0/0/0 |
| polybench/2MM | 176.00 | 126.32 | 1.39x | 2 / 2 | 0.07 / 0.00 | identical | 124.99 | 1.41x | 2 / 2 | 0.08 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/3DCONV | 0.72 | 0.73 | 0.99x | 0 / 254 | 0.00 / 0.46 | identical | 0.73 | 0.99x | 0 / 254 | 0.00 / 0.46 | identical | 0/0/0/0/0/252 |
| polybench/3MM | 2.20 | 1.67 | 1.32x | 2 / 3 | 0.00 / 0.40 | identical | 1.67 | 1.31x | 2 / 3 | 0.00 / 0.40 | identical | 0/0/0/0/0/0 |
| polybench/ADI | 0.46 | 0.47 | 0.98x | 0 / 2049 | 0.00 / 0.38 | identical | 0.46 | 0.99x | 0 / 2049 | 0.00 / 0.38 | identical | 0/0/0/0/0/2041 |
| polybench/ATAX | 1.09 | 1.08 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 1.09 | 0.99x | 0 / 2 | 0.00 / 0.99 | identical | 0/0/0/0/0/0 |
| polybench/BICG | 0.27 | 0.27 | 0.98x | 0 / 2 | 0.00 / 0.13 | identical | 0.27 | 1.00x | 0 / 2 | 0.00 / 0.12 | identical | 0/0/0/0/0/0 |
| polybench/CORR | 60.10 | 36.96 | 1.63x | 1 / 4 | 1.67 / 0.20 | identical | 36.85 | 1.63x | 1 / 4 | 1.67 / 0.20 | identical | 0/0/0/0/0/0 |
| polybench/COVAR | 57.15 | 36.52 | 1.56x | 1 / 3 | 1.67 / 0.02 | identical | 36.90 | 1.55x | 1 / 3 | 1.67 / 0.02 | identical | 0/0/0/0/0/0 |
| polybench/FDTD-2D | 94.71 | 4.64 | 20.41x | 1494 / 1500 | 0.22 / 0.36 | identical | 4.65 | 20.38x | 1494 / 1500 | 0.21 / 0.37 | identical | 0/0/0/0/0/0 |
| polybench/GEMM | 0.72 | 0.72 | 1.00x | 0 / 1 | 0.00 / 0.43 | identical | 0.73 | 0.99x | 0 / 1 | 0.00 / 0.43 | identical | 0/0/0/0/0/0 |
| polybench/GEMVER | 0.63 | 0.63 | 0.99x | 0 / 3 | 0.00 / 0.39 | identical | 0.68 | 0.93x | 0 / 3 | 0.00 / 0.40 | identical | 0/0/0/0/0/0 |
| polybench/GESUMMV | 0.38 | 0.38 | 1.01x | 0 / 1 | 0.00 / 0.18 | identical | 0.38 | 1.00x | 0 / 1 | 0.00 / 0.18 | identical | 0/0/0/0/0/0 |
| polybench/GRAMSCHM | 172.39 | 123.49 | 1.40x | 6118 / 6144 | 2.69 / 0.28 | identical | 115.56 | 1.49x | 6118 / 6144 | 2.68 / 0.28 | identical | 0/0/0/0/0/20 |
| polybench/JACOBI1D | 2.29 | 1.51 | 1.52x | 11528 / 20000 | 0.18 / 0.91 | identical | 2.34 | 0.98x | 0 / 20000 | 0.00 / 2.20 | identical | 0/0/0/0/0/8468 |
| polybench/JACOBI2D | 0.59 | 0.55 | 1.06x | 21 / 40 | 0.00 / 0.25 | identical | 0.55 | 1.06x | 21 / 40 | 0.00 / 0.26 | identical | 0/0/0/0/0/15 |
| polybench/LU | 27.90 | 1.34 | 20.82x | 4083 / 4096 | 0.10 / 0.29 | identical | 1.34 | 20.83x | 4083 / 4096 | 0.10 / 0.29 | identical | 0/0/0/0/2/7 |
| polybench/MVT | 1.22 | 1.22 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 1.22 | 1.00x | 0 / 2 | 0.00 / 0.99 | identical | 0/0/0/0/0/0 |
| polybench/SYR2K | 9.65 | 2.32 | 4.16x | 1 / 1 | 0.02 / 0.00 | identical | 2.32 | 4.16x | 1 / 1 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/SYRK | 5.19 | 1.30 | 4.00x | 1 / 1 | 0.01 / 0.00 | identical | 1.29 | 4.01x | 1 / 1 | 0.01 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/backprop | 0.68 | 0.79 | 0.86x | 1 / 2 | 0.02 / 0.33 | identical | 0.79 | 0.85x | 1 / 2 | 0.02 / 0.34 | identical | 0/0/0/0/0/0 |
| rodinia/gaussian | 3.80 | 0.53 | 7.16x | 987 / 1022 | 0.02 / 0.25 | identical | 0.53 | 7.11x | 985 / 1022 | 0.02 / 0.26 | identical | 0/0/0/0/0/31 |
| rodinia/lud | 3.26 | 2.75 | 1.19x | 170 / 190 | 0.01 / 0.29 | identical | 2.80 | 1.17x | 170 / 190 | 0.01 / 0.28 | identical | 0/0/0/0/0/14 |
| rodinia/needle | 0.32 | 0.33 | 0.98x | 0 / 255 | 0.00 / 0.28 | identical | 0.33 | 0.98x | 0 / 255 | 0.00 / 0.28 | identical | 0/0/0/0/0/251 |
| rodinia/pathfinder | 1.03 | 1.03 | 1.00x | 0 / 5 | 0.00 / 0.20 | identical | 1.04 | 1.00x | 0 / 5 | 0.00 / 0.19 | identical | 0/0/0/0/0/3 |
| rodinia/srad_v2 | 0.07 | 0.06 | 1.02x | 0 / 1 | 0.00 / 0.00 | identical | 0.07 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/particlefilter_float | 2.93 | 0.99 | 2.97x | 28 / 36 | 0.04 / 0.40 | different (abs 2.29, rel 5.75) | 0.93 | 3.13x | 28 / 36 | 0.04 / 0.72 | different (abs 2.41, rel 5.5) | 0/0/0/0/0/0 |
| rodinia/sc_gpu | 0.95 | 0.85 | 1.12x | 813 / 1432 | 0.08 / 0.38 | identical | 0.96 | 0.99x | 0 / 1432 | 0.00 / 0.86 | identical | 0/0/0/0/0/617 |
| rodinia/hotspot | 1.34 | 0.56 | 2.40x | 787 / 1000 | 0.01 / 0.28 | identical | 0.56 | 2.40x | 777 / 1000 | 0.01 / 0.30 | identical | 0/0/0/0/0/211 |
| rodinia/bfs | 0.01 | 0.01 | 0.97x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 0.97x | 0 / 26 | 0.00 / 0.01 | identical | 0/0/0/0/0/22 |
| rodinia/nn | 0.02 | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/lavaMD | 1.56 | 1.56 | 1.00x | 0 / 1 | 0.00 / 1.53 | identical | 1.56 | 1.00x | 0 / 1 | 0.00 / 1.53 | identical | 0/0/0/0/0/0 |

turbo: geometric mean speed-up over sim 1.61x (32 programs).
turbos0: geometric mean speed-up over sim 1.58x (32 programs).

Notes:

* rodinia/particlefilter_float: the sim runs do not print the same output twice: its comparisons mean little
