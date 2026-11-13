Host-GPU benchmark: nvidia/a10g, 1 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default, VGPU_HOST_GPU_START_COST=0.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | 0.96 | 0.93 | 1.03x | 1 / 1 | 0.04 / 0.00 | identical | 0.88 | 1.09x | 1 / 1 | 0.04 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/3DCONV | 0.76 | 0.76 | 1.00x | 0 / 254 | 0.00 / 0.51 | identical | 0.76 | 1.00x | 0 / 254 | 0.00 / 0.50 | identical | 0/0/0/0/0/252 |
| polybench/3MM | 2.30 | 1.42 | 1.62x | 3 / 3 | 0.00 / 0.00 | identical | 1.44 | 1.60x | 3 / 3 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/ADI | 0.39 | 0.40 | 0.99x | 0 / 2049 | 0.00 / 0.32 | identical | 0.40 | 0.98x | 0 / 2049 | 0.00 / 0.32 | identical | 0/0/0/0/0/2041 |
| polybench/ATAX | 1.08 | 1.08 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 1.08 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 0/0/0/0/0/0 |
| polybench/BICG | 0.26 | 0.26 | 1.01x | 0 / 2 | 0.00 / 0.13 | identical | 0.27 | 0.97x | 0 / 2 | 0.00 / 0.13 | identical | 0/0/0/0/0/0 |
| polybench/FDTD-2D | 108.79 | 4.64 | 23.43x | 1494 / 1500 | 0.23 / 0.42 | identical | 4.62 | 23.55x | 1494 / 1500 | 0.22 / 0.42 | identical | 0/0/0/0/0/0 |
| polybench/GEMM | 0.85 | 0.65 | 1.30x | 1 / 1 | 0.00 / 0.00 | identical | 0.67 | 1.27x | 1 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/GEMVER | 0.76 | 0.73 | 1.03x | 1 / 3 | 0.03 / 0.13 | identical | 0.72 | 1.05x | 1 / 3 | 0.03 / 0.14 | identical | 0/0/0/0/0/0 |
| polybench/GESUMMV | 0.38 | 0.38 | 1.01x | 0 / 1 | 0.00 / 0.18 | identical | 0.38 | 1.00x | 0 / 1 | 0.00 / 0.18 | identical | 0/0/0/0/0/0 |
| polybench/JACOBI1D | 2.43 | 2.46 | 0.99x | 0 / 20000 | 0.00 / 2.45 | identical | 2.46 | 0.99x | 0 / 20000 | 0.00 / 2.32 | identical | 0/0/0/0/0/19996 |
| polybench/JACOBI2D | 0.62 | 0.37 | 1.70x | 36 / 40 | 0.01 / 0.06 | identical | 0.36 | 1.73x | 36 / 40 | 0.01 / 0.06 | identical | 0/0/0/0/0/0 |
| polybench/LU | 46.51 | 12.94 | 3.59x | 1407 / 4096 | 4.57 / 0.81 | identical | 11.03 | 4.22x | 1407 / 4096 | 4.55 / 0.79 | identical | 0/0/0/0/0/2685 |
| polybench/MVT | 1.36 | 1.28 | 1.06x | 0 / 2 | 0.00 / 0.98 | identical | 1.26 | 1.08x | 0 / 2 | 0.00 / 0.99 | identical | 0/0/0/0/0/0 |
| polybench/SYR2K | 9.78 | 2.36 | 4.15x | 1 / 1 | 0.02 / 0.00 | identical | 2.30 | 4.25x | 1 / 1 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/SYRK | 5.19 | 1.29 | 4.04x | 1 / 1 | 0.01 / 0.00 | identical | 1.28 | 4.05x | 1 / 1 | 0.01 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/backprop | 0.61 | 0.70 | 0.87x | 1 / 2 | 0.02 / 0.27 | identical | 0.44 | 1.37x | 2 / 2 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/gaussian | 4.12 | 0.71 | 5.83x | 509 / 1022 | 0.18 / 0.05 | identical | 0.72 | 5.70x | 509 / 1022 | 0.19 / 0.05 | identical | 0/0/0/0/0/509 |
| rodinia/lud | 3.57 | 2.90 | 1.23x | 53 / 190 | 0.04 / 0.28 | identical | 2.76 | 1.29x | 102 / 190 | 0.04 / 0.14 | identical | 0/0/0/0/0/82 |
| rodinia/needle | 0.36 | 0.36 | 1.01x | 0 / 255 | 0.00 / 0.40 | identical | 0.37 | 0.99x | 0 / 255 | 0.00 / 0.32 | identical | 0/0/0/0/0/251 |
| rodinia/pathfinder | 1.08 | 1.27 | 0.85x | 3 / 5 | 0.02 / 0.09 | identical | 1.26 | 0.86x | 3 / 5 | 0.02 / 0.09 | identical | 0/0/0/0/0/0 |
| rodinia/srad_v2 | 0.06 | 0.06 | 0.98x | 0 / 1 | 0.00 / 0.00 | identical | 0.44 | 0.14x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/particlefilter_float | 2.56 | 2.60 | 0.98x | 0 / 36 | 0.00 / 2.81 | different (abs 0.126, rel 0.167) | 2.56 | 1.00x | 0 / 36 | 0.00 / 2.55 | different (abs 0.104, rel 0.167) | 0/0/0/0/36/0 |
| rodinia/sc_gpu | 1.01 | 0.56 | 1.79x | 1430 / 1432 | 0.13 / 0.00 | identical | 0.57 | 1.77x | 1430 / 1432 | 0.12 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/hotspot | 1.18 | 1.19 | 1.00x | 0 / 1000 | 0.00 / 1.65 | identical | 0.33 | 3.61x | 998 / 1000 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/bfs | 0.01 | 0.01 | 1.01x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 1.00x | 0 / 26 | 0.00 / 0.01 | identical | 0/0/0/0/0/22 |
| rodinia/nn | 0.02 | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0.02 | 0.99x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/lavaMD | 1.79 | 1.80 | 1.00x | 0 / 1 | 0.00 / 1.77 | identical | 1.80 | 1.00x | 0 / 1 | 0.00 / 1.77 | identical | 0/0/0/0/0/0 |

exact: geometric mean speed-up over sim 1.47x (28 programs).
turbo: geometric mean speed-up over sim 1.47x (28 programs).

Notes:

* rodinia/srad_v2 turbo: _Z11srad_cuda_1PfS_S_S_S_S_iif failed on the card (error 700); mode off, the simulator runs it
