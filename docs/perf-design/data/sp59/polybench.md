Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 1200 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small | verify words different / outside (largest) | verify kernels outside | verify output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | 0.96 | 0.96 | 0.99x | 0 / 1 | 0.00 / 0.38 | identical | 0.96 | 0.99x | 0 / 1 | - | identical | 0.96 | 0.99x | 0 / 1 | 0.00 / 0.38 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/2MM | 251.16 | 151.79 | 1.66x | 1 / 2 | - | identical | 149.92 | 1.68x | 1 / 2 | - | identical | 152.05 | 1.65x | 1 / 2 | 0.05 / 29.15 | identical | 0/0/0/0/0/0 | TIMEOUT | - | - |
| polybench/3DCONV | 0.76 | 0.77 | 0.99x | 0 / 254 | 0.00 / 0.53 | identical | 0.77 | 0.99x | 0 / 254 | - | identical | 0.77 | 0.99x | 0 / 254 | 0.00 / 0.51 | identical | 0/0/0/0/0/252 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/3MM | 2.42 | 1.82 | 1.33x | 2 / 3 | 0.00 / 0.39 | identical | 1.83 | 1.32x | 2 / 3 | - | identical | 1.79 | 1.35x | 2 / 3 | 0.00 / 0.39 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/ADI | 0.39 | 0.39 | 1.00x | 0 / 2049 | 0.00 / 0.33 | identical | 0.40 | 0.99x | 0 / 2049 | - | identical | 0.40 | 0.99x | 0 / 2049 | 0.00 / 0.31 | identical | 0/0/0/0/0/2041 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/ATAX | 1.08 | 1.08 | 1.00x | 0 / 2 | 0.00 / 1.00 | identical | 1.08 | 1.00x | 0 / 2 | - | identical | 1.08 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/BICG | 0.26 | 0.26 | 1.00x | 0 / 2 | 0.00 / 0.12 | identical | 0.26 | 1.01x | 0 / 2 | - | identical | 0.26 | 1.00x | 0 / 2 | 0.00 / 0.12 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/CORR | 78.11 | 63.28 | 1.23x | 0 / 4 | - | identical | 65.13 | 1.20x | 0 / 4 | - | identical | 62.73 | 1.25x | 0 / 4 | 0.00 / 23.35 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/COVAR | 62.82 | 62.79 | 1.00x | 0 / 3 | - | identical | 63.52 | 0.99x | 0 / 3 | - | identical | 62.94 | 1.00x | 0 / 3 | 0.00 / 23.29 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/FDTD-2D | 106.05 | 4.64 | 22.88x | 1494 / 1500 | - | identical | 4.89 | 21.69x | 1494 / 1500 | - | identical | 4.64 | 22.86x | 1494 / 1500 | 0.21 / 0.40 | identical | 0/0/0/0/0/0 | TIMEOUT | - | - |
| polybench/GEMM | 0.84 | 0.81 | 1.04x | 0 / 1 | 0.00 / 0.44 | identical | 0.85 | 0.99x | 0 / 1 | - | identical | 0.86 | 0.97x | 0 / 1 | 0.00 / 0.44 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/GEMVER | 0.65 | 0.65 | 1.00x | 0 / 3 | 0.00 / 0.42 | identical | 0.65 | 1.00x | 0 / 3 | - | identical | 0.65 | 1.00x | 0 / 3 | 0.00 / 0.42 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/GESUMMV | 0.39 | 0.38 | 1.01x | 0 / 1 | 0.00 / 0.18 | identical | 0.38 | 1.01x | 0 / 1 | - | identical | 0.38 | 1.00x | 0 / 1 | 0.00 / 0.19 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/GRAMSCHM | 228.56 | 274.90 | 0.83x | 2041 / 6144 | - | identical | 268.18 | 0.85x | 2041 / 6144 | - | identical | 261.77 | 0.87x | 2043 / 6144 | 21.52 / 1.25 | identical | 0/0/0/0/0/4095 | TIMEOUT | - | - |
| polybench/JACOBI1D | 2.41 | 2.46 | 0.98x | 0 / 20000 | 0.00 / 2.44 | identical | 2.45 | 0.99x | 0 / 20000 | - | identical | 2.45 | 0.98x | 0 / 20000 | 0.00 / 2.31 | identical | 0/0/0/0/0/19996 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/JACOBI2D | 0.62 | 0.55 | 1.12x | 22 / 40 | 0.00 / 0.26 | identical | 0.55 | 1.12x | 22 / 40 | - | identical | 0.60 | 1.03x | 22 / 40 | 0.00 / 0.26 | identical | 0/0/0/0/0/14 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/LU | 30.68 | 11.10 | 2.76x | 1405 / 4096 | - | identical | 11.11 | 2.76x | 1405 / 4096 | - | identical | 11.14 | 2.75x | 1405 / 4096 | 4.54 / 0.92 | identical | 0/0/0/0/0/2687 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/MVT | 1.21 | 1.21 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 1.21 | 1.00x | 0 / 2 | - | identical | 1.21 | 1.00x | 0 / 2 | 0.00 / 0.98 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/SYR2K | 9.77 | 9.77 | 1.00x | 0 / 1 | 0.00 / 7.74 | identical | 9.77 | 1.00x | 0 / 1 | - | identical | 9.78 | 1.00x | 0 / 1 | 0.00 / 7.73 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| polybench/SYRK | 6.48 | 5.19 | 1.25x | 0 / 1 | 0.00 / 4.17 | identical | 5.20 | 1.25x | 0 / 1 | - | identical | 5.19 | 1.25x | 0 / 1 | 0.00 / 4.16 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |

exact: geometric mean speed-up over sim 1.30x (20 programs).
exactdiv: geometric mean speed-up over sim 1.30x (20 programs).
turbo: geometric mean speed-up over sim 1.30x (20 programs).
