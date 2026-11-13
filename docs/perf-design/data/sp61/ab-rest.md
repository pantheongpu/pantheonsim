Host-GPU benchmark: nvidia/a10g, 1 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactw s | x sim | exactw card / all launches | exactw card s / sim s | exactw output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbow s | x sim | turbow card / all launches | turbow card s / sim s | turbow output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | 0.96 | 0.96 | 1.00x | 0 / 1 | 0.00 / 0.38 | identical | 0.95 | 1.01x | 0 / 1 | - | identical | 0.95 | 1.01x | 0 / 1 | 0.00 / 0.38 | identical | 0.95 | 1.01x | 0 / 1 | 0.00 / 0.38 | identical | 0/0/0/0/0/0 |
| polybench/3DCONV | 0.76 | 0.76 | 1.00x | 0 / 254 | 0.00 / 0.56 | identical | 0.76 | 1.00x | 0 / 254 | - | identical | 0.76 | 0.99x | 0 / 254 | 0.00 / 0.49 | identical | 0.76 | 1.00x | 0 / 254 | 0.00 / 0.49 | identical | 0/0/0/0/0/252 |
| polybench/3MM | 2.34 | 1.74 | 1.34x | 2 / 3 | 0.00 / 0.39 | identical | 1.77 | 1.32x | 2 / 3 | - | identical | 1.87 | 1.26x | 2 / 3 | 0.00 / 0.39 | identical | 1.77 | 1.33x | 2 / 3 | 0.00 / 0.39 | identical | 0/0/0/0/0/0 |
| polybench/ADI | 0.40 | 0.40 | 1.00x | 0 / 2049 | 0.00 / 0.33 | identical | 0.40 | 1.01x | 0 / 2049 | - | identical | 0.40 | 1.01x | 0 / 2049 | 0.00 / 0.32 | identical | 0.40 | 1.01x | 0 / 2049 | 0.00 / 0.31 | identical | 0/0/0/0/0/2041 |
| polybench/GEMM | 0.83 | 0.85 | 0.97x | 0 / 1 | 0.00 / 0.43 | identical | 0.87 | 0.96x | 0 / 1 | - | identical | 0.78 | 1.06x | 0 / 1 | 0.00 / 0.43 | identical | 0.85 | 0.98x | 0 / 1 | 0.00 / 0.43 | identical | 0/0/0/0/0/0 |
| polybench/JACOBI1D | 2.44 | 2.47 | 0.99x | 0 / 20000 | 0.00 / 2.47 | identical | 2.48 | 0.98x | 0 / 20000 | - | identical | 2.48 | 0.98x | 0 / 20000 | 0.00 / 2.34 | identical | 2.47 | 0.99x | 0 / 20000 | 0.00 / 2.33 | identical | 0/0/0/0/0/19996 |
| polybench/JACOBI2D | 0.61 | 0.55 | 1.11x | 22 / 40 | 0.00 / 0.26 | identical | 0.55 | 1.10x | 22 / 40 | - | identical | 0.56 | 1.09x | 22 / 40 | 0.00 / 0.26 | identical | 0.56 | 1.10x | 22 / 40 | 0.00 / 0.26 | identical | 0/0/0/0/0/14 |
| polybench/SYR2K | 9.76 | 2.37 | 4.11x | 1 / 1 | 0.02 / 0.00 | identical | 2.37 | 4.12x | 1 / 1 | - | identical | 2.32 | 4.21x | 1 / 1 | 0.02 / 0.00 | identical | 2.33 | 4.20x | 1 / 1 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/SYRK | 5.18 | 1.29 | 4.01x | 1 / 1 | 0.01 / 0.00 | identical | 1.34 | 3.85x | 1 / 1 | - | identical | 1.29 | 4.01x | 1 / 1 | 0.01 / 0.00 | identical | 1.29 | 4.01x | 1 / 1 | 0.01 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/backprop | 0.61 | 0.61 | 1.00x | 0 / 2 | 0.00 / 0.44 | identical | 0.61 | 1.00x | 0 / 2 | - | identical | 0.71 | 0.87x | 1 / 2 | 0.02 / 0.26 | identical | 0.76 | 0.81x | 1 / 2 | 0.02 / 0.26 | identical | 0/0/0/0/0/0 |
| rodinia/gaussian | 4.10 | 0.97 | 4.24x | 495 / 1022 | 0.18 / 0.27 | identical | 0.91 | 4.50x | 495 / 1022 | - | identical | 0.93 | 4.39x | 495 / 1022 | 0.18 / 0.28 | identical | 0.99 | 4.16x | 495 / 1022 | 0.18 / 0.27 | identical | 0/0/0/0/0/523 |
| rodinia/lud | 3.59 | 2.97 | 1.21x | 50 / 190 | 0.04 / 0.44 | identical | 3.08 | 1.17x | 50 / 190 | - | identical | 2.71 | 1.32x | 95 / 190 | 0.03 / 0.30 | identical | 2.69 | 1.34x | 95 / 190 | 0.04 / 0.30 | identical | 0/0/0/0/0/89 |
| rodinia/needle | 0.37 | 0.37 | 1.00x | 0 / 255 | 0.00 / 0.48 | identical | 0.36 | 1.02x | 0 / 255 | - | identical | 0.37 | 1.00x | 0 / 255 | 0.00 / 0.32 | identical | 0.37 | 0.99x | 0 / 255 | 0.00 / 0.32 | identical | 0/0/0/0/0/251 |
| rodinia/pathfinder | 1.08 | 1.08 | 1.00x | 0 / 5 | 0.00 / 0.22 | identical | 1.06 | 1.01x | 0 / 5 | - | identical | 1.08 | 1.00x | 0 / 5 | 0.00 / 0.22 | identical | 1.07 | 1.01x | 0 / 5 | 0.00 / 0.22 | identical | 0/0/0/0/0/3 |
| rodinia/srad_v2 | 0.06 | 0.07 | 0.97x | 0 / 1 | 0.00 / 0.00 | identical | 0.06 | 1.01x | 0 / 1 | - | identical | 0.07 | 0.96x | 0 / 1 | 0.00 / 0.00 | identical | 0.07 | 0.97x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/particlefilter_float | 2.93 | 3.09 | 0.95x | 0 / 36 | 0.00 / 3.11 | different (abs 0.161, rel 0.386) | 3.40 | 0.86x | 0 / 36 | - | different (abs 0.464, rel 0.386) | 2.91 | 1.01x | 0 / 36 | 0.00 / 2.89 | different (abs 0.136, rel 0.343) | 2.82 | 1.04x | 0 / 36 | 0.00 / 2.81 | different (abs 0.109, rel 0.386) | 0/0/0/0/36/0 |
| rodinia/hotspot | 1.19 | 1.19 | 1.00x | 0 / 1000 | 0.00 / 1.42 | identical | 1.19 | 1.00x | 0 / 1000 | - | identical | 0.52 | 2.30x | 789 / 1000 | 0.02 / 0.25 | identical | 0.52 | 2.31x | 795 / 1000 | 0.02 / 0.25 | identical | 0/0/0/0/0/209 |
| rodinia/bfs | 0.01 | 0.01 | 1.01x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 1.05x | 0 / 26 | - | identical | 0.01 | 1.03x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 1.04x | 0 / 26 | 0.00 / 0.01 | identical | 0/0/0/0/0/22 |
| rodinia/nn | 0.02 | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0.02 | 1.00x | 0 / 1 | - | identical | 0.02 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0.02 | 0.96x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/lavaMD | 1.78 | 1.78 | 1.00x | 0 / 1 | 0.00 / 1.78 | identical | 1.78 | 1.00x | 0 / 1 | - | identical | 1.78 | 1.00x | 0 / 1 | 0.00 / 1.75 | identical | 1.78 | 1.00x | 0 / 1 | 0.00 / 1.75 | identical | 0/0/0/0/0/0 |

exact: geometric mean speed-up over sim 1.27x (20 programs).
exactw: geometric mean speed-up over sim 1.27x (20 programs).
turbo: geometric mean speed-up over sim 1.33x (20 programs).
turbow: geometric mean speed-up over sim 1.32x (20 programs).
