Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 1200 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small | verify words different / outside (largest) | verify kernels outside | verify output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rodinia/backprop | 0.61 | 0.61 | 0.99x | 0 / 2 | 0.00 / 0.44 | identical | 0.61 | 0.99x | 0 / 2 | - | identical | 0.71 | 0.86x | 1 / 2 | 0.02 / 0.27 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/gaussian | 4.13 | 0.92 | 4.47x | 495 / 1022 | 0.17 / 0.28 | identical | 0.92 | 4.51x | 495 / 1022 | - | identical | 0.92 | 4.49x | 495 / 1022 | 0.18 / 0.28 | identical | 0/0/0/0/0/523 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/lud | 3.52 | 2.96 | 1.19x | 50 / 190 | 0.04 / 0.43 | identical | 2.70 | 1.30x | 95 / 190 | - | identical | 2.86 | 1.23x | 95 / 190 | 0.03 / 0.30 | identical | 0/0/0/0/0/89 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/needle | 0.36 | 0.37 | 0.98x | 0 / 255 | 0.00 / 0.38 | identical | 0.37 | 0.97x | 0 / 255 | - | identical | 0.36 | 0.98x | 0 / 255 | 0.00 / 0.32 | identical | 0/0/0/0/0/251 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/pathfinder | 1.08 | 1.08 | 0.99x | 0 / 5 | 0.00 / 0.22 | identical | 1.08 | 0.99x | 0 / 5 | - | identical | 1.08 | 0.99x | 0 / 5 | 0.00 / 0.22 | identical | 0/0/0/0/0/3 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/srad_v2 | 0.06 | 0.06 | 1.01x | 0 / 1 | 0.00 / 0.00 | identical | 0.06 | 1.01x | 0 / 1 | - | identical | 0.06 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/particlefilter_float | 2.97 | 2.92 | 1.01x | 0 / 36 | 0.00 / 2.95 | different (abs 0.324, rel 0.6) | 3.09 | 0.96x | 0 / 36 | - | different (abs 0.32, rel 0.447) | 2.95 | 1.00x | 0 / 36 | 0.00 / 2.86 | different (abs 0.112, rel 0.298) | 0/0/0/0/36/0 | 0 / 0 (abs 0, rel 0) | 0 | different (abs 0.617, rel 0.2) |
| rodinia/sc_gpu | 1.03 | 0.78 | 1.32x | 1087 / 1432 | 0.11 / 0.25 | identical | 0.78 | 1.32x | 1054 / 1432 | - | identical | 0.74 | 1.39x | 1053 / 1432 | 0.10 / 0.25 | identical | 0/0/0/0/0/377 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/hotspot | 1.18 | 1.19 | 0.99x | 0 / 1000 | 0.00 / 1.53 | identical | 1.20 | 0.99x | 0 / 1000 | - | identical | 0.51 | 2.30x | 789 / 1000 | 0.01 / 0.25 | identical | 0/0/0/0/0/209 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/bfs | 0.01 | 0.01 | 0.96x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 0.96x | 0 / 26 | - | identical | 0.01 | 0.96x | 0 / 26 | 0.00 / 0.01 | identical | 0/0/0/0/0/22 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/nn | 0.02 | 0.02 | 0.99x | 0 / 1 | 0.00 / 0.00 | identical | 0.02 | 1.00x | 0 / 1 | - | identical | 0.02 | 0.99x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| rodinia/lavaMD | 1.80 | 1.80 | 1.00x | 0 / 1 | 0.00 / 1.77 | identical | 1.81 | 1.00x | 0 / 1 | - | identical | 1.81 | 1.00x | 0 / 1 | 0.00 / 1.77 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |

exact: geometric mean speed-up over sim 1.17x (12 programs).
exactdiv: geometric mean speed-up over sim 1.18x (12 programs).
turbo: geometric mean speed-up over sim 1.25x (12 programs).

Notes:

* rodinia/particlefilter_float: the sim runs do not print the same output twice: its comparisons mean little
