Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactl0 s | x sim | exactl0 card / all launches | exactl0 card s / sim s | exactl0 output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbol0 s | x sim | turbol0 card / all launches | turbol0 card s / sim s | turbol0 output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2MM | 212.40 | 133.42 | 1.59x | 2 / 2 | - | identical | 98.52 | 2.16x | 2 / 2 | - | identical | 100.81 | 2.11x | 2 / 2 | 0.07 / 0.00 | identical | 105.11 | 2.02x | 2 / 2 | 0.07 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/3MM | 2.47 | 1.73 | 1.43x | 2 / 3 | 0.00 / 0.47 | identical | 1.79 | 1.38x | 2 / 3 | - | identical | 1.76 | 1.40x | 2 / 3 | 0.00 / 0.53 | identical | 1.75 | 1.41x | 2 / 3 | 0.00 / 0.50 | identical | 0/0/0/0/0/0 |
| polybench/CORR | 62.21 | 38.07 | 1.63x | 1 / 4 | - | identical | 36.46 | 1.71x | 1 / 4 | - | identical | 37.65 | 1.65x | 1 / 4 | 1.69 / 0.18 | identical | 38.62 | 1.61x | 1 / 4 | 1.69 / 0.18 | identical | 0/0/0/0/0/0 |
| polybench/COVAR | 58.06 | 36.92 | 1.57x | 1 / 3 | - | identical | 36.66 | 1.58x | 1 / 3 | - | identical | 37.12 | 1.56x | 1 / 3 | 1.69 / 0.03 | identical | 36.38 | 1.60x | 1 / 3 | 1.69 / 0.03 | identical | 0/0/0/0/0/0 |
| polybench/FDTD-2D | 103.87 | 4.73 | 21.96x | 1494 / 1500 | - | identical | 4.63 | 22.43x | 1494 / 1500 | - | identical | 4.76 | 21.83x | 1494 / 1500 | 0.21 / 0.41 | identical | 4.75 | 21.89x | 1494 / 1500 | 0.21 / 0.40 | identical | 0/0/0/0/0/0 |
| polybench/LU | 30.62 | 10.62 | 2.88x | 1406 / 4096 | - | identical | 10.67 | 2.87x | 1413 / 4096 | - | identical | 1.28 | 23.89x | 4085 / 4096 | 0.10 / 0.26 | identical | 1.28 | 23.89x | 4085 / 4096 | 0.10 / 0.26 | identical | 0/0/0/0/2/5 |
| polybench/SYR2K | 9.69 | 2.32 | 4.18x | 1 / 1 | 0.02 / 0.00 | identical | 2.32 | 4.18x | 1 / 1 | - | identical | 2.32 | 4.18x | 1 / 1 | 0.02 / 0.00 | identical | 2.32 | 4.18x | 1 / 1 | 0.02 / 0.00 | identical | 0/0/0/0/0/0 |
| polybench/SYRK | 5.12 | 1.29 | 3.96x | 1 / 1 | 0.01 / 0.00 | identical | 1.29 | 3.97x | 1 / 1 | - | identical | 1.29 | 3.97x | 1 / 1 | 0.01 / 0.00 | identical | 1.29 | 3.97x | 1 / 1 | 0.01 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/gaussian | 4.09 | 0.92 | 4.47x | 495 / 1022 | 0.18 / 0.27 | identical | 0.91 | 4.48x | 495 / 1022 | - | identical | 0.58 | 7.11x | 989 / 1022 | 0.03 / 0.25 | identical | 0.53 | 7.78x | 989 / 1022 | 0.02 / 0.26 | identical | 0/0/0/0/0/29 |
| rodinia/lud | 3.60 | 2.97 | 1.21x | 50 / 190 | 0.04 / 0.45 | identical | 2.94 | 1.22x | 50 / 190 | - | identical | 2.76 | 1.30x | 173 / 190 | 0.01 / 0.28 | identical | 2.69 | 1.34x | 174 / 190 | 0.01 / 0.28 | identical | 0/0/0/0/0/11 |
| rodinia/particlefilter_float | 3.16 | 2.24 | 1.41x | 7 / 36 | 0.02 / 1.96 | different (abs 1.18, rel 5.5) | 2.27 | 1.39x | 7 / 36 | - | different (abs 1.22, rel 6.75) | 1.04 | 3.06x | 28 / 36 | 0.04 / 0.74 | different (abs 2.37, rel 5.5) | 1.05 | 3.01x | 28 / 36 | 0.04 / 0.70 | different (abs 2.38, rel 7.38) | 0/0/0/0/0/0 |
| hecbench/asyncAllocation-cuda | 3.95 | 3.94 | 1.00x | 0 / 800 | 0.00 / 3.71 | identical | 3.95 | 1.00x | 0 / 800 | - | identical | 1.01 | 3.91x | 743 / 800 | 0.10 / 0.27 | identical | 0.99 | 3.99x | 747 / 800 | 0.10 / 0.25 | identical | 0/0/0/0/0/55 |
| hecbench/atomicAggregate-cuda | 3.58 | 3.58 | 1.00x | 0 / 6 | 0.00 / 3.58 | identical | 3.58 | 1.00x | 0 / 6 | - | identical | 1.46 | 2.45x | 4 / 6 | 0.00 / 1.21 | identical | 1.46 | 2.45x | 4 / 6 | 0.00 / 1.21 | identical | 0/0/0/0/0/0 |
| hecbench/atomicPerf-cuda | 1.17 | 1.17 | 0.99x | 0 / 18 | 0.00 / 1.15 | identical | 1.17 | 0.99x | 0 / 18 | - | identical | 0.80 | 1.46x | 12 / 18 | 0.00 / 0.55 | identical | 0.80 | 1.46x | 12 / 18 | 0.00 / 0.55 | identical | 0/0/0/0/0/0 |
| hecbench/black-scholes-cuda | 25.30 | 25.32 | 1.00x | 0 / 1 | - | identical | 25.30 | 1.00x | 0 / 1 | - | identical | 6.06 | 4.17x | 1 / 1 | 0.39 / 0.00 | identical | 6.02 | 4.21x | 1 / 1 | 0.39 / 0.00 | identical | 0/0/0/0/0/0 |

exact: geometric mean speed-up over sim 2.05x (15 programs).
exactl0: geometric mean speed-up over sim 2.09x (15 programs).
turbo: geometric mean speed-up over sim 3.43x (15 programs).
turbol0: geometric mean speed-up over sim 3.46x (15 programs).

Notes:

* rodinia/particlefilter_float: the sim runs do not print the same output twice: its comparisons mean little
