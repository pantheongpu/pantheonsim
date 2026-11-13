Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactn s | x sim | exactn card / all launches | exactn card s / sim s | exactn output | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/LU | 31.03 | 11.68 | 2.66x | 1390 / 4096 | 4.70 / 0.98 | identical | 11.66 | 2.66x | 1403 / 4096 | - | identical | 1.48 | 20.94x | 4085 / 4095 | - | identical | 1.53 | 20.23x | 4085 / 4095 | 0.10 / 0.26 | identical | 11.64 | 2.67x | 1389 / 4096 | 4.67 / 0.97 | identical | 0/0/0/0/0/5 |
| rodinia/gaussian | 4.07 | 0.94 | 4.32x | 494 / 1022 | 0.19 / 0.29 | identical | 0.94 | 4.34x | 495 / 1022 | - | identical | 0.57 | 7.17x | 989 / 1022 | - | identical | 0.53 | 7.73x | 987 / 1022 | 0.03 / 0.26 | identical | 0.99 | 4.12x | 495 / 1022 | 0.19 / 0.27 | identical | 0/0/0/0/0/31 |
| rodinia/lud | 3.56 | 2.99 | 1.19x | 50 / 190 | 0.05 / 0.44 | identical | 2.92 | 1.22x | 50 / 190 | - | identical | 2.89 | 1.23x | 96 / 190 | - | identical | 2.78 | 1.28x | 174 / 190 | 0.01 / 0.28 | identical | 2.82 | 1.26x | 95 / 190 | 0.04 / 0.30 | identical | 0/0/0/0/0/10 |
| rodinia/sc_gpu | 1.05 | 0.79 | 1.33x | 1179 / 1432 | 0.12 / 0.25 | identical | 0.79 | 1.33x | 1060 / 1432 | - | identical | 0.79 | 1.34x | 1064 / 1432 | - | identical | 0.80 | 1.32x | 1061 / 1432 | 0.10 / 0.25 | identical | 0.77 | 1.38x | 1062 / 1432 | 0.12 / 0.25 | identical | 0/0/0/0/0/369 |
| rodinia/hotspot | 1.19 | 1.19 | 1.00x | 0 / 1000 | 0.00 / 1.68 | identical | 1.20 | 0.99x | 0 / 1000 | - | identical | 1.20 | 0.99x | 0 / 1000 | - | identical | 0.52 | 2.28x | 795 / 1000 | 0.02 / 0.25 | identical | 0.52 | 2.30x | 794 / 1000 | 0.02 / 0.25 | identical | 0/0/0/0/0/203 |

exact: geometric mean speed-up over sim 1.79x (5 programs).
exactn: geometric mean speed-up over sim 1.79x (5 programs).
exactdiv: geometric mean speed-up over sim 3.01x (5 programs).
turbo: geometric mean speed-up over sim 3.60x (5 programs).
turbon: geometric mean speed-up over sim 2.13x (5 programs).
