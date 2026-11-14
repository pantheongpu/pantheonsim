Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactn s | x sim | exactn card / all launches | exactn card s / sim s | exactn output | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/LU | 31.27 | 13.04 | 2.40x | 1386 / 4096 | 4.72 / 0.99 | identical | 13.11 | 2.39x | 1380 / 4096 | - | identical | 1.78 | 17.56x | 4085 / 4095 | - | identical | 1.74 | 17.94x | 4085 / 4095 | 0.10 / 0.26 | identical | 13.00 | 2.41x | 1389 / 4096 | 4.71 / 0.97 | identical | 0/0/0/0/0/5 |
| rodinia/gaussian | 4.10 | 0.99 | 4.13x | 495 / 1022 | 0.19 / 0.28 | identical | 0.94 | 4.38x | 495 / 1022 | - | identical | 0.52 | 7.89x | 989 / 1022 | - | identical | 0.53 | 7.70x | 989 / 1022 | 0.02 / 0.25 | identical | 0.97 | 4.24x | 495 / 1022 | 0.19 / 0.28 | identical | 0/0/0/0/0/29 |
| rodinia/lud | 3.56 | 2.97 | 1.20x | 50 / 190 | 0.05 / 0.46 | identical | 3.02 | 1.18x | 50 / 190 | - | identical | 2.91 | 1.23x | 95 / 190 | - | identical | 2.78 | 1.28x | 174 / 190 | 0.01 / 0.28 | identical | 2.81 | 1.27x | 95 / 190 | 0.04 / 0.30 | identical | 0/0/0/0/0/10 |

exact: geometric mean speed-up over sim 2.28x (3 programs).
exactn: geometric mean speed-up over sim 2.31x (3 programs).
exactdiv: geometric mean speed-up over sim 5.54x (3 programs).
turbo: geometric mean speed-up over sim 5.62x (3 programs).
turbon: geometric mean speed-up over sim 2.35x (3 programs).
