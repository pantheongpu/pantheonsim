Host-GPU benchmark: nvidia/a10g, 10 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactn s | x sim | exactn card / all launches | exactn card s / sim s | exactn output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/3DCONV | 0.77 | 0.78 | 0.99x | 0 / 254 | - | identical | 0.78 | 0.99x | 0 / 254 | - | identical | 0.78 | 0.99x | 0 / 254 | 0.00 / 0.50 | identical | 0.78 | 0.98x | 0 / 254 | 0.00 / 0.51 | identical | 0/0/0/0/0/252 |
| polybench/JACOBI2D | 0.63 | 0.56 | 1.13x | 22 / 40 | - | identical | 0.56 | 1.13x | 22 / 40 | - | identical | 0.56 | 1.13x | 22 / 40 | 0.00 / 0.26 | identical | 0.56 | 1.12x | 22 / 40 | 0.00 / 0.26 | identical | 0/0/0/0/0/14 |
| polybench/MVT | 1.23 | 1.23 | 1.00x | 0 / 2 | - | identical | 1.22 | 1.00x | 0 / 2 | - | identical | 1.23 | 1.00x | 0 / 2 | 0.00 / 0.99 | identical | 1.23 | 1.00x | 0 / 2 | 0.00 / 1.03 | identical | 0/0/0/0/0/0 |
| rodinia/backprop | 0.62 | 0.62 | 1.00x | 0 / 2 | - | identical | 0.62 | 1.00x | 0 / 2 | - | identical | 0.72 | 0.85x | 1 / 2 | 0.02 / 0.27 | identical | 0.72 | 0.85x | 1 / 2 | 0.02 / 0.27 | identical | 0/0/0/0/0/0 |
| rodinia/lud | 3.58 | 3.00 | 1.19x | 50 / 190 | - | identical | 3.00 | 1.19x | 50 / 190 | - | identical | 2.77 | 1.29x | 174 / 190 | 0.01 / 0.28 | identical | 2.86 | 1.25x | 95 / 190 | 0.04 / 0.30 | identical | 0/0/0/0/0/10 |
| rodinia/needle | 0.36 | 0.37 | 0.98x | 0 / 255 | - | identical | 0.37 | 0.96x | 0 / 255 | - | identical | 0.37 | 0.98x | 0 / 255 | 0.00 / 0.32 | identical | 0.37 | 0.97x | 0 / 255 | 0.00 / 0.32 | identical | 0/0/0/0/0/251 |
| rodinia/sc_gpu | 1.06 | 0.80 | 1.33x | 1059 / 1432 | - | identical | 0.79 | 1.34x | 1063 / 1432 | - | identical | 0.80 | 1.32x | 1065 / 1432 | 0.11 / 0.25 | identical | 0.76 | 1.39x | 1059 / 1432 | 0.10 / 0.25 | identical | 0/0/0/0/0/365 |
| hecbench/atan2-cuda | 0.03 | 0.03 | 0.98x | 0 / 3 | - | identical | 0.03 | 0.98x | 0 / 3 | - | identical | 0.03 | 0.99x | 0 / 3 | 0.00 / 0.02 | identical | 0.03 | 0.99x | 0 / 3 | 0.00 / 0.02 | identical | 0/0/0/0/0/0 |
| hecbench/attention-cuda | 0.04 | 0.04 | 1.00x | 0 / 3 | - | identical | 0.04 | 0.99x | 0 / 3 | - | identical | 0.04 | 1.00x | 0 / 3 | 0.00 / 0.01 | identical | 0.04 | 1.00x | 0 / 3 | 0.00 / 0.01 | identical | 0/0/0/0/3/0 |
| hecbench/bgmv-cuda | 0.02 | 0.02 | 0.99x | 0 / 12 | - | identical | 0.02 | 0.98x | 0 / 12 | - | identical | 0.02 | 1.00x | 0 / 12 | 0.00 / 0.01 | identical | 0.02 | 1.00x | 0 / 12 | 0.00 / 0.01 | identical | 0/0/0/0/12/0 |

exact: geometric mean speed-up over sim 1.05x (10 programs).
exactn: geometric mean speed-up over sim 1.05x (10 programs).
turbo: geometric mean speed-up over sim 1.05x (10 programs).
turbon: geometric mean speed-up over sim 1.05x (10 programs).
