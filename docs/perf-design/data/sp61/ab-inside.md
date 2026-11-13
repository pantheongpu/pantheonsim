Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactw s | x sim | exactw card / all launches | exactw card s / sim s | exactw output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbow s | x sim | turbow card / all launches | turbow card s / sim s | turbow output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/ATAX | 1.07 | 1.07 | 1.00x | 0 / 2 | 0.00 / 0.97 | identical | 1.07 | 1.00x | 0 / 2 | - | identical | 1.07 | 1.00x | 0 / 2 | 0.00 / 0.97 | identical | 1.08 | 1.00x | 0 / 2 | 0.00 / 0.97 | identical | 0/0/0/0/0/0 |
| polybench/BICG | 0.27 | 0.26 | 1.03x | 0 / 2 | 0.00 / 0.14 | identical | 0.27 | 1.00x | 0 / 2 | - | identical | 0.27 | 1.02x | 0 / 2 | 0.00 / 0.13 | identical | 0.27 | 1.01x | 0 / 2 | 0.00 / 0.14 | identical | 0/0/0/0/0/0 |
| polybench/CORR | 76.96 | 42.21 | 1.82x | 1 / 4 | - | identical | 64.63 | 1.19x | 0 / 4 | - | identical | 41.90 | 1.84x | 1 / 4 | 1.67 / 0.18 | identical | 65.94 | 1.17x | 0 / 4 | 0.00 / 25.54 | identical | 0/0/0/0/0/0 |
| polybench/COVAR | 80.20 | 42.09 | 1.91x | 1 / 3 | - | identical | 64.67 | 1.24x | 0 / 3 | - | identical | 40.91 | 1.96x | 1 / 3 | 1.67 / 0.03 | identical | 64.67 | 1.24x | 0 / 3 | 0.00 / 24.83 | identical | 0/0/0/0/0/0 |
| polybench/GEMVER | 0.65 | 0.66 | 1.00x | 0 / 3 | 0.00 / 0.42 | identical | 0.65 | 1.00x | 0 / 3 | - | identical | 0.65 | 1.00x | 0 / 3 | 0.00 / 0.42 | identical | 0.66 | 0.99x | 0 / 3 | 0.00 / 0.41 | identical | 0/0/0/0/0/0 |
| polybench/GESUMMV | 0.38 | 0.38 | 1.00x | 0 / 1 | 0.00 / 0.19 | identical | 0.38 | 0.99x | 0 / 1 | - | identical | 0.38 | 0.99x | 0 / 1 | 0.00 / 0.18 | identical | 0.38 | 0.99x | 0 / 1 | 0.00 / 0.18 | identical | 0/0/0/0/0/0 |
| polybench/MVT | 1.32 | 1.30 | 1.01x | 0 / 2 | 0.00 / 0.99 | identical | 1.30 | 1.01x | 0 / 2 | - | identical | 1.29 | 1.02x | 0 / 2 | 0.00 / 0.98 | identical | 1.34 | 0.98x | 0 / 2 | 0.00 / 0.97 | identical | 0/0/0/0/0/0 |
| rodinia/sc_gpu | 1.04 | 0.73 | 1.41x | 1158 / 1432 | 0.11 / 0.25 | identical | 0.78 | 1.32x | 1056 / 1432 | - | identical | 0.74 | 1.40x | 1057 / 1432 | 0.10 / 0.25 | identical | 0.78 | 1.32x | 1057 / 1432 | 0.10 / 0.25 | identical | 0/0/0/0/0/373 |
| hecbench/aop-cuda | 1.86 | 1.86 | 1.00x | 0 / 301 | 0.00 / 1.76 | identical | 1.86 | 1.00x | 0 / 301 | - | identical | 1.87 | 1.00x | 0 / 301 | 0.00 / 1.76 | identical | 1.87 | 0.99x | 0 / 301 | 0.00 / 1.76 | identical | 0/201/0/0/0/97 |
| hecbench/bh-cuda | 0.87 | 0.87 | 1.00x | 0 / 17 | 0.00 / 0.87 | identical | 0.88 | 0.99x | 0 / 17 | - | identical | 0.70 | 1.25x | 1 / 17 | 0.00 / 0.45 | identical | 0.75 | 1.16x | 1 / 17 | 0.00 / 0.45 | identical | 0/2/0/0/0/0 |

exact: geometric mean speed-up over sim 1.18x (10 programs).
exactw: geometric mean speed-up over sim 1.07x (10 programs).
turbo: geometric mean speed-up over sim 1.21x (10 programs).
turbow: geometric mean speed-up over sim 1.08x (10 programs).

Kernels turbo mode kept in the simulator, by class and first refused instruction:

* atomics `ATOMG.E.CAS.STRONG.GPU`: 1 kernels (hecbench/bh-cuda)
* atomics `ATOMS.CAST.SPIN.64`: 5 kernels (hecbench/aop-cuda)
