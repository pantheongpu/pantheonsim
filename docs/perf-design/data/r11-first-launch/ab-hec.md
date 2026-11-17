Host-GPU benchmark: nvidia/a10g, 9 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turborules s | x sim | turborules card / all launches | turborules card s / sim s | turborules output | turbof0 s | x sim | turbof0 card / all launches | turbof0 card s / sim s | turbof0 output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/aop-cuda | 1.98 | 0.90 | 2.19x | 291 / 301 | 0.03 / 0.51 | identical | 0.93 | 2.14x | 291 / 301 | 0.03 / 0.52 | identical |
| hecbench/bilateral-cuda | 1.90 | 1.57 | 1.21x | 0 / 3 | 0.00 / 1.37 | identical | 1.51 | 1.26x | 0 / 3 | 0.00 / 1.36 | identical |
| hecbench/radixsort2-cuda | 3.53 | 0.58 | 6.10x | 558 / 600 | 0.03 / 0.25 | identical | 0.57 | 6.14x | 552 / 600 | 0.03 / 0.28 | identical |
| hecbench/silu-cuda | 0.65 | 0.60 | 1.08x | 25 / 50 | 0.00 / 0.30 | identical | 0.64 | 1.02x | 25 / 50 | 0.00 / 0.31 | identical |

turborules: geometric mean speed-up over sim 2.05x (4 programs).
turbof0: geometric mean speed-up over sim 2.02x (4 programs).
