Host-GPU benchmark: nvidia/a10g, 9 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turborules s | x sim | turborules card / all launches | turborules card s / sim s | turborules output | turbof0 s | x sim | turbof0 card / all launches | turbof0 card s / sim s | turbof0 output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/3MM | 2.22 | 1.67 | 1.33x | 2 / 3 | 0.00 / 0.40 | identical | 1.70 | 1.30x | 2 / 3 | 0.00 / 0.40 | identical |
| polybench/ATAX | 1.10 | 1.10 | 1.00x | 0 / 2 | 0.00 / 1.01 | identical | 1.10 | 1.00x | 0 / 2 | 0.00 / 1.00 | identical |
| polybench/GEMM | 0.72 | 0.73 | 0.99x | 0 / 1 | 0.00 / 0.43 | identical | 0.72 | 1.00x | 0 / 1 | 0.00 / 0.43 | identical |
| polybench/MVT | 1.22 | 1.23 | 0.99x | 0 / 2 | 0.00 / 0.98 | identical | 1.22 | 1.00x | 0 / 2 | 0.00 / 1.01 | identical |

turborules: geometric mean speed-up over sim 1.07x (4 programs).
turbof0: geometric mean speed-up over sim 1.07x (4 programs).
