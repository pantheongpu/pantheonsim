Host-GPU benchmark: nvidia/a10g, 9 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turborules s | x sim | turborules card / all launches | turborules card s / sim s | turborules output | turbof0 s | x sim | turbof0 card / all launches | turbof0 card s / sim s | turbof0 output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rodinia/pathfinder | 1.03 | 1.04 | 0.98x | 0 / 5 | 0.00 / 0.19 | identical | 1.02 | 1.01x | 0 / 5 | 0.00 / 0.19 | identical |
| rodinia/lavaMD | 1.53 | 1.53 | 1.00x | 0 / 1 | 0.00 / 1.50 | identical | 1.54 | 0.99x | 0 / 1 | 0.00 / 1.50 | identical |

turborules: geometric mean speed-up over sim 0.99x (2 programs).
turbof0: geometric mean speed-up over sim 1.00x (2 programs).
