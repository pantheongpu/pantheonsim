Host-GPU benchmark: nvidia/a10g, 9 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turborules s | x sim | turborules card / all launches | turborules card s / sim s | turborules output | turbof0 s | x sim | turbof0 card / all launches | turbof0 card s / sim s | turbof0 output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples/interval | 14.40 | 2.66 | 5.42x | 97 / 100 | 0.08 / 0.37 | identical | 2.68 | 5.37x | 97 / 100 | 0.08 / 0.37 | identical |
| samples/reduction | 5.62 | 0.87 | 6.45x | 192 / 201 | 0.06 / 0.27 | identical | 0.88 | 6.42x | 192 / 201 | 0.06 / 0.26 | identical |
| samples/segmentationTreeThrust | 14.22 | 4.62 | 3.08x | 284 / 514 | 0.48 / 2.48 | identical | 4.58 | 3.10x | 284 / 514 | 0.49 / 2.50 | identical |

turborules: geometric mean speed-up over sim 4.76x (3 programs).
turbof0: geometric mean speed-up over sim 4.74x (3 programs).
