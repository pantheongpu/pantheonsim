Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 900 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactl0 s | x sim | exactl0 card / all launches | exactl0 card s / sim s | exactl0 output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2MM | 150.55 | 97.63 | 1.54x | 2 / 2 | 0.07 / 0.00 | identical | 97.29 | 1.55x | 2 / 2 | - | identical |
| polybench/CORR | 58.29 | 37.20 | 1.57x | 1 / 4 | 1.69 / 0.17 | identical | 36.84 | 1.58x | 1 / 4 | - | identical |

exact: geometric mean speed-up over sim 1.55x (2 programs).
exactl0: geometric mean speed-up over sim 1.56x (2 programs).
