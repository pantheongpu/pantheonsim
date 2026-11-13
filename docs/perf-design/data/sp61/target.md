Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactw s | x sim | exactw card / all launches | exactw card s / sim s | exactw output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbow s | x sim | turbow card / all launches | turbow card s / sim s | turbow output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/CORR | 65.14 | 42.01 | 1.55x | 1 / 4 | 1.67 / 0.18 | identical | 64.84 | 1.00x | 0 / 4 | - | identical | 42.07 | 1.55x | 1 / 4 | 1.67 / 0.18 | identical | 64.77 | 1.01x | 0 / 4 | 0.00 / 24.80 | identical | 0/0/0/0/0/0 |
| polybench/COVAR | 64.83 | 41.82 | 1.55x | 1 / 3 | 1.67 / 0.03 | identical | 64.65 | 1.00x | 0 / 3 | - | identical | 41.95 | 1.54x | 1 / 3 | 1.67 / 0.03 | identical | 64.39 | 1.01x | 0 / 3 | 0.00 / 24.55 | identical | 0/0/0/0/0/0 |

exact: geometric mean speed-up over sim 1.55x (2 programs).
exactw: geometric mean speed-up over sim 1.00x (2 programs).
turbo: geometric mean speed-up over sim 1.55x (2 programs).
turbow: geometric mean speed-up over sim 1.01x (2 programs).
