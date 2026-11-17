Host-GPU benchmark: nvidia/a10g, 1 runs (median wall seconds), timeout 400 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/GRAMSCHM | 164.04 | 148.76 | 1.10x | 2041 / 6144 | 20.62 / 1.21 | identical | 121.06 | 1.35x | 6118 / 6144 | 2.71 / 0.28 | identical | 0/0/0/0/0/20 |

exact: geometric mean speed-up over sim 1.10x (1 programs).
turbo: geometric mean speed-up over sim 1.35x (1 programs).
