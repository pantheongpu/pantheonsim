Host-GPU benchmark: nvidia/a10g, 1 runs (median wall seconds), timeout 900 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/GRAMSCHM | 254.84 | 224.96 | 1.13x | 6121 / 6144 | 2.68 / 0.28 | identical | 271.14 | 0.94x | 2042 / 6144 | 22.69 / 1.31 | identical | 0/0/0/0/0/17 |

turbo: geometric mean speed-up over sim 1.13x (1 programs).
turbon: geometric mean speed-up over sim 0.94x (1 programs).
