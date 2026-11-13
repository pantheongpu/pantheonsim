Host-GPU benchmark: nvidia/a10g, 1 runs (median wall seconds), timeout 900 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactn s | x sim | exactn card / all launches | exactn card s / sim s | exactn output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/GRAMSCHM | 283.97 | 265.71 | 1.07x | 2041 / 6144 | - | identical | 279.32 | 1.02x | 2041 / 6144 | - | identical | 260.67 | 1.09x | 4480 / 6144 | 13.90 / 0.76 | identical | 282.09 | 1.01x | 2046 / 6144 | 26.51 / 1.49 | identical | 0/0/0/0/0/1658 |

exact: geometric mean speed-up over sim 1.07x (1 programs).
exactn: geometric mean speed-up over sim 1.02x (1 programs).
turbo: geometric mean speed-up over sim 1.09x (1 programs).
turbon: geometric mean speed-up over sim 1.01x (1 programs).
