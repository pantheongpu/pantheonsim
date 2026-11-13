Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/LU | 31.05 | 1.48 | 20.99x | 4085 / 4096 | - | identical | 1.51 | 20.57x | 4085 / 4096 | 0.11 / 0.26 | identical | 11.63 | 2.67x | 1389 / 4096 | 4.66 / 0.97 | identical | 0/0/0/0/2/5 |

exactdiv: geometric mean speed-up over sim 20.99x (1 programs).
turbo: geometric mean speed-up over sim 20.57x (1 programs).
turbon: geometric mean speed-up over sim 2.67x (1 programs).
