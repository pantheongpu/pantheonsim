Host-GPU benchmark: nvidia/a10g, 5 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbod0 s | x sim | turbod0 card / all launches | turbod0 card s / sim s | turbod0 output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/3MM | 2.18 | 1.63 | 1.33x | 2 / 3 | 0.00 / 0.40 | identical | 1.64 | 1.33x | 2 / 3 | 0.00 / 0.39 | identical | 1.64 | 1.33x | 2 / 3 | 0.00 / 0.39 | identical | 0/0/0/0/0/0 |
| rodinia/lud | 3.37 | 2.96 | 1.14x | 47 / 190 | 0.03 / 0.45 | identical | 2.80 | 1.20x | 170 / 190 | 0.01 / 0.27 | identical | 2.69 | 1.25x | 170 / 190 | 0.01 / 0.27 | identical | 0/0/0/0/0/14 |
| hecbench/radixsort2-cuda | 2.92 | 2.94 | 0.99x | 0 / 600 | 0.00 / 3.29 | identical | 0.58 | 5.04x | 455 / 600 | 0.04 / 0.28 | identical | 0.58 | 5.04x | 455 / 600 | 0.04 / 0.27 | identical | 0/0/0/0/0/139 |

exact: geometric mean speed-up over sim 1.15x (3 programs).
turbo: geometric mean speed-up over sim 2.01x (3 programs).
turbod0: geometric mean speed-up over sim 2.03x (3 programs).
