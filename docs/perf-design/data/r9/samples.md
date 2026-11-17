Host-GPU benchmark: nvidia/a10g, 10 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbod0 s | x sim | turbod0 card / all launches | turbod0 card s / sim s | turbod0 output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples/mergeSort | 6.87 | 6.88 | 1.00x | 0 / 49 | 0.00 / 6.73 | identical | 1.89 | 3.64x | 42 / 49 | 0.05 / 1.39 | identical | 2.80 | 2.45x | 34 / 49 | 0.06 / 2.28 | identical | 0/0/0/0/0/2 |
| samples/histogram | 11.49 | 11.50 | 1.00x | 0 / 68 | 0.00 / 10.46 | identical | 2.12 | 5.41x | 62 / 68 | 0.09 / 0.62 | identical | 2.72 | 4.22x | 58 / 68 | 0.11 / 1.24 | identical | 0/0/0/0/0/0 |
| samples/segmentationTreeThrust | 13.95 | 10.84 | 1.29x | 60 / 514 | 0.21 / 9.32 | identical | 4.50 | 3.10x | 284 / 514 | 0.48 / 2.47 | identical | 5.62 | 2.48x | 282 / 514 | 0.46 / 3.64 | identical | 0/0/0/0/0/173 |

exact: geometric mean speed-up over sim 1.09x (3 programs).
turbo: geometric mean speed-up over sim 3.94x (3 programs).
turbod0: geometric mean speed-up over sim 2.95x (3 programs).
