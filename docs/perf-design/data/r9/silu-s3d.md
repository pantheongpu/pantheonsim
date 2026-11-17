Host-GPU benchmark: nvidia/a10g, 15 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo0 s | x sim | turbo0 card / all launches | turbo0 card s / sim s | turbo0 output | turbod0 s | x sim | turbod0 card / all launches | turbod0 card s / sim s | turbod0 output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/s3d-cuda | 0.36 | 0.39 | 0.93x | 0 / 54 | 0.00 / 0.29 | identical | 0.39 | 0.93x | 0 / 54 | 0.00 / 0.29 | identical | 0.39 | 0.93x | 0 / 54 | 0.00 / 0.28 | identical | 0.39 | 0.93x | 0 / 54 | 0.00 / 0.29 | identical | 0/0/0/0/0/0 |
| hecbench/silu-cuda | 0.65 | 0.65 | 1.00x | 0 / 50 | 0.00 / 0.61 | identical | 0.59 | 1.09x | 25 / 50 | 0.00 / 0.31 | identical | 0.58 | 1.11x | 25 / 50 | 0.00 / 0.31 | identical | 0.59 | 1.10x | 25 / 50 | 0.00 / 0.31 | identical | 0/0/0/0/0/15 |

exact: geometric mean speed-up over sim 0.97x (2 programs).
turbo: geometric mean speed-up over sim 1.01x (2 programs).
turbo0: geometric mean speed-up over sim 1.02x (2 programs).
turbod0: geometric mean speed-up over sim 1.01x (2 programs).
