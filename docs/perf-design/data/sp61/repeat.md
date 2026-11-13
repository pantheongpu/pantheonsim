Host-GPU benchmark: nvidia/a10g, 10 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactw s | x sim | exactw card / all launches | exactw card s / sim s | exactw output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbow s | x sim | turbow card / all launches | turbow card s / sim s | turbow output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/3MM | 2.33 | 1.81 | 1.29x | 2 / 3 | - | identical | 1.86 | 1.25x | 2 / 3 | - | identical | 1.80 | 1.29x | 2 / 3 | 0.00 / 0.39 | identical | 1.84 | 1.27x | 2 / 3 | 0.00 / 0.39 | identical | 0/0/0/0/0/0 |
| rodinia/gaussian | 4.09 | 0.92 | 4.46x | 495 / 1022 | - | identical | 0.92 | 4.45x | 495 / 1022 | - | identical | 0.92 | 4.43x | 495 / 1022 | 0.18 / 0.27 | identical | 0.92 | 4.43x | 495 / 1022 | 0.18 / 0.27 | identical | 0/0/0/0/0/523 |
| rodinia/srad_v2 | 0.06 | 0.06 | 1.00x | 0 / 1 | - | identical | 0.06 | 0.99x | 0 / 1 | - | identical | 0.06 | 0.96x | 0 / 1 | 0.00 / 0.00 | identical | 0.06 | 0.97x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| rodinia/particlefilter_float | 3.09 | 2.96 | 1.04x | 0 / 36 | - | different (abs 0.729, rel 0.843) | 3.31 | 0.93x | 0 / 36 | - | different (abs 0.0433, rel 0.463) | 3.07 | 1.00x | 0 / 36 | 0.00 / 3.27 | different (abs 0.169, rel 0.593) | 2.99 | 1.03x | 0 / 36 | 0.00 / 2.96 | different (abs 1.18, rel 0.353) | 0/0/0/0/36/0 |
| rodinia/bfs | 0.01 | 0.01 | 0.98x | 0 / 26 | - | identical | 0.01 | 0.99x | 0 / 26 | - | identical | 0.01 | 0.98x | 0 / 26 | 0.00 / 0.01 | identical | 0.01 | 0.99x | 0 / 26 | 0.00 / 0.01 | identical | 0/0/0/0/0/22 |

exact: geometric mean speed-up over sim 1.42x (5 programs).
exactw: geometric mean speed-up over sim 1.39x (5 programs).
turbo: geometric mean speed-up over sim 1.40x (5 programs).
turbow: geometric mean speed-up over sim 1.41x (5 programs).

Notes:

* rodinia/particlefilter_float: the sim runs do not print the same output twice: its comparisons mean little
