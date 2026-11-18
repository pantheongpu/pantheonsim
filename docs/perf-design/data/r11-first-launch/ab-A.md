Host-GPU benchmark: nvidia/a10g, 9 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | turborules s | x sim | turborules card / all launches | turborules card s / sim s | turborules output | turbof0 s | x sim | turbof0 card / all launches | turbof0 card s / sim s | turbof0 output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples/mergeSort | 8.88 | 0.58 | 15.17x | 43 / 49 | 0.08 / 0.02 | identical | 2.30 | 3.86x | 42 / 49 | 0.06 / 1.80 | identical |
| samples/MC_EstimatePiInlineP | 4.15 | 0.29 | 14.54x | 2 / 2 | 0.00 / 0.00 | identical | 4.09 | 1.02x | 0 / 2 | 0.00 / 4.10 | identical |
| samples/MC_SingleAsianOptionP | 4.49 | 0.77 | 5.85x | 1 / 3 | 0.02 / 0.34 | identical | 4.09 | 1.10x | 0 / 3 | 0.00 / 3.42 | identical |
| samples/histogram | 11.70 | 2.14 | 5.46x | 62 / 68 | 0.10 / 0.78 | identical | 2.13 | 5.49x | 62 / 68 | 0.09 / 0.62 | identical |
| samples/warpAggregatedAtomicsCG | 2.04 | 2.05 | 1.00x | 0 / 3 | 0.00 / 1.78 | identical | 2.05 | 0.99x | 0 / 3 | 0.00 / 1.63 | identical |

turborules: geometric mean speed-up over sim 5.88x (5 programs).
turbof0: geometric mean speed-up over sim 1.88x (5 programs).
