Host-GPU benchmark: nvidia/a10g, 10 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactl0 s | x sim | exactl0 card / all launches | exactl0 card s / sim s | exactl0 output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbol0 s | x sim | turbol0 card / all launches | turbol0 card s / sim s | turbol0 output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| samples/MC_EstimatePiInlineP | 3.28 | 3.28 | 1.00x | 0 / 2 | 0.00 / 3.27 | identical | 3.28 | 1.00x | 0 / 2 | - | identical | 3.28 | 1.00x | 0 / 2 | 0.00 / 3.27 | identical | 3.52 | 0.93x | 1 / 2 | 0.00 / 3.26 | identical | 0/0/0/0/0/0 |
| samples/dct8x8 | 1.08 | 1.08 | 1.00x | 0 / 118 | 0.00 / 0.98 | identical | 1.21 | 0.89x | 31 / 118 | - | identical | 1.08 | 0.99x | 0 / 118 | 0.00 / 0.97 | identical | 1.19 | 0.91x | 37 / 118 | 0.00 / 0.84 | identical | 11/0/0/0/0/99 |
| samples/threadFenceReduction | 0.38 | 0.38 | 0.99x | 0 / 101 | 0.00 / 0.36 | identical | 0.38 | 1.00x | 0 / 101 | - | identical | 0.38 | 0.99x | 0 / 101 | 0.00 / 0.34 | identical | 0.53 | 0.71x | 27 / 101 | 0.00 / 0.25 | identical | 0/0/0/0/0/98 |
| samples/LargeKernelParameter | 0.49 | 0.51 | 0.97x | 0 / 2002 | 0.00 / 0.49 | identical | 0.71 | 0.69x | 163 / 2002 | - | identical | 0.51 | 0.97x | 0 / 2002 | 0.00 / 0.47 | identical | 0.74 | 0.67x | 778 / 2002 | 0.07 / 0.34 | identical | 0/0/0/0/0/1998 |

exact: geometric mean speed-up over sim 0.99x (4 programs).
exactl0: geometric mean speed-up over sim 0.88x (4 programs).
turbo: geometric mean speed-up over sim 0.99x (4 programs).
turbol0: geometric mean speed-up over sim 0.80x (4 programs).

Kernels turbo mode kept in the simulator, by class and first refused instruction:

* never `TEX.SCR.LL`: 2 kernels (samples/dct8x8)
