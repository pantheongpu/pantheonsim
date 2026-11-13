Host-GPU benchmark: nvidia/a10g, 10 runs (median wall seconds), timeout 600 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactn s | x sim | exactn card / all launches | exactn card s / sim s | exactn output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbon s | x sim | turbon card / all launches | turbon card s / sim s | turbon output | turbo kept in sim: never/atomics/off/exact/safety/small |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/aop-cuda | 1.87 | 1.87 | 1.00x | 0 / 301 | - | identical | 1.87 | 1.00x | 0 / 301 | - | identical | 0.84 | 2.23x | 291 / 301 | 0.03 / 0.43 | identical | 1.87 | 1.00x | 0 / 301 | 0.00 / 1.80 | identical | 0/0/0/0/0/0 |
| hecbench/asyncAllocation-cuda | 3.86 | 3.87 | 1.00x | 0 / 800 | - | identical | 3.87 | 1.00x | 0 / 800 | - | identical | 0.99 | 3.90x | 745 / 800 | 0.10 / 0.25 | identical | 3.87 | 1.00x | 0 / 800 | 0.00 / 3.57 | identical | 0/0/0/0/0/53 |
| samples/simpleCooperativeGroups | 0.00 | 0.00 | 0.97x | 0 / 1 | - | identical | 0.00 | 1.00x | 0 / 1 | - | identical | 0.00 | 0.94x | 0 / 1 | 0.00 / 0.00 | identical | 0.00 | 1.00x | 0 / 1 | 0.00 / 0.00 | identical | 0/0/0/0/0/0 |
| samples/simplePrintf | 0.00 | 0.00 | 1.00x | 0 / 1 | - | different ('0' against '2') | 0.00 | 1.02x | 0 / 1 | - | different ('0' against '2') | 0.00 | 1.02x | 0 / 1 | 0.00 / 0.00 | different ('0' against '2') | 0.00 | 1.02x | 0 / 1 | 0.00 / 0.00 | different ('0' against '2') | 0/0/0/0/0/0 |
| samples/MC_EstimatePiInlineP | 3.29 | 3.29 | 1.00x | 0 / 2 | - | identical | 3.29 | 1.00x | 0 / 2 | - | identical | 3.53 | 0.93x | 1 / 2 | 0.00 / 3.27 | identical | 3.29 | 1.00x | 0 / 2 | 0.00 / 3.28 | identical | 0/0/0/0/0/0 |
| samples/dct8x8 | 1.08 | 1.22 | 0.89x | 30 / 118 | - | identical | 1.08 | 1.00x | 0 / 118 | - | identical | 1.22 | 0.89x | 36 / 118 | 0.00 / 0.83 | identical | 1.08 | 1.00x | 0 / 118 | 0.00 / 0.97 | identical | 11/0/0/0/0/64 |
| samples/interval | 14.85 | 14.93 | 0.99x | 0 / 100 | - | identical | 14.83 | 1.00x | 0 / 100 | - | identical | 2.59 | 5.74x | 98 / 100 | 0.08 / 0.28 | identical | 2.64 | 5.62x | 98 / 100 | 0.08 / 0.26 | identical | 0/0/0/0/0/0 |
| samples/threadFenceReduction | 0.38 | 0.38 | 0.99x | 0 / 101 | - | identical | 0.38 | 1.00x | 0 / 101 | - | identical | 0.53 | 0.71x | 27 / 101 | 0.00 / 0.25 | identical | 0.38 | 1.00x | 0 / 101 | 0.00 / 0.34 | identical | 0/0/0/0/0/71 |
| samples/LargeKernelParameter | 0.49 | 0.72 | 0.68x | 134 / 2002 | - | identical | 0.51 | 0.96x | 0 / 2002 | - | identical | 0.75 | 0.65x | 1 / 2002 | 0.00 / 0.46 | identical | 0.51 | 0.96x | 0 / 2002 | 0.00 / 0.47 | identical | 0/0/0/0/0/1997 |

exact: geometric mean speed-up over sim 0.94x (9 programs).
exactn: geometric mean speed-up over sim 1.00x (9 programs).
turbo: geometric mean speed-up over sim 1.38x (9 programs).
turbon: geometric mean speed-up over sim 1.21x (9 programs).

Kernels turbo mode kept in the simulator, by class and first refused instruction:

* never `TEX.SCR.LL`: 2 kernels (samples/dct8x8)

Notes:

* samples/simplePrintf: the sim runs do not print the same output twice: its comparisons mean little
