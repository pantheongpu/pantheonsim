Host-GPU benchmark: nvidia/a10g, 3 runs (median wall seconds), timeout 300 s, atol 1e-08, rtol 1e-05, threads default.

| program | sim s | exact s | x sim | exact card / all launches | exact card s / sim s | exact output | exactdiv s | x sim | exactdiv card / all launches | exactdiv card s / sim s | exactdiv output | turbo s | x sim | turbo card / all launches | turbo card s / sim s | turbo output | turbo kept in sim: never/atomics/off/exact/safety/small | verify words different / outside (largest) | verify kernels outside | verify output |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hecbench/adam-cuda | 0.03 | 0.03 | 0.99x | 0 / 1 | 0.00 / 0.02 | identical | 0.03 | 0.98x | 0 / 1 | - | identical | 0.03 | 0.99x | 0 / 1 | 0.00 / 0.02 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/aidw-cuda | TIMEOUT | TIMEOUT | - | - | - | - | TIMEOUT | - | - | - | - | TIMEOUT | - | - | - | - | - | TIMEOUT | - | - |
| hecbench/aligned-types-cuda | TIMEOUT | 8.37 | - | 12974 / 13000 | - | - | 8.31 | - | 12974 / 13000 | - | - | 8.44 | - | 12974 / 13000 | 3.42 / 3.50 | - | 0/0/0/0/0/0 | TIMEOUT | - | - |
| hecbench/aobench-cuda | 0.54 | 0.54 | 1.00x | 0 / 1 | 0.00 / 0.54 | identical | 0.54 | 1.00x | 0 / 1 | - | identical | 0.55 | 0.99x | 0 / 1 | 0.00 / 0.54 | identical | 0/0/0/0/0/0 | 4 / 2 (abs 1.07e+242, rel 3) | 1 | identical |
| hecbench/aop-cuda | 1.86 | 1.86 | 1.00x | 0 / 301 | 0.00 / 1.76 | identical | 1.87 | 1.00x | 0 / 301 | - | identical | 1.87 | 1.00x | 0 / 301 | 0.00 / 1.76 | identical | 0/201/0/0/0/97 | 32648 / 0 (abs 1.62e+33, rel 3.46e-05) | 0 | identical |
| hecbench/asmooth-cuda | 0.25 | 0.25 | 1.00x | 0 / 3 | 0.00 / 0.22 | identical | 0.25 | 1.00x | 0 / 3 | - | identical | 0.25 | 1.00x | 0 / 3 | 0.00 / 0.22 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/asyncAllocation-cuda | 3.86 | 3.88 | 0.99x | 0 / 800 | 0.00 / 3.60 | identical | 3.87 | 1.00x | 0 / 800 | - | identical | 3.87 | 1.00x | 0 / 800 | 0.00 / 3.58 | identical | 0/0/0/0/800/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/atan2-cuda | 0.03 | 0.03 | 1.03x | 0 / 3 | 0.00 / 0.02 | identical | 0.03 | 0.99x | 0 / 3 | - | identical | 0.03 | 0.94x | 0 / 3 | 0.00 / 0.02 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/atomicAggregate-cuda | 3.58 | 3.59 | 1.00x | 0 / 6 | 0.00 / 3.59 | identical | 3.59 | 1.00x | 0 / 6 | - | identical | 1.46 | 2.46x | 4 / 6 | 0.00 / 1.22 | identical | 0/0/0/0/0/0 | 17 / 0 (abs 8.41e-45, rel 0) | 0 | different ('FAIL' against 'PASS') |
| hecbench/atomicCost-cuda | 22.64 rc[-9] | 22.92 rc[-9] | 0.99x | - | - | identical | 22.58 rc[-9] | 1.00x | - | - | identical | 22.81 rc[-9] | 0.99x | - | - | identical | - | - | - | identical |
| hecbench/atomicIntrinsics-cuda | 0.18 | 0.18 | 1.00x | 0 / 4 | 0.00 / 0.17 | identical | 0.18 | 1.00x | 0 / 4 | - | identical | 0.18 | 1.01x | 0 / 4 | 0.00 / 0.17 | identical | 0/0/0/0/0/0 | 1 / 0 (abs 4.48e-43, rel 0) | 0 | different (9 words against 2) |
| hecbench/atomicPerf-cuda | 1.13 | 1.13 | 1.00x | 0 / 18 | 0.00 / 1.11 | identical | 1.13 | 1.00x | 0 / 18 | - | identical | 1.34 | 0.84x | 1 / 18 | 0.00 / 1.09 | identical | 0/6/0/0/0/0 | 32 / 0 (abs 4.58e-05, rel 6.32e-06) | 0 | different ('FAIL' against 'PASS') |
| hecbench/atomicReduction-cuda | 296.99 | 296.80 | 1.00x | 0 / 2100 | - | identical | 296.82 | 1.00x | 0 / 2100 | - | identical | 3.81 | 77.98x | 2090 / 2100 | 1.15 / 1.36 | identical | 0/0/0/0/0/0 | TIMEOUT | - | - |
| hecbench/attention-cuda | 0.03 | 0.04 | 0.97x | 0 / 3 | 0.00 / 0.01 | identical | 0.04 | 0.97x | 0 / 3 | - | identical | 0.03 | 1.00x | 0 / 3 | 0.00 / 0.01 | identical | 0/0/0/0/3/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/axpby-cuda | 0.47 | 0.65 | 0.73x | 6 / 35 | 0.00 / 0.32 | identical | 0.65 | 0.73x | 5 / 35 | - | identical | 0.65 | 0.73x | 5 / 35 | 0.00 / 0.33 | identical | 0/0/0/0/0/28 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/babelstream-cuda | TIMEOUT | 13.46 | - | 1188 / 1214 | - | - | 13.50 | - | 1188 / 1214 | - | - | 13.47 | - | 1188 / 1214 | 2.24 / 8.62 | - | 0/0/0/0/0/0 | TIMEOUT | - | - |
| hecbench/background-subtract-cuda | 0.00 | 0.00 | 0.97x | - | - | identical | 0.00 | 0.97x | - | - | identical | 0.00 | 0.97x | - | - | identical | - | - | - | identical |
| hecbench/backprop-cuda | 0.01 | 0.01 | 1.01x | 0 / 2 | 0.00 / 0.01 | identical | 0.01 | 1.01x | 0 / 2 | - | identical | 0.01 | 1.01x | 0 / 2 | 0.00 / 0.01 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/base64e-cuda | TIMEOUT | 8.75 | - | 514 / 516 | - | - | 8.75 | - | 514 / 516 | - | - | 8.76 | - | 514 / 516 | 2.02 / 0.80 | - | 0/0/0/0/0/0 | TIMEOUT | - | - |
| hecbench/bgmv-cuda | 0.02 | 0.02 | 0.98x | 0 / 12 | 0.00 / 0.01 | identical | 0.02 | 0.98x | 0 / 12 | - | identical | 0.02 | 0.99x | 0 / 12 | 0.00 / 0.01 | identical | 0/0/0/0/12/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/bh-cuda | 0.88 | 0.89 | 0.99x | 0 / 17 | 0.00 / 0.89 | identical | 0.89 | 0.98x | 0 / 17 | - | identical | 0.71 | 1.25x | 1 / 17 | 0.00 / 0.46 | identical | 0/2/0/0/0/0 | 40351 / 12392 (abs 4.17e-07, rel 0.000318) | 2 | identical |
| hecbench/bilateral-cuda | 1.28 | 1.28 | 1.00x | 0 / 3 | 0.00 / 1.18 | identical | 1.28 | 1.00x | 0 / 3 | - | identical | 1.28 | 1.00x | 0 / 3 | 0.00 / 1.18 | identical | 0/0/0/0/0/0 | 642 / 0 (abs 3.05e-05, rel 1.97e-07) | 0 | identical |
| hecbench/binomial-cuda | TIMEOUT | TIMEOUT | - | - | - | - | TIMEOUT | - | - | - | - | 7.45 | - | 998 / 1000 | 0.76 / 6.10 | - | 0/0/0/0/0/0 | TIMEOUT | - | - |
| hecbench/bitonic-sort-cuda | 0.32 | 0.33 | 0.99x | 0 / 136 | 0.00 / 0.33 | identical | 0.32 | 1.00x | 0 / 136 | - | identical | 0.52 | 0.62x | 22 / 136 | 0.00 / 0.25 | identical | 0/0/0/0/0/112 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/black-scholes-cuda | 24.90 | 24.90 | 1.00x | 0 / 1 | 0.00 / 19.65 | identical | 24.90 | 1.00x | 0 / 1 | - | identical | 24.89 | 1.00x | 0 / 1 | 0.00 / 19.69 | identical | 0/0/0/0/0/0 | 1351352 / 0 (abs 2.86e-06, rel 5.61e-07) | 0 | identical |
| hecbench/blockAccess-cuda | 0.02 | 0.02 | 0.99x | 0 / 4 | 0.00 / 0.01 | identical | 0.02 | 0.98x | 0 / 4 | - | identical | 0.02 | 0.99x | 0 / 4 | 0.00 / 0.01 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/perplexity-cuda | 0.34 | 0.33 | 1.02x | 0 / 1 | 0.00 / 0.31 | identical | 0.33 | 1.02x | 0 / 1 | - | identical | 0.34 | 0.99x | 0 / 1 | 0.00 / 0.31 | identical | 0/0/0/0/0/0 | 164972 / 0 (abs 5.22e-08, rel 8.69e-07) | 0 | identical |
| hecbench/radixsort2-cuda | 3.37 | 3.28 | 1.03x | 0 / 600 | 0.00 / 3.34 | identical | 3.29 | 1.03x | 0 / 600 | - | identical | 3.30 | 1.02x | 0 / 600 | 0.00 / 3.28 | identical | 0/0/0/0/600/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/s3d-cuda | 0.36 | 0.37 | 0.98x | 0 / 54 | 0.00 / 0.28 | identical | 0.38 | 0.95x | 0 / 54 | - | identical | 0.37 | 0.97x | 0 / 54 | 0.00 / 0.28 | identical | 0/0/0/0/0/0 | 940032 / 0 (abs 3.52e+13, rel 3.74e-07) | 0 | identical |
| hecbench/shmembench-cuda | 6.15 | 6.15 | 1.00x | 0 / 1 | 0.00 / 6.14 | identical | 6.15 | 1.00x | 0 / 1 | - | identical | 6.15 | 1.00x | 0 / 1 | 0.00 / 6.13 | identical | 0/0/0/0/0/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |
| hecbench/silu-cuda | 0.53 | 0.53 | 0.99x | 0 / 50 | 0.00 / 0.49 | identical | 0.53 | 1.00x | 0 / 50 | - | identical | 0.54 | 0.97x | 24 / 50 | 0.00 / 0.25 | identical | 0/0/0/0/0/16 | 1532110 / 0 (abs 1.79e-07, rel 4.7e-07) | 0 | identical |
| hecbench/sortKV-cuda | 0.28 | 0.27 | 1.02x | 0 / 48 | 0.00 / 0.24 | identical | 0.27 | 1.02x | 0 / 48 | - | identical | 0.27 | 1.02x | 0 / 48 | 0.00 / 0.24 | identical | 0/0/0/0/48/0 | 0 / 0 (abs 0, rel 0) | 0 | identical |

exact: geometric mean speed-up over sim 0.99x (27 programs).
exactdiv: geometric mean speed-up over sim 0.98x (27 programs).
turbo: geometric mean speed-up over sim 1.18x (27 programs).

Kernels turbo mode kept in the simulator, by class and first refused instruction:

* atomics `ATOMG.E.CAS.STRONG.GPU`: 1 kernels (hecbench/bh-cuda)
* atomics `ATOMS.CAST.SPIN`: 3 kernels (hecbench/atomicPerf-cuda)
* atomics `ATOMS.CAST.SPIN.64`: 8 kernels (hecbench/aop-cuda, hecbench/atomicPerf-cuda)
