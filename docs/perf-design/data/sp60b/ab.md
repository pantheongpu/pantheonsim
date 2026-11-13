| program | sim s | exact s | exact0 s | exact0 / exact | turbo s | turbo0 s | turbo0 / turbo | sampled, sent (exact; turbo) | on card of launches (exact; turbo) | output (exact; turbo) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| rodinia/backprop | 0.61 | 0.61 | 0.61 | 1.00x | 0.76 | 0.71 | 0.93x | 0, 0; 0, 0 | 0 of 2; 1 of 2 | identical; identical |
| rodinia/gaussian | 4.12 | 0.95 | 0.95 | 0.99x | 0.95 | 0.96 | 1.00x | 0, 0; 0, 0 | 495 of 1022; 495 of 1022 | identical; identical |
| rodinia/lud | 3.58 | 2.98 | 3.03 | 1.02x | 2.85 | 2.90 | 1.02x | 0, 0; 0, 0 | 50 of 190; 95 of 190 | identical; identical |
| rodinia/needle | 0.36 | 0.36 | 0.37 | 1.01x | 0.36 | 0.36 | 1.00x | 0, 0; 0, 0 | 0 of 255; 0 of 255 | identical; identical |
| rodinia/pathfinder | 1.07 | 1.07 | 1.07 | 1.00x | 1.07 | 1.08 | 1.01x | 0, 0; 0, 0 | 0 of 5; 0 of 5 | identical; identical |
| rodinia/srad_v2 | 0.06 | 0.06 | 0.06 | 0.99x | 0.06 | 0.06 | 0.98x | 0, 0; 1, 0 | 0 of 1; 0 of 1 | identical; identical |
| rodinia/particlefilter_float | 2.89 | 2.84 | 3.10 | 1.09x | 3.15 | 3.07 | 0.98x | 0, 0; 0, 0 | 0 of 36; 0 of 36 | different; different |
| rodinia/sc_gpu | 1.02 | 0.78 | 0.73 | 0.94x | 0.75 | 0.74 | 0.99x | 0, 0; 0, 0 | 1075 of 1432; 1055 of 1432 | identical; identical |
| rodinia/hotspot | 1.18 | 1.19 | 1.18 | 0.99x | 0.52 | 0.51 | 1.00x | 0, 0; 0, 0 | 0 of 1000; 785 of 1000 | identical; identical |
| rodinia/bfs | 0.01 | 0.01 | 0.01 | 1.00x | 0.01 | 0.01 | 1.01x | 0, 0; 0, 0 | 0 of 26; 0 of 26 | identical; identical |
| rodinia/nn | 0.02 | 0.02 | 0.02 | 1.00x | 0.02 | 0.02 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| rodinia/lavaMD | 1.77 | 1.77 | 1.77 | 1.00x | 1.77 | 1.77 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/2DCONV | 0.96 | 0.96 | 0.96 | 1.00x | 0.96 | 0.96 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/2MM | 240.67 | 122.20 | 171.77 | 1.41x | 122.04 | 154.36 | 1.26x | -; 1, 1 | 2 of 2; 2 of 2 | identical; identical |
| polybench/3DCONV | 0.76 | 0.77 | 0.77 | 1.00x | 0.77 | 0.77 | 1.00x | 0, 0; 0, 0 | 0 of 254; 0 of 254 | identical; identical |
| polybench/3MM | 2.34 | 1.85 | 1.84 | 0.99x | 1.89 | 1.87 | 0.99x | 0, 0; 0, 0 | 2 of 3; 2 of 3 | identical; identical |
| polybench/ADI | 0.40 | 0.39 | 0.40 | 1.01x | 0.40 | 0.40 | 1.00x | 0, 0; 0, 0 | 0 of 2049; 0 of 2049 | identical; identical |
| polybench/ATAX | 1.08 | 1.08 | 1.08 | 1.00x | 1.08 | 1.08 | 1.00x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| polybench/BICG | 0.26 | 0.26 | 0.27 | 1.01x | 0.26 | 0.27 | 1.00x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| polybench/CORR | 63.12 | 63.22 | 63.55 | 1.01x | 63.35 | 63.94 | 1.01x | -; 1, 0 | 0 of 4; 0 of 4 | identical; identical |
| polybench/COVAR | 62.81 | 64.51 | 64.12 | 0.99x | 64.02 | 63.62 | 0.99x | -; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| polybench/FDTD-2D | 105.34 | 4.81 | 4.68 | 0.97x | 4.69 | 4.93 | 1.05x | -; 0, 0 | 1494 of 1500; 1494 of 1500 | identical; identical |
| polybench/GEMM | 0.88 | 0.86 | 0.88 | 1.03x | 0.89 | 0.89 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/GEMVER | 0.66 | 0.66 | 0.66 | 1.00x | 0.66 | 0.66 | 1.00x | 0, 0; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| polybench/GESUMMV | 0.38 | 0.38 | 0.38 | 1.00x | 0.38 | 0.38 | 1.01x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/JACOBI1D | 2.50 | 2.49 | 2.48 | 0.99x | 2.49 | 2.49 | 1.00x | 0, 0; 0, 0 | 0 of 20000; 0 of 20000 | identical; identical |
| polybench/JACOBI2D | 0.62 | 0.60 | 0.55 | 0.91x | 0.56 | 0.55 | 0.99x | 0, 0; 0, 0 | 22 of 40; 22 of 40 | identical; identical |
| polybench/LU | 30.88 | 11.18 | 11.25 | 1.01x | 11.18 | 11.19 | 1.00x | -; 0, 0 | 1405 of 4096; 1405 of 4096 | identical; identical |
| polybench/MVT | 1.23 | 1.21 | 1.21 | 1.00x | 1.21 | 1.22 | 1.00x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| polybench/SYR2K | 9.73 | 2.37 | 9.81 | 4.13x | 2.37 | 9.72 | 4.10x | 1, 1; 1, 1 | 1 of 1; 1 of 1 | identical; identical |
| polybench/SYRK | 5.16 | 1.30 | 5.16 | 3.98x | 1.30 | 5.19 | 3.99x | 1, 1; 1, 1 | 1 of 1; 1 of 1 | identical; identical |
| hecbench/adam-cuda | 0.03 | 0.03 | 0.03 | 1.00x | 0.03 | 0.03 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| hecbench/aligned-types-cuda | - | 8.39 | 8.39 | 1.00x | 8.49 | 8.68 | 1.02x | -; 0, 0 | 12974 of 13000; 12974 of 13000 | None; None |
| hecbench/aobench-cuda | 0.55 | 0.54 | 0.54 | 1.00x | 0.54 | 0.54 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| hecbench/aop-cuda | 1.86 | 1.86 | 1.86 | 1.00x | 1.87 | 1.86 | 1.00x | 0, 0; 0, 0 | 0 of 301; 0 of 301 | identical; identical |
| hecbench/asmooth-cuda | 0.25 | 0.26 | 0.26 | 1.00x | 0.26 | 0.26 | 1.00x | 0, 0; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| hecbench/asyncAllocation-cuda | 3.89 | 3.92 | 3.94 | 1.00x | 3.90 | 3.92 | 1.01x | 0, 0; 0, 0 | 0 of 800; 0 of 800 | identical; identical |
| hecbench/atan2-cuda | 0.03 | 0.03 | 0.03 | 0.98x | 0.03 | 0.03 | 0.97x | 0, 0; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| hecbench/atomicAggregate-cuda | 3.59 | 3.59 | 3.59 | 1.00x | 1.46 | 1.46 | 1.00x | 0, 0; 1, 0 | 0 of 6; 4 of 6 | identical; identical |
| hecbench/atomicIntrinsics-cuda | 0.18 | 0.18 | 0.18 | 0.99x | 0.18 | 0.18 | 1.00x | 0, 0; 0, 0 | 0 of 4; 0 of 4 | identical; identical |
| hecbench/atomicPerf-cuda | 1.13 | 1.13 | 1.13 | 1.00x | 1.36 | 1.35 | 0.99x | 3, 0; 8, 0 | 0 of 18; 1 of 18 | identical; identical |
| hecbench/atomicReduction-cuda | - | - | - | - | 3.84 | 3.86 | 1.00x | -; 0, 0 | None of None; 2090 of 2100 | None; None |
| hecbench/attention-cuda | 0.03 | 0.03 | 0.03 | 1.00x | 0.03 | 0.03 | 1.00x | 0, 0; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| hecbench/axpby-cuda | 0.47 | 0.65 | 0.64 | 0.99x | 0.65 | 0.64 | 0.99x | 0, 0; 0, 0 | 6 of 35; 5 of 35 | identical; identical |
| hecbench/babelstream-cuda | - | 13.77 | 13.88 | 1.01x | 13.72 | 13.79 | 1.00x | -; 0, 0 | 1188 of 1214; 1188 of 1214 | None; None |
| hecbench/background-subtract-cuda | 0.00 | 0.00 | 0.00 | 1.00x | 0.00 | 0.00 | 1.00x | -; - | None of None; None of None | identical; identical |
| hecbench/backprop-cuda | 0.01 | 0.01 | 0.01 | 1.02x | 0.01 | 0.01 | 1.03x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| hecbench/base64e-cuda | - | 8.81 | 8.81 | 1.00x | 8.81 | 8.81 | 1.00x | -; 0, 0 | 514 of 516; 514 of 516 | None; None |
| hecbench/bgmv-cuda | 0.02 | 0.02 | 0.02 | 1.01x | 0.02 | 0.02 | 0.99x | 0, 0; 0, 0 | 0 of 12; 0 of 12 | identical; identical |
| hecbench/bh-cuda | 0.86 | 0.88 | 0.87 | 0.99x | 0.69 | 0.69 | 1.00x | 0, 0; 0, 0 | 0 of 17; 1 of 17 | identical; identical |
| hecbench/bilateral-cuda | 1.27 | 1.27 | 1.27 | 1.00x | 1.27 | 1.27 | 1.00x | 0, 0; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| hecbench/binomial-cuda | - | - | - | - | 1.36 | 7.45 | 5.46x | -; 1, 1 | None of None; 1000 of 1000 | None; None |
| hecbench/bitonic-sort-cuda | 0.32 | 0.32 | 0.32 | 0.99x | 0.52 | 0.57 | 1.10x | 0, 0; 0, 0 | 0 of 136; 20 of 136 | identical; identical |
| hecbench/black-scholes-cuda | 25.05 | 25.05 | 25.10 | 1.00x | 25.18 | 25.06 | 1.00x | 0, 0; 1, 0 | 0 of 1; 0 of 1 | identical; identical |
| hecbench/blockAccess-cuda | 0.02 | 0.02 | 0.02 | 1.00x | 0.02 | 0.02 | 1.00x | 0, 0; 0, 0 | 0 of 4; 0 of 4 | identical; identical |
| hecbench/perplexity-cuda | 0.34 | 0.33 | 0.33 | 0.99x | 0.34 | 0.34 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| hecbench/radixsort2-cuda | 3.28 | 3.26 | 3.26 | 1.00x | 3.25 | 3.22 | 0.99x | 0, 0; 0, 0 | 0 of 600; 0 of 600 | identical; identical |
| hecbench/s3d-cuda | 0.35 | 0.38 | 0.38 | 1.00x | 0.38 | 0.38 | 1.01x | 0, 0; 0, 0 | 0 of 54; 0 of 54 | identical; identical |
| hecbench/shmembench-cuda | 6.20 | 6.17 | 6.20 | 1.00x | 0.30 | 6.17 | 20.27x | 0, 0; 1, 1 | 0 of 1; 1 of 1 | identical; identical |
| hecbench/silu-cuda | 0.53 | 0.53 | 0.53 | 1.00x | 0.54 | 0.58 | 1.08x | 0, 0; 0, 0 | 0 of 50; 24 of 50 | identical; identical |
| hecbench/sortKV-cuda | 0.28 | 0.27 | 0.28 | 1.02x | 0.27 | 0.27 | 1.00x | 0, 0; 0, 0 | 0 of 48; 0 of 48 | identical; identical |

hecbench: 30 programs. Over the simulator (geometric mean; programs with a simulator time): exact 0.986x (25), exact0 0.986x, turbo 1.130x (25), turbo0 0.996x. exact0 / exact 1.001x over 28; turbo0 / turbo 1.177x over 30. Summed seconds: exact 81.7, exact0 81.9, turbo 79.3, turbo0 91.5
polybench: 19 programs. Over the simulator (geometric mean; programs with a simulator time): exact 1.510x (19), exact0 1.286x, turbo 1.512x (19), turbo0 1.286x. exact0 / exact 1.174x over 19; turbo0 / turbo 1.176x over 19. Summed seconds: exact 281.1, exact0 341.8, turbo 280.5, turbo0 324.5
rodinia: 12 programs. Over the simulator (geometric mean; programs with a simulator time): exact 1.173x (12), exact0 1.169x, turbo 1.234x (12), turbo0 1.241x. exact0 / exact 1.004x over 12; turbo0 / turbo 0.994x over 12. Summed seconds: exact 12.6, exact0 12.9, turbo 12.3, turbo0 12.2
all: 61 programs. Over the simulator (geometric mean; programs with a simulator time): exact 1.182x (56), exact0 1.119x, turbo 1.271x (56), turbo0 1.139x. exact0 / exact 1.054x over 59; turbo0 / turbo 1.138x over 61. Summed seconds: exact 375.5, exact0 436.6, turbo 372.1, turbo0 428.2

More than 3% slower than the unsampled form (median), with the runs of both:
* rodinia/backprop turbo: 0.758 s against 0.708 s (+7.0%); runs [0.7058, 0.7621, 0.7577] against [0.7118, 0.7057, 0.7081]; launches sampled 0, sample cost 0.0 s
* rodinia/sc_gpu exact: 0.775 s against 0.728 s (+6.6%); runs [0.7755, 0.7738, 0.7796] against [0.7784, 0.7268, 0.7276]; launches sampled 0, sample cost 0.0 s
* polybench/JACOBI2D exact: 0.605 s against 0.552 s (+9.5%); runs [0.6046, 0.6059, 0.5548] against [0.5511, 0.5594, 0.5522]; launches sampled 0, sample cost 0.0 s

Fork checks, summed over the 170 runs whose summary line was kept: sampled=22, sent=7, timeouts=2, after_driver=0, copies_running_at_exit=0, copies_left=0
Outputs that are not identical to the simulator's: [('rodinia/particlefilter_float', 'exact'), ('rodinia/particlefilter_float', 'exact0'), ('rodinia/particlefilter_float', 'turbo'), ('rodinia/particlefilter_float', 'turbo0')]
