| program | sim s | exact s | exactw s | exactw / exact | turbo s | turbow s | turbow / turbo | sampled, sent (exact; turbo) | on card of launches (exact; turbo) | output (exact; turbo) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/CORR | 65.14 | 42.01 | 64.84 | 1.54x | 42.07 | 64.77 | 1.54x | 1, 1; 2, 1 | 1 of 4; 1 of 4 | identical; identical |
| polybench/COVAR | 64.83 | 41.82 | 64.65 | 1.55x | 41.95 | 64.39 | 1.54x | 1, 1; 1, 1 | 1 of 3; 1 of 3 | identical; identical |
| polybench/ATAX | 1.07 | 1.07 | 1.07 | 1.00x | 1.07 | 1.08 | 1.00x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| polybench/BICG | 0.27 | 0.26 | 0.27 | 1.03x | 0.27 | 0.27 | 1.01x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| polybench/CORR | 76.96 | 42.21 | 64.63 | 1.53x | 41.90 | 65.94 | 1.57x | -; 2, 1 | 1 of 4; 1 of 4 | identical; identical |
| polybench/COVAR | 80.20 | 42.09 | 64.67 | 1.54x | 40.91 | 64.67 | 1.58x | -; 1, 1 | 1 of 3; 1 of 3 | identical; identical |
| polybench/GEMVER | 0.65 | 0.66 | 0.65 | 0.99x | 0.65 | 0.66 | 1.01x | 0, 0; 0, 0 | 0 of 3; 0 of 3 | identical; identical |
| polybench/GESUMMV | 0.38 | 0.38 | 0.38 | 1.01x | 0.38 | 0.38 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/MVT | 1.32 | 1.30 | 1.30 | 1.00x | 1.29 | 1.34 | 1.04x | 0, 0; 0, 0 | 0 of 2; 0 of 2 | identical; identical |
| rodinia/sc_gpu | 1.04 | 0.73 | 0.78 | 1.07x | 0.74 | 0.78 | 1.06x | 0, 0; 0, 0 | 1158 of 1432; 1057 of 1432 | identical; identical |
| hecbench/aop-cuda | 1.86 | 1.86 | 1.86 | 1.00x | 1.87 | 1.87 | 1.00x | 0, 0; 0, 0 | 0 of 301; 0 of 301 | identical; identical |
| hecbench/bh-cuda | 0.87 | 0.87 | 0.88 | 1.01x | 0.70 | 0.75 | 1.07x | 0, 0; 1, 0 | 0 of 17; 1 of 17 | identical; identical |
| polybench/2DCONV | 0.96 | 0.96 | 0.95 | 0.99x | 0.95 | 0.95 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/3DCONV | 0.76 | 0.76 | 0.76 | 1.00x | 0.76 | 0.76 | 0.99x | 0, 0; 0, 0 | 0 of 254; 0 of 254 | identical; identical |
| polybench/3MM | 2.34 | 1.74 | 1.77 | 1.02x | 1.87 | 1.77 | 0.95x | 0, 0; 0, 0 | 2 of 3; 2 of 3 | identical; identical |
| polybench/ADI | 0.40 | 0.40 | 0.40 | 0.99x | 0.40 | 0.40 | 1.00x | 0, 0; 0, 0 | 0 of 2049; 0 of 2049 | identical; identical |
| polybench/GEMM | 0.83 | 0.85 | 0.87 | 1.02x | 0.78 | 0.85 | 1.08x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/JACOBI1D | 2.44 | 2.47 | 2.48 | 1.00x | 2.48 | 2.47 | 0.99x | 0, 0; 0, 0 | 0 of 20000; 0 of 20000 | identical; identical |
| polybench/JACOBI2D | 0.61 | 0.55 | 0.55 | 1.01x | 0.56 | 0.56 | 0.99x | 0, 0; 0, 0 | 22 of 40; 22 of 40 | identical; identical |
| polybench/SYR2K | 9.76 | 2.37 | 2.37 | 1.00x | 2.32 | 2.33 | 1.00x | 1, 1; 1, 1 | 1 of 1; 1 of 1 | identical; identical |
| polybench/SYRK | 5.18 | 1.29 | 1.34 | 1.04x | 1.29 | 1.29 | 1.00x | 1, 1; 1, 1 | 1 of 1; 1 of 1 | identical; identical |
| rodinia/backprop | 0.61 | 0.61 | 0.61 | 1.00x | 0.71 | 0.76 | 1.07x | 0, 0; 0, 0 | 0 of 2; 1 of 2 | identical; identical |
| rodinia/gaussian | 4.10 | 0.97 | 0.91 | 0.94x | 0.93 | 0.99 | 1.05x | 0, 0; 0, 0 | 495 of 1022; 495 of 1022 | identical; identical |
| rodinia/lud | 3.59 | 2.97 | 3.08 | 1.04x | 2.71 | 2.69 | 0.99x | 0, 0; 0, 0 | 50 of 190; 95 of 190 | identical; identical |
| rodinia/needle | 0.37 | 0.37 | 0.36 | 0.98x | 0.37 | 0.37 | 1.00x | 0, 0; 0, 0 | 0 of 255; 0 of 255 | identical; identical |
| rodinia/pathfinder | 1.08 | 1.08 | 1.06 | 0.99x | 1.08 | 1.07 | 1.00x | 0, 0; 0, 0 | 0 of 5; 0 of 5 | identical; identical |
| rodinia/srad_v2 | 0.06 | 0.07 | 0.06 | 0.96x | 0.07 | 0.07 | 1.00x | 0, 0; 1, 0 | 0 of 1; 0 of 1 | identical; identical |
| rodinia/particlefilter_float | 2.93 | 3.09 | 3.40 | 1.10x | 2.91 | 2.82 | 0.97x | 0, 0; 0, 0 | 0 of 36; 0 of 36 | different; different |
| rodinia/hotspot | 1.19 | 1.19 | 1.19 | 1.00x | 0.52 | 0.52 | 1.00x | 0, 0; 0, 0 | 0 of 1000; 789 of 1000 | identical; identical |
| rodinia/bfs | 0.01 | 0.01 | 0.01 | 0.96x | 0.01 | 0.01 | 0.99x | 0, 0; 0, 0 | 0 of 26; 0 of 26 | identical; identical |
| rodinia/nn | 0.02 | 0.02 | 0.02 | 1.00x | 0.02 | 0.02 | 1.04x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| rodinia/lavaMD | 1.78 | 1.78 | 1.78 | 1.00x | 1.78 | 1.78 | 1.00x | 0, 0; 0, 0 | 0 of 1; 0 of 1 | identical; identical |
| polybench/3MM | 2.33 | 1.81 | 1.86 | 1.03x | 1.80 | 1.84 | 1.02x | -; 0, 0 | 2 of 3; 2 of 3 | identical; identical |
| rodinia/gaussian | 4.09 | 0.92 | 0.92 | 1.00x | 0.92 | 0.92 | 1.00x | -; 0, 0 | 495 of 1022; 495 of 1022 | identical; identical |
| rodinia/srad_v2 | 0.06 | 0.06 | 0.06 | 1.01x | 0.06 | 0.06 | 0.99x | -; 1, 0 | 0 of 1; 0 of 1 | identical; identical |
| rodinia/particlefilter_float | 3.09 | 2.96 | 3.31 | 1.12x | 3.07 | 2.99 | 0.98x | -; 0, 0 | 0 of 36; 0 of 36 | different; different |
| rodinia/bfs | 0.01 | 0.01 | 0.01 | 0.99x | 0.01 | 0.01 | 0.99x | -; 0, 0 | 0 of 26; 0 of 26 | identical; identical |

hecbench: 2 programs. Over the simulator (geometric mean; programs with a simulator time): exact 0.998x (2), exactw 0.995x, turbo 1.115x (2), turbow 1.074x. exactw / exact 1.003x over 2; turbow / turbo 1.038x over 2. Summed seconds: exact 2.7, exactw 2.7, turbo 2.6, turbow 2.6
polybench: 19 programs. Over the simulator (geometric mean; programs with a simulator time): exact 1.341x (19), exactw 1.217x, turbo 1.346x (19), turbow 1.220x. exactw / exact 1.102x over 19; turbow / turbo 1.103x over 19. Summed seconds: exact 185.0, exactw 275.8, turbo 183.7, turbow 276.7
rodinia: 16 programs. Over the simulator (geometric mean; programs with a simulator time): exact 1.239x (16), exactw 1.229x, turbo 1.303x (16), turbow 1.292x. exactw / exact 1.008x over 16; turbow / turbo 1.008x over 16. Summed seconds: exact 16.8, exactw 17.6, turbo 15.9, turbow 15.9
all: 37 programs. Over the simulator (geometric mean; programs with a simulator time): exact 1.276x (37), exactw 1.209x, turbo 1.313x (37), turbow 1.242x. exactw / exact 1.055x over 37; turbow / turbo 1.057x over 37. Summed seconds: exact 204.6, exactw 296.1, turbo 202.2, turbow 295.2

More than 3% slower than the form that samples whole blocks only (VGPU_HOST_GPU_SAMPLE=whole) (median), with the runs of both:
* polybench/3MM turbo: 1.867 s against 1.767 s (+5.6%); runs [1.8665] against [1.7672]; launches sampled 0, sample cost 0.0 s
* rodinia/gaussian exact: 0.966 s against 0.911 s (+6.0%); runs [0.9658] against [0.9108]; launches sampled 0, sample cost 0.0 s
* rodinia/srad_v2 exact: 0.066 s against 0.063 s (+3.6%); runs [0.0656] against [0.0633]; launches sampled 0, sample cost 0.0 s
* rodinia/particlefilter_float turbo: 2.909 s against 2.824 s (+3.0%); runs [2.909] against [2.824]; launches sampled 0, sample cost 0.0 s
* rodinia/bfs exact: 0.014 s against 0.014 s (+4.4%); runs [0.0142] against [0.0136]; launches sampled 0, sample cost 0.0 s

Fork checks, summed over the 104 runs whose summary line was kept: sampled=21, sent=12, timeouts=4, after_driver=0, copies_running_at_exit=0, copies_left=0
Outputs that are not identical to the simulator's: [('rodinia/particlefilter_float', 'exact'), ('rodinia/particlefilter_float', 'exactw'), ('rodinia/particlefilter_float', 'turbo'), ('rodinia/particlefilter_float', 'turbow'), ('rodinia/particlefilter_float', 'exact'), ('rodinia/particlefilter_float', 'exactw'), ('rodinia/particlefilter_float', 'turbo'), ('rodinia/particlefilter_float', 'turbow')]
