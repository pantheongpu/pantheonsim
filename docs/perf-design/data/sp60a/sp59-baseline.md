### 1. sim / form where the form sent 0 launches to the card

| program | sim s (runs) | exact | exact + DIV | turbo |
|---|---:|---:|---:|---:|
| rodinia/backprop | 0.61 (3) | 0.995 | 0.993 | - |
| rodinia/needle | 0.36 (3) | 0.976 | 0.974 | 0.983 |
| rodinia/pathfinder | 1.08 (3) | 0.994 | 0.995 | 0.995 |
| rodinia/srad_v2 | 0.06 (3) | 1.015 | 1.015 | 1.003 |
| rodinia/particlefilter_float | 2.97 (3) | 1.015 | 0.960 | 1.005 |
| rodinia/hotspot | 1.18 (3) | 0.991 | 0.989 | - |
| rodinia/bfs | 0.01 (3) | 0.957 | 0.964 | 0.964 |
| rodinia/nn | 0.02 (3) | 0.995 | 1.000 | 0.995 |
| rodinia/lavaMD | 1.80 (3) | 1.001 | 0.999 | 1.000 |
| polybench/2DCONV | 0.96 (3) | 0.992 | 0.994 | 0.995 |
| polybench/3DCONV | 0.76 (3) | 0.991 | 0.991 | 0.989 |
| polybench/ADI | 0.39 (3) | 0.997 | 0.992 | 0.994 |
| polybench/ATAX | 1.08 (3) | 1.000 | 0.999 | 1.004 |
| polybench/BICG | 0.26 (3) | 1.003 | 1.006 | 1.003 |
| polybench/CORR | 78.11 (1) | 1.234 | 1.199 | 1.245 |
| polybench/COVAR | 62.82 (1) | 1.000 | 0.989 | 0.998 |
| polybench/GEMM | 0.84 (3) | 1.041 | 0.991 | 0.974 |
| polybench/GEMVER | 0.65 (3) | 0.996 | 0.996 | 0.999 |
| polybench/GESUMMV | 0.39 (3) | 1.011 | 1.013 | 1.003 |
| polybench/JACOBI1D | 2.41 (3) | 0.981 | 0.987 | 0.984 |
| polybench/MVT | 1.21 (3) | 1.001 | 1.002 | 1.002 |
| polybench/SYR2K | 9.77 (3) | 1.000 | 1.000 | 0.999 |
| polybench/SYRK | 6.48 (3) | 1.248 | 1.248 | 1.248 |
| hecbench/adam-cuda | 0.03 (3) | 0.991 | 0.985 | 0.988 |
| hecbench/aobench-cuda | 0.54 (3) | 0.997 | 0.999 | 0.992 |
| hecbench/aop-cuda | 1.86 (3) | 1.000 | 0.998 | 0.998 |
| hecbench/asmooth-cuda | 0.25 (3) | 0.998 | 0.999 | 0.998 |
| hecbench/asyncAllocation-cuda | 3.86 (3) | 0.994 | 0.997 | 0.997 |
| hecbench/atan2-cuda | 0.03 (3) | 1.033 | 0.986 | 0.940 |
| hecbench/atomicAggregate-cuda | 3.58 (3) | 1.000 | 0.999 | - |
| hecbench/atomicIntrinsics-cuda | 0.18 (3) | 1.002 | 1.002 | 1.008 |
| hecbench/atomicPerf-cuda | 1.13 (3) | 1.002 | 1.003 | - |
| hecbench/atomicReduction-cuda | 296.99 (1) | 1.001 | 1.001 | - |
| hecbench/attention-cuda | 0.03 (3) | 0.975 | 0.975 | 0.997 |
| hecbench/backprop-cuda | 0.01 (3) | 1.015 | 1.015 | 1.015 |
| hecbench/bgmv-cuda | 0.02 (3) | 0.976 | 0.982 | 0.994 |
| hecbench/bh-cuda | 0.88 (3) | 0.993 | 0.985 | - |
| hecbench/bilateral-cuda | 1.28 (3) | 1.002 | 1.000 | 1.002 |
| hecbench/bitonic-sort-cuda | 0.32 (3) | 0.994 | 1.001 | - |
| hecbench/black-scholes-cuda | 24.90 (3) | 1.000 | 1.000 | 1.000 |
| hecbench/blockAccess-cuda | 0.02 (3) | 0.991 | 0.982 | 0.991 |
| hecbench/perplexity-cuda | 0.34 (3) | 1.018 | 1.019 | 0.989 |
| hecbench/radixsort2-cuda | 3.37 (3) | 1.028 | 1.026 | 1.022 |
| hecbench/s3d-cuda | 0.36 (3) | 0.977 | 0.950 | 0.968 |
| hecbench/shmembench-cuda | 6.15 (3) | 1.000 | 1.000 | 1.001 |
| hecbench/silu-cuda | 0.53 (3) | 0.995 | 0.996 | - |
| hecbench/sortKV-cuda | 0.28 (3) | 1.020 | 1.023 | 1.022 |

exact: 47 programs, geometric mean 1.0081; of 0.3 s or more: 34, geometric mean 1.0122, median 1.000, more than 5 % from 1: CORR 1.234, SYRK 1.248
exactdiv: 47 programs, geometric mean 1.0036; of 0.3 s or more: 34, geometric mean 1.0070, median 0.999, more than 5 % from 1: CORR 1.199, SYRK 1.248, s3d-cuda 0.950
turbo: 39 programs, geometric mean 1.0064; of 0.3 s or more: 26, geometric mean 1.0129, median 0.999, more than 5 % from 1: CORR 1.245, SYRK 1.248

### 2. The plain simulator against itself (main pass / START_COST=0 pass)

| program | sim, main pass | sim, START_COST=0 pass (1 run) | ratio |
|---|---:|---:|---:|
| polybench/2DCONV | 0.96 | 0.96 | 0.998 |
| polybench/3DCONV | 0.76 | 0.76 | 0.998 |
| polybench/3MM | 2.42 | 2.30 | 1.053 |
| polybench/ADI | 0.39 | 0.39 | 1.004 |
| polybench/ATAX | 1.08 | 1.08 | 1.005 |
| polybench/FDTD-2D | 106.05 | 108.79 | 0.975 |
| polybench/GEMM | 0.84 | 0.85 | 0.988 |
| polybench/GEMVER | 0.65 | 0.76 | 0.861 |
| polybench/GESUMMV | 0.39 | 0.38 | 1.013 |
| polybench/JACOBI1D | 2.41 | 2.43 | 0.993 |
| polybench/JACOBI2D | 0.62 | 0.62 | 0.995 |
| polybench/LU | 30.68 | 46.51 | 0.660 |
| polybench/MVT | 1.21 | 1.36 | 0.890 |
| polybench/SYR2K | 9.77 | 9.78 | 0.999 |
| polybench/SYRK | 6.48 | 5.19 | 1.248 |
| rodinia/backprop | 0.61 | 0.61 | 1.002 |
| rodinia/gaussian | 4.13 | 4.12 | 1.002 |
| rodinia/lud | 3.52 | 3.57 | 0.986 |
| rodinia/needle | 0.36 | 0.36 | 0.984 |
| rodinia/pathfinder | 1.08 | 1.08 | 0.994 |
| rodinia/particlefilter_float | 2.97 | 2.56 | 1.159 |
| rodinia/sc_gpu | 1.03 | 1.01 | 1.021 |
| rodinia/hotspot | 1.18 | 1.18 | 0.999 |
| rodinia/lavaMD | 1.80 | 1.79 | 1.006 |

24 programs of 0.3 s or more: ratio from 0.660 to 1.248; 6 of them more than 5 % from 1

### 3. Headline numbers of sp59, as published and with a form that used no card capped at 1.00

| form | programs | geometric mean, published | corrected | summed sim s / form s, published | corrected |
|---|---:|---:|---:|---:|---:|
| exact | 59 | 1.123x | 1.115x | 1172 / 978 = 1.199x | 1156 / 978 = 1.183x |
|  | changed: CORR 1.234, SYRK 1.248 | | | | |
| exactdiv | 59 | 1.121x | 1.113x | 1172 / 972 = 1.207x | 1158 / 972 = 1.192x |
|  | changed: CORR 1.199, SYRK 1.248 | | | | |
| turbo | 59 | 1.231x | 1.221x | 1172 / 669 = 1.753x | 1156 / 669 = 1.728x |
|  | changed: CORR 1.245, SYRK 1.248 | | | | |
