| program | kernel | blocks | live MiB | static est s | age s | sample cost ms | fork ms | blocks timed | first block ms | timed runs ms | plain extrapolation s | launch took s | ratio | launch / static | estimate s | would send |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | _Z20convolution2D_kerneliiPfS_ | 65536 | 128.0 | 0.2936 | 0.121 | 2.74 | 1.16 | 32 | 0.585 | 0.176 | 0.0901 | 0.1319 | 0.68 | 0.4 | 0.0901 | 0 |
| polybench/2MM | _Z11mm2_kernel1iiiiffPfS_S_ | 16384 | 80.0 | 0.3355 | 0.049 | 9.73 | 1.50 | 2 | 2.650 | 3.294 | 6.7461 | 10.5157 | 0.64 | 31.3 | 6.7461 | 1 |
| polybench/3DCONV | _Z20convolution3D_kerneliiiPfS_i | 256 | 128.0 | 0.0015 | 0.065 | 3.28 | 1.91 | 32 | 0.190 | 0.227 | 0.0005 | 0.0019 | 0.24 | 1.3 | 0.0005 | 0 |
| polybench/3MM | _Z11mm3_kernel1iiiiiPfS_S_ | 1024 | 7.0 | 0.0190 | 0.011 | 5.87 | 0.22 | 6 | 1.264 | 2.155 | 0.0919 | 0.1289 | 0.71 | 6.8 | 0.0919 | 0 |
| polybench/3MM | _Z11mm3_kernel2iiiiiPfS_S_ | 1024 | 7.0 | 0.0190 | 0.147 | 5.94 | 0.33 | 4 | 0.971 | 2.115 | 0.1354 | 0.1270 | 1.07 | 6.7 | 0.1354 | 0 |
| polybench/GEMM | _Z11gemm_kerneliiiffPfS_S_ | 1024 | 3.0 | 0.0210 | 0.010 | 4.95 | 0.16 | 5 | 0.830 | 2.019 | 0.1034 | 0.1401 | 0.74 | 6.7 | 0.1034 | 0 |
| polybench/GEMVER | _Z14gemver_kernel1iffPfS_S_S_S_ | 65536 | 64.1 | 0.2097 | 0.043 | 3.07 | 1.35 | 32 | 0.292 | 0.124 | 0.0635 | 0.0962 | 0.66 | 0.5 | 0.0635 | 0 |
| polybench/JACOBI2D | _Z21runJacobiCUDA_kernel1iPfS_ | 4000 | 7.6 | 0.0128 | 0.013 | 1.94 | 0.24 | 32 | 0.335 | 0.181 | 0.0057 | 0.0053 | 1.07 | 0.4 | 0.0057 | 0 |
| polybench/JACOBI2D | _Z21runJacobiCUDA_kernel2iPfS_ | 4000 | 7.6 | 0.0102 | 0.020 | 2.59 | 0.49 | 32 | 0.223 | 0.228 | 0.0071 | 0.0036 | 2.00 | 0.3 | 0.0071 | 0 |
| polybench/LU | _Z10lu_kernel2iPfi | 16384 | 16.0 | 0.0524 | 0.019 | 2.28 | 0.53 | 32 | 0.214 | 0.219 | 0.0280 | 0.0215 | 1.30 | 0.4 | 0.0280 | 0 |
| polybench/SYR2K | _Z12syr2k_kerneliiffPfS_S_ | 4096 | 12.0 | 0.0446 | 0.016 | 6.10 | 0.39 | 1 | 2.953 | 2.280 | 2.3347 | 3.3882 | 0.69 | 76.0 | 2.3347 | 1 |
| polybench/SYRK | _Z11syrk_kerneliiffPfS_ | 4096 | 8.0 | 0.0839 | 0.013 | 5.90 | 0.26 | 2 | 1.646 | 2.354 | 1.2052 | 1.8919 | 0.64 | 22.6 | 1.2052 | 1 |
| rodinia/backprop | _Z24bpnn_adjust_weights_cudaPfiS_iS_S_ | 16384 | 36.0 | 0.1049 | 0.154 | 4.46 | 1.98 | 32 | 0.397 | 0.589 | 0.0754 | 0.0669 | 1.13 | 0.6 | 0.0754 | 0 |
| rodinia/gaussian | _Z4Fan2PfS_S_iii | 16384 | 2.0 | 0.0092 | 0.003 | 1.02 | 0.18 | 32 | 0.318 | 0.088 | 0.0113 | 0.0058 | 1.95 | 0.6 | 0.0113 | 0 |
| rodinia/lud | _Z12lud_internalPfii | 3969 | 4.0 | 0.0254 | 0.009 | 3.44 | 0.56 | 32 | 0.369 | 0.483 | 0.0150 | 0.0162 | 0.92 | 0.6 | 0.0150 | 0 |
| rodinia/pathfinder | _Z14dynproc_kerneliPiS_S_iiii | 463 | 38.5 | 0.0036 | 0.333 | 6.27 | 0.92 | 17 | 0.599 | 2.060 | 0.0140 | 0.0126 | 1.12 | 3.5 | 0.0140 | 0 |

launches 16; ratio min 0.24, median 0.92, 90th percentile 1.95, max 2.00; estimated high (ratio over 1.0): 7
