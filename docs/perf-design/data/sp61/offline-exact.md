| program | kernel | blocks | inside | live MiB | static est s | age s | affordable | cost ms | warps followed/not | turns left read | steps a turn | estimate s | launch took s | ratio | would send |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2DCONV | _Z20convolution2D_kerneliiPfS_ | 65536 | 0 | 128.0 | 0.2936 | 0.123 | 0 | 3.1 | - | - | - | 0.0913 | 0.1088 | 0.84 | 0 |
| polybench/3DCONV | _Z20convolution3D_kerneliiiPfS_i | 256 | 0 | 128.0 | 0.0015 | 0.066 | 0 | 3.4 | - | - | - | 0.0005 | 0.0019 | 0.25 | 0 |
| polybench/3MM | _Z11mm3_kernel1iiiiiPfS_S_ | 1024 | 0 | 7.0 | 0.0190 | 0.012 | 0 | 5.8 | - | - | - | 0.0900 | 0.1281 | 0.70 | 0 |
| polybench/3MM | _Z11mm3_kernel2iiiiiPfS_S_ | 1024 | 0 | 7.0 | 0.0190 | 0.146 | 0 | 6.2 | - | - | - | 0.1234 | 0.1269 | 0.97 | 0 |
| polybench/ATAX | _Z12atax_kernel1iiPfS_S_ | 128 | 1 | 64.0 | 0.0022 | 0.046 | 0 | 15.7 | 24/0 | 144 | 76 | 0.1668 | 0.2082 | 0.80 | 0 |
| polybench/ATAX | _Z12atax_kernel2iiPfS_S_ | 128 | 1 | 64.0 | 0.0022 | 0.270 | 0 | 16.6 | 16/0 | 96 | 76 | 0.2018 | 0.2227 | 0.91 | 0 |
| polybench/BICG | _Z12bicg_kernel1iiPfS_S_ | 16 | 1 | 64.1 | 0.0003 | 0.042 | 0 | 14.0 | 16/0 | 96 | 76 | 0.0247 | 0.0293 | 0.84 | 0 |
| polybench/BICG | _Z12bicg_kernel2iiPfS_S_ | 16 | 1 | 64.1 | 0.0003 | 0.086 | 0 | 18.1 | 24/0 | 144 | 76 | 0.0238 | 0.0280 | 0.85 | 0 |
| polybench/CORR | _Z11corr_kerneliiPfS_ | 8 | 1 | 32.0 | 0.0001 | 0.084 | 1 | 13.9 | 16/0 | 28912 | 12235 | 16.7609 | 10.3506 | 1.62 | 1 |
| polybench/COVAR | _Z13reduce_kerneliiPfS_ | 4096 | 0 | 32.0 | 0.0105 | 0.031 | 0 | 3.0 | - | - | - | 0.0054 | 0.0059 | 0.92 | 0 |
| polybench/COVAR | _Z12covar_kerneliiPfS_ | 8 | 1 | 32.0 | 0.0001 | 0.041 | 1 | 5.6 | 64/0 | 66560 | 13988 | 10.2710 | 10.0925 | 1.02 | 1 |
| polybench/GEMM | _Z11gemm_kerneliiiffPfS_S_ | 1024 | 0 | 3.0 | 0.0210 | 0.010 | 0 | 6.1 | - | - | - | 0.0974 | 0.1408 | 0.69 | 0 |
| polybench/GEMVER | _Z14gemver_kernel1iffPfS_S_S_S_ | 65536 | 0 | 64.1 | 0.2097 | 0.045 | 0 | 3.3 | - | - | - | 0.0749 | 0.0770 | 0.97 | 0 |
| polybench/GEMVER | _Z14gemver_kernel2iffPfS_S_S_ | 16 | 1 | 64.1 | 0.0003 | 0.126 | 0 | 4.1 | 128/0 | 32768 | 88 | 0.0532 | 0.0299 | 1.78 | 0 |
| polybench/GEMVER | _Z14gemver_kernel3iffPfS_S_ | 16 | 1 | 64.1 | 0.0003 | 0.161 | 0 | 3.4 | 128/0 | 32768 | 88 | 0.0370 | 0.0278 | 1.33 | 0 |
| polybench/GESUMMV | _Z14gesummv_kerneliffPfS_S_S_S_ | 16 | 1 | 128.0 | 0.0002 | 0.076 | 0 | 5.0 | 128/0 | 131072 | 52 | 0.0567 | 0.0845 | 0.67 | 0 |
| polybench/JACOBI2D | _Z21runJacobiCUDA_kernel1iPfS_ | 4000 | 0 | 7.6 | 0.0128 | 0.014 | 0 | 2.7 | - | - | - | 0.0107 | 0.0055 | 1.94 | 0 |
| polybench/JACOBI2D | _Z21runJacobiCUDA_kernel2iPfS_ | 4000 | 0 | 7.6 | 0.0102 | 0.022 | 0 | 2.0 | - | - | - | 0.0052 | 0.0038 | 1.37 | 0 |
| polybench/MVT | _Z11mvt_kernel1iPfS_S_ | 128 | 1 | 64.1 | 0.0022 | 0.045 | 0 | 2.8 | 256/0 | 65536 | 75 | 0.1483 | 0.2094 | 0.71 | 0 |
| polybench/MVT | _Z11mvt_kernel2iPfS_S_ | 128 | 1 | 64.1 | 0.0022 | 0.258 | 0 | 4.7 | 256/0 | 65536 | 75 | 0.3540 | 0.2221 | 1.59 | 0 |
| polybench/SYR2K | _Z12syr2k_kerneliiffPfS_S_ | 4096 | 0 | 12.0 | 0.0446 | 0.015 | 1 | 6.2 | - | - | - | 2.2398 | 3.4706 | 0.65 | 1 |
| polybench/SYRK | _Z11syrk_kerneliiffPfS_ | 4096 | 0 | 8.0 | 0.0839 | 0.013 | 1 | 7.3 | - | - | - | 1.6536 | 1.8177 | 0.91 | 1 |
| rodinia/backprop | _Z24bpnn_adjust_weights_cudaPfiS_iS_ | 16384 | 0 | 36.0 | 0.1049 | 0.163 | 0 | 4.3 | - | - | - | 0.0642 | 0.0677 | 0.95 | 0 |
| rodinia/gaussian | _Z4Fan2PfS_S_iii | 16384 | 0 | 2.0 | 0.0092 | 0.002 | 0 | 0.9 | - | - | - | 0.0092 | 0.0051 | 1.80 | 0 |
| rodinia/lud | _Z12lud_internalPfii | 3969 | 0 | 4.0 | 0.0254 | 0.009 | 0 | 3.4 | - | - | - | 0.0149 | 0.0141 | 1.06 | 0 |
| rodinia/pathfinder | _Z14dynproc_kerneliPiS_S_iiii | 463 | 0 | 38.5 | 0.0036 | 0.332 | 0 | 6.2 | - | - | - | 0.0087 | 0.0123 | 0.71 | 0 |
| rodinia/sc_gpu | _Z19kernel_compute_costiilP5PointiiP | 8 | 1 | 0.6 | 0.0004 | 0.003 | 0 | 1.9 | 0/0 | 0 | 0 | 0.0000 | 0.0011 | - | 0 |

launches 26; ratio min 0.25, median 0.92, max 1.94; over 2: 0; under 0.5: 1
