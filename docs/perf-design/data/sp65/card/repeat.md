Ten runs a form (twenty for sc_gpu) of every reading more than 3 % apart in the A/B; medians. Full rows: summary.py over repeat*.json on the rig.
| program | mode | without the rule s | with it s | with / without | sent by the rule | card launches with / without | outputs with / without |
|---|---|---:|---:|---:|---:|---:|---|
| hecbench/atan2-cuda | exact | 0.030 | 0.030 | 1.007 | - | 0 / 0 | identical / identical |
| hecbench/atan2-cuda | turbo | 0.029 | 0.029 | 1.000 | 0 | 0 / 0 | identical / identical |
| hecbench/attention-cuda | exact | 0.036 | 0.036 | 0.997 | - | 0 / 0 | identical / identical |
| hecbench/attention-cuda | turbo | 0.036 | 0.036 | 0.997 | 0 | 0 / 0 | identical / identical |
| hecbench/bgmv-cuda | exact | 0.018 | 0.017 | 0.989 | - | 0 / 0 | identical / identical |
| hecbench/bgmv-cuda | turbo | 0.017 | 0.017 | 1.000 | 0 | 0 / 0 | identical / identical |
| polybench/3DCONV | exact | 0.778 | 0.778 | 1.000 | - | 0 / 0 | identical / identical |
| polybench/3DCONV | turbo | 0.780 | 0.776 | 0.995 | 0 | 0 / 0 | identical / identical |
| polybench/JACOBI2D | exact | 0.562 | 0.563 | 1.002 | - | 22 / 22 | identical / identical |
| polybench/JACOBI2D | turbo | 0.564 | 0.562 | 0.998 | 0 | 22 / 22 | identical / identical |
| polybench/MVT | exact | 1.224 | 1.228 | 1.004 | - | 0 / 0 | identical / identical |
| polybench/MVT | turbo | 1.227 | 1.227 | 1.000 | 0 | 0 / 0 | identical / identical |
| rodinia/backprop | exact | 0.616 | 0.616 | 1.001 | - | 0 / 0 | identical / identical |
| rodinia/backprop | turbo | 0.722 | 0.723 | 1.002 | 0 | 1 / 1 | identical / identical |
| rodinia/lud | exact | 3.001 | 3.004 | 1.001 | - | 50 / 50 | identical / identical |
| rodinia/lud | turbo | 2.861 | 2.767 | 0.967 | 63 | 174 / 95 | identical / identical |
| rodinia/needle | exact | 0.374 | 0.369 | 0.988 | - | 0 / 0 | identical / identical |
| rodinia/needle | turbo | 0.371 | 0.369 | 0.994 | 0 | 0 / 0 | identical / identical |
| rodinia/sc_gpu | exact | 0.793 | 0.797 | 1.005 | - | 1059 / 1063 | identical / identical |
| rodinia/sc_gpu | turbo | 0.764 | 0.802 | 1.050 | 0 | 1065 / 1059 | identical / identical |
| hecbench/black-scholes-cuda | exact | 25.194 | 25.174 | 0.999 | - | 0 / 0 | identical / identical |
| rodinia/sc_gpu | turbo | 0.798 | 0.754 | 0.945 | 0 | 1063 / 1063 | identical / identical |
| rodinia/gaussian | exact | 0.936 | 0.939 | 1.003 | - | 495 / 495 | identical / identical |
| rodinia/lud | exactdiv | 2.882 | 2.844 | 0.987 | - | 95 / 95 | identical / identical |

sc_gpu, 20 runs: two levels about 0.75 s and 0.80 s in every form (sp60b saw the same); turbo 14 of 20 runs low, turbon 7 of 20.
