| program | mode | without the rule s | with it s | with / without | sent by the rule | card launches with / without | outputs with / without |
|---|---|---:|---:|---:|---:|---:|---|
| hecbench/atan2-cuda | turbo | 0.029 | 0.030 | 1.048 | 0 | 0 / 0 | identical / identical |
| hecbench/attention-cuda | exact | 0.036 | 0.037 | 1.030 | 0 | 0 / 0 | identical / identical |
| hecbench/axpby-cuda | exact | 0.653 | 0.574 | 0.878 | 3 | 8 / 5 | identical / identical |
| hecbench/axpby-cuda | turbo | 0.655 | 0.591 | 0.902 | 3 | 8 / 5 | identical / identical |
| hecbench/background-subtract-cuda | turbo | 0.004 | 0.003 | 0.943 | - | None / None | identical / identical |
| hecbench/backprop-cuda | turbo | 0.016 | 0.014 | 0.872 | 0 | 0 / 0 | identical / identical |
| hecbench/bgmv-cuda | turbo | 0.017 | 0.018 | 1.036 | 0 | 0 / 0 | identical / identical |
| hecbench/black-scholes-cuda | exact | 25.250 | 26.085 | 1.033 | 0 | 0 / 0 | identical / identical |
| hecbench/blockAccess-cuda | turbo | 0.024 | 0.023 | 0.942 | 0 | 0 / 0 | identical / identical |
| polybench/3DCONV | turbo | 0.771 | 0.954 | 1.236 | 0 | 0 / 0 | identical / identical |
| polybench/3MM | exact | 1.836 | 1.770 | 0.964 | 0 | 2 / 2 | identical / identical |
| polybench/GRAMSCHM | exact | 279.316 | 265.712 | 0.951 | - | 2041 / 2041 | identical / identical |
| polybench/GRAMSCHM | turbo | 271.141 | 224.956 | 0.830 | 3288 | 6121 / 2042 | identical / identical |
| polybench/JACOBI2D | exact | 0.567 | 0.614 | 1.083 | 0 | 22 / 22 | identical / identical |
| polybench/LU | turbo | 13.000 | 1.743 | 0.134 | 2138 | 4085 / 1389 | identical / identical |
| polybench/LU | exactdiv | 11.690 | 1.781 | 0.152 | - | 4085 / 1390 | identical / identical |
| polybench/MVT | exact | 1.453 | 1.223 | 0.842 | 0 | 0 / 0 | identical / identical |
| polybench/MVT | turbo | 1.569 | 1.633 | 1.041 | 0 | 0 / 0 | identical / identical |
| polybench/SYRK | exact | 1.506 | 1.438 | 0.955 | 0 | 1 / 1 | identical / identical |
| rodinia/backprop | turbo | 0.714 | 0.770 | 1.080 | 0 | 1 / 1 | identical / identical |
| rodinia/bfs | exact | 0.015 | 0.015 | 0.960 | 0 | 0 / 0 | identical / identical |
| rodinia/gaussian | exact | 0.936 | 0.992 | 1.059 | 0 | 495 / 495 | identical / identical |
| rodinia/gaussian | turbo | 0.967 | 0.532 | 0.551 | 494 | 989 / 495 | identical / identical |
| rodinia/gaussian | exactdiv | 0.948 | 0.519 | 0.548 | - | 989 / 495 | identical / identical |
| rodinia/lud | turbo | 2.815 | 2.778 | 0.987 | 63 | 174 / 95 | identical / identical |
| rodinia/lud | exactdiv | 2.812 | 2.906 | 1.033 | - | 95 / 95 | identical / identical |
| rodinia/needle | exact | 0.371 | 0.423 | 1.140 | 0 | 0 / 0 | identical / identical |
| rodinia/needle | turbo | 0.365 | 0.377 | 1.031 | 0 | 0 / 0 | identical / identical |
| rodinia/particlefilter_float | exact | 3.292 | 2.680 | 0.814 | 0 | 0 / 0 | different / different |
| rodinia/particlefilter_float | turbo | 3.186 | 2.839 | 0.891 | 0 | 0 / 0 | different / different |
| rodinia/sc_gpu | turbo | 0.765 | 0.796 | 1.041 | 0 | 1061 / 1062 | identical / identical |
| rodinia/srad_v2 | exact | 0.067 | 0.063 | 0.947 | 0 | 0 / 0 | identical / identical |

81 more rows: nothing sent by the rule, within 3 %, outputs identical (all rows: --json)

more than 3 % slower with the rule: hecbench/atan2-cuda turbo 1.048, hecbench/attention-cuda exact 1.030, hecbench/bgmv-cuda turbo 1.036, hecbench/black-scholes-cuda exact 1.033, polybench/3DCONV turbo 1.236, polybench/JACOBI2D exact 1.083, polybench/MVT turbo 1.041, rodinia/backprop turbo 1.080, rodinia/gaussian exact 1.059, rodinia/lud exactdiv 1.033, rodinia/needle exact 1.140, rodinia/needle turbo 1.031, rodinia/sc_gpu turbo 1.041
