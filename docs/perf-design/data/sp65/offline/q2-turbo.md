| program | sp59 card launches | replay card launches | card path s (sp59) | path per launch us | pushes per launch us (model) | kernel per launch us | sim per launch us (rig) | sim / kernel |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench/2MM | 1 | 1 | 0.0492 | 49160.0 | 17476.3 | 31676.7 | 25597642 | 808 |
| polybench/3MM | 2 | 2 | 0.0027 | 1360.0 | 764.6 | 588.4 | 269646 | 458 |
| polybench/FDTD-2D | 1494 | 1494 | 0.2108 | 141.1 | 7.0 | 127.1 | 66582 | 524 |
| polybench/GRAMSCHM | 2043 | 2038 | 21.5230 | 10535.0 | 10485.8 | 42.2 | 16275 | 385 |
| polybench/JACOBI2D | 22 | 23 | 0.0042 | 192.9 | 72.5 | 113.4 | 14801 | 130 |
| polybench/LU (counts differ) | 1405 | 1211 | 4.5372 | 3229.3 | 3495.3 | -272.9 | 17904 | inf |
| rodinia/backprop | 1 | 1 | 0.0177 | 17667.0 | 7864.4 | 9795.6 | 256580 | 26 |
| rodinia/gaussian | 495 | 495 | 0.1795 | 362.6 | 437.3 | -81.7 | 7935 | inf |
| rodinia/lud (counts differ) | 95 | 39 | 0.0338 | 355.9 | 739.4 | -390.5 | 3833 | inf |
| rodinia/sc_gpu | 1053 | 1050 | 0.0985 | 93.5 | 62.5 | 24.0 | 604 | 25 |
| rodinia/hotspot | 789 | 795 | 0.0148 | 18.8 | 0.0 | 11.8 | 1452 | 123 |
| hecbench/atomicAggregate-cuda | 4 | 4 | 0.0032 | 795.5 | 0.0 | 788.5 | 564550 | 716 |
| hecbench/axpby-cuda | 5 | 4 | 0.0028 | 553.2 | 646.3 | -100.1 | 19801 | inf |
| hecbench/bh-cuda | 1 | 1 | 0.0025 | 2544.0 | 430.3 | 2106.7 | 343267 | 163 |
| hecbench/bitonic-sort-cuda (counts differ) | 22 | 13 | 0.0009 | 40.4 | 4.2 | 29.2 | 1857 | 64 |
| hecbench/silu-cuda | 24 | 25 | 0.0019 | 78.5 | 35.0 | 36.5 | 9516 | 260 |

programs fitted: 11; kernel per launch: min 11.8 us, median 127.1 us; sim / kernel: min 25, median 260, max 808
