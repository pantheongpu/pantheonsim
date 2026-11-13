lock: 2 holds, 277 s waited
| program | runs | default s | default B/N | base JIT cold | JIT cold | JIT warm | JIT second run | objects (warm fill) | compile s | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecbench/asmooth-cuda | 3 | 0.082 | 0.978 | 0.855 | 0.903 | 0.982 | 0.993 | 0 | 0.0 | same |
| hecbench/asyncAllocation-cuda | 3 | 0.991 | 0.987 | 0.842 | 0.853 | 1.238 | 0.832 | 1 | 2.7 | same |
| hecbench/atomicAggregate-cuda | 3 | 0.918 | 0.972 | 0.936 | 0.966 | 1.109 | 0.930 | 1 | 0.8 | same |
| hecbench/bh-cuda | 3 | 0.207 | 0.984 | 0.941 | 0.971 | 1.006 | 0.977 | 0 | 0.0 | default differs from itself |
| hecbench/bilateral-cuda | 3 | 0.292 | 0.988 | 0.971 | 0.995 | 1.012 | 0.989 | 0 | 0.0 | same |
| hecbench/bitonic-sort-cuda | 3 | 0.087 | 0.987 | 0.946 | 0.969 | 0.997 | 0.997 | 0 | 0.0 | same |
| hecbench/radixsort2-cuda | 3 | 0.895 | 0.967 | 0.957 | 1.023 | 0.995 | 1.026 | 0 | 0.0 | default differs from itself |
| polybench/2DCONV | 3 | 0.526 | 0.910 | 0.883 | 0.995 | 1.002 | 1.018 | 0 | 0.0 | same |
| polybench/ADI | 3 | 0.138 | 1.050 | 0.926 | 1.012 | 1.042 | 0.981 | 0 | 0.0 | same |
| polybench/GEMVER | 3 | 0.342 | 0.827 | 0.851 | 0.993 | 1.063 | 0.984 | 0 | 0.0 | same |
| polybench/JACOBI1D | 3 | 0.749 | 1.018 | 0.953 | 1.032 | 1.032 | 0.968 | 0 | 0.0 | same |
| polybench/JACOBI2D | 3 | 0.169 | 1.002 | 0.915 | 0.925 | 0.972 | 0.974 | 0 | 0.0 | same |
| polybench/SYR2K | 3 | 3.356 | 0.994 | 0.894 | 0.923 | 1.118 | 0.976 | 1 | 6.7 | same |
| polybench/SYRK | 3 | 1.800 | 0.994 | 0.927 | 0.936 | 1.164 | 0.910 | 1 | 2.6 | same |
| rodinia/backprop | 3 | 0.246 | 0.916 | 0.875 | 0.943 | 1.061 | 1.077 | 0 | 0.0 | same |
| rodinia/gaussian | 3 | 0.915 | 0.993 | 0.796 | 0.886 | 1.099 | 0.823 | 1 | 2.3 | same |
| rodinia/hotspot | 3 | 0.301 | 0.996 | 0.982 | 0.982 | 0.991 | 0.991 | 0 | 0.0 | same |
| rodinia/lud | 3 | 2.087 | 1.000 | 0.999 | 0.995 | 1.003 | 1.000 | 0 | 0.0 | same |

| programs | n | default B/N | base JIT cold | JIT cold | JIT warm | JIT second run |
|---|---:|---:|---:|---:|---:|---:|
| polybench | 7 | 0.968 | 0.906 | 0.973 | 1.055 | 0.973 |
| rodinia | 4 | 0.976 | 0.909 | 0.950 | 1.037 | 0.968 |
| hecbench | 7 | 0.981 | 0.920 | 0.953 | 1.045 | 0.961 |
| all | 18 | 0.975 | 0.912 | 0.960 | 1.047 | 0.967 |
| with objects | 5 | 0.988 | 0.877 | 0.912 | 1.144 | 0.892 |
| without | 13 | 0.969 | 0.926 | 0.979 | 1.012 | 0.998 |
| 0.2 s and longer | 14 | 0.966 | 0.913 | 0.962 | 1.061 | 0.962 |

default B/N: 6 of 18 more than 2 % slower: polybench/GEMVER 0.827, polybench/2DCONV 0.910, rodinia/backprop 0.916, hecbench/radixsort2-cuda 0.967, hecbench/atomicAggregate-cuda 0.972, hecbench/asmooth-cuda 0.978
base JIT cold: 16 of 18 more than 2 % slower: rodinia/gaussian 0.796, hecbench/asyncAllocation-cuda 0.842, polybench/GEMVER 0.851, hecbench/asmooth-cuda 0.855, rodinia/backprop 0.875, polybench/2DCONV 0.883, polybench/SYR2K 0.894, polybench/JACOBI2D 0.915, polybench/ADI 0.926, polybench/SYRK 0.927, hecbench/atomicAggregate-cuda 0.936, hecbench/bh-cuda 0.941, hecbench/bitonic-sort-cuda 0.946, polybench/JACOBI1D 0.953, hecbench/radixsort2-cuda 0.957, hecbench/bilateral-cuda 0.971
JIT cold: 10 of 18 more than 2 % slower: hecbench/asyncAllocation-cuda 0.853, rodinia/gaussian 0.886, hecbench/asmooth-cuda 0.903, polybench/SYR2K 0.923, polybench/JACOBI2D 0.925, polybench/SYRK 0.936, rodinia/backprop 0.943, hecbench/atomicAggregate-cuda 0.966, hecbench/bitonic-sort-cuda 0.969, hecbench/bh-cuda 0.971
JIT warm: 1 of 18 more than 2 % slower: polybench/JACOBI2D 0.972
JIT second run: 8 of 18 more than 2 % slower: rodinia/gaussian 0.823, hecbench/asyncAllocation-cuda 0.832, polybench/SYRK 0.910, hecbench/atomicAggregate-cuda 0.930, polybench/JACOBI1D 0.968, polybench/JACOBI2D 0.974, polybench/SYR2K 0.976, hecbench/bh-cuda 0.977
programs of 0.2 s or more more than 3 % slower on the changed build (cold, warm, second): [(0.8230796905918332, 'rodinia/gaussian', 'Nbs'), (0.8323675149071974, 'hecbench/asyncAllocation-cuda', 'Nbs'), (0.8525591397849461, 'hecbench/asyncAllocation-cuda', 'Nbc'), (0.885780660149066, 'rodinia/gaussian', 'Nbc'), (0.9097064322166641, 'polybench/SYRK', 'Nbs'), (0.9229119111135557, 'polybench/SYR2K', 'Nbc'), (0.9300628420839245, 'hecbench/atomicAggregate-cuda', 'Nbs'), (0.9356129501636959, 'polybench/SYRK', 'Nbc'), (0.9428680981595093, 'rodinia/backprop', 'Nbc'), (0.9661998525850268, 'hecbench/atomicAggregate-cuda', 'Nbc'), (0.9678419217357613, 'polybench/JACOBI1D', 'Nbs')]
same bits: 18 programs, 16 equal in every run of every side, default differs from itself: ['hecbench/bh-cuda', 'hecbench/radixsort2-cuda'], DIFFERENT: []
left out: {}
objects in the cache after each timed run (cold and second-run sides): {}
