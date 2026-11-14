| program | runs | default s | default B/N | base JIT cold | base JIT warm | base JIT second run | JIT cold | JIT cold, header there | JIT warm | JIT second run | N/B cold | N/B warm | N/B second run | objects (warm fill) | compile s | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecbench/asmooth-cuda | 5 | 0.082 | 1.004 | 0.906 | 0.975 | 0.975 | 0.898 | 0.973 | 0.967 | 0.980 | 0.991 | 0.992 | 1.005 | 0 | 0.0 | same |
| hecbench/asyncAllocation-cuda | 5 | 0.988 | 1.012 | 0.858 | 1.245 | 0.851 | 0.903 | 0.904 | 1.232 | 0.909 | 1.053 | 0.990 | 1.068 | 1 | 2.3 | same |
| hecbench/atomicAggregate-cuda | 3 | 0.908 | 1.010 | 0.962 | 1.100 | 0.921 | 0.974 | 0.971 | 1.080 | 0.952 | 1.012 | 0.981 | 1.034 | 1 | 0.8 | same |
| hecbench/bh-cuda | 5 | 0.205 | 1.006 | 0.978 | 0.985 | 0.972 | 0.961 | 0.963 | 0.990 | 0.998 | 0.983 | 1.005 | 1.027 | 0 | 0.0 | default differs from itself |
| hecbench/bilateral-cuda | 5 | 0.294 | 0.995 | 0.988 | 1.011 | 0.996 | 0.958 | 1.009 | 1.007 | 1.010 | 0.970 | 0.996 | 1.014 | 0 | 0.0 | same |
| hecbench/bitonic-sort-cuda | 5 | 0.097 | 0.983 | 0.975 | 1.000 | 1.002 | 0.972 | 1.009 | 1.018 | 1.023 | 0.996 | 1.018 | 1.021 | 0 | 0.0 | same |
| hecbench/radixsort2-cuda | 5 | 0.961 | 0.989 | 0.982 | 0.973 | 0.973 | 1.013 | 1.066 | 1.055 | 1.075 | 1.031 | 1.084 | 1.105 | 0 | 0.0 | default differs from itself |
| polybench/2DCONV | 5 | 0.534 | 0.992 | 0.964 | 0.992 | 1.005 | 0.979 | 0.991 | 1.014 | 1.003 | 1.016 | 1.023 | 0.998 | 0 | 0.0 | same |
| polybench/ADI | 5 | 0.147 | 1.006 | 0.947 | 0.998 | 0.991 | 0.945 | 0.982 | 0.991 | 0.997 | 0.998 | 0.993 | 1.006 | 0 | 0.0 | same |
| polybench/GEMVER | 5 | 0.307 | 1.020 | 0.899 | 1.019 | 1.011 | 0.881 | 0.867 | 0.974 | 0.930 | 0.980 | 0.957 | 0.919 | 0 | 0.0 | same |
| polybench/JACOBI1D | 5 | 0.759 | 1.009 | 1.023 | 0.986 | 0.980 | 0.971 | 0.972 | 0.976 | 0.967 | 0.949 | 0.990 | 0.987 | 0 | 0.0 | same |
| polybench/JACOBI2D | 5 | 0.173 | 1.026 | 0.953 | 1.004 | 1.010 | 0.921 | 0.967 | 0.973 | 0.980 | 0.966 | 0.969 | 0.970 | 0 | 0.0 | same |
| polybench/SYR2K | 3 | 3.551 | 0.979 | 0.933 | 1.130 | 1.146 | 0.955 | 0.980 | 1.166 | 1.166 | 1.023 | 1.031 | 1.017 | 1 | 2.6 | same |
| polybench/SYRK | 3 | 1.808 | 1.003 | 0.935 | 1.113 | 0.908 | 0.942 | 0.933 | 1.112 | 0.928 | 1.007 | 0.999 | 1.022 | 1 | 4.6 | same |
| rodinia/backprop | 5 | 0.228 | 0.996 | 0.896 | 0.997 | 0.997 | 0.887 | 1.001 | 0.998 | 0.998 | 0.990 | 1.001 | 1.001 | 0 | 0.0 | same |
| rodinia/gaussian | 5 | 0.895 | 1.006 | 0.893 | 1.106 | 0.836 | 0.929 | 0.946 | 1.096 | 0.904 | 1.041 | 0.991 | 1.082 | 1 | 0.8 | same |
| rodinia/hotspot | 5 | 0.289 | 1.018 | 0.994 | 0.992 | 1.004 | 0.982 | 0.990 | 0.989 | 0.992 | 0.988 | 0.997 | 0.988 | 0 | 0.0 | same |
| rodinia/lud | 3 | 2.052 | 1.016 | 0.997 | 1.009 | 0.999 | 0.990 | 0.991 | 0.988 | 0.984 | 0.993 | 0.979 | 0.985 | 0 | 0.0 | same |

| programs | n | default B/N | base JIT cold | base JIT warm | base JIT second run | JIT cold | JIT cold, header there | JIT warm | JIT second run | N/B cold | N/B warm | N/B second run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench | 7 | 1.005 | 0.950 | 1.033 | 1.005 | 0.941 | 0.955 | 1.027 | 0.993 | 0.991 | 0.994 | 0.988 |
| rodinia | 4 | 1.009 | 0.944 | 1.025 | 0.956 | 0.946 | 0.982 | 1.017 | 0.969 | 1.003 | 0.992 | 1.013 |
| hecbench | 7 | 1.000 | 0.949 | 1.038 | 0.954 | 0.953 | 0.984 | 1.047 | 0.991 | 1.005 | 1.009 | 1.039 |
| all | 18 | 1.004 | 0.948 | 1.033 | 0.974 | 0.947 | 0.972 | 1.032 | 0.987 | 0.999 | 0.999 | 1.013 |
| with objects | 5 | 1.002 | 0.916 | 1.138 | 0.926 | 0.940 | 0.946 | 1.136 | 0.967 | 1.027 | 0.998 | 1.044 |
| without | 13 | 1.005 | 0.961 | 0.995 | 0.993 | 0.950 | 0.982 | 0.995 | 0.995 | 0.988 | 1.000 | 1.001 |
| 0.2 s and longer | 14 | 1.004 | 0.949 | 1.044 | 0.969 | 0.951 | 0.969 | 1.046 | 0.985 | 1.002 | 1.001 | 1.017 |

default B/N: 1 of 18 more than 2 % slower: polybench/SYR2K 0.979
base JIT cold: 13 of 18 more than 2 % slower: hecbench/asyncAllocation-cuda 0.858, rodinia/gaussian 0.893, rodinia/backprop 0.896, polybench/GEMVER 0.899, hecbench/asmooth-cuda 0.906, polybench/SYR2K 0.933, polybench/SYRK 0.935, polybench/ADI 0.947, polybench/JACOBI2D 0.953, hecbench/atomicAggregate-cuda 0.962, polybench/2DCONV 0.964, hecbench/bitonic-sort-cuda 0.975, hecbench/bh-cuda 0.978
base JIT warm: 2 of 18 more than 2 % slower: hecbench/radixsort2-cuda 0.973, hecbench/asmooth-cuda 0.975
base JIT second run: 8 of 18 more than 2 % slower: rodinia/gaussian 0.836, hecbench/asyncAllocation-cuda 0.851, polybench/SYRK 0.908, hecbench/atomicAggregate-cuda 0.921, hecbench/bh-cuda 0.972, hecbench/radixsort2-cuda 0.973, hecbench/asmooth-cuda 0.975, polybench/JACOBI1D 0.980
JIT cold: 15 of 18 more than 2 % slower: polybench/GEMVER 0.881, rodinia/backprop 0.887, hecbench/asmooth-cuda 0.898, hecbench/asyncAllocation-cuda 0.903, polybench/JACOBI2D 0.921, rodinia/gaussian 0.929, polybench/SYRK 0.942, polybench/ADI 0.945, polybench/SYR2K 0.955, hecbench/bilateral-cuda 0.958, hecbench/bh-cuda 0.961, polybench/JACOBI1D 0.971, hecbench/bitonic-sort-cuda 0.972, hecbench/atomicAggregate-cuda 0.974, polybench/2DCONV 0.979
JIT cold, header there: 10 of 18 more than 2 % slower: polybench/GEMVER 0.867, hecbench/asyncAllocation-cuda 0.904, polybench/SYRK 0.933, rodinia/gaussian 0.946, hecbench/bh-cuda 0.963, polybench/JACOBI2D 0.967, hecbench/atomicAggregate-cuda 0.971, polybench/JACOBI1D 0.972, hecbench/asmooth-cuda 0.973, polybench/SYR2K 0.980
JIT warm: 4 of 18 more than 2 % slower: hecbench/asmooth-cuda 0.967, polybench/JACOBI2D 0.973, polybench/GEMVER 0.974, polybench/JACOBI1D 0.976
JIT second run: 8 of 18 more than 2 % slower: rodinia/gaussian 0.904, hecbench/asyncAllocation-cuda 0.909, polybench/SYRK 0.928, polybench/GEMVER 0.930, hecbench/atomicAggregate-cuda 0.952, polybench/JACOBI1D 0.967, hecbench/asmooth-cuda 0.980, polybench/JACOBI2D 0.980
N/B cold: 4 of 18 more than 2 % slower: polybench/JACOBI1D 0.949, polybench/JACOBI2D 0.966, hecbench/bilateral-cuda 0.970, polybench/GEMVER 0.980
N/B warm: 3 of 18 more than 2 % slower: polybench/GEMVER 0.957, polybench/JACOBI2D 0.969, rodinia/lud 0.979
N/B second run: 2 of 18 more than 2 % slower: polybench/GEMVER 0.919, polybench/JACOBI2D 0.970
the keep bar: programs of 0.2 s or more more than 3 % slower on the changed build than on the base build (cold, warm, second run, each against its own default): [(0.919, 'polybench/GEMVER', 'N/B second run'), (0.949, 'polybench/JACOBI1D', 'N/B cold'), (0.957, 'polybench/GEMVER', 'N/B warm'), (0.97, 'hecbench/bilateral-cuda', 'N/B cold')]
same bits: 18 programs, 16 equal in every run of every side, default differs from itself: ['hecbench/bh-cuda', 'hecbench/radixsort2-cuda'], DIFFERENT: []
left out: {}
objects in the cache after each timed run (cold and second-run sides): {'polybench/SYR2K': {'Bbc': [1, 1, 1], 'Bbw': [1, 1, 1], 'Bbs': [1, 1, 1], 'Nbc': [1, 1, 1], 'Nbp': [1, 1, 1], 'Nbs': [1, 1, 1]}, 'polybench/SYRK': {'Bbw': [1, 1, 1]}, 'rodinia/gaussian': {'Bbw': [1, 1, 1, 1, 1]}, 'hecbench/asyncAllocation-cuda': {'Bbw': [1, 1, 1, 1, 1]}, 'hecbench/atomicAggregate-cuda': {'Bbw': [1, 1, 1]}}
