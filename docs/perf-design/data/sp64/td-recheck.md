| program | runs | default s | default B/N | base JIT cold | base JIT warm | base JIT second run | JIT cold | JIT warm | JIT second run | N/B cold | N/B warm | N/B second run | objects (warm fill) | compile s | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecbench/bilateral-cuda | 9 | 0.287 | 1.007 | 0.963 | 0.987 | 0.986 | 0.982 | 0.999 | 0.992 | 1.020 | 1.013 | 1.005 | 0 | 0.0 | same |
| polybench/GEMVER | 9 | 0.310 | 1.024 | 0.910 | 0.934 | 1.007 | 0.854 | 0.995 | 0.939 | 0.938 | 1.065 | 0.933 | 0 | 0.0 | same |
| polybench/JACOBI1D | 9 | 0.755 | 0.996 | 0.977 | 0.966 | 0.956 | 0.959 | 0.958 | 0.971 | 0.981 | 0.992 | 1.016 | 0 | 0.0 | same |
| polybench/JACOBI2D | 9 | 0.172 | 0.990 | 0.912 | 0.963 | 0.963 | 0.924 | 0.973 | 0.968 | 1.013 | 1.011 | 1.005 | 0 | 0.0 | same |

| programs | n | default B/N | base JIT cold | base JIT warm | base JIT second run | JIT cold | JIT warm | JIT second run | N/B cold | N/B warm | N/B second run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| polybench | 3 | 1.003 | 0.933 | 0.954 | 0.975 | 0.911 | 0.975 | 0.959 | 0.977 | 1.022 | 0.984 |
| rodinia | 0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| hecbench | 1 | 1.007 | 0.963 | 0.987 | 0.986 | 0.982 | 0.999 | 0.992 | 1.020 | 1.013 | 1.005 |
| all | 4 | 1.004 | 0.940 | 0.962 | 0.978 | 0.928 | 0.981 | 0.967 | 0.988 | 1.020 | 0.989 |
| with objects | 0 | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |
| without | 4 | 1.004 | 0.940 | 0.962 | 0.978 | 0.928 | 0.981 | 0.967 | 0.988 | 1.020 | 0.989 |
| 0.2 s and longer | 3 | 1.009 | 0.950 | 0.962 | 0.983 | 0.930 | 0.984 | 0.967 | 0.979 | 1.023 | 0.984 |

default B/N: 0 of 4 more than 2 % slower
base JIT cold: 4 of 4 more than 2 % slower: polybench/GEMVER 0.910, polybench/JACOBI2D 0.912, hecbench/bilateral-cuda 0.963, polybench/JACOBI1D 0.977
base JIT warm: 3 of 4 more than 2 % slower: polybench/GEMVER 0.934, polybench/JACOBI2D 0.963, polybench/JACOBI1D 0.966
base JIT second run: 2 of 4 more than 2 % slower: polybench/JACOBI1D 0.956, polybench/JACOBI2D 0.963
JIT cold: 3 of 4 more than 2 % slower: polybench/GEMVER 0.854, polybench/JACOBI2D 0.924, polybench/JACOBI1D 0.959
JIT warm: 2 of 4 more than 2 % slower: polybench/JACOBI1D 0.958, polybench/JACOBI2D 0.973
JIT second run: 3 of 4 more than 2 % slower: polybench/GEMVER 0.939, polybench/JACOBI2D 0.968, polybench/JACOBI1D 0.971
N/B cold: 1 of 4 more than 2 % slower: polybench/GEMVER 0.938
N/B warm: 0 of 4 more than 2 % slower
N/B second run: 1 of 4 more than 2 % slower: polybench/GEMVER 0.933
the keep bar: programs of 0.2 s or more more than 3 % slower on the changed build than on the base build (cold, warm, second run, each against its own default): [(0.933, 'polybench/GEMVER', 'N/B second run'), (0.938, 'polybench/GEMVER', 'N/B cold')]
same bits: 4 programs, 4 equal in every run of every side, default differs from itself: [], DIFFERENT: []
left out: {}
objects in the cache after each timed run (cold and second-run sides): {}
GEMVER Ba [0.3034, 0.308, 0.311, 0.3129, 0.3175, 0.3233, 0.326, 0.3455, 0.3833]
GEMVER Bbc [0.3359, 0.3418, 0.3456, 0.3472, 0.3488, 0.3602, 0.3732, 0.3737, 0.4025]
GEMVER Bbw [0.3141, 0.315, 0.316, 0.3298, 0.3398, 0.3464, 0.3752, 0.3787, 0.3895]
GEMVER Bbs [0.3038, 0.3072, 0.3095, 0.3113, 0.3153, 0.3206, 0.326, 0.3292, 0.342]
GEMVER Na [0.3038, 0.3045, 0.3049, 0.31, 0.31, 0.3124, 0.315, 0.3226, 0.3585]
GEMVER Nbc [0.3351, 0.3426, 0.3457, 0.352, 0.3629, 0.3661, 0.3679, 0.3804, 0.3992]
GEMVER Nbw [0.3053, 0.3059, 0.308, 0.3114, 0.3116, 0.3144, 0.3194, 0.3343, 0.3696]
GEMVER Nbs [0.31, 0.3114, 0.3222, 0.324, 0.3301, 0.3374, 0.3495, 0.3529, 0.355]
