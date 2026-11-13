| program | runs | default s | default B/N | base JIT cold | JIT cold | JIT warm | JIT second run | objects (warm fill) | compile s | same bits |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| hecbench/asyncAllocation-cuda | 5 | 3.263 | 1.020 | 0.934 | 0.911 | 1.240 | 0.996 | 1 | 2.0 | same |
| hecbench/bgmv-cuda | 5 | 0.012 | 0.952 | 0.894 | 0.976 | 0.954 | 0.947 | 0 | 0.0 | same |
| polybench/JACOBI1D | 5 | 1.059 | 0.995 | 0.978 | 0.989 | 0.990 | 0.988 | 0 | 0.0 | same |
| polybench/SYRK | 5 | 5.402 | 0.998 | 0.924 | 0.924 | 1.127 | 1.149 | 1 | 2.3 | same |
| rodinia/gaussian | 5 | 3.693 | 1.014 | 0.946 | 0.934 | 1.081 | 0.934 | 1 | 0.9 | same |

| programs | n | default B/N | base JIT cold | JIT cold | JIT warm | JIT second run |
|---|---:|---:|---:|---:|---:|---:|
| polybench | 2 | 0.996 | 0.951 | 0.956 | 1.056 | 1.065 |
| rodinia | 1 | 1.014 | 0.946 | 0.934 | 1.081 | 0.934 |
| hecbench | 2 | 0.985 | 0.914 | 0.943 | 1.088 | 0.971 |
| all | 5 | 0.995 | 0.935 | 0.946 | 1.074 | 1.000 |
| with objects | 3 | 1.010 | 0.935 | 0.923 | 1.147 | 1.022 |
| without | 2 | 0.973 | 0.935 | 0.983 | 0.972 | 0.967 |
| 0.2 s and longer | 4 | 1.007 | 0.945 | 0.939 | 1.106 | 1.013 |

default B/N: 1 of 5 more than 2 % slower: hecbench/bgmv-cuda 0.952
base JIT cold: 5 of 5 more than 2 % slower: hecbench/bgmv-cuda 0.894, polybench/SYRK 0.924, hecbench/asyncAllocation-cuda 0.934, rodinia/gaussian 0.946, polybench/JACOBI1D 0.978
JIT cold: 4 of 5 more than 2 % slower: hecbench/asyncAllocation-cuda 0.911, polybench/SYRK 0.924, rodinia/gaussian 0.934, hecbench/bgmv-cuda 0.976
JIT warm: 1 of 5 more than 2 % slower: hecbench/bgmv-cuda 0.954
JIT second run: 2 of 5 more than 2 % slower: rodinia/gaussian 0.934, hecbench/bgmv-cuda 0.947
programs of 0.2 s or more more than 3 % slower on the changed build (cold, warm, second): [(0.9105014930371446, 'hecbench/asyncAllocation-cuda', 'Nbc'), (0.9236779565388362, 'polybench/SYRK', 'Nbc'), (0.9337125521425863, 'rodinia/gaussian', 'Nbs'), (0.9338778193587538, 'rodinia/gaussian', 'Nbc')]
same bits: 5 programs, 5 equal in every run of every side, default differs from itself: [], DIFFERENT: []
left out: {}
objects in the cache after each timed run (cold and second-run sides): {'polybench/SYRK': {'Bbc': [1, 1, 1, 1, 1], 'Nbc': [1, 1, 1, 1, 1], 'Nbs': [1, 1, 1, 1, 1]}, 'rodinia/gaussian': {'Bbc': [1, 0, 0, 1, 1], 'Nbc': [0, 0, 0, 1, 1], 'Nbs': [1, 1, 1, 1, 1]}, 'hecbench/asyncAllocation-cuda': {'Nbs': [1, 1, 1, 1, 1]}}
