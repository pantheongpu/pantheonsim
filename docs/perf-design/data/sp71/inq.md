# Round sp71: inq with the hiprtc cache off, cold and warm

`all_unit_tests "[dir][test]"` on the simulated MI300X (`inq_one_mode.sh`), server3 shared with another round's builds and measurements. Wall
seconds / CPU-seconds (user plus system). "Cold" is one cache for the six tests, empty before the first; "warm" is the same cache afterwards.
`inq-runs.txt` has every run: `<test> <wall> <user> <system> rc pass out=<md5 of the output without timestamps> c=<compiles> mem= disk= real=
real_s=<seconds in ROCm's compiler> nocache=<not cacheable>`. The `out=` of a test is the same in every run.

## Median of three passes

| test | cache off | cold cache | warm cache |
|---|---|---|---|
| laplacian | 6.7 s / 6 | 8.8 s / 8 | 2.0 s / 2 |
| gradient | 11.8 s / 10 | 8.0 s / 7 | 1.7 s / 1 |
| transform | 46.2 s / 61 | 46.4 s / 56 | 6.6 s / 22 |
| ks_hamiltonian | 24.9 s / 49 | 21.8 s / 48 | 12.4 s / 41 |
| initial_guess | 54.4 s / 177 | 40.8 s / 151 | 25.4 s / 121 |
| electrons | 73.6 s / 107 | 46.0 s / 74 | 16.0 s / 49 |
| all six | 217.4 s / 410 | 171.8 s / 345 | 64.2 s / 237 |

## Each pass (wall s / CPU-s)

| test | off 1 | off 2 | off 3 | cold 1 | cold 2 | cold 3 | warm 1 | warm 2 | warm 3 |
|---|---|---|---|---|---|---|---|---|---|
| laplacian | 5.8 / 5 | 6.7 / 6 | 12.3 / 7 | 6.7 / 7 | 8.8 / 9 | 10.6 / 8 | 1.9 / 2 | 2.1 / 3 | 2.0 / 2 |
| gradient | 11.8 / 9 | 11.5 / 10 | 14.4 / 11 | 6.7 / 6 | 8.5 / 8 | 8.0 / 7 | 1.7 / 1 | 1.9 / 2 | 1.7 / 1 |
| transform | 46.2 / 61 | 43.0 / 57 | 62.5 / 70 | 30.4 / 47 | 46.4 / 56 | 48.1 / 62 | 5.6 / 22 | 6.6 / 30 | 7.5 / 21 |
| ks_hamiltonian | 23.6 / 46 | 24.9 / 49 | 35.7 / 59 | 18.9 / 41 | 21.8 / 48 | 23.9 / 54 | 11.1 / 35 | 13.1 / 43 | 12.4 / 41 |
| initial_guess | 38.0 / 132 | 61.1 / 203 | 54.4 / 177 | 35.5 / 135 | 40.8 / 155 | 49.1 / 151 | 21.7 / 121 | 25.5 / 121 | 25.4 / 141 |
| electrons | 51.2 / 84 | 86.5 / 131 | 73.6 / 107 | 40.4 / 74 | 46.8 / 92 | 46.0 / 74 | 14.7 / 49 | 16.0 / 54 | 16.4 / 48 |

## Where the compiles came from (the same in every pass)

| test | compiles | cold: ROCm / memory / files | warm: files / memory |
|---|---|---|---|
| laplacian | 24 | 24 / 0 / 0 | 24 / 0 |
| gradient | 48 | 24 / 0 / 24 | 48 / 0 |
| transform | 224 | 130 / 94 / 0 | 130 / 94 |
| ks_hamiltonian | 68 | 44 / 24 / 0 | 44 / 24 |
| initial_guess | 98 | 71 / 27 / 0 | 71 / 27 |
| electrons | 200 | 126 / 74 / 0 | 126 / 74 |

## Four tests at once (laplacian, gradient, transform, ks_hamiltonian), `pass3_and_four_at_once.sh`

| | seconds until all four are done | CPU-s | compiles by ROCm / from memory / from files |
|---|---|---|---|
| cache off | 70.0, 63.5 | 166, 167 | 364 / 0 / 0 |
| cold (one empty cache for the four) | 39.3, 42.7 | 137, 125 | 231 / 118 / 15; 230 / 118 / 16 |
| warm | 29.5, 16.6 | 73, 75 | 0 / 118 / 246 |

## ROCm's own cache (`AMD_COMGR_CACHE=1`), transform, one run each

| | wall | CPU-s |
|---|---|---|
| without | 60.1 s | 72.6 |
| cold | 83.8 s | 66.5 |
| warm | 67.4 s | 62.3 |

## The last commit, once more

The passes above ran the library as it was before two small changes (`hiprtcDestroyProgram` no longer clears the caller's handle; one
redundant test of the off switch went). The library of the last commit, cold then warm, one run: the same outputs, the same counts of
compiles; warm 2.1, 2.0, 6.2, 11.5, 26.4 and 18.2 s (`inq-runs.txt`).
