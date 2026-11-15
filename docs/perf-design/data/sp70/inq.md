# Round sp70: the eight inq tests, build by build

Wall seconds / CPU-seconds (user plus system) of `all_unit_tests "[dir][test]"` on the simulated MI300X, server3 with its 16 hardware threads free,
one run a cell (`inq8.sh`). Each column is the build with the changes up to it: base c7af104d, 1 the register-file pool, 2 one host thread a
physical core, 3 fmin and fmax of doubles, 4 quads of managed memory. The base, 1 and 2 are pass 3 of `inq-passes.txt`, 3 and 4 pass 4 (the
commits as they are; passes 1 and 2 are the first form of changes 1 and 3). The md5 of each test's output without its timestamp lines is the
same in every build and every pass (the `out=` field of `inq-passes.txt`).

| test | base | 1 pool | 2 cores | 3 fmin/fmax | 4 managed quads |
|---|---|---|---|---|---|
| laplacian | 5.8 s / 7 | 5.9 s / 7 | 5.8 s / 6 | 5.8 s / 6 | 5.8 s / 6 |
| gradient | 9.9 s / 10 | 10.0 s / 9 | 9.9 s / 9 | 9.9 s / 9 | 10.2 s / 9 |
| steepest_descent | 16.8 s / 55 | 16.6 s / 54 | 16.4 s / 45 | 15.5 s / 41 | 15.3 s / 41 |
| ks_hamiltonian | 21.0 s / 68 | 20.9 s / 67 | 20.9 s / 42 | 22.3 s / 42 | 21.1 s / 43 |
| initial_guess | 44.3 s / 374 | 38.3 s / 280 | 37.2 s / 134 | 35.4 s / 126 | 36.9 s / 125 |
| transform | 42.7 s / 78 | 42.7 s / 76 | 42.7 s / 58 | 43.3 s / 58 | 44.9 s / 57 |
| electrons | 50.4 s / 162 | 48.2 s / 122 | 48.0 s / 82 | 48.3 s / 79 | 48.6 s / 79 |
| poisson | 64.2 s / 326 | 62.9 s / 327 | 63.5 s / 189 | 63.4 s / 190 | 64.5 s / 188 |
| all eight | 255.1 s / 1080 | 245.6 s / 942 | 244.4 s / 565 | 243.8 s / 551 | 247.2 s / 547 |

`gradient` on the base is the mean of passes 1 and 2 (9.8 and 10.1 s); its run in pass 3 (12.9 s) overlapped a file copy.

## One host thread (`VGPU_THREADS=1`), `one_thread.sh`

Wall seconds. "3" and "4" here are the builds measured before the SDWA comparison handler was taken out of change 3 (it is in both).

| test | base | 3 | 4 |
|---|---|---|---|
| ks_hamiltonian | 30.5, 30.9, 31.5 | 30.7, 30.8, 30.7 | 27.4, 27.5, 27.5 |
| steepest_descent | 30.7, 30.3 | 26.6, 26.5 | 26.2, 26.3 |
| transform | 49.9 | 49.7, 49.4 | 49.6, 49.1 |
| poisson | 122.5 | 122.6 | 115.1 |

## Four copies at once (`four_at_once.sh`): commit 1 against commit 2

Seconds until all four are done (CPU-seconds of the four together).

| test | 16 threads each (commit 1) | 8 threads each (commit 2) |
|---|---|---|
| steepest_descent | 34.6 (338), 36.8 (341) | 34.6 (337), 34.5 (334) |
| initial_guess | 98.0 (1218), 97.5 (1217) | 93.7 (1129), 94.0 (1124) |
| poisson | 144.3 (1610), 146.0 (1624) | 139.4 (1494) |

(The second `poisson` run of commit 2, 141.2 s, overlapped the start of another measurement.)

## The two halves of the first form of change 3 on initial_guess (16 threads, wall / CPU-seconds)

| build | run 1 | run 2 |
|---|---|---|
| commit 2 | 36.1 / 134 | 36.4 / 133 |
| plus fmin/fmax | 36.0 / 126 | |
| plus the SDWA comparison handler alone | 36.1 / 133 | 36.5 / 133 |
