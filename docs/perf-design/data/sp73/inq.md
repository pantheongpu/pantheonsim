# inq's unit tests with blocks off and on (round sp73)

inq f939e63 (round sp70's build, `/tmp/sp69-inq`, read only), the final library of this round with `VGPU_AMD_BLOCKS` unset and `=1`, the
two sides of a test one after the other (`../sp72/scripts/inqab.sh` with `BUILD=p3/build`), on a quiet host. CPU seconds are user plus
system of the whole process. `out=` is the checksum of the test's output without its timestamps.

### One host thread (VGPU_THREADS=1), three passes, each run pinned to one physical core

| test | off: wall s / CPU s (best of n) | on: wall s / CPU s | CPU on / off | runs | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 26.3 / 25.8 | 25.4 / 25.0 | 0.967 | 3 + 3 | yes |
| ks_hamiltonian | 27.7 / 26.2 | 27.2 / 25.9 | 0.988 | 3 + 3 | yes |
| initial_guess | 66.5 / 65.2 | 64.2 / 62.8 | 0.964 | 3 + 3 | yes |
| poisson | 115.7 / 114.1 | 112.5 / 111.0 | 0.972 | 3 + 3 | yes |

Every run:

```
off1 steepest_descent 26.35 25.51 0.37 rc=0 pass=1 out=988a2f2e
on1 steepest_descent 25.40 24.55 0.41 rc=0 pass=1 out=988a2f2e
off1 ks_hamiltonian 27.72 25.59 0.81 rc=0 pass=1 out=4918124d
on1 ks_hamiltonian 27.95 25.26 0.80 rc=0 pass=1 out=4918124d
off1 initial_guess 67.00 64.94 0.66 rc=0 pass=1 out=def4f106
on1 initial_guess 65.38 63.38 0.56 rc=0 pass=1 out=def4f106
off1 poisson 116.16 112.95 1.18 rc=0 pass=1 out=bd6e875a
on1 poisson 113.24 110.41 1.12 rc=0 pass=1 out=bd6e875a
on2 steepest_descent 25.42 24.59 0.39 rc=0 pass=1 out=988a2f2e
off2 steepest_descent 26.47 25.63 0.38 rc=0 pass=1 out=988a2f2e
on2 ks_hamiltonian 27.32 25.16 0.82 rc=0 pass=1 out=4918124d
off2 ks_hamiltonian 27.67 25.45 0.81 rc=0 pass=1 out=4918124d
on2 initial_guess 65.04 62.99 0.56 rc=0 pass=1 out=def4f106
off2 initial_guess 66.78 64.62 0.69 rc=0 pass=1 out=def4f106
on2 poisson 112.49 109.80 1.19 rc=0 pass=1 out=bd6e875a
off2 poisson 115.84 113.08 1.16 rc=0 pass=1 out=bd6e875a
off3 steepest_descent 26.27 25.45 0.36 rc=0 pass=1 out=988a2f2e
on3 steepest_descent 25.52 24.68 0.40 rc=0 pass=1 out=988a2f2e
off3 ks_hamiltonian 27.86 25.45 0.79 rc=0 pass=1 out=4918124d
on3 ks_hamiltonian 27.24 25.10 0.83 rc=0 pass=1 out=4918124d
off3 initial_guess 66.51 64.54 0.66 rc=0 pass=1 out=def4f106
on3 initial_guess 64.15 62.26 0.58 rc=0 pass=1 out=def4f106
off3 poisson 115.69 113.21 1.15 rc=0 pass=1 out=bd6e875a
on3 poisson 113.32 110.28 1.18 rc=0 pass=1 out=bd6e875a
```
### The default host threads (8 on this host), two passes

| test | off: wall s / CPU s (best of n) | on: wall s / CPU s | CPU on / off | runs | all pass, same output |
|---|---:|---:|---:|---:|---|
| laplacian | 5.8 / 5.6 | 5.9 / 5.5 | 0.977 | 2 + 2 | yes |
| gradient | 10.0 / 9.0 | 10.3 / 8.9 | 0.998 | 2 + 2 | yes |
| steepest_descent | 15.1 / 40.4 | 14.8 / 39.3 | 0.973 | 2 + 2 | yes |
| ks_hamiltonian | 21.1 / 42.0 | 21.0 / 41.1 | 0.978 | 2 + 2 | yes |
| initial_guess | 35.9 / 126.3 | 35.5 / 125.7 | 0.995 | 2 + 2 | yes |
| transform | 42.9 / 56.8 | 42.7 / 56.5 | 0.994 | 2 + 2 | yes |
| electrons | 48.6 / 79.3 | 48.1 / 77.7 | 0.980 | 2 + 2 | yes |
| poisson | 64.0 / 188.3 | 63.2 / 182.5 | 0.969 | 2 + 2 | yes |

Every run:

```
off1 laplacian 5.83 5.39 0.22 rc=0 pass=1 out=fb35b478
on1 laplacian 5.93 5.29 0.21 rc=0 pass=1 out=fb35b478
off1 gradient 10.52 8.73 0.22 rc=0 pass=1 out=cada440c
on1 gradient 10.34 8.69 0.24 rc=0 pass=1 out=cada440c
off1 steepest_descent 15.13 39.54 0.88 rc=0 pass=1 out=988a2f2e
on1 steepest_descent 14.81 38.45 0.86 rc=0 pass=1 out=988a2f2e
off1 ks_hamiltonian 21.09 40.54 1.50 rc=0 pass=1 out=4918124d
on1 ks_hamiltonian 29.58 39.51 1.59 rc=0 pass=1 out=4918124d
off1 initial_guess 35.89 124.80 1.95 rc=0 pass=1 out=def4f106
on1 initial_guess 35.47 123.80 1.89 rc=0 pass=1 out=def4f106
off1 transform 42.90 56.69 0.86 rc=0 pass=1 out=9116dd42
on1 transform 42.71 55.57 0.89 rc=0 pass=1 out=9116dd42
off1 electrons 49.08 78.48 0.94 rc=0 pass=1 out=ce3cf4a9
on1 electrons 48.49 77.37 0.92 rc=0 pass=1 out=ce3cf4a9
off1 poisson 63.99 187.53 1.36 rc=0 pass=1 out=bd6e875a
on1 poisson 63.16 182.44 1.36 rc=0 pass=1 out=bd6e875a
on2 laplacian 6.03 5.25 0.22 rc=0 pass=1 out=fb35b478
off2 laplacian 5.93 5.39 0.21 rc=0 pass=1 out=fb35b478
on2 gradient 10.26 8.68 0.26 rc=0 pass=1 out=cada440c
off2 gradient 10.04 8.80 0.25 rc=0 pass=1 out=cada440c
on2 steepest_descent 14.75 38.57 0.88 rc=0 pass=1 out=988a2f2e
off2 steepest_descent 15.36 40.60 0.81 rc=0 pass=1 out=988a2f2e
on2 ks_hamiltonian 21.02 40.17 1.54 rc=0 pass=1 out=4918124d
off2 ks_hamiltonian 21.48 40.95 1.57 rc=0 pass=1 out=4918124d
on2 initial_guess 35.89 124.30 1.94 rc=0 pass=1 out=def4f106
off2 initial_guess 35.91 124.42 1.91 rc=0 pass=1 out=def4f106
on2 transform 43.19 56.12 0.87 rc=0 pass=1 out=9116dd42
off2 transform 42.98 55.91 0.91 rc=0 pass=1 out=9116dd42
on2 electrons 48.13 76.83 0.88 rc=0 pass=1 out=ce3cf4a9
off2 electrons 48.65 78.34 0.92 rc=0 pass=1 out=ce3cf4a9
on2 poisson 63.72 181.12 1.35 rc=0 pass=1 out=bd6e875a
off2 poisson 65.47 186.94 1.35 rc=0 pass=1 out=bd6e875a
```
