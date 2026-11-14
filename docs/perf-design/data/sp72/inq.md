# inq's unit tests with blocks off and on (round sp72)

inq f939e63, `all_unit_tests` built for gfx942 against ROCm 6.4.4 (round sp70's build), on the simulated MI300X, this branch's library with
`VGPU_AMD_BLOCKS` unset and `=1`, the two sides of a test one after the other (`scripts/inqab.sh`). CPU seconds are user plus system of the
whole process (the host program and the run-time compile of VkFFT's kernels included). `out=` is the checksum of the test's output without
its timestamps. The host was shared (1-minute load 5 to 11), so the best run of each side is given and single runs are in the listing: the
same test with the same knob differs by 10 to 20 % between passes.

### One host thread (VGPU_THREADS=1), two passes, each run pinned to one physical core

| test | off: wall s / CPU s (best of n) | on: wall s / CPU s | CPU on / off | runs | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 39.6 / 39.3 | 33.2 / 33.1 | 0.843 | 2 + 2 | yes |
| ks_hamiltonian | 44.5 / 33.1 | 38.8 / 29.7 | 0.897 | 2 + 2 | yes |
| initial_guess | 101.4 / 92.2 | 93.2 / 88.5 | 0.960 | 2 + 2 | yes |
| poisson | 158.8 / 156.8 | 167.6 / 157.0 | 1.002 | 2 + 2 | yes |

Every run:

```
off1 steepest_descent 44.51 43.45 0.64 rc=0 pass=1 out=988a2f2e
on1 steepest_descent 39.14 38.50 0.60 rc=0 pass=1 out=988a2f2e
off1 ks_hamiltonian 55.20 47.07 1.40 rc=0 pass=1 out=4918124d
on1 ks_hamiltonian 57.25 37.35 1.27 rc=0 pass=1 out=4918124d
off1 initial_guess 102.91 94.18 0.74 rc=0 pass=1 out=def4f106
on1 initial_guess 95.03 89.42 0.97 rc=0 pass=1 out=def4f106
off1 poisson 191.58 168.11 1.62 rc=0 pass=1 out=bd6e875a
on1 poisson 167.57 155.48 1.57 rc=0 pass=1 out=bd6e875a
on2 steepest_descent 33.22 32.56 0.55 rc=0 pass=1 out=988a2f2e
off2 steepest_descent 39.62 38.69 0.59 rc=0 pass=1 out=988a2f2e
on2 ks_hamiltonian 38.82 28.77 0.95 rc=0 pass=1 out=4918124d
off2 ks_hamiltonian 44.45 32.19 0.95 rc=0 pass=1 out=4918124d
on2 initial_guess 93.23 87.55 0.91 rc=0 pass=1 out=def4f106
off2 initial_guess 101.41 91.43 0.72 rc=0 pass=1 out=def4f106
on2 poisson 176.41 160.68 1.54 rc=0 pass=1 out=bd6e875a
off2 poisson 158.84 155.21 1.57 rc=0 pass=1 out=bd6e875a
```
### The default host threads (8 on this host), one pass (poisson twice: its first run with the knob on met the library being rebuilt and is left out)

| test | off: wall s / CPU s (best of n) | on: wall s / CPU s | CPU on / off | runs | all pass, same output |
|---|---:|---:|---:|---:|---|
| laplacian | 6.8 / 6.3 | 6.5 / 6.2 | 0.987 | 1 + 1 | yes |
| gradient | 11.5 / 10.1 | 10.9 / 10.1 | 0.999 | 1 + 1 | yes |
| steepest_descent | 16.9 / 46.3 | 16.0 / 43.2 | 0.933 | 1 + 1 | yes |
| ks_hamiltonian | 24.0 / 46.4 | 23.6 / 46.2 | 0.995 | 1 + 1 | yes |
| initial_guess | 40.7 / 137.6 | 39.6 / 131.0 | 0.952 | 1 + 1 | yes |
| transform | 47.9 / 64.1 | 50.3 / 67.3 | 1.050 | 1 + 1 | yes |
| electrons | 71.6 / 104.5 | 74.9 / 97.5 | 0.933 | 1 + 1 | yes |
| poisson | 81.1 / 189.3 | 106.0 / 198.9 | 1.051 | 2 + 1 | yes |

Every run:

```
off1 laplacian 6.77 6.08 0.22 rc=0 pass=1 out=fb35b478
on1 laplacian 6.52 5.98 0.24 rc=0 pass=1 out=fb35b478
off1 gradient 11.46 9.90 0.23 rc=0 pass=1 out=cada440c
on1 gradient 10.92 9.85 0.27 rc=0 pass=1 out=cada440c
off1 steepest_descent 16.93 45.35 0.93 rc=0 pass=1 out=988a2f2e
on1 steepest_descent 16.02 42.26 0.90 rc=0 pass=1 out=988a2f2e
off1 ks_hamiltonian 24.02 44.82 1.57 rc=0 pass=1 out=4918124d
on1 ks_hamiltonian 23.59 44.65 1.53 rc=0 pass=1 out=4918124d
off1 initial_guess 40.72 135.51 2.10 rc=0 pass=1 out=def4f106
on1 initial_guess 39.62 128.89 2.07 rc=0 pass=1 out=def4f106
off1 transform 47.94 63.10 0.96 rc=0 pass=1 out=9116dd42
on1 transform 50.33 66.25 1.01 rc=0 pass=1 out=9116dd42
off1 electrons 71.63 103.31 1.15 rc=0 pass=1 out=ce3cf4a9
on1 electrons 74.92 96.45 1.04 rc=0 pass=1 out=ce3cf4a9
off1 poisson 111.49 187.88 1.46 rc=0 pass=1 out=bd6e875a
off2 poisson 81.07 195.44 1.43 rc=0 pass=1 out=bd6e875a
on2 poisson 105.97 197.44 1.47 rc=0 pass=1 out=bd6e875a
```
