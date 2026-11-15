# inq's unit tests (round sp74)

`inq2.sh <tag> <passes> <threads|default> <build A> <build B> -- <tests>`: the two builds take turns, the order turned over each pass;
a table gives the best run of each side (`inq2tab.py`). CPU seconds are user plus system of the whole process (the test's own host code,
the ROCm runtime and the simulator). One host thread: pinned to cpus 14,15, `VGPU_THREADS=1`. Default: 8 host threads, not pinned. *base*:
origin/sp69-hotspot-start a33ead36; *final*: this round (`w4`). inq is round sp70's build for gfx942, used read-only.

## One host thread, knob off: the base against the final library

| test | base: wall s / CPU s | final: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 26.5 / 26.0 | 21.6 / 21.2 | 0.815 | 2 | yes |
| ks_hamiltonian | 27.8 / 26.4 | 27.3 / 25.9 | 0.984 | 2 | yes |
| initial_guess | 66.8 / 65.4 | 58.6 / 57.2 | 0.876 | 2 | yes |
| poisson | 117.8 / 116.2 | 112.7 / 110.8 | 0.954 | 2 | yes |

## One host thread, `VGPU_AMD_BLOCKS=1` on both sides

| test | base, blocks: wall s / CPU s | final, blocks: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 25.4 / 25.0 | 20.4 / 19.9 | 0.797 | 2 | yes |
| ks_hamiltonian | 27.4 / 25.9 | 26.9 / 25.5 | 0.985 | 2 | yes |
| initial_guess | 64.9 / 63.6 | 56.9 / 55.3 | 0.869 | 2 | yes |
| poisson | 113.3 / 111.8 | 109.0 / 107.1 | 0.958 | 2 | yes |

## One host thread: the base against the no-op control (one pass)

| test | base: wall s / CPU s | control: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 26.5 / 26.0 | 26.7 / 26.3 | 1.009 | 1 | yes |
| ks_hamiltonian | 27.8 / 26.3 | 27.7 / 26.4 | 1.003 | 1 | yes |
| initial_guess | 66.6 / 65.2 | 66.9 / 65.6 | 1.006 | 1 | yes |
| poisson | 117.2 / 115.0 | 117.3 / 115.9 | 1.007 | 1 | yes |

## Default threads (8), knob off

| test | base: wall s / CPU s | final: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| laplacian | 5.9 / 5.6 | 5.7 / 5.5 | 0.986 | 2 | yes |
| gradient | 9.9 / 9.0 | 9.8 / 8.9 | 0.986 | 2 | yes |
| steepest_descent | 15.4 / 41.5 | 13.2 / 34.0 | 0.821 | 2 | yes |
| ks_hamiltonian | 20.8 / 41.4 | 20.8 / 41.8 | 1.008 | 2 | yes |
| initial_guess | 35.3 / 126.9 | 33.7 / 114.3 | 0.901 | 2 | yes |
| transform | 42.7 / 57.4 | 41.9 / 56.2 | 0.980 | 2 | yes |
| electrons | 47.4 / 79.7 | 47.4 / 73.8 | 0.926 | 2 | yes |
| poisson | 63.6 / 189.9 | 62.9 / 182.3 | 0.960 | 2 | yes |

## Default threads (8), `VGPU_AMD_BLOCKS=1` on both sides

| test | base, blocks: wall s / CPU s | final, blocks: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| laplacian | 5.8 / 5.5 | 5.7 / 5.4 | 0.984 | 2 | yes |
| gradient | 9.9 / 9.0 | 10.0 / 8.9 | 0.999 | 2 | yes |
| steepest_descent | 14.8 / 39.7 | 12.8 / 32.5 | 0.820 | 2 | yes |
| ks_hamiltonian | 20.8 / 41.7 | 20.9 / 41.3 | 0.990 | 2 | yes |
| initial_guess | 35.4 / 125.7 | 33.5 / 112.4 | 0.894 | 2 | yes |
| transform | 43.4 / 56.5 | 43.4 / 55.0 | 0.973 | 2 | yes |
| electrons | 48.1 / 78.4 | 47.4 / 72.7 | 0.928 | 2 | yes |
| poisson | 63.3 / 183.6 | 61.7 / 175.0 | 0.953 | 2 | yes |

## The steps, one host thread, knob off

Step 1 (the row handlers) against the base:

| test | base: wall s / CPU s | step 1: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 26.5 / 26.0 | 22.1 / 21.7 | 0.834 | 2 | yes |
| ks_hamiltonian | 28.2 / 26.3 | 27.5 / 26.1 | 0.992 | 2 | yes |
| initial_guess | 67.3 / 65.4 | 60.3 / 58.8 | 0.899 | 2 | yes |
| poisson | 117.4 / 115.1 | 115.7 / 113.0 | 0.981 | 2 | yes |

Step 3 as kept against step 1:

| test | step 1: wall s / CPU s | step 3: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 22.2 / 21.8 | 21.7 / 21.2 | 0.975 | 2 | yes |
| ks_hamiltonian | 27.6 / 26.1 | 27.4 / 25.9 | 0.996 | 2 | yes |
| initial_guess | 60.2 / 58.6 | 59.3 / 57.6 | 0.982 | 2 | yes |
| poisson | 115.6 / 113.0 | 112.6 / 110.4 | 0.977 | 2 | yes |

Step 3 as first written against step 1 (not kept in that form; `bench.md`):

| test | step 1: wall s / CPU s | w3: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 22.2 / 21.7 | 22.0 / 21.5 | 0.992 | 2 | yes |
| ks_hamiltonian | 27.5 / 26.1 | 27.5 / 26.1 | 1.000 | 2 | yes |
| initial_guess | 60.5 / 58.7 | 60.0 / 58.0 | 0.987 | 2 | yes |
| poisson | 115.2 / 112.8 | 113.7 / 111.1 | 0.985 | 2 | yes |

## One host thread, knob off, the library after the test fix (`w5`, one pass)

| test | base: wall s / CPU s | final: wall s / CPU s | CPU, second / first | runs a side | all pass, same output |
|---|---:|---:|---:|---:|---|
| steepest_descent | 26.6 / 26.2 | 21.6 / 21.1 | 0.805 | 1 | yes |
| ks_hamiltonian | 27.9 / 26.4 | 28.2 / 25.9 | 0.981 | 1 | yes |
| initial_guess | 67.1 / 65.2 | 59.2 / 57.7 | 0.885 | 1 | yes |
| poisson | 118.0 / 115.8 | 113.1 / 110.5 | 0.954 | 1 | yes |
