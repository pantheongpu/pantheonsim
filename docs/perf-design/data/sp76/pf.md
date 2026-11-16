# Round sp76: Rodinia particlefilter_float, the switch of Round sp75 on (turbo) against off (turbob0)

server3, the stand-in driver (it refuses every module: a launch the mode sends is simulated), profile `nvidia/a10g`, the round's build,
`-x 128 -y 128 -z 10 -np 10000` (the benchmark's arguments). `fixtime.c` in `LD_PRELOAD` makes `time()` answer `FIXTIME`: the program's one
use of the clock is `seed[i] = time(0) * i`. The machine was shared: a pass made at another time reads up to 30 % apart; within a pass
the forms are interleaved run by run.

## One host thread, the clock fixed (`pf.py --threads 1 --runs 5`)

Median of 5 runs a form (shortest to longest); "outputs": how many different answers (XE, YE and the distance) the five runs of each form
printed, and over all fifteen.

| seed | off s (min to max) | turbo s (min to max) | turbob0 s (min to max) | turbo / turbob0 | outputs seen: off, turbo, turbob0 (all forms) | launches kept / failed on the card |
|---|---:|---:|---:|---:|---|---|
| 1763200000 | 1.390 (1.372 to 1.408) | 1.395 (1.377 to 1.414) | 1.382 (1.373 to 1.513) | 1.009 | 1, 1, 1 (1) | 15 / 21 |
| 1763200001 | 1.744 (1.687 to 1.848) | 1.749 (1.691 to 1.959) | 1.815 (1.727 to 1.892) | 0.964 | 1, 1, 1 (1) | 15 / 21 |
| 1763200002 | 1.893 (1.780 to 1.995) | 1.900 (1.771 to 1.921) | 1.900 (1.774 to 1.929) | 1.000 | 1, 1, 1 (1) | 15 / 21 |
| 1763200003 | 1.888 (1.883 to 1.920) | 1.879 (1.862 to 1.888) | 1.899 (1.868 to 1.944) | 0.989 | 1, 1, 1 (1) | 15 / 21 |
| 1763200004 | 1.876 (1.872 to 1.912) | 1.887 (1.874 to 1.935) | 1.880 (1.865 to 1.884) | 1.004 | 1, 1, 1 (1) | 15 / 21 |
| 1763200005 | 1.873 (1.857 to 1.903) | 1.869 (1.853 to 2.066) | 1.876 (1.865 to 1.896) | 0.996 | 1, 1, 1 (1) | 15 / 21 |

off: 30 runs, median 1.874 s, mean 1.785 s, min 1.372, max 1.995
turbo: 30 runs, median 1.869 s, mean 1.789 s, min 1.377, max 2.066
turbob0: 30 runs, median 1.875 s, mean 1.794 s, min 1.373, max 1.944

An earlier pass of the same (a quieter hour; 1.39 to 1.43 s a run) read turbo over turbob0 1.001, 0.995, 0.991, 1.006, 1.004 and 1.004.

The mode's decisions, one logged run a form (`pf.py --runs 0`): every line that says what a launch did, with what a sample measured
taken out. Seeds 1763200000 to 1763200004, 1763200006 and 1763200007: the same (61 lines; 7 launches said not to be sampled in each
form). Seed 1763200005, four times: the same twice, and twice the one sample's line differed in how many warps the copy found in a
followed loop after its millisecond (288 against 292 of 304, once each way round): a count read within a time, now taken out too. What
does differ between the forms, always, is the text of the reason for three first launches that neither form samples: with the switch on a
grid of 20 blocks is a candidate for bounded blocks and is refused by the floor (`static estimate under the floor and the process too
young`) or by its earlier launch, with it off it is refused as `a grid under 256 blocks and no counted loop to follow`; and with that,
whether `host_gpu_kept` prints two such launches of a kernel on one line or two.

## Three host threads, one seed, three times over (`pf.py --threads 3 --runs 7`)

| seed | off s (min to max) | turbo s (min to max) | turbob0 s (min to max) | turbo / turbob0 | outputs seen: off, turbo, turbob0 (all forms) |
|---|---:|---:|---:|---:|---|
| 1763200005 | 1.172 (1.028 to 1.258) | 1.129 (0.940 to 1.294) | 1.082 (0.977 to 1.413) | 1.044 | 4, 7, 6 (15) |
| 1763200005 | 1.170 (1.040 to 1.361) | 1.054 (0.747 to 1.360) | 1.193 (1.045 to 1.331) | 0.884 | 5, 7, 6 (14) |
| 1763200005 | 1.168 (1.038 to 1.301) | 1.192 (1.024 to 1.383) | 1.092 (0.932 to 1.300) | 1.092 | 6, 5, 6 (15) |

The same seed, the same decisions, and medians of seven runs that put turbo 4 % over, 12 % under and 9 % over turbob0: the program prints
4 to 7 different answers in 7 runs of a form.

## Four host threads, the clock fixed at 1763200000, run by run (`pf_race.py --threads 4 --runs 30`)

The first launch of `find_index_kernel` (the simulator's time for it, from its sample's line) falls in two groups, cut at 60 ms (the
widest gap between two times next to each other, a factor of 1.8). `turbo1` is turbo with `VGPU_HOST_GPU_SAMPLE_BLOCK=1` in the
environment: the rule on, the environment the size of turbob0's.

| form | runs | short first launch | long first launch | short: ms | long: ms | answers seen | most frequent answer, runs |
|---|---:|---:|---:|---|---|---:|---:|
| turbo | 30 | 6 | 24 | 7 to 31 | 82 to 216 | 5 | 23 |
| turbo1 | 30 | 10 | 20 | 8 to 44 | 88 to 193 | 8 | 16 |
| turbob0 | 30 | 9 | 21 | 8 to 41 | 90 to 235 | 9 | 18 |

The answers of the 90 runs, and the first launch that went with each:

| answer (XE, YE) | runs | first launch of find_index_kernel ms |
|---|---:|---|
| 48.507576 72.415690 | 57 | 82 to 235 |
| 48.453971 72.428539 | 17 | 7 to 22 |
| 48.454830 72.428936 | 2 | 22 to 31 |
| 48.498079 72.426825 | 2 | 41 to 44 |
| 48.544914 72.433630 | 1 | 8 to 8 |
| (other answers) | 11 | |

Fisher's exact test on short against long: turbo against turbob0 p = 0.55, turbo against the other two together p = 0.32 (Round sp75's
card run: 12 of 30 turbo runs and 19 of 30 turbob0 runs in the fast group, p = 0.12).

## One host thread, the clock fixed, run by run (`pf_race.py --threads 1 --runs 5`)

| form | runs | first launch of find_index_kernel ms | answers |
|---|---:|---|---|
| turbo | 5 | 19.2 to 20.4 | 48.562348 72.892182 |
| turbo1 | 5 | 19.5 to 21.1 | 48.562348 72.892182 |
| turbob0 | 5 | 20.0 to 21.9 | 48.562348 72.892182 |

## Four host threads, the clock free (12 turbo runs)

First launch of `find_index_kernel`: 9, 9, 88, 88, 90, 95, 97, 97, 100, 101, 102, 112 ms; 12 answers.

## Why

`normalize_weights_kernel` has a race between blocks: block 0 writes `u[0]`, thread 0 of every other block reads it, and nothing orders
the two (the docstring of `pf_race.py` and the round's text have the source). With one host thread block 0 runs first.
