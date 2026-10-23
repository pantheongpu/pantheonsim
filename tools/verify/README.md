# tools/verify: corpus verification of two simulator builds

Runs the external program corpora on two simulator builds (A = a baseline, B = the build under
test) and reports, per corpus, how many programs gave identical output and counters, how many
differ, and how many failed. It changes no simulator source. It runs simulator builds only, unless you ask
for the optional real-card configuration `C` (see "The real card" below).

Written for the round 7 + 8 speed work (A = `origin/main` before round 8, B = `origin/r8-a`, with
`VGPU_TIER=1` and the PTX engine as extra configurations), but nothing in it is specific to that pair.

## Pieces

| file | what it does |
| --- | --- |
| `run_corpora.sh` | runs every program under every configuration, one at a time, and stores stdout, stderr, exit code, counters and output files |
| `summarize.py` | compares configurations pairwise and writes `report.md` and `pairs.tsv` into the results directory |
| `build_polybench_small.sh` | builds small-size copies of PolyBench/GPU (see below) |
| `build_pantheon.sh` | builds the pantheon workloads once, for both builds |
| `nvcc_locked.sh` | nvcc under the machine-wide build lock, niced and memory-capped, with `-cudart shared` |
| `probes/round9_probe.cu` | a small program that exercises the deliberate round 9 changes (tensor-core sums, NaN selection, conversions, -0 under .sat/.ftz, bf16 HFMA2, 64-bit saturation, VABSDIFF, launch errors) and prints raw bits; `round9_probe full` also dumps every double result. Build with `nvcc -arch=sm_86 -cudart shared`, run it on the card and under every configuration, diff the text |
| `fixed_time.c` | an `LD_PRELOAD` library that pins `time()` and `clock()`, for programs that seed their input from them (see below) |
| `compat.h` | maps `cudaThreadSynchronize` (removed in CUDA 12) for the older benchmark sources |

## Corpora

The programs are not part of this repository. Defaults point at the copies set up under `~/.cache`;
each has an environment override.

| corpus | default location | program list |
| --- | --- | --- |
| `hecbench` | `HECBENCH_DIR=~/.cache/HeCBench` | `reduced.txt` there: 34 benchmarks with reduced arguments (`HECBENCH_LIST`); aidw runs at `4 1 1`, aligned-types uses its `small` build and binomial its `binomial-small` build |
| `rodinia` | `RODINIA_DIR=~/.cache/gpu-rodinia/bin` | the 12 apps and arguments of that directory's `run.sh`, `run2.sh` and lavaMD (`-boxes1d 4`) |
| `polybench` | `POLYBENCH_DIR`; use `build_polybench_small.sh` output | every executable in the directory |
| `samples` | `SAMPLES_DIR=~/.cache/cuda-samples/build` | `exes.txt` there (98 samples built for sm_86 with a shared runtime); `SAMPLES_TIMEOUT` seconds each (default 120) |
| `pantheon` | `PANTHEON_BIN`; use `build_pantheon.sh` output | every built workload except `rt_virus` (OptiX) and `media_enc_virus` (NVENC shim not built here); run for `WL_SECONDS` seconds (default 2) with the arguments of the nightly workflow (`--kernel_loops 2 --warmup_iters 1 --grid_size 4`, 2% memory, 2 devices of 1024 MB, `--gemm_size 256` for mma_virus and omni_virus) |

PolyBench/GPU: in the checkout, the `MINI_DATASET` block holds full-size numbers (2MM 2048 cubed,
ATAX 4096 squared, JACOBI1D 10000 steps), so one run takes minutes, and the verification runs every program
six times on one host thread. `build_polybench_small.sh` caps the dimensions (cubic programs 256,
3DCONV 96, the rest 512, time steps 10). The programs still compare the GPU result with a CPU
reference. DOITGEN does not compile and is skipped.

## Run

```
tools/verify/build_pantheon.sh ~/tools/pantheon/pantheon /path/to/work/pantheon
tools/verify/build_polybench_small.sh ~/.cache/polybenchGpu /path/to/work/polybench-small
export PANTHEON_BIN=/path/to/work/pantheon/bin POLYBENCH_DIR=/path/to/work/polybench-small/bin

tools/verify/run_corpora.sh --a <A build dir> --b <B build dir> --out <results dir> --skip-timeouts A
tools/verify/run_corpora.sh --a <A build dir> --b <B build dir> --out <results dir>   # then the timeouts too
python3 -I tools/verify/summarize.py <results dir>
```

A build directory is a CMake build with `shim/` in it (`libcuda`, `libcudart`, `libcublas`, ... plus whatever the
programs link). Build both the same way (Release, CUDA 13.0 toolkit, GCC 12, LTO) under the build lock.
The run resumes: a run that has an `rc` file is skipped (`--force` reruns). `--list` prints the programs,
`--dry-run` the commands, `--only 'corpus/name'` selects, `--configs 'Bf0=B+VGPU_FASTPATH=0'` adds a
configuration (NAME=BUILD[+VAR=value...]; this is how a difference is bisected), `--max-load N` waits for a quiet machine.

Configurations (all with `VGPU_THREADS=1`, `VGPU_COUNTERS=1`, `VGPU_COUNTERS_FILE`, the RTX 3060 profile,
`nice -n 10`, `ulimit -v 12582912`, `timeout -k 10`):

| name | build | extra environment |
| --- | --- | --- |
| `A` | A | |
| `A2` | A | the same again: finds programs that differ from themselves |
| `B` | B | |
| `Bt` | B | `VGPU_TIER=1 VGPU_TIER_STATS=1` |
| `Bx` | B | `VGPU_TC_EXACT=0` (the earlier sequential tensor-core sums; only for builds that have the switch) |
| `Ap` | A | `VGPU_SASS=0` (PTX engine) |
| `Bp` | B | `VGPU_SASS=0` |

## What is compared, and what is not

`summarize.py` compares the pairs `B:A`, `Bt:A`, `Bt:B`, `Bp:Ap` and `A2:A` (`--pairs` changes them) on:

- the exit status;
- stdout, and stderr without the counter blocks and tier statistics;
- the files the program wrote (md5 of every file created or changed);
- the counters: the `[vgpu][counters]` blocks of stderr and the `VGPU_COUNTERS_FILE` records (without `pid`).

Lines that carry a timing or a rate (the words listed in `TIMING_WORDS`) have their numbers replaced by `N`
before comparing. A program equal only after that gets a `*`. This can hide a real difference on a line that happens to
contain one of those words: the report lists the compared lines of every difference, and the raw runs are kept.
Programs of the `pantheon` corpus are time-based, so the number of launches depends on the speed of the host: only their
verification-looking lines (`VERIFY`) and the set of distinct counter records are compared, not the sequence or the totals.

Verdicts: `identical`, `identical-fail` (both sides exit non-zero the same way: a failure that is not a difference; this also
counts the samples that waive themselves with exit code 2), `nondeterministic` (differs, but A and A2 differ in the same
component), `different`, `timeout-vs-done` (one side hit the timeout, the other did not), `both-timeout`, `n/a`.
`Bp:Ap` has no PTX self-comparison; a PTX difference in a program that is nondeterministic on the SASS engine is labelled
`nondeterministic` only by that proxy and should be rerun.

Programs that differ from themselves because they seed their input from the clock (CUDA Samples `fp16ScalarProduct`,
`radixSortThrust`, `UnifiedMemoryStreams`, Rodinia `particlefilter_float`) can be compared by running them with the
pinned clock: build `fixed_time.c` (`gcc -shared -fPIC fixed_time.c -o fixed_time.so`) and add
`+LD_PRELOAD=/path/fixed_time.so` to every configuration spec, for example
`--configs 'A=A+LD_PRELOAD=...,B=B+LD_PRELOAD=...'`. Programs that print a process id (the IPC samples) or
uninitialised memory (Rodinia `backprop`'s `output.dat` tail) stay different from themselves.

The real card: a configuration whose build is `C` (for example `--configs C=C`) runs the program against NVIDIA's CUDA 13.0
libraries on the local RTX 3060 (`/usr/local/cuda-13.0/lib64`; the default 13.2 runtime segfaults on these cards) under
`flock ~/.cache/pantheonsim-card.lock`, with no memory cap. `summarize.py` compares a pair that contains `C` on exit code, stdout and files
only (the card has no simulator diagnostics and no counters). Do not run the card over the whole corpora unattended: `simpleMPI` starts an
`orted` that keeps the lock's file descriptor open after the timeout; kill it, or the next card run waits for the lock forever.
Do not run two `run_corpora.sh` instances on one results directory at the same time unless their program selections do not overlap (the
runs would otherwise remove each other's results); each instance reads its own copy of the program list.

Not covered: the real card in the default configurations (only an explicit `C` runs it), more than one host thread (`VGPU_THREADS=1` is what makes the order, and so the output, deterministic),
programs the corpora do not contain, and anything that needs more than the per-corpus timeout.
