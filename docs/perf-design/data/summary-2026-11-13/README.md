# Host-GPU modes against the plain simulator: one summary run (2026-11-13)

One pass of the whole benchmark corpus in four forms on a real A10G, for a single table of what the host-GPU modes
([../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md)) gain over the plain simulator. The tables are in [summary.md](summary.md); the
harness records, compacted, in `bench.json`; `summary.py` makes both from the harness JSON of the passes.

## Code

Commit 0d959bf8 (branch `speed-round-sp64-sp68`: origin/main with PR #641, the integration of sp64-sp68), shipped to the rig with `git archive`,
built there with `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` (GCC 13.3, LTO, CUDA 13.0 headers), targets `vgpucudart` and
`vgpucuda` (the runtime and driver shims; the CUPTI shim did not build there, the rig's CUDA 13.0 keeps `cupti.h` under `extras/`, and nothing in the
corpus loads it). The harness is `tools/bench/hostgpu/run.py` with one addition made for this run (same commit as this directory): the form `exacti`.

## Rig

AWS g5.xlarge: one NVIDIA A10G (24 GB, driver 595.112.07), 4 vCPUs (AMD EPYC 7R32), 16 GB, Ubuntu 24.04, Python 3.12. Nothing else ran on it.
The simulator ran with its default thread count and `VGPU_GPU=nvidia/a10g`, the profile of the card in the machine.

## Forms

| form | environment | what it is |
|---|---|---|
| simulator (`sim`) | none | every launch interpreted on the CPU |
| exact | `VGPU_HOST_GPU=exact` | a launch goes to the card only when every instruction of its kernel gives bit-identical results there |
| exact + idioms (`exacti`) | `VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_SQRT=1` | exact mode with both opt-in idioms: the IEEE float division and square root sequences (Rounds sp44, sp68) |
| turbo | `VGPU_HOST_GPU=turbo`, `VGPU_HOST_GPU_ATOL=1e-08`, `VGPU_HOST_GPU_RTOL=1e-05` | the card also runs the approximate instructions (`MUFU` and the like): results within tolerance, not bit-exact |

Every other setting is the commit's default (atomics, sampling, the streak rule, the wider rules of sp66, the start-share rule of sp64-sp68).

## Method

    run.py --build build --corp ~/corp --gpu nvidia/a10g --forms sim,exact,exacti,turbo --runs 3 --long-secs L --long-runs 1 --timeout 600

For each program the four forms run one after another, then in the reverse order, and so on: interleaved, one process at a time. A program whose
first simulator run took longer than L seconds gets one run of each form, the others three; a time is the median wall time of a form's runs (process
start to exit, the program's own host code included). A program that got three runs also gets one more, untimed, exact run with
`VGPU_HOST_GPU_LOG=1` for the exact form's launch counts. A run past 600 s is killed and recorded as a timeout; a form that timed out is not run again,
and a simulator timeout makes the speed-up a lower bound (reported separately, left out of the geometric means).

L was 30 s for PolyBench, Rodinia and HeCBench, and **5 s** for the CUDA Samples and the two late HeCBench passes: so that the whole run fit in the
rig's time, the 10 Samples between 5 and 30 s in the simulator (convolutionSeparable, cudaTensorCoreGemm, histogram, interval, jacobiCudaGraphs,
mergeSort, reduction, segmentationTreeThrust, simpleHyperQ, sortingNetworks) got one run per form instead of three. (The late HeCBench programs are
all over 30 s in the simulator, so L changed nothing there.)

Outputs: stdout and stderr, without the simulator's lines and the lines that carry timings, and for Rodinia the files the program wrote, of every run
of a mode against the first simulator run: `identical`, `numeric` (the same words and every number within 1e-8 + 1e-5 |sim|) or `different`.
A `different` is put in its own column when the simulator's own runs differ from each other (rodinia/particlefilter_float seeds its random numbers
from the clock, samples/cudaGraphsPerfScaling prints timings, samples/simplePrintf prints in thread order) or when the program's output carries
timings the filter does not catch (hecbench/babelstream-cuda's bandwidth table; hecbench/base64e-cuda repeats its test when the measured stream overlap
is not acceptable, so its line count depends on timing).

## Corpus

The sm_86 binaries (nvcc 13.0, `-cudart shared`), copied from the dev machine's caches with `tools/bench/real/pack_corpora.sh`, in the layout of
`tools/bench/real/real.py` (whose manifest gives the arguments: Rodinia at its scaled-up sizes).

| suite | programs | which |
|---|---:|---|
| PolyBench/GPU | 20 | all |
| Rodinia | 12 | all of `real.py`'s list |
| HeCBench (reduced list) | 32 | the 34 of `reduced.txt` without `fpdc-cuda` and `zerocopy-cuda` (known simulator problems) |
| CUDA Samples 13.0 | 36 | the Samples of Round sp66's card run (`data/sp66/card-run.sh`, `CHANGED`) |

Passes, in this order: polybench, rodinia, hecbench (without the five below), samples, then the four HeCBench programs whose simulator run
passed 300 s in Round sp66 (`aligned-types-cuda`, `babelstream-cuda`, `base64e-cuda`, `binomial-cuda`), then `aidw-cuda` (past 300 s in every form
in Round sp66).

Known non-results: `hecbench/atomicCost-cuda` is killed (exit -9) in every run of every form, as in earlier rounds; its times are of the killed runs.
`hecbench/background-subtract-cuda` exits after 5 ms in every form (no launch), so it times process start-up only.

## When

The rig was launched at 22:45 UTC on 2026-11-13. Passes (UTC): polybench 22:53-23:25, rodinia 23:25-23:30, hecbench 23:30-23:57, samples
23:57-00:32, the four slow HeCBench programs 00:32-01:24, aidw 01:24-02:01. Nothing was cut beyond the 5 s threshold of the Samples above; every
program of the corpus ran in every form.
