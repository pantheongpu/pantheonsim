# Round sp60b: the first blocks of a launch, timed (data)

The text is *Round sp60b* in [../../rounds/11-rounds-sp36-sp41.md](../../rounds/11-rounds-sp36-sp41.md); the design is section 8 of
[../../12-host-gpu-turbo.md](../../12-host-gpu-turbo.md).

## Before the card (server3: 16 hardware threads, GCC 13.3, `Release`, `-Werror`, no LTO, `VGPU_THREADS=4`, profile `nvidia/a10g`, no driver)

Every build and run under the machine's build lock, `nice 19`. The programs are the sm_86 binaries of PolyBench/GPU, Rodinia and the reduced
HeCBench list, with the arguments of `tools/bench/real/real.py`.

| file | what |
|---|---|
| `samplelog.py` | runs the corpus in forms of the mode (off, `exact`, `turbo`, `VGPU_HOST_GPU_SAMPLE=0`, `=log`, a stand-in driver that refuses, one that keeps memory and computes nothing) and keeps wall time, an output hash, the mode's lines and every `host_gpu_sample:` line |
| `model.py` | per sampled launch: the static estimate, the sample's cost, the estimate from the sample, what the launch took |
| `model-exact.md`, `model-turbo.md` | its tables for `VGPU_HOST_GPU_SAMPLE=log` in exact and in turbo mode (every candidate launch is sampled in `log`, whatever the loop test and the floor say): 16 and 52 launches. Made before the loop test was added, so they have no column for it |
| `mutants.py`, `mutants.txt` | 69 mutants of the rule in `include/vgpu/host_gpu.hpp` (how many blocks, the stride, the floor, the loop test, the estimate, the card path, when to sample), each a copy of the header outside the tree: 69 killed by `test_host_gpu_logic`, none by failing to compile |
| `card-run.sh` | the card part, by stage: `build` (the card CI), `ci`, `ab` (the A/B of `exact` and `turbo` against `exact0` and `turbo0`, which do not sample, and the log of what was sent), `early` and `exit` (the early-start experiment, whose switch was then removed), `repeat` |
| `summary.py` | the A/B's tables and checks from the harness's JSON |

What the tables say: the eight launches the rule would send (SYRK, SYR2K, the first kernel of 2MM, in both modes; black-scholes, binomial,
shmembench in turbo) took 0.9 to 1.6 times the sample's estimate; no launch under 0.25 s was estimated over the bar. A sample cost 0.9 to 12 ms
(27 ms with 1.9 GiB live: the fork). That is the time waited; the page faults after the fork cost more (Rodinia backprop: 4 ms waited and 12 ms
more over the run, median of 12 interleaved runs), which is why the rule also asks for a loop and scales the floor with the live memory.

Against the stand-in driver: a sampled launch that the driver refuses is run by the simulator and the output is the mode-off output (SYRK,
SYR2K, GEMVER, black-scholes, shmembench); a sampled launch sent to a stand-in that keeps memory and computes nothing leaves the program
reading the memory of before the launch (SYRK and SYR2K: every one of the 1,046,529 compared elements differs from the CPU reference).

Not reached: PolyBench COVAR and CORR (their long kernel runs on 8 blocks of 256 threads; in exact mode CORR's kernels are not admitted at all).

## On the card (AWS g5.xlarge, us-east-1, instance i-004f225995427c7ac: A10G, driver 595.112.07, 4 vCPU, nvcc 13.0; 2026-11-12)

Simulator: `RelWithDebInfo`, no LTO (what `tools/card-ci/run-card-ci.sh` configures), profile `nvidia/a10g`; the same corpus binaries.

| file | what |
|---|---|
| `ab.md` | `summary.py --md` over the three suites: per program the simulator's seconds, `exact`, `exact0`, `turbo`, `turbo0` (medians of 3 runs, 1 for a program over 30 s), the launches sampled and sent, the launches on the card, the outputs; the geometric means; the programs more than 3% apart; the fork checks |
| `ab.json` | the same records, compact: per program and form the median seconds, `<form>_sampled` = [sampled, sent, copies that did not answer, cost in ms], `<form>_card` the launches on the card, `<form>_out` only where it is not `identical` |
| `sample-log.txt` | `VGPU_HOST_GPU_LOG=1` runs of nine programs: the `host_gpu_sample:` line of each sampled launch (blocks, static estimate, the process's age, the fork, the timed runs, the estimate) and what the card then took; 4 lines a program at most |
| `repeat.txt` | ten runs of each form of the three programs that the A/B showed more than 3% apart with no launch sampled, and of hotspot; then the zombies and corpus processes left on the rig after everything (0 and 0) |
| `final.txt` | the seven programs run again on the last build, black-scholes's log there, the tests on that build, what was left on the rig |
| `early.txt` | `VGPU_HOST_GPU_EARLY_START=1` against the default on the 26 PolyBench and Rodinia programs under 30 s |

Order of the runs: the card CI and `e2e_host_gpu_stub` passed on the first build; a first A/B of Rodinia (the floor alone, no loop test) showed
backprop 4% slower in exact mode and was stopped; the rule got the loop test and the floor its memory term, the rig was rebuilt, `ci` passed
again, and the A/B in `ab.md` ran on that build. The timeout that grows with the live memory and the removal of the early-start switch came
after it: the rig was rebuilt once more and seven programs were run again (the round's text has their numbers), with the unit tests,
`e2e_host_gpu` and `e2e_host_gpu_stub`.

Cut for the rig's hours: PolyBench GRAMSCHM was not run; HeCBench runs were limited to 120 s (five programs have no simulator time); no verify pass.

## What was built first and removed

| variant | what it did | why it went | numbers (server3, 4 threads) |
|---|---|---|---|
| move a running launch | copy the whole live memory, run the launch, a watcher thread reads the blocks done; when the rest repays the card, stop the simulator between two blocks and run the launch on the card from the copy (or go on in the simulator from the first block not taken) | the copy is paid by every watched launch | 128 MiB: 66 ms the first time, 16 ms after; 3DCONV 0.33 s to 0.43 s, ATAX +6%; stopping and resuming worked (ATAX's second kernel stopped at block 12 of 128, output unchanged) |
| sample in the process, undo through the card gate | the first blocks run in the process with every allocation's slot raised, the gate saves each 4 KiB page before its first access, the pages are put back | a block that runs through the gate is 14 times (SYR2K) to 26 times (GEMM) slower than a block of the launch: that is the cost (29 ms for 2 blocks of GEMM, a 0.47 s program) and the error of the estimate (GEMM: 0.88 s estimated, 0.13 s taken) | the undo was exact: with every byte of live memory compared before and after, GEMM and SYR2K passed |
| a sample of the first blocks, each run once | as kept, but blocks 0 to 31 and one timed run | Rodinia gaussian's first blocks do 9 times the work of its average block; spread over the grid, PolyBench GEMVER's blocks each wrote a fresh 64 KiB chunk and looked 7 times as long as they are (0.82 s estimated, 0.11 s taken: it would have been sent) | fixed by spreading the blocks and timing each block's second run |
| the process's age alone as the guard | a sample only when the samples' cost is within 2% of the time the process has run | programs launch 2 to 20 ms after they start (SYRK 13 ms, SYR2K 16 ms): nothing that matters would be sampled | kept as the second of two conditions, behind the static floor |
