# Round r11: the first launch of a kernel in a young process (#656, #659; A10G card run, 2026-11-17)

Scope: the size rules (`VGPU_HOST_GPU_FIT=0`, form `turborules`) and exact mode. Turbo's default since #693 is `FIT=1`, which starts the driver at device
creation and sends every fitting launch, so it does not meet this refusal ("static estimate under the floor and the process too young").
Tags: **measured** is this run (one A10G, g5.xlarge, 4 vCPUs, shared with another job; `Release` build of this branch); **inferred** is anything not run.

## What the old rule kept (measured: `candidates-off.txt`, one logged `FIT=0` run of every program of the four corpora, `logs.py`)

Every first launch refused as "too young", biggest first. The launches the issues name are all big: mergeSort's `mergeSortSharedKernel` is 4096 x 512
threads (2.1 M; 1.4 to 2.5 s in the simulator), `initRNG` of both Monte Carlo samples 782 x 128 (100 k; 6 s with the log on), histogram64 4370 x 64 (280 k),
histogram256 240 x 192 (46 k), warpAggregatedAtomicsCG's two kernels 19531 x 512 (10 M). The launches that are short are big too: silu's 262 k threads take
0.03 s, getRepresentatives (1.2 M) 0.09 s, pathfinder (119 k) 0.07 s. The static count does not tell them apart (initRNG counts 9.5 ms and takes 6 s, silu counts
13 ms and takes 30 ms), the thread count does not either, so the launch has to be measured: the sample of Round sp60b, which the old rule would not pay for in a young process.

## The rule (`VGPU_HOST_GPU_FIRST_BIG=<threads>`, default 65536; 0 is the old rule)

A launch of at least 65,536 threads is asked no floor and no process age for its sample, **while the process has spent nothing on samples**: one sample, 3 to 10 ms
(measured: 3 to 14 ms in the logs), at most, for a program whose big launches are all short. Everything else (a loop in the kernel, blocks to sample, an
earlier short launch of the kernel, the sample's own verdict against the card path with the start) is asked as before, so the rule only moves a launch between two
paths the mode already has. A second excused sample was tried (warpAggregatedAtomicsCG: its first big launch is a 0.4 s loser, its second a 1 s launch that the
sample puts at 0.34 s against a card path of over 0.5 s; 14 ms for no send) and dropped. The threshold is round, not fitted: 100,096 (initRNG) and 2.1 M (mergeSort)
are the launches that gain, and 16,384 (tried first) gave the same results with more programs paying a sample.

## A/B (measured; `ab-*.md`; medians of 9 interleaved runs, `turborules` = the rule, `turbof0` = `VGPU_HOST_GPU_FIRST_BIG=0`; outputs identical to the simulator's in every row)

| program | sim s | rule s | old rule s | rule / old | card launches rule / old |
|---|---:|---:|---:|---:|---|
| samples/mergeSort | 8.88 | 0.58 | 2.30 | 0.25 | 43 / 42 of 49 |
| samples/MC_EstimatePiInlineP | 4.15 | 0.29 | 4.09 | 0.07 | 2 / 0 of 2 |
| samples/MC_SingleAsianOptionP | 4.49 | 0.77 | 4.09 | 0.19 | 1 / 0 of 3 |
| samples/histogram | 11.70 | 2.14 | 2.13 | 1.00 | 62 / 62 of 68 |
| samples/warpAggregatedAtomicsCG | 2.04 | 2.05 | 2.05 | 1.00 | 0 / 0 of 3 |
| samples/interval | 14.40 | 2.66 | 2.68 | 0.99 | 97 / 97 of 100 |
| samples/reduction | 5.62 | 0.87 | 0.88 | 0.99 | 192 / 192 of 201 |
| samples/segmentationTreeThrust | 14.22 | 4.62 | 4.58 | 1.01 | 284 / 284 of 514 |
| hecbench/aop-cuda, bilateral, radixsort2, silu | | | | 0.97 / 1.04 / 1.02 / 0.94 | |
| polybench/3MM, ATAX, GEMM, MVT | | | | 0.98 / 1.00 / 1.01 / 1.01 | |
| rodinia/pathfinder, lavaMD | | | | 1.02 / 1.00 | |

Which programs the rule changes (`on-check.txt`, `cmp.py`: the logged runs of all 59 programs with the rule on against the rule off): 9 take a sample they
did not take before: mergeSort (sent, estimate 0.80 s), both Monte Carlo samples (initRNG, sent, estimate 0.95 s; the rest of EstimatePi follows it to the card),
and radixsort2, silu, 3MM, GEMM, pathfinder (sampled, not sent: 5 to 7 ms each; the estimates 0.007 to 0.27 s are under the card path) and histogram
(its first launch is sampled and not sent, 0.20 s against 0.32 s taken, 7.7 ms). The 50 others, including every program of polybench not named, run the same
code in both forms. Differences in the number of samples of dct8x8, particlefilter and segmentationTreeThrust are run to run (the old rule's own sampling depends on the
process's age). Rig noise is 4 % at 9 runs (hecbench/bilateral, 32,768 threads, below the threshold, same decisions in both forms: 1.57 s against 1.51 s), above the 3 % the
issue asks, so the claim for the programs that pay one sample is about the medians here: 0.94 to 1.04, none of them a program that sends a different number of launches.

## What is not done

* warpAggregatedAtomicsCG, lavaMD and histogram's first launches stay in the simulator (the first and third: the sample is taken and says no; estimates 0.20 to
  0.34 s against card paths of 0.37 s and over, launches that took 0.32 to 1 s; lavaMD has 216 blocks of 128 threads, under the threshold, and was not
  looked at further). A time box on the first simulator run with a
  speculative driver start was not built: the sample answers in 3 to 10 ms where a box would run into the launch's 0.1 to 2 s, and it needs the start (0.25 s)
  before the first answer. Not measured.
* `FIT=1` (the default turbo) is not changed.

## Tests and mutation evidence (measured, on the rig)

* `tests/unit/test_host_gpu_logic.cpp` `a_big_launch_is_sampled_in_a_young_process` (142 of 142 pass). Four mutants of `sample_big_launch` / `sample_refusal` each fail it:
  the rule not asked, `>=` to `>`, the spent-nothing clause dropped, the rule always on (0 not off; three tests fail).
* `nvidia/tests/e2e/run_host_gpu_stub.sh` step 20 (pinned with `VGPU_HOST_GPU_FIRST_BIG=0`) and the new 20b: the big launch of `acct` (262,144 threads) in a young process is sampled
  exactly once and starts no driver; a threshold above it samples nothing; the output is the same. The script passes; with `c.big_threads = 0` in `consider_sample` (the
  wiring, which the unit test cannot see) 20b fails ("sampled=0").
* `tools/bench/hostgpu/run.py --dry-run`: `turbof0` form env.
Scripts as run: `logs.py` (logged run of every program), `cmp.py` (on against off), `ab.sh` (the A/B sets).
