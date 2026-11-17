# Round r9: the census of a loop-weighted static count (#656; analysis only, no decision changed)

The question of [../turbo-left/README.md](../turbo-left/README.md) ("#656 the rest"): if the static count the floor is asked about weighed a
loop's body by its turns where the bound is an immediate (the `initRNG` case of MC_SingleAsianOptionP and MC_EstimatePiInlineP), which kernels of
the whole corpus would newly be sampled about 5 ms into their process, and what would that cost them? "Newly" means: the launch has no estimate,
the grid and the loop rule let a sample through, and the only thing that keeps it in the simulator today is the floor (`NoSample::Floor`,
"static estimate under the floor and the process too young"); with the weighted count over the floor it would be sampled.

Nothing in `include/` or `src/` was touched, and no decision, digest or output changed: the census is a mode of
[../sp75/census.py](../sp75/census.py) (`--weighted`, `--report`) that reads what the mode already prints (`VGPU_HOST_GPU_LOG=2`: the text of every
admitted kernel's code, the `host_gpu_nosample` lines, the `host_gpu_sample` lines) and does the weighing in Python. Tags: **measured** is a
run on server3 (the numbers below; files named), **inferred** is anything about the A10G rig or any card, **unknown** is what was not measured.

## What was run (measured)

* Build: `origin/main` at e3b8f6ab, `Release`, the stand-in driver (`nvidia/tests/e2e/host_gpu_stub_driver.c`, which refuses every module),
  `VGPU_GPU=nvidia/a10g`, `VGPU_THREADS=3`, `VGPU_HOST_GPU=turbo`, `VGPU_HOST_GPU_LOG=2`; the sm_86 corpora of
  `tools/bench/real/pack_corpora.sh` (polybench 20, rodinia 12, hecbench 34, samples 58: 124 programs), one run each, a run ended at 60 s.
  11 programs were cut at the time limit (hecbench aidw, aligned-types, atomicCost, atomicReduction, babelstream, base64e, binomial, fpdc;
  polybench 2MM, GRAMSCHM; samples tf32TensorCoreGemm). A kernel first launched after 60 s is not in the census, and such a launch is not
  "young" (the process can afford a sample by its age), so the cut loses nothing of what is asked here.
* The corpus was packed on the dev machine (files only) and unpacked in a scratch directory on server3; every run held the machine's build
  lock, `nice -n 19`.
* The weighted count: every loop of the kernel (a back edge makes a range; ranges that cross are left out) whose counter is found as the
  simulator's `find_probe_loops` finds it (the compare that decides a branch, one side an induction register written once by an add of a
  constant, directly or through `iv + offset`) and whose bound is an immediate, has `floor(|bound| / |step|)` turns (`bound_turns`, the
  simulator's formula, which assumes the loop starts at 0). An instruction in a nest is counted once per turn of every loop around it
  (the body is weighed, the prologue is not; the simulator's own estimate for a sample inside the blocks multiplies the whole count by the
  largest loop's turns). A loop with a register or parameter bound, or none that can be found, weighs 1. Two things differ from the simulator's
  finder on purpose, because they are what the design in section 13 and 14 asks for: loops inside loops are all weighed, and a loop left through a
  guarded `CALL` (`@P CALL.REL.NOINC` to the kernel's `EXIT`) is followed (today any call makes every loop of the kernel "not followed").
* The estimate in seconds is the simulator's: `blocks x warps per block x count / 100e6` (`Costs::sim_warp_instr_per_s`); the floor is the
  line's own (`floor_s`: 30 ms, or three times the page faults of the live memory when that is more).
* The sample cost column is from a second pass over the five programs with `VGPU_HOST_GPU_SAMPLE=always` (neither the floor nor the process's
  age is asked, so each kernel is sampled at its first launch): `cost_s` and `fork_s` of its `host_gpu_sample` line, and the estimate it gave.

Two checks of the census itself (both in the report's header, `census-weighted-all.md`):

* **Against the simulator's own finder:** for every kernel with no call the simulator prints its followed loops (`followed loop [lo, hi]: iv step
  bound`). 124 loops, and the census finds the same range and, for an immediate bound, the same turns, for all 124 (a first version that took the
  first branch it could read instead of the first the simulator reads differed for one loop of pathfinder's `dynproc_kernel`; fixed).
* **Twice:** the whole census was run twice (two runs at once, so the time limit and the age-dependent reasons fell differently). For the 108
  programs that completed in both, the kernels' counts, loops and cross-check are identical; the floor lines are identical but for two programs
  (dct8x8 and segmentationTreeThrust: one launch each whose reason depends on the process's age) and the programs cut at the limit in one run and
  not the other (hecbench bh-cuda). None of the five kernels below is among the differences: the second run names the same five, with the same
  counts and estimates (`census-weighted-all.md` is of the first run, the one that reached more kernels).

## The census (measured)

253 launched kernels with a loop that the mode admits (248 distinct names) in 124 programs. For 13 of them the weighted count is above the
plain count (a loop with an immediate bound the plain count sees once). 177 launches (176 kernels) are kept today by the floor; only these five of
them have a weighted estimate over the floor (their programs, plain count, weighted count and the estimates of the first launch):

| program | kernel | grid x block | plain count (instrs) | weighted count | today's estimate | weighted estimate | floor | cost of its sample (fork) | what the sample says here |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| samples/MC_EstimatePiInlineP | `initRNG` | 782 x 128 | 304 | 2,315 | 9.5 ms | 72.4 ms | 30.0 ms | 6.2 ms (0.2) | 595 ms |
| samples/MC_SingleAsianOptionP | `initRNG` | 782 x 128 | 304 | 2,315 | 9.5 ms | 72.4 ms | 71.7 ms (38 MiB live) | 5.5 ms (0.2) | 569 ms |
| rodinia/lavaMD | `kernel_gpu_cuda` | 216 x 128 | 392 | 6,223 | 3.4 ms | 53.8 ms | 30.0 ms | 6.8 ms (0.1) | 197 ms |
| hecbench/bilateral-cuda | `bilateralFilter<6>` | 128 x 256 | 416 | 5,206 | 4.3 ms | 53.3 ms | 30.0 ms | 9.4 ms (0.4) | 170 ms |
| hecbench/aop-cuda | `prepare_svd_kernel<256, PayoffPut>` | 99 x 256 | 2,344 | 10,384 | 18.6 ms | 82.2 ms | 30.0 ms | 8.9 ms (0.7) | 111 ms |

The loops that make the weights: `initRNG` 5 x 2 turns (an immediate `ISETP.GE.U32 R14, 0x5` around `IADD3 R21, R21, 0x10` against
`ISETP.NE R21, 0x20`: the counter steps by 16, so 2 turns, not 32), lavaMD 50 turns on its inner loop, bilateral 7 x 3, prepare_svd 16
(and 33 calls). In no other kernel of the 248 does the weighted estimate reach the floor. Near misses (weighted estimate of the first launch,
over a floor of 30 ms): bilateralFilter<9> 27.6 ms, jacobiCudaGraphs `JacobiMethod` 16.4 ms, asyncAllocation `process` 11.9 ms, lud_perimeter 2.4 ms
(its launch is on 63 blocks of 32 threads). Every other launch the floor keeps is unchanged by the weights. The full table, launch by launch, is
[census-weighted-all.md](census-weighted-all.md) (the sample columns only for the five programs of the second pass).

## What it would cost, and what it would buy

* **The cost of being sampled (measured, server3):** 5.5 to 9.4 ms per kernel (0.1 to 0.7 ms of it the fork), taken at the process's age
  of the floor line: 3 to 6 ms for lavaMD and the two `initRNG`, 51 to 96 ms for bilateral and prepare_svd (both runs), plus the page faults after
  the fork that the floor's own term estimates (not measured here: 37 MiB live for MC_SingleAsianOptionP, 4 for EstimatePi, 50 for aop, 1 for
  lavaMD). Against the programs' times here that is a fraction of a percent to about 1 % (lavaMD takes 0.63 s here, `launches.md`) for a sample
  whose answer is "keep".
* **What the samples say (measured, server3):** all five are "keep" here against the card path with the start (0.25 s): the estimates are
  111 to 595 ms against "needs" of about 0.55 to 0.7 s (twice the card path with the start; the figures `launches.md` gives for the two
  `initRNG` and lavaMD, whose samples said 0.56 s, 0.79 s and 0.20 s there and 0.60 s, 0.57 s and 0.20 s here; bilateral and prepare_svd have no
  figure there, "needs" for them was not read from a log). So on server3 the rule would change no launch's path, and cost each of the five 5 to 9 ms.
* **On the rig (inferred, not run):** the simulator there is 2.5 to 3 times slower (documented, `launches.md`), so the two `initRNG` samples
  would say about 1.4 to 1.7 s against 0.64 to 0.71 s needed and the launches would be sent: about 3 s each from the summary run of
  2026-11-13, as the README says (inferred from that run; this census does not measure the card). lavaMD, bilateral and prepare_svd would
  say 0.28 to 0.5 s: lavaMD is "at the bar" by `launches.md` (about 0.5 s on the rig against 0.55 needed), bilateral and prepare_svd are unknown
  (not measured on a card).

## Verdict

By the README's test, five kernels in four programs (of 248 kernels and 124 programs) is a handful, not "more than a handful": the census does not
show the rule to be a net loss of sample time, and by that test it is worth building. The numbers say what that is worth, and it is less than the
README's reading of it:

1. The gain is the two `initRNG` launches on the rig: two programs, about 3 s each (inferred). The other three kernels cost 7 to 9 ms and
   gain nothing known (lavaMD, bilateral and prepare_svd are kept or at the bar).
2. **The weights do not reach the 160-times the README reads in `initRNG`.** The weighted count is 7.6 times the plain one (2,315 against 304
   instructions: the nest is 5 x 2 turns, because the counter steps by 16 to a bound of 32), an estimate of 72.4 ms against the launch's 1.4 to 1.6 s
   (4.5 %): most of `initRNG`'s time is in loops whose bounds are not immediates (the skip-ahead loops over the thread's index). The rule would put
   the two launches over the floor by a factor of 2.4 (EstimatePi) and by **1 %** (SingleAsianOption: 72.4 ms against 71.7 ms, a floor made of its
   38 MiB of live memory). A little more live memory, or a count one instruction lower, and SingleAsian is not sampled.
3. A kernel the simulator follows today already has its count multiplied by its loop's turns (`sample_static_estimate`); the weighted count
   must be combined with it by taking the larger (for samples/LargeKernelParameter the weighted estimate, 0.1 ms, is under today's 3.5 ms).

So: building it is not a loss, and nothing in the corpus argues against it, but what it buys is two programs and one of the two by a 1 % margin.
The step that matters first is a measured one nobody has made: what the second table's `initRNG` weight is worth with the unknown-bound loops
included (a loop whose bound is a register the thread computes cannot be weighed statically, and it is where the 1.5 s is).

## Residual risks

* The census sees what ran: 11 programs were cut at 60 s, and a kernel a program launches only for other inputs or never is absent (the
  corpus is the reduced set of the benchmarks).
* The floor lines are the first two launches of each kernel (the mode says no more); a kernel whose first launch has a history-free estimate
  but a different grid later is not covered.
* The weights assume a loop starts at 0, as `bound_turns` does: a count-down loop to an immediate 0 gets weight 1 here (its turns are unknown), and a
  loop that starts near its bound is overweighed.
* The census reads the mode's printed text of the code (the same text the simulator decodes), not the decoded instructions; the cross-check
  against the simulator's finder is the evidence that the reading is faithful (124 of 124 loops), not a proof for the loops it alone finds
  (those inside loops, and those left through a call).
* Ages in the floor lines depend on server3's speed (a process on the rig is slower, so "5 ms" there is fewer instructions in).
* Nothing here is a card measurement: the rig's behaviour is inferred.

## Reproducing

```
census.py --weighted --corp CORP --stub libstubcuda.so --out DIR --form new=/path/build --timeout 60        # the census
census.py --weighted ... --form always=/path/build,VGPU_HOST_GPU_SAMPLE=always --programs a,b --out DIR2    # the cost pass
census.py --report DIR [--cost-from DIR2]                                                                    # the table
```
