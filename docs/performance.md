# Simulator performance

How fast the simulator is, which switches change that, and where the evidence is. This page is the overview; the measurements
behind every statement are in the round-by-round log, [perf-design/rounds/](perf-design/rounds/README.md) (the unedited text of what used to
be this page), and the design studies in [perf-design/](perf-design/README.md).

The rule behind all of it ([ARCHITECTURE.md](../ARCHITECTURE.md)): correctness, determinism and diagnosability outrank speed. A speed-up
must not change a result bit, a reported counter or the order of an observable effect; every round checks that with same-bits tests and
counter comparisons before it looks at a stopwatch. The simulator is a functional model with no timing model, so "fast" means host seconds
per simulated instruction.

## How fast it is

The simulator executes warp instructions on the host CPU. Rates depend on the instruction, the engine and the machine, so treat the numbers
below as orders of magnitude and as ratios between two builds on one machine.

* Baseline at the start of the speed work (build 443e79c0, one host thread, i9-12900K, shared machine): 0.5 to 40 million warp instructions
  per second depending on the workload; about 150,000 launches per second and about 4 million API calls per second
  ([baseline table](perf-design/rounds/01-harness-baseline-profiles.md#baseline)).
* Since then, at one host thread on the same kind of workloads (A/B of the SASS engine, counters identical, p < 0.001): straight-line integer and
  float code 2.3 to 2.8x, `warp` 3.4x, `local` 1.9x, `divergent` 1.9x, launch 1.2x ([Tier 1 together](perf-design/rounds/04-speedup-branches-tier2-amd-tier.md#tier-1-together-branch-speedup-tier-on-by-default-ptx-atomics-and-texel-reads-amd-baseline)),
  then tensor-core kernels 1.5x and the AMD engine 1.7 to 3.7x depending on the instruction class ([Tier 2 together](perf-design/rounds/04-speedup-branches-tier2-amd-tier.md#tier-2-together-branch-speedup-all-pieces-merged)).
  Later rounds (sp to sp39) added gains on named programs, from a few percent to several times (memory, atomics, counting, launch, AMD float division); each is in the log's index.
  On a program of doubles and reductions the AMD engine (inq's unit tests) uses about half the CPU time since [Round sp70](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp70-the-amd-engine-on-inq-the-register-file-pool-one-host-thread-a-physical-core-fmin-and-fmax-of-doubles-quads-of-managed-memory): one host thread a physical core, a larger register-file pool.
* Host threads: blocks of a grid run on a pool of host threads (`VGPU_THREADS`). Large grids scaled 7.7x on 16 threads when measured
  ([study 5](perf-design/05-sharding-and-sampling.md)); small launches stay off the pool ([Round sp12](perf-design/rounds/08-rounds-sp11-sp15.md#round-sp12-small-launches-stay-off-the-worker-pool-branch-sp12-pool-base-2c68075d--sp11-tensor-pr)).
* Noise: the same build against itself differs by about 4 percent at the median and up to 20 percent at worst on a shared machine, so a
  difference under 5 percent needs 9 or more alternating runs ([Noise](perf-design/rounds/01-harness-baseline-profiles.md#noise)).

## The knobs

All variables are in [cli.md](cli.md); this is when each one helps or hurts. Results, counters and errors are the same with any setting
unless a line says so.

**Exact mode and turbo mode.** Both are host-GPU modes: on a machine whose real NVIDIA GPU is the simulated model, kernel launches that are worth it run on that card and the rest stays simulated. *Exact mode* (`VGPU_HOST_GPU=exact`) sends only kernels the card computes bit for bit as the simulator does, so the program gets the simulator's bits. *Turbo mode* (`VGPU_HOST_GPU=turbo`) is the fast mode: it also sends kernels with approximate math, half precision, tensor-core instructions and integer atomics, and the program gets the card's own values, which can differ from the simulator's in the last bits. Safety checks are the same in both. See [12-host-gpu-turbo.md](perf-design/12-host-gpu-turbo.md).

| Switch | Default | Helps when | Hurts or does not help when |
| --- | --- | --- | --- |
| `VGPU_TIER` | on (SASS engine) | Always: the pre-decoded tier is 1.2 to 2.8x on the benchmark workloads ([why it is the default](perf-design/rounds/02-sass-tier-rounds-8-9.md#the-tier-on-by-default-vgpu_tier0-turns-it-off)). | Set `0` only to rule the tier out when a result looks wrong. |
| `VGPU_AMD_TIER` | on (AMD engines) | Always: whole-register-row handlers for the hot AMD instruction forms ([AMD tier](perf-design/rounds/04-speedup-branches-tier2-amd-tier.md#the-amd-pre-decoded-tier-branch-speedup4-amdtier-vgpu_amd_tier)). | `0` to rule it out. |
| `VGPU_SIMD` | best the CPU has | AVX2 and AVX-512 lane and tensor-core kernels; the 512-bit tile kernels are about 8 percent faster and up to 2x on integer tensor forms, and the 512-bit binary64 lane kernel makes programs that compute in `double` 3 to 39 percent faster on a Zen 4 host (round sp54). | `base` or `off` for ruling the vector kernels out; slower. |
| `VGPU_THREADS` | one per CPU the process may use (NVIDIA engines); one per physical core (AMD engine, Round sp70) | Grids with many blocks. | Tiny grids and launch-bound programs: the pool costs more than it saves. Cap it when sharing a machine. On the AMD engine a value above the number of physical cores bought nothing on the host it was measured on (8 cores, 16 threads) and doubled the CPU time. |
| `VGPU_EXACT_THREADS` | off | Beside `VGPU_THREADS=1`, when the result must be the one of blocks in order and there are 4 or more cores: 1.59x at 4 threads and 2.01x at 8 (geometric mean, 53 Rodinia, PolyBench and HeCBench programs), same output, counters and memory ([exact-threads.md](exact-threads.md), [Round sp47](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp47-the-blocks-of-one-launch-on-several-host-threads-with-the-bits-of-running-them-in-order-vgpu_exact_threads-branch-sp47-parallel-blocks-base-originmain-27f09add)). | 2 threads on kernels that mostly load (PolyBench 0.93x): the proof costs about 1.4x per thread. Kernels with atomics shared by blocks, device printf, textures, clusters or mapped memory run in order as before. SASS engine only. |
| `VGPU_JIT` | off | Long-running programs with hot loops (tier 2 compiles hot stretches to C++ in the background; warm-cache geometric mean 1.10 over 22 programs, goal 1.15 not met, [round 15](perf-design/round15-jit.md)). | Programs that never reach the compile threshold of 100 M region instructions (45 of 57 corpus programs, all those under three seconds): nothing is compiled and nothing is gained; the counting on the way cost them 0 to 16 % until round sp57, 0 to 8 % until [round sp63](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp63-the-jits-cold-run-the-wait-before-the-lookup-history-the-cache-cap-the-notices-branch-sp63-jit-cold) and 0 to 3 % since for those of 0.2 s or more (geometric mean 0.992 cold over 45, one thread). Programs of 3 to 10 s on a cold cache: the compile arrives late or is cancelled when the program ends (0.89 to 0.98 of the tier's speed in their first run; since round sp63 a cancelled compile starts at the next run's lookup, so asyncAllocation is 0.97 in its second run and warm, 1.33, from its third; since [round sp64](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp64-the-jits-compile-and-the-run-around-it-the-plain-loop-while-a-compile-runs-the-precompiled-runtime-header-measured-at-1-and-16-threads-branch-sp64-jit-threads) the section runs on the plain loop while its compile runs, and with GCC a precompiled runtime header takes 0.33 to 0.37 s off each compile after the first). At the default thread count a compile is longer than a program of 1 to 3 s and never finishes, so such a program is never warm (its later runs cost 5 to 8 % at sixteen threads on server3, 8 to 18 % before round sp64). Needs a host C++ compiler; a filled cache is of no use without one. [Why it is not the default](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp57-the-jit-as-the-default-measured-not-flipped-the-count-on-the-way-to-the-threshold-moved-to-the-plain-loop-branch-sp57-jit-default) (geometric mean over 57 programs with the defaults: 0.954 cold and 0.985 warm before that round's fix; the 12 programs that compile: 1.12 warm, 1.19 with `VGPU_JIT_LINK=1`). |
| `VGPU_JIT_LINK` (with `VGPU_JIT=1`) | off | Programs whose warps diverge or whose kernels are short and launched many times: the generated code goes on across branches, BSSY, untaken EXITs, calls and returns where the run loop would only pick the same group again, and the regions of a section are one function (black-scholes 1.23, JACOBI1D 1.14, hotspot 1.13, bilateral 1.13; nine of 18 programs gain 5 to 23 %, geometric mean 1.055; a section with a loop and no barrier or call is left unlinked, so SYRK and SYR2K are unchanged: [round sp52](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp52-linking-keeping-execution-inside-generated-code-across-region-exits-branch-sp52-region-linking-base-originmain-7b90cee5), [round sp53](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp53-linking-the-follow-up-why-syr2k-lost-bsync-inside-generated-code-what-the-linked-compile-costs-branch-sp53-link-followup-base-sp52-region-linking-1fce4fd5)). | The first run: the linked object compiles 2 to 3 times as long. |
| `VGPU_JIT_FUSE` (with `VGPU_JIT=1`) | off | Float-heavy kernels: runs of register-only ops are computed block by block of eight lanes with the rows in host registers (black-scholes 1.11 alone; with `VGPU_JIT_LINK=1` four programs gain 5 to 10 % more, black-scholes 1.33 with both, [round sp52](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp52-linking-keeping-execution-inside-generated-code-across-region-exits-branch-sp52-region-linking-base-originmain-7b90cee5)). | Short kernels with few such runs: no change. AVX2 hosts only. |
| `VGPU_MEMO` | off | Programs that repeat launches on identical memory (24 of 39 pantheon workloads ran 2x or more; [round sp35](perf-design/rounds/10-rounds-sp19-sp27.md#round-sp35-launch-level-result-caching-built-tested-measured-branch-sp35-memo-base-originmain-9f80d40e)); with `VGPU_MEMO_DIR` a second run hits from its first launch. | Programs whose launches never repeat: hashing is bounded by a budget (3 percent of kernel time by default) but not free. Never with a diagnostic that needs the real run. |
| `VGPU_HOST_GPU` | off, experimental | A machine whose real GPU is the simulated model: integer and IEEE kernels run on the card with the simulator's bits ([study 7](perf-design/07-host-gpu.md), [spike](perf-design/09-host-gpu-spike.md)); measured on an A10G: 4x to over 70x on long many-launch programs (FDTD-2D 23x, babelstream 30x), see [round sp42](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp42-host-gpu-measured-on-a10g). | Counters, faults and the simulator's checks do not apply to those launches; the card's driver now starts only for a launch that is offloaded and the start cost (`VGPU_HOST_GPU_START_COST`, seconds, `0` = the old rule) is charged to the decision, so programs under about 0.3 s stay on the simulator (a launch that would start it only because the program has run that long must also repay the start within 50 launches like it, or, once the simulator has run its kernel 200 times, within as many launches as it has run it, at most 500: `VGPU_HOST_GPU_START_LAUNCHES`, `VGPU_HOST_GPU_START_REPEATS`); and kernels using `MUFU.RCP`/`RSQ` outside the division idiom are refused. `turbo` also sends kernels with approximate functions, half and tensor arithmetic and atomics, whose result is the card's and not the simulator's bits ([design](perf-design/12-host-gpu-turbo.md)); on an A10G, geometric mean 1.23x over 59 programs against 1.12x for `exact`, outputs identical ([Round sp59](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp59-turbo-host-gpu-mode-on-an-a10g)) |
| `VGPU_TC_EXACT` | on (sm_86) | Never about speed: it makes tensor-core sums equal the card's. | `0` is 3.5 to 4.5x faster on those instructions and gives different bits. |
| `VGPU_FASTPATH`, `VGPU_FASTCOUNT`, `VGPU_LOOKUP_CACHE`, `VGPU_LAUNCH_CACHE`, `VGPU_SASS_CACHE` | on | Speed, without changing a result. | Set to `0` only to rule one out when a result or counter looks wrong. |
| Build type | Release | Always benchmark a Release build. | PGO builds ([pgo.md](pgo.md)) gave no speed-up on held-out programs and are not released. |

Diagnostics that need the real execution order (`VGPU_SASS_TRACE`, the race detector, a trace, an armed fault, a profiler) switch the tiers,
the JIT and the result cache off or restart them; the same page lists which.

## What was tried and did not pay off

One line each; the evidence is in the linked section. These are worth knowing before repeating them.

* Recognising library kernels (cuBLAS-style) in user code: not done, needs bit-exact accumulation-order proofs ([study 4](perf-design/04-library-kernel-recognition.md)).
* Sharding and sampling a grid: not done, changes results ([study 5](perf-design/05-sharding-and-sampling.md)).
* An x86 template JIT, and the copy-and-patch route: not built ([study 1](perf-design/01-whole-kernel-translation.md), [copy-and-patch](perf-design/copy-and-patch.md)).
* Keeping rows resident across a JIT escape: no gain ([round 13](perf-design/round13-jit.md)).
* Memoising kernels by default: result caching is opt-in because the input set is the hard part ([study 3](perf-design/03-result-memoisation.md), [sp32 measurement](perf-design/rounds/10-rounds-sp19-sp27.md#round-sp32-memoisation-measured-not-built-branch-sp32-memo-base-originmain-7d8d98a4)).
* Release-wide PGO: geometric mean 0.993 on 47 held-out programs, not enabled ([Round sp39](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp39-release-wide-pgo-no-speed-up-on-held-out-programs-stays-opt-in-branch-sp39-pgo)).
* Wide row stores against store-forwarding stalls, and removing the register-file clear: pooled 0.989x and at most 1.02x, nothing kept ([Round sp40](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp40-store-forwarding-stalls-in-the-simd-kernels-register-file-clear-at-block-setup-both-measured-neither-kept-branch-sp40-simd-stalls-base-originmain-d462bc2f)).
* A tier handler for the AMD SDWA integer comparisons (`v_cmp_*_u16_sdwa`): no gain on the program it was written for and the DPP probes 3 to 5.6% slower with it, removed; and the AMD work-group admission after a barrier release, which is faster but changes the order groups run in ([Round sp70](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp70-the-amd-engine-on-inq-the-register-file-pool-one-host-thread-a-physical-core-fmin-and-fmax-of-doubles-quads-of-managed-memory)).
* Counting memory accesses faster in the PTX interpreter and the AMD engine: all leads below the bar, nothing built ([Round sp41](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp41-counting-of-memory-accesses-in-the-ptx-interpreter-and-the-amd-engine-ceilings-measured-first-all-three-leads-below-the-bar-nothing-built-branch-sp41-ptx-amd-counting-base-originmain-d462bc2f)).
* Compiling register-only runs of a block with the guest registers in host registers (fused runs in the JIT's generated code): black-scholes 1.11, four other programs 1.01 to 1.05, below the bar alone (with linking, round sp52, it clears it and is in the tree as `VGPU_JIT_FUSE=1`); with the JIT on the generated code is 9 to 19 % of a compute-bound program and the run loop between region calls 24 to 40 %, which is where a block compiler has to go next ([Round sp48](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp48-compiling-whole-blocks-to-host-code-the-ceiling-measured-per-instruction-kind-the-design-of-the-next-stage-and-fused-runs-of-register-only-ops-built-and-measured-branch-sp48-block-compiler-base-originmain-27f09add)).
* A persistent on-disk cache of decoded modules for short programs: load, decode and compile are 2.9 % of a short program (ceiling 1.031x), not built ([Round sp51](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp51-persistent-on-disk-caches-for-short-programs-the-split-measured-first-load-decode-and-compile-are-3-percent-of-a-short-program-nothing-built-branch-sp51-disk-cache-base-originmain-27f09add)).
* Dispatch-loop variants other than one removed hot line: measured, not kept ([Round sp37](perf-design/rounds/11-rounds-sp36-sp41.md#round-sp37-dispatch-loop-and-block-setup-measured-per-line-one-hot-line-removed-the-rest-measured-and-not-kept-branch-sp37-dispatch-base-originspeedups-combined-48d3e8df)).
* Code placement: results depend on where a loop lands in a 64-byte line, so the hot units sit outside LTO on fixed alignment and a layout lint holds them ([Round sp14](perf-design/rounds/08-rounds-sp11-sp15.md); `tests/lint/check_hot_unit_layout.py`). Judge a change with an A/A control next to the A/B.
* The same local-memory layout in SASS that sped up PTX local arrays: slower, not kept ([local memory layout](perf-design/rounds/04-speedup-branches-tier2-amd-tier.md#local-memory-layout-and-the-ptx-local-regression-branch-speedup2-localmem)).

Many smaller dropped ideas sit under "Not done, and negative results" subheadings in the rounds files.

## Measuring a change

```
tools/bench/bench.sh --build build                       # time every workload, SASS and PTX
tools/bench/bench.sh --ab buildA buildB --runs 9         # alternating A B A B on one program
tools/bench/bench.sh --check-counters --build build      # counts (not times) against tools/bench/expected
tools/bench/bench.sh mix --build build                   # instruction-class mix per workload
tools/bench/bench.sh profile --build build-g             # perf: top functions per engine and group
tools/bench-amd/bench.py --help                          # the AMD engine's harness
```

Rules that cost the most to learn: pin to a quiet CPU (`--cpus`) and check the load; use 9 or more alternating runs and an A/A control; a
speed-up that changes a counter or a bit is a bug, not a speed-up; keep an unmodified build of the base to compare against; every fix ships
with a test that holds it to the interpreter's bits (`tests/unit/test_*same_bits*`). The harness is described in full in
[The harness](perf-design/rounds/01-harness-baseline-profiles.md#the-harness) and "Using this to check a change" in
[the AMD baseline](perf-design/rounds/03-amd-engine-baseline.md#using-this-to-check-a-change).

## Where the detail lives

* [perf-design/rounds/README.md](perf-design/rounds/README.md): index of every round with a one-line result and the full text.
* [perf-design/README.md](perf-design/README.md): the design studies (whole-kernel translation, tiering, memoisation, sharding, JIT, host GPU) and later round notes.
* [pgo.md](pgo.md): the opt-in PGO build. [cli.md](cli.md): every `VGPU_*` variable. [ci.md](ci.md): how CI keeps the same-bits tests honest.
