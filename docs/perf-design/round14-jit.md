# Round 14: tier 2 (the JIT), an adaptive rule for the short-run losses. Measured: the rule removes them with a warm cache; with a cold cache four winners still pay 2% to 5%

Branch `jit-adaptive` (from `origin/main` 840bbf6c), 2026-10-25. Design: [06-tier2-jit.md](06-tier2-jit.md); the milestones this is measured against:
[round11-jit.md](round11-jit.md) (m1 to m3, the code on `main`, called **m3** below) and round 13 (branch `jit-m4`, milestone 4a: no speed-up, not merged, but it showed the
loss is not store-and-reload). Raw output: `data/round14/`.

Rig: c7i.4xlarge (Sapphire Rapids, 8 cores / 16 threads), builds on CPUs 0-3,8-11 in the CI toolchain image (CUDA 13.0, gcc, Release), every timing pinned to CPU 4 with its
sibling 12 idle (5-7 and 13-15 idle). **The gates ran on CPUs 0-3,8-11 at the same time as some of the timings** (the table of measurements says which): the rest of the machine was
not idle for those. Timings are the harness workloads of `tools/bench/kernels/bench.cu` at full size (`mx.py`, written for this round, not in the repository: it runs each
configuration `RUNS` times per pass in a rotating order, one untimed run of each first, and takes the median of the wall seconds of the whole process; the spread is
(max - min) / median). 36 samples per cell (12 passes of 3 runs) unless it says otherwise.

## Verdict

**The rule works where the loss was, and the goal is met only with a warm cache.**

* **Warm cache** (the protocol of rounds 11 to 13: `VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1`, the cache filled by a first run): all 16 workloads are within +0.9% of the
  tier *of the same binary* (36 samples each; the largest: `shared` +0.9%, `mma_ptx` +0.7%, `atomics` +0.3%). m3 against the same tier lost 17.4% (`warp`), 19.4% (`f64_fma`), 21.8%
  (`shared`), 8.0% (`divergent`), 7.8% (`f64_subnormal`), 6.3% (`texture`). The winners did not lose, they gained: `int_alu` 0.416 of the tier's time (m3 0.511), `f32_fma` 0.566
  (0.704), `local` 0.797 (0.854), `mem` 0.814 (0.863); pantheon `int_virus` 2.51x the tier's rate (m3 2.50x), `compute_virus` 1.26x (1.23x), `memory_bank_thrash` 1.24x (1.24x),
  `memory_retention_bake` 2.13x (2.14x, -0.7%, inside the spread), `fp64_virus` 1.004x (m3 0.995x). Most of the winners' extra gain over m3 is not the rule but the second change
  of this branch (the compiler's identity is remembered, section "What was built"), which took 9 to 17 ms of every lookup out of runs of 50 to 300 ms.
* **Cold cache, default knobs** (`VGPU_JIT=1` only, an empty cache: the compile threshold of 100 M region instructions is never reached by these 5 M to 11 M instruction runs, so nothing
  compiles and only the cost of the machinery is paid): nothing is slower than the tier by more than 1.1% **except the four workloads the JIT is for**: `int_alu` +5.4%, `f32_fma`
  +4.9%, `local` +3.6%, `mem` +2.1% (m3: +14.4%, +22.8%, +10.3%, +8.8%). The rule cannot help them, by construction (their regions are 70% to 99% of their work, so they are kept), and
  the cost they pay until the section is compiled (hook loop 1.9 points, counting 1.3, dispatch 2.1 on `int_alu`) is structural. Two ways to lower it were tried and dropped (below).
* **So: "no harness workload more than 2% slower than the tier with `VGPU_JIT=1`" holds for 12 of the 16 workloads in both protocols and for all 16 with a warm cache; it does not
  hold for `int_alu`, `f32_fma`, `local` and `mem` in a short cold run.** A run long enough to reach the threshold (the pantheon workloads: seconds) pays that cost for a fraction
  of a second and then gains.

## Step 1: where the loss sits (measured before the rule was written)

The brief's four candidates, switched off one at a time in a throwaway build of `main` (the patch is `~/.cache/pantheonsim-briefs/jitx-experiment.patch`, not in the repository: the
env knob `VGPU_JITX` bits: 8 no unit at all, 4 the manager only, 32 units built but `step_warp` never uses them, 2 `tier_run_jit` runs the plain loop, 16 the hook loop whose hook returns at
once, 1 the hook without counting). Everything with `VGPU_JIT=1 VGPU_JIT_THRESHOLD=10^12` (nothing compiles), a valid empty cache directory, the same number of environment variables
in every configuration, 30 samples per cell (10 passes of 3), medians of wall seconds, `data/round14/decomposition-main-plus-knobs-10x3.txt`. **No hardware-counter or callgrind profile
decomposes it** (the rig has `perf` with working counters; I used it only for the checks below). The cells are *percentage points of the tier's time*; a difference of 1 to 1.5 points
is at the edge of what 30 samples with a 3% to 10% spread can show.

| workload | tier (s) | unit built, not used (`unit32`) | dispatch through `tier_run_jit` (`plain2` - `unit32`) | hook loop (`hook16` - `plain2`) | region admission (`nocount1` - `hook16`) | counting (`nolook` - `nocount1`) | cache lookup (`full` - `nolook`) | all (`full`) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| warp | 0.0885 | +0.3 | +3.9 | +1.4 | +0.6 | +1.3 | +9.9 | +17.4 |
| shared | 0.0760 | -0.4 | +1.7 | +2.2 | +0.5 | +1.4 | -0.5 | +4.9 |
| f64_fma | 0.0700 | +0.2 | +1.8 | +1.7 | +0.0 | +0.2 | +15.4 | +19.3 |
| divergent | 0.3436 | -0.2 | +4.4 | -0.3 | +0.2 | +0.1 | -0.1 | +4.1 |
| texture | 0.1428 | +0.2 | +0.7 | -0.7 | +0.2 | +0.1 | +0.1 | +0.6 |
| atomics | 0.0442 | -0.1 | +1.6 | +0.1 | -0.1 | +0.3 | +0.5 | +2.3 |
| f64_subnormal | 0.1568 | -0.2 | +0.9 | -0.2 | +0.1 | -0.3 | +0.1 | +0.4 |
| int_alu | 0.1327 | -0.2 | +2.3 | +1.9 | -0.5 | +1.3 | +10.2 | +15.0 |
| mem | 0.2961 | -0.3 | +1.1 | +0.5 | +0.3 | +1.0 | +5.7 | +8.3 |

Reading it:

* **(e) the per-section setup is free**: `JIT=1` with nothing (`off8`), the manager (`mgr4`) and the units built (`unit32`) are all 1.000 +- 0.004. `make_unit` is not the cost.
* **(c) the dispatch is a cost of its own**: running the *plain* loop from inside `tier_run_jit` (a separate `noinline` copy of the loop, called for every section that has a unit)
  costs 0.7 to 4.4 points (`divergent` +4.4, `warp` +3.9, `int_alu` +2.3). `perf stat` on `warp` (`data/round14/dispatch-perf-stat.txt`): +0.7% user instructions (1.53104 G to 1.54169 G),
  +3.5% to +4.1% cycles, and **no change** in L1 instruction-cache misses (about 0.29 M both) or branch misses (about 0.31 M both). So the cost is the code of the outlined loop (a call, a
  larger frame and its register allocation for every pass), not a cache effect; I did not take it further.
* **(b) the hook loop** (the `region_at` test of every op and the call into `jit_run_region`) adds -0.7 to 2.2 points; **admission** (the checks, `fn()`) -0.5 to 0.6; **(a) counting** -0.3 to 1.4.
  Together with (c): 3 to 7 points on a section whose regions are not the work, every run, from the first instruction.
* **(d) the cache lookup (the compiler-identity queries, the section key, the `.sel` file)** is the largest single piece where it happens, and it happens once a section has run 1 M
  region instructions: +9.9 points on `warp` (8.8 ms), +15.4 on `f64_fma` (10.8 ms), +10.2 on `int_alu` (13.5 ms), +5.7 on `mem` (16.8 ms for its three sections); none on `shared`,
  `divergent`, `texture`, which never reach 1 M region instructions. `strace` shows two child processes (`c++ --version`, `c++ -dumpmachine`) per process; `perf record` puts about
  7 ms of kernel time (fork/exec of a process with the simulator's address space, on the core the warps run on) in them. In a run on one core the worker thread of the default
  asynchronous mode takes that time from the warps.
* The sum for `warp`: 0.3 + 3.9 + 1.4 + 0.6 + 1.3 + 9.9 = 17.4, the 17% the earlier rounds saw. Round 13's own "hook only" figure (1.07x `shared`, 1.19x `warp`, 1.18x `f64_fma` with
  `VGPU_JIT_THRESHOLD=10^12`) cannot be read as the hook alone: with the threshold that high the lookup still happens at 1 M region instructions, and with a cache that already holds
  the section (a default cache directory filled by earlier runs) it **loads and runs the compiled regions**. My first attempt at this decomposition did exactly that (`warp` with
  716,770 warp instructions computed by generated code at `THRESHOLD=10^12`); I do not know whether theirs did, and did not look.

## What was built

| file | what |
| --- | --- |
| `src/sass/jit.hpp` | `State::Declined`; `ShareRule`, `share_ok`, `share_rule()`; `Unit::sampling`, `sampled`, `sample()`, `gen_work()`, `hook()`; `Stats::declined/accepted` |
| `src/sass/jit.cpp` | the rule's settings in the Manager; `Unit::sample` (the decision); `Unit::note` does not ask the cache before the rule has decided; the report line and `VGPU_JIT_STATS` print the decision; **`Engine::compiler_id` remembers the compiler's identity in the cache directory**; mutation hooks |
| `src/sass/tier_ops.inc` | `JitSampler` (adds the warp instructions of a hook-loop call to the unit's sample on the way out, also when a fault leaves); `tier_run_jit` is the hook loop only |
| `src/sass/exec.cpp` | `step_warp` calls `tier_run_jit` only for a section whose `Unit::hook()` is true; a declined or failed section takes the inlined plain loop, as a section without a unit |
| `tests/unit/test_jit_share.cpp` (new), `tests/unit/test_jit.cpp`, `CMakeLists.txt`, `nvidia/tests/e2e/run_sass_jit.sh` | tests below |

### The rule (the share rule)

1. **Static, at `tier_program()` time (unchanged, now tested)**: `make_unit` returns no unit, so the section never gets the hook and never counts, when it has no region (a run of
   at least `VGPU_JIT_MIN_REGION`, default 4, straight ops with handlers) that is at least half generated ops.
2. **Dynamic, once, per section**: while a section has a unit and no decision, every call of the hook loop adds the warp instructions it ran (`w.steps` before and after, regions or not)
   to `Unit::sampled`; the call that brings it to `sample_after` (`VGPU_JIT_SAMPLE`, default 50,000 warp instructions; at most the lookup threshold) decides: the **generated share**
   is `sum over regions (entries x generated ops) / sampled instructions`, the entries being the region counts the hook loop kept anyway. Below `VGPU_JIT_MIN_SHARE` (percent, default 40)
   the section is **Declined**: `step_warp` stops calling the hook loop, nothing is counted, and neither the cache nor the compiler is ever asked. At or above, it goes on as in m3.
   `Unit::note` does not start the lookup before the decision (a section that reaches 1 M region instructions in one pass would otherwise look first).
3. **The knobs and the switches**: `VGPU_JIT_MIN_SHARE=0` turns the rule off; so does `VGPU_JIT_HOT_PERMILLE=0` ("compile every region": every differential test and the e2e
   scripts set it), and so does a sample under 1,000 instructions (`VGPU_JIT_THRESHOLD=1`, "compile everything now", gives the lookup point 1). The rule never changes what a region does.
4. **Why 40% and 50,000** (`data/round14/share-at-sample.txt`; per thousand of the sample): `warp` 350, `shared` 278/106/106, `atomics` 143, `fp64_virus` 121, `divergent` 73 (at 5,000
   instructions it reads 735: the prologue; at 20,000, 184; from 50,000 on it is stable), `f64_fma` 46, `f64_subnormal` 23, `wmma` 2, `mma_ptx` 1: declined. Kept: `memory_retention_bake` 444
   and 896, `memory_bank_thrash` 507, `mem` 719/733/691, `compute_virus` 666, `texture` 755, `local` 819, `int_alu` 913, `int_virus` 938, `f32_fma` 992. The bar has to sit between 350 (`warp`
   loses with it kept, +3% even after this branch's other changes) and 444 (`memory_retention_bake`, which gains 2.1x); I first set 60% and `memory_bank_thrash` (507, 1.24x) and half of
   `memory_retention_bake` would have been declined. **The margin to 444 is 4 points; this is a heuristic with a knob, not a law.** The share is a share of instructions, not of time
   (`texture` is kept at 755 and still gains nothing).
5. **Costs and limits**: the sample is the first 50,000 warp instructions of the section, which is a few warps of the first blocks; a kernel whose first 50,000 look unlike the rest (a long
   setup phase before the hot loop) is judged by its setup and **declined for good** (the plain loop does not count, so nothing can change the decision). Until the decision the section pays
   the full cost of (b) and (c) and (a) for 50,000 instructions. NOT MEASURED: a kernel built to be misjudged.

### The compiler's identity is remembered

`Engine::compiler_id` ran `c++ --version` and `c++ -dumpmachine` for every process that looked a section up. It now keeps the answer in a file `cid-<hash>.txt` in the cache directory (so
under the directory's owner/mode checks), named by a hash of the compiler's path, size, modification and change times and inode, and believes it only if its first lines match the path; a
different file is another name and is asked again; a damaged note is not believed and is rewritten. The key of an object is unchanged (the identity is the same text), so every existing
cache stays valid. This is where most of the winners' gain over m3 comes from in the warm table (the lookup of a section is 9 to 17 ms cheaper).

## Tests

* `build/test_jit_share` (new, CMake-wired, 6 tests, no compiler needed): the rule's arithmetic (the bar exactly, nothing sampled, rule off, no overflow at 10^13); the settings read; **the static
  decisions** (a region of escapes only, 1 of 7 generated, regions shorter than the minimum, 8 of 23 and 10 of 26 generated: no unit; 6 of 10: a unit that starts sampled and hooked); **the dynamic
  decision** (590 of 2000 declined: state `Declined`, hook off, counter, and 2 M more region instructions ask neither the cache nor the compiler; 1200 of 2000 kept exactly at the bar);
  and **the lookup waits for the decision** (1.5 M region instructions before any sample: still `Idle`, no cache access; after the sample the next note starts it).
* `build/test_jit` (22 tests now, 21 before): `jit_the_compilers_identity_is_asked_once_and_remembered_in_the_cache_directory` uses a script as the compiler that logs each call: 2 calls in the
  first engine, none in a second engine on the same cache, 2 more for a replaced compiler file, 2 more after the note is damaged.
* `nvidia/tests/e2e/run_sass_jit.sh` (the `ctest e2e_sass_jit` gate): on `sass_tier` a run with the rule **on** (sample 5000, bar 1%, 60% and 100%; no `HOT_PERMILLE=0`) must print what the tier prints.

### Mutation checks (`-DVGPU_JIT_MUTANTS=ON`, `data/round14/mutation-checks.log`; `VGPU_JIT_MUTANT=<name>`)

| mutant | what it does | result |
| --- | --- | --- |
| `share_all_on` | the rule keeps every section | `test_jit_share`: **FAIL** (`..._declined_and_asks_nobody`), 5/6 |
| `share_all_off` | the rule declines every section | **FAIL** (`..._kept_and_asks_after_the_sample`, `..._not_asked_before_the_share_rule_has_decided`), 4/6 |
| `share_no_defer` | a section may ask the cache before the rule has decided | **FAIL** (`..._not_asked_before_...`), 5/6 |
| `cid_nocache` | a remembered compiler identity is never believed | `test_jit`: **FAIL** (the identity test), 21/22 |

The speed gate would catch the first two as well: "rule off" (`VGPU_JIT_MIN_SHARE=0`, the "keeps every section" behaviour) is the `rule off` column of the tables below: `shared` +3.8%, `warp` +3.0%,
`divergent` +4.2%, `f64_fma` +3.5% against the tier.

## Measurements

### Warm cache (`VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1`, one cache directory per configuration, filled by the first untimed run), 36 samples per cell

Wall seconds of the whole process (ratio to the tier of the same binary, `VGPU_JIT` unset). The environment of every column holds the same number of `VGPU_` variables (see "Traps").
`data/round14/rule-only-warm-12x3.txt`. Gates were running on CPUs 0-3,8-11 during this run (`e2e_sass_jit`).

| workload | tier, this binary, JIT unset: s (ratio) | tier, main build | m3 (main build, JIT) | this binary, rule off (`VGPU_JIT_MIN_SHARE=0`) | **this binary, rule on** |
| --- | --- | --- | --- | --- | --- |
| int_alu | 0.1313 (1.000) | 0.1309 (0.997) | 0.0671 (0.511) | 0.0547 (0.417) | 0.0546 (0.416) |
| f32_fma | 0.0842 (1.000) | 0.0842 (1.000) | 0.0593 (0.704) | 0.0478 (0.567) | 0.0477 (0.566) |
| f64_fma | 0.0701 (1.000) | 0.0702 (1.002) | 0.0837 (1.194) | 0.0725 (1.035) | 0.0700 (0.998) |
| f64_subnormal | 0.1564 (1.000) | 0.1567 (1.002) | 0.1685 (1.078) | 0.1563 (1.000) | 0.1560 (0.998) |
| wmma | 0.1877 (1.000) | 0.1924 (1.025) | 0.1918 (1.022) | 0.1882 (1.002) | 0.1879 (1.001) |
| mma_ptx | 0.0962 (1.000) | 0.0992 (1.031) | 0.0993 (1.032) | 0.0969 (1.007) | 0.0970 (1.007) |
| mem | 0.2917 (1.000) | 0.2919 (1.001) | 0.2518 (0.863) | 0.2371 (0.813) | 0.2375 (0.814) |
| shared | 0.0742 (1.000) | 0.0743 (1.001) | 0.0904 (1.218) | 0.0770 (1.038) | 0.0749 (1.009) |
| atomics | 0.0459 (1.000) | 0.0442 (0.963) | 0.0445 (0.968) | 0.0461 (1.003) | 0.0461 (1.003) |
| texture | 0.1425 (1.000) | 0.1421 (0.997) | 0.1515 (1.063) | 0.1402 (0.984) | 0.1408 (0.988) |
| local | 0.1938 (1.000) | 0.1946 (1.004) | 0.1655 (0.854) | 0.1549 (0.799) | 0.1544 (0.797) |
| divergent | 0.3376 (1.000) | 0.3418 (1.013) | 0.3647 (1.080) | 0.3518 (1.042) | 0.3377 (1.000) |
| warp | 0.0883 (1.000) | 0.0885 (1.003) | 0.1036 (1.174) | 0.0909 (1.030) | 0.0884 (1.002) |
| launch | 0.0949 (1.000) | 0.0957 (1.009) | 0.0952 (1.003) | 0.0938 (0.988) | 0.0936 (0.987) |
| api | 0.1955 (1.000) | 0.1959 (1.002) | 0.1959 (1.002) | 0.1956 (1.001) | 0.1953 (0.999) |
| startup | 0.0040 (1.000) | 0.0041 (1.023) | 0.0040 (1.001) | 0.0040 (1.005) | 0.0040 (0.994) |

* `rule on` against its own tier: 0.987 to 1.009 on the 12 workloads that are not winners (spreads 3% to 8%; `startup` 37%, a 4 ms process); the four winners at 0.41 to 0.81. Against m3 it is faster on every workload except `atomics` (1.003 against 0.968, the layout difference of the two builds that shows with the JIT unset too).
* The tier of `main`'s build (`tier, main build`) differs from this binary's tier by -3.7% to +3.1% on workloads where the JIT is not involved at all (`atomics` 0.963, `wmma` 1.025,
  `mma_ptx` 1.031): code layout between two builds. That is why every ratio above is against the same binary with the JIT unset, and why a +-3% movement between builds means nothing.
* The winners: `int_alu`, `f32_fma`, `local`, `mem` are within 0.3 points of `rule off` (which is m3 plus the remembered identity): the rule does not touch them.

### Warm cache with four CPUs (4-7, siblings idle) instead of one, 24 samples per cell

The same configurations, `MX_CPU=4,5,6,7` (the worker thread that does the lookup can run on another core). `data/round14/rule-only-warm-four-cpus-8x3.txt`.

| workload | tier (s) | m3 | **rule on** |
| --- | --- | --- | --- |
| warp | 0.0890 | 1.077 | **1.001** |
| f64_fma | 0.0705 | 1.080 | **1.002** |
| shared | 0.0751 | 1.082 | **1.007** |
| divergent | 0.3392 | 1.056 | **1.001** |
| texture | 0.1431 | 1.006 | **0.988** |
| int_alu | 0.1324 | 0.441 | **0.417** |
| mem | 0.2965 | 0.829 | **0.814** |

With free cores the lookup no longer costs the warps' time and m3's loss is 5.6% to 8.2% instead of 8% to 22%: that is the hook loop and the dispatch (the rest of the decomposition), which the rule removes
for these sections too.

### Cold cache, defaults (`VGPU_JIT=1` only; an empty cache directory per configuration; nothing compiles), 36 samples per cell

`data/round14/rule-only-cold-12x3.txt`. Gates were running on the other CPUs.

| workload | tier, this binary, JIT unset: s (ratio) | tier, main build | m3 (main build, JIT) | this binary, rule off | **this binary, rule on** |
| --- | --- | --- | --- | --- | --- |
| int_alu | 0.1307 (1.000) | 0.1309 (1.001) | 0.1495 (1.144) | 0.1379 (1.055) | 0.1377 (1.054) |
| f32_fma | 0.0834 (1.000) | 0.0837 (1.003) | 0.1024 (1.228) | 0.0883 (1.058) | 0.0875 (1.049) |
| f64_fma | 0.0693 (1.000) | 0.0697 (1.005) | 0.0834 (1.203) | 0.0717 (1.034) | 0.0691 (0.996) |
| f64_subnormal | 0.1558 (1.000) | 0.1559 (1.001) | 0.1553 (0.997) | 0.1559 (1.001) | 0.1561 (1.002) |
| wmma | 0.1858 (1.000) | 0.1911 (1.029) | 0.1913 (1.029) | 0.1846 (0.994) | 0.1855 (0.999) |
| mma_ptx | 0.0956 (1.000) | 0.0990 (1.036) | 0.0987 (1.033) | 0.0966 (1.010) | 0.0960 (1.005) |
| mem | 0.2917 (1.000) | 0.2909 (0.997) | 0.3173 (1.088) | 0.2969 (1.018) | 0.2977 (1.021) |
| shared | 0.0742 (1.000) | 0.0744 (1.003) | 0.0788 (1.062) | 0.0781 (1.053) | 0.0748 (1.008) |
| atomics | 0.0462 (1.000) | 0.0445 (0.964) | 0.0446 (0.966) | 0.0465 (1.008) | 0.0465 (1.007) |
| texture | 0.1419 (1.000) | 0.1423 (1.003) | 0.1425 (1.005) | 0.1438 (1.014) | 0.1435 (1.011) |
| local | 0.1931 (1.000) | 0.1940 (1.004) | 0.2129 (1.103) | 0.1998 (1.034) | 0.2000 (1.036) |
| divergent | 0.3366 (1.000) | 0.3406 (1.012) | 0.3534 (1.050) | 0.3532 (1.049) | 0.3370 (1.001) |
| warp | 0.0884 (1.000) | 0.0886 (1.002) | 0.1031 (1.167) | 0.0930 (1.053) | 0.0881 (0.997) |
| launch | 0.0904 (1.000) | 0.0912 (1.009) | 0.0906 (1.003) | 0.0911 (1.008) | 0.0899 (0.995) |
| api | 0.1954 (1.000) | 0.1960 (1.003) | 0.1952 (0.999) | 0.1952 (0.999) | 0.1950 (0.998) |
| startup | 0.0040 (1.000) | 0.0040 (1.013) | 0.0039 (0.992) | 0.0040 (1.011) | 0.0040 (0.997) |

* **Still slower than the tier by more than 2%** (rule on): `int_alu` 1.054, `f32_fma` 1.049, `local` 1.036, `mem` 1.021. All kept by the rule (91%, 99%, 82%, 70% generated); the cost is m3's
  (2.3 + 1.9 + 1.3 points for `int_alu`), paid until the section is compiled, which in a default run is after 100 M region instructions. Every other workload is within 1.1%.
* m3 in the same table: `int_alu` 1.144, `f32_fma` 1.228, `local` 1.103, `mem` 1.088, `warp` 1.167, `f64_fma` 1.203, `shared` 1.062, `divergent` 1.050: the lookup, which this branch makes cheap, was
  half of it for the short ones.

### Pantheon stress workloads (10 s runs, `--kernel_loops 2 --grid_size 4`, warm cache and `THRESHOLD=100000 SYNC=1`), Mi warp instructions per CPU second, 4 samples per cell

`data/round14/rule-only-pantheon-warm-4x1.txt` (`mxp.py`, not in the repository; counters on in every run as `bench.sh pantheon`; rate = the run's own instructions over its own CPU seconds).
The sources are the unmodified `kernels/<w>/<w>.cpp`, built with `-cudart shared` for sm_86 on the rig.

| workload | tier | m3 | rule off | **rule on** | rule on / m3 | decision |
| --- | --- | --- | --- | --- | --- | --- |
| int_virus | 63.61 | 159.31 (2.504x) | 159.00 (2.500x) | 159.78 (**2.512x**) | 1.003 | kept (938) |
| compute_virus | 39.00 | 48.04 (1.232x) | 48.89 (1.254x) | 49.07 (**1.258x**) | 1.021 | kept (666) |
| fp64_virus | 39.25 | 39.04 (0.995x) | 39.54 (1.007x) | 39.42 (1.004x) | 1.010 | declined (121) |
| memory_bank_thrash | 24.85 | 30.77 (1.238x) | 30.73 (1.237x) | 30.82 (**1.240x**) | 1.002 | kept (507) |
| memory_retention_bake | 63.66 | 136.42 (2.143x) | 136.34 (2.142x) | 135.45 (**2.128x**) | 0.993 | kept (444, 896) |

Spreads 0.4% to 8.2% (`fp64_virus` m3); with 4 samples a movement of 1% to 2% is not distinguishable (`memory_retention_bake` -0.7% against m3, `compute_virus` +2.1%).

## Dropped (negative results, not shipped)

* **Counting with a load and a store instead of an atomic add** (`Unit::note`, no locked instruction on the hot path): cold, `int_alu` 1.051 with the locked add against 1.050 without; `f32_fma`
  1.042 / 1.049; `mem` 1.026 / 1.017; `local` 1.019 / 1.033 (10 passes of 3, `data/round14/cold-counting-nonatomic-dropped-10x3.txt`). No effect outside the noise.
* **Counting one pass in 16** (the hook loop only for passes picked by a hash of the warp's step count, a counted pass counting for 16; the other passes on the inlined plain loop; patch in
  `~/.cache/pantheonsim-briefs/sparse-counting-dropped.patch`): `mem` and `f32_fma` fell to 0.998 and 1.043, but `int_alu` rose to 1.140 and `local` to 1.082. A call of the hook loop
  runs a whole chain of passes (the tier chains them), so the passes are few and long and a sample of them is a poor estimate: the estimate reached the compile threshold of 100 M early
  (a compile was started in an 11 M instruction run: `1 compiles (54.2 ms)` in the statistics) and the compile steals the single core. `data/round14/sparse-counting-dropped-cold-12x3.txt`.
  A sampled count would need the share of the call, not the call, as the unit.
* Not tried: a hook loop with a cheaper `region_at` test, or keeping the regions' first ops in a side table.

## Traps found on the way (they cost me a wrong measurement each)

* **A cache directory that is group-writable is refused** ("writable by others"). `mkdir -p` under `umask 002` makes one; the JIT then fails silently (a `VGPU_QUIET=1` run prints nothing),
  `disabled` is set and later sections get no unit at all. Three of my first experiments ran in that state. Create cache directories with `mkdir -m 700`.
* **The same, as root**: a `sudo perf` run does not accept a cache directory owned by the user; it then also spawns the compiler for every lookup.
* **Extra environment variables slow `launch` by 3.7%** for two variables (`getenv` scans the environment on the launch path: `__strncmp_evex` 5.7% of its profile). A JIT configuration has
  3 to 5 more `VGPU_` variables than the tier's: pad the tier's environment with the same number of dummy `VGPU_PAD<n>` variables, or `launch` shows a loss that is not the JIT's
  (`data/round14/launch-env-size-15x3.txt`).
* **Two builds of the same source differ by +-3%** on small workloads (layout), so compare a binary with its own JIT-unset runs.
* The brief's "hook only" experiment is contaminated by a warm cache (above).

## Gates (the same-bits rule)

All on the binary measured above (the commit `badc5e14`, built once in the CUDA 13.0 CI image; `~/jitR` on the rig) unless the line says otherwise. Logs: `data/round14/gates-*.log`.

* `ctest -R '^(test_jit|test_jit_resid|test_jit_share|test_tier|test_fastpath|test_simd)$'`: **6 of 6 passed** (test_jit 22 tests, test_jit_share 6, test_jit_resid 12 s, test_tier, test_fastpath, test_simd).
* `ctest -R '^e2e_sass_jit$'` (the PR gate: counters, 1 and 4 threads, warm cache, background compile, chain off, faults tier against JIT, the new rule-on stage, one random seed, the failure modes): **PASS** (1038 s).
* `ctest -R '^e2e_sass_tier$'`: **PASS** (79 s).
* `bench.sh --check-counters`: **PASS** with the JIT unset, with every region forced (`THRESHOLD=1 SYNC=1 HOT_PERMILLE=0 MIN_REGION=2 MAX_OPS=10^8`), and with the rule on (`THRESHOLD=100000 SYNC=1`): 16 workloads x SASS and PTX each.
* Random kernels (`jit_random.py` through `run_sass_jit.sh`, `VGPU_JIT_ONLY_RANDOM=1 VGPU_JIT_RANDOM_SEEDS=8`, tier against JIT with every region compiled, 1 and 4 threads): **8 of 8 seeds ok**.
* `e2e_sass_archs` with the JIT forced (`VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2`): (running when this commit was made: the result is in the next commit)
* CI-style build (CUDA 12.0 image, Ninja, `-DVGPU_WERROR=ON`, a `git init`ed copy) and `ctest -R '^lint_'`: **PASS**: the build is clean with `-Werror`; `ctest -R ^lint_` 24 of 24 passed (the `git init`ed copy, so `lint_curl_retries` runs and passes), including `lint_odr_globals` and its self-test; the same six unit tests pass on that build.

## Not done / NOT MEASURED / for the parent

* **The cold-cache cost of the four winners (2% to 5%) is not removed.** Decision for the parent: accept it (it is bounded by the 100 M-instruction threshold and is paid once per section), or
  lower the cost of the hook loop itself (a different structure for the first ops of a region), which is a milestone of its own.
* NOT MEASURED: clang++; AVX-512 code in the generated objects (the rig has AVX-512, the key says `avx2`); `VGPU_THREADS>1` timings (the correctness runs use 4 threads); GPU profiles other than
  `nvidia/rtx3060` and architectures other than sm_86 for timing; PyTorch (51 M warp instructions in 100 s: the sample is never reached with any effect); runs longer than 10 s; a kernel with a long
  setup phase before its hot loop (declined for good, above).
* `mx.py`, `mxp.py` and the `VGPU_JITX` patch are not in the repository.
* The `VGPU_JIT_MIN_SHARE` and `VGPU_JIT_SAMPLE` defaults come from 21 sections of 21 workloads on one machine; a different workload may sit between 350 and 444 per thousand.
