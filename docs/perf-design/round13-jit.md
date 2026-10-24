# Round 13: tier 2 (the JIT), milestone 4a: rows stay resident across an escape. Measured: no gain. M4b and M4c not started.

Branch `jit-m4` (from `origin/main` d8ec6128), 2026-10-24. Design: [06-tier2-jit.md](06-tier2-jit.md) section 6, milestone 4. Earlier milestones and the numbers
this one is compared with: [round11-jit.md](round11-jit.md) (m1 skeleton, m2 ALU forms, m3 register residence). Raw output: `data/round13/`.

Rig: the round-13 c7i.4xlarge (Sapphire Rapids, 8 cores / 16 threads, nothing else running during any timing), builds on CPUs 0-3,8-11, every timing pinned to CPU 4
with its sibling (12) idle, CI toolchain image (CUDA 13.0, gcc 12), Release build. Builds compared: `b-base` = `origin/main` (m3), `b-m4` = this branch.

## Verdict

**Milestone 4a is correct and does nothing for speed.** Keeping the rows an escape does not touch in locals across it (instead of storing every dirty row before every
escape and trusting nothing after it) gives, against the m3 build and against the *same m4 binary with the feature switched off* (`VGPU_JIT_ESCAPE_SETS=0`, which
generates m3's code), differences inside the noise on every workload, including the three the brief names (`mem`, `local`, `compute_virus`). It does not remove the 2%
to 17% losses of short runs against the tier either: the data below show that those are not store-and-reload costs. By the rule of the brief ("stop at the first stage
that does not beat the previous milestone outside the noise") **M4b (inline LDG) and M4c (LDS/STS/LDL/STL) were not started**; the reason each could not be expected to
do better is in "Why" below.

**Slower than the tier, in m3 and in m4 alike (5 passes of 3 runs, `m3/tier` and `m4/tier` of wall time):** `shared` 1.16 / 1.17, `f64_fma` 1.10 / 1.18, `warp` 1.12 / 1.13,
`f64_subnormal` 1.05 / 1.07, `divergent` 1.07 / 1.05, `texture` 1.05 / 1.05, `launch` 1.05 / 1.03, `atomics` 1.01 / 1.03 (15 passes of 5 runs for `shared` 1.16 / 1.18, `warp` 1.13 /
1.14, `f64_fma` 1.17 / 1.11, `divergent` 1.06 / 1.05). Faster than the tier: `int_alu` 0.47 / 0.46, `f32_fma` 0.64 / 0.63, `local` 0.84 / 0.80, `mem` 0.83 / 0.83, and on the pantheon
workloads `int_virus` 2.56x / 2.51x the tier's rate, `memory_retention_bake` 2.2x / 2.2x, `memory_bank_thrash` 1.25x / 1.25x, `compute_virus` 1.23x / 1.21x.

Recommendation to the parent: do not merge the generator change (about 700 lines for no measured speed). If the tests and the verifier are of value on their own, they
are separable: `VGPU_JIT_VERIFY_SETS` and its report are independent of the generator.

## What was built

| file | what |
| --- | --- |
| `src/sass/jit.hpp` | `EscapeSets` (rows 0-255 read and written, predicates P0-P6 read and written, `known`), `OpFacts::sets`, `escape_sets(const tier::MOp&)`, `Config::escape_sets` (`VGPU_JIT_ESCAPE_SETS=0` turns it off), `RowCtx::sync_escape`, `any_dirty`, `dirty_stores`. Generator version 2 -> 3. |
| `src/sass/jit_gen.cpp` | `escape_sets`: the table (below); `RowCtx::sync_escape`: before an escape, stores the dirty rows and predicates the escape reads *or* writes, forgets the copies of the ones it writes, keeps every other row and predicate (dirty or not) in its local. Mutation hooks. |
| `src/sass/jit.cpp` | `make_unit` fills the sets of every escape (an `@!PT` op reads and writes nothing; an escape with a non-trivial guard also reads its guard predicate); the sets are part of the object's metadata; `Engine::source` emits the **landing code** (below). |
| `src/sass/tier_ops.inc` | `jit_escape_verified` (`VGPU_JIT_VERIFY_SETS=1`, report file `VGPU_JIT_VERIFY_REPORT`). |
| `tests/unit/test_jit_resid.cpp` | escapes that declare their sets (below). |
| `nvidia/tests/e2e/run_sass_jit.sh` | on `sass_tier`: a run with `VGPU_JIT_ESCAPE_SETS=0` and a run with every escape verified, both must print what the tier prints. |

### Rule

An escape runs the tier's handler on the real register file. Before it: every *dirty* row or predicate in the escape's read set **or write set** is stored (a masked write
keeps the lanes outside the executing mask, so the old value must be there). After it: the copies of rows and predicates in the write set are forgotten (reloaded or read
in place at their next use); everything else stays in its local, dirty or clean. An escape with no entry in the table reads and writes everything, as in m3.

**Landing code.** A row that stays dirty in a local across a call that throws would be lost (the tier has it in the register file at that point). Each region that has
such a call gets `unsigned esc` (set before the call to the call's number, 0 when nothing is dirty) and a `try { ... } catch (...) { switch (esc) { stores } throw; }`: the
stores of what was dirty at that call. The first version of this forgot `esc = 0` at a later call with nothing dirty and stored stale rows over the handler's results;
`test_jit_resid` found it (its throwing runs compare the whole register file), see below.

### The table

Derived from the handlers in `tier_ops.inc` and the operands the MOp resolved; conservative where unsure; a form not listed is unknown. **Only forms that were checked
dynamically are listed** (next section): mov popc i2fp mufu f2i i2f unary iabs iadd3 imad ffma prmt fadd fmul f2fp hop2 lop3 iadd3x imad32 imadwide isetp fsetp isetpex
fmnmx dop dsetp ldc ldg lds ldl sts stl, and the uniform forms uiadd3 uimad uisetp ulea ulop3 ushf uunary (rows of a register source, every predicate read and written).
Forms whose entry I wrote and then removed because no escape of them ever ran under the verifier: lea shf imnmx fsel sel imad.hi s2r cs2r ldsm hmma imma umov uldc usel
(the first five are forms the JIT generates itself and which are escapes only for a source it cannot make; the next five are not straight-line ops and never in a
region; umov, uldc and usel did not occur as escapes in any program).

## Verifying the table: `VGPU_JIT_VERIFY_SETS=1`

The escape of this mode snapshots the register file, predicates, uniform registers and statistics, runs the handler, and checks (1) that no row or predicate outside the
write set changed, then (2) restores the snapshot, **scrambles every row and predicate that is in neither set**, runs the handler again, and requires that the rows and
predicates of the write set come out as the first time (so the handler reads nothing outside the read set) and that it does not throw. Stores cannot be run twice and get
check (1) only. A violation is printed once (`JIT VERIFY SETS: LDG (pc 0x...) ...`) on stderr, which fails the differential comparisons. It found, while the table was being
written: F2FP reads two sources, not one (`sass_tier`); the uniform datapath must be restored before the second run (the verifier's own bug, found because `UIADD3 UR5, UR5, ...`
ran twice).

Coverage (`data/round13/m4a-verify-sets-coverage.txt`: escapes checked, summed over every test program of `run_sass_tier.sh` on all 7 architectures plus `sass_tier_steps`,
the 16 harness workloads at quick size, and the five pantheon workloads for 4 s each): all 38 forms of the table were checked at least once; the smallest counts are
`i2fp` 1, `fadd` 50, `iadd3` 50, `isetp` 50, `lop3` 100, `mov` 158. The largest: `dop` 6.4 M, `ldg` 3.7 M, `isetpex` 1.2 M. Not covered: a particular flavour of a form
(an `FSETP` with `|x|` is the interpreter's path; whether the harness's `FSETP` programs include one is not known) and any sm_100/sm_120 form the programs do not use. Every
test program that passes with the JIT passes with the verifier on (0 violations: `data/round13/m4a-verify-sets-test-programs.log`).

## Tests

* `build/test_jit_resid`: now 65% of its escapes declare random sets (0 to 3 rows read, 0 to 3 written, 0 to 2 predicates each way, over 16 rows). A declared escape hashes
  only its read set and changes only its write set (executing lanes only); an undeclared one is the old "any row, hash everything". Register file, predicates, counters and the
  escape log must equal the code without residence, with 6 trials per region (whole warp, half, one lane, random, stats on and off) and a throw from an escape in the middle in
  one trial of two regions. 64 units: 768 regions, 4608 runs, 14257 generated ops, 1455 escapes that declare their sets, 12483 escapes executed, **0 failures**.
* `test_jit` 21/21, `test_tier` 8/8, `test_fastpath` 38/38, `test_simd` 25/25 on the final build.
* `run_sass_jit.sh`: see "Gates" below.

### Mutation checks (`-DVGPU_JIT_MUTANTS=ON`; `data/round13/m4a-mutation-checks.log`)

| mutant | what it does | `test_jit_resid` (32 units, 2304 runs) | `sass_tier.sm_86` JIT output vs tier | `sass_tier_steps.sm_86` | verifier lines (tier / steps) |
| --- | --- | --- | --- | --- | --- |
| `esc_small_read` | the generator drops one row from every escape's read set | FAIL, 591 failing runs | n/a | n/a | n/a |
| `esc_small_write` | ... one row from the write set | FAIL, 592 | n/a | n/a | n/a |
| `esc_small_pread` | ... one predicate from the read set | FAIL, 421 | n/a | n/a | n/a |
| `esc_small_pwrite` | ... one predicate from the write set | FAIL (predicates differ) | n/a | n/a | n/a |
| `res_nostore_escape`, `res_noreload` (m3's) | | FAIL, 1283 / 361 | | | |
| `set_mem_addr` | the table's LDG/LDS/LDL entry forgets the address register | | output differs | output differs | 0 / 1 |
| `set_mem_dst` | ... forgets the destination registers of a load | | output differs | output differs | 266 / 43 |
| `set_mem_data` | ... forgets the data registers of STS/STL | | output differs | output differs | 0 / 0 (stores get no read check) |
| `set_alu_src` | the entry of MOV/MUFU/F2I/I2F/POPC/... forgets its source | | output differs | output differs | 31 / 35 |
| `set_dop_dst` | DFMA/DADD/DMUL forgets the high register of the pair | | output differs | **identical** (nothing read the stale half) | 63 / 3 |

`set_dop_dst` on `sass_tier_steps` is the only mutant whose output is not different; the verifier reports it (3 lines), and `sass_tier` shows it in the output.

## Gates (the same-bits rule)

All on the final code (commit `ecd7fb8e`) unless it says otherwise. The speed measurements were taken on the binary of `e8d35026`, which differs from it by one static function
(`escape_sets_on`: no retained rows across a call in code compiled with `-fno-exceptions`), a change that does not touch the text generated with the default flags.

* **CUDA 13.0 build** (CI image, Release): `test_jit` 21/21, `test_jit_resid` PASS (64 units: 768 regions, 4608 runs, 1455 declared escapes, 0 failures; 300 units: 3600 regions, 21600 runs, 67045 generated ops, 6964 escapes that declare their sets, 59760 escapes executed, 0 failures), `test_tier` 8/8,
  `test_fastpath` 38/38, `test_simd` 25/25.
* **`run_sass_jit.sh`, the PR gate** (`sass_tier` and `sass_tier_steps` on sm_86 with every variant: counters, 1 and 4 threads, warm cache, background compile, chain off, sets off,
  every escape verified; the fault programs, tier against JIT, with and without `-fexceptions`; the sweeps of `VGPU_MAX_STEPS`; regions capped at 255/256/257; one seed of random kernels;
  the failure modes): **PASS**. `sass_tier_steps` ran 557 M warp instructions in 342 regions (556 M of them computed by generated code). The first run of the gate on the branch **failed**: the
  fault programs compiled with `-fno-exceptions` (generated code with a `try`) did not build and printed a compile notice; fixed by not retaining rows across calls in that mode
  (`escape_sets_on`); the second run passed. (A run of the gate started while the build tree was being rebuilt under it failed in `jit_random`; that run is not counted.)
* **Random CUDA kernels** (`jit_random.py` through `run_sass_jit.sh`, `VGPU_JIT_ONLY_RANDOM=1 VGPU_JIT_RANDOM_SEEDS=8`: 8 seeds x 16 kernels x blocks of 128, 48 and 33 threads, every region compiled, tier against JIT, 1 and 4 threads): **PASS** (about 90% of the region instructions computed by generated code). These kernels have loads but no new generated form, so they test the sets only through the escapes they contain.
* **`e2e_sass_tier`** (tier against the interpreter, `run_sass_tier.sh`; I kept the last six lines of its output): **PASS**.
* **`e2e_sass_archs` with `VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2`**: **PASS** (`nvidia/tests/e2e/run_sass_archs.sh`, every program on every generation from sm_75 to sm_120a; I kept only the last 15 lines of its output, so the count of `ok` lines is not recorded, and I did not capture the JIT's statistics of this run: that regions ran in it is inferred from the environment, not shown). 1 h 32 min on 16 CPUs, concurrently with other runs.
* **Harness counter check** (`bench.sh --check-counters`): PASS without the JIT (16 workloads x SASS and PTX) and PASS with the JIT forced on every region
  (`VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MIN_REGION=2`).
* **Every escape verified** (`VGPU_JIT_VERIFY_SETS=1`) on every test program of `run_sass_tier.sh` (7 architectures plus the 44 others), the 16 harness workloads at quick size and the five
  pantheon workloads: 0 violations (the run was with the table before the unverified forms were removed from it; the forms that remained were exercised as listed above).
* **CI-style build** (the rig has `cuda12.0-30b119500e5a`, not the `9f7f0f8400f7` of the brief; Ninja, `-DVGPU_WERROR=ON`, a `git init`ed copy): clean build, `ctest -R '^lint_'`:
  22 of 23 pass, `lint_curl_retries` fails (the container runs as root; known from round 11; I did not run it on `origin/main` here); `lint_odr_globals` and its self-test pass
  (the new code has no global with external linkage: the verifier's `VerifyCount` is in the anonymous namespace of `exec.cpp`). `test_jit`, `test_jit_resid`, `test_tier`,
  `test_fastpath`, `test_simd` pass on that build too.


## Measurements

Medians of wall seconds of the whole process, the harness's workloads at full size, warm cache, `VGPU_JIT=1 VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1` as in rounds 11 (the
cache of each build kept between passes); alternating passes in rotating order; the harness's counting run of every workload and configuration gave **identical counters
in every configuration** (the same-bits check of the harness). The spread is (max-min)/median over all samples; `m4nosets` is the m4 binary with
`VGPU_JIT_ESCAPE_SETS=0`, which generates m3's text, so it separates the feature from the difference between two binaries.

**All 16 workloads, 5 passes of 3 runs** (`m4a-harness-tier-m3-m4-5x3.txt`):

| workload | tier | m3 | m4 | m3/tier | m4/tier | m4/m3 |
| --- | --- | --- | --- | --- | --- | --- |
| int_alu | 0.1426 | 0.0664 | 0.0656 | 0.466 | 0.460 | 0.988 |
| f32_fma | 0.0908 | 0.0582 | 0.0576 | 0.641 | 0.634 | 0.990 |
| f64_fma | 0.0760 | 0.0836 | 0.0894 | 1.100 | 1.175 | 1.069 |
| f64_subnormal | 0.1707 | 0.1790 | 0.1823 | 1.049 | 1.068 | 1.018 |
| wmma | 0.2148 | 0.2157 | 0.2121 | 1.004 | 0.988 | 0.983 |
| mma_ptx | 0.1112 | 0.1101 | 0.1082 | 0.990 | 0.973 | 0.983 |
| **mem** | 0.3106 | 0.2582 | 0.2591 | 0.831 | 0.834 | 1.003 |
| shared | 0.0776 | 0.0900 | 0.0911 | 1.160 | 1.174 | 1.012 |
| atomics | 0.0455 | 0.0461 | 0.0470 | 1.013 | 1.032 | 1.020 |
| texture | 0.1491 | 0.1563 | 0.1570 | 1.048 | 1.053 | 1.004 |
| **local** | 0.2047 | 0.1716 | 0.1646 | 0.838 | 0.804 | 0.959 |
| divergent | 0.3603 | 0.3836 | 0.3793 | 1.065 | 1.053 | 0.989 |
| warp | 0.0925 | 0.1032 | 0.1047 | 1.116 | 1.132 | 1.015 |
| launch | 0.0903 | 0.0946 | 0.0934 | 1.047 | 1.034 | 0.987 |
| api | 0.2049 | 0.2050 | 0.2047 | 1.000 | 0.999 | 0.998 |
| startup | 0.0037 | 0.0036 | 0.0036 | 0.990 | 0.987 | 0.997 |

Spreads were 2% to 20% (the JIT's columns 3% to 19%: `int_alu` 15-19%, `shared` 7-14%, `f64_fma` 12-13%): single movements of up to 7% (`f64_fma`) are inside them.

**The workloads where m4 could have moved, 15 passes of 5 runs = 75 samples per cell** (`m4a-harness-short-runs-15x5.txt`; all within the same experiment):

| workload | tier | m3 | m4 | m4nosets | m4/m3 | m4nosets/m3 | m4/m4nosets | spread of m4 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| mem | 0.3096 | 0.2602 | 0.2617 | 0.2600 | 1.006 | 0.999 | 1.007 | 4.9% |
| local | 0.2044 | 0.1716 | 0.1684 | 0.1697 | 0.981 | 0.989 | 0.992 | 8.4% |
| shared | 0.0771 | 0.0894 | 0.0909 | 0.0908 | 1.017 | 1.016 | 1.001 | 15.2% |
| warp | 0.0922 | 0.1041 | 0.1051 | 0.1048 | 1.010 | 1.007 | 1.003 | 11.2% |
| f64_fma | 0.0758 | 0.0887 | 0.0841 | 0.0864 | 0.948 | 0.974 | 0.973 | 14.7% |
| divergent | 0.3605 | 0.3822 | 0.3794 | 0.3797 | 0.993 | 0.993 | 1.000 | 3.8% |
| int_alu | 0.1427 | 0.0663 | 0.0658 | 0.0657 | 0.992 | 0.991 | 1.001 | 20.8% |

`m4/m4nosets` (the feature alone, same binary) is between 0.973 and 1.007: nothing. The two cases that look like gains (`local` -1.9%, `f64_fma` -5.2% against m3)
are equally present in `m4nosets` against m3 (-1.1%, -2.6%) and have spreads of 8% and 15%.

**Pantheon stress workloads**, Mi warp instructions per CPU second, the harness's `pantheon` mode, `nvidia/rtx3060`, 10 s per run, 1 warm-up run, warm cache, SASS engine only,
`VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1` (`m4a-pantheon-m3-m4-m4nosets-4x3.txt`, 4 passes x 3 runs = 12 samples per cell, and `m4a-pantheon-workloads-3x2.txt`, tier included, 6 samples):

| workload | tier | m3 | m4 | m4nosets | m4/m3 | m4nosets/m3 |
| --- | --- | --- | --- | --- | --- | --- |
| int_virus | 59.56 | 152.58 (152.40 in the 3x2 run) | 149.19 (149.74) | 149.58 | 0.978 | 0.980 |
| **compute_virus** | 38.43 | 46.90 (47.07) | 46.53 (46.45) | 46.36 | 0.992 (0.987) | 0.989 |
| memory_retention_bake | 58.81 | 129.43 (129.68) | 128.12 (127.92) | 128.17 | 0.990 (0.986) | 0.990 |
| memory_bank_thrash | 23.59 | 29.38 | 29.38 | not run | 1.000 | |
| fp64_virus | 35.87 | 36.27 | 36.60 | not run | 1.009 | |

m4 is 1% to 2% **slower** than m3 on three of the five, with spreads of 1% to 3% (the 3x2 spreads are 0.5% to 1.6%): outside the noise, but the same movement is in
`m4nosets`, which generates m3's text: it is the two binaries (code layout; the feature is not the cause), the same kind of movement as the "launch +1.6%" of round 11.
The brief's three workloads: `mem` +0.6% and `local` -1.9% (inside spreads of 5% and 8%) and `compute_virus` -0.8% to -1.3%: **m4 does not beat m3 on any of them.**

**JIT off** (`m4a-jit-off-vs-base-all-workloads-7x3.txt`, 16 workloads, 7 alternations of 3 runs, base build against this branch, `VGPU_JIT` unset): ratios 0.978 to 1.009,
counters identical. (`launch` 0.990, the workload where rounds 11 saw +1.6%.)

**Share of warp instructions computed by generated code, compile times** (`m3-m4-share-and-compile.txt`; cold compile, then a warm run; the numbers of `VGPU_JIT_STATS`, one run each):

| workload | warp instr. | in regions | computed by generated code | cold compile m3 / m4 (ms) |
| --- | --- | --- | --- | --- |
| int_alu | 11.14 M | 97% | 90.5% | 1487 / 1635 |
| f32_fma | 5.39 M | 97% | 97.2% | 2527 / 2516 |
| f64_fma | 2.30 M | 0.2% | 0.2% | 847 / 851 |
| mem | 4.90 M | 79% | 65.7% | 2471 / 2642 |
| local | 6.96 M | 97% | 80.7% | 4170 / 4489 |
| shared | 1.40 M | 9.1% | 7.2% | 795 / 832 |
| warp | 5.13 M | 33% | 33.0% | 1070 / 1115 |
| divergent | 9.97 M | 2.3% | 2.3% | 877 / 914 |
| texture | 0.48 M | 54% | 54.0% | (not measured) |
| atomics, wmma, mma_ptx | 0.31 M, 0.31 M, 0.30 M | 0% | 0% (no region ran 100 k instructions) | none compiled |
| f64_subnormal | 0.85 M | 0.03% | 0.03% | none compiled |
| launch, api, startup | | no unit | | |

(Single cold compiles; m4 is 0% to 10% slower to compile, which is its landing code: every region with a retained dirty row has a switch of stores.)

## Why there is no gain (what the numbers say)

1. **The stores M4a removes are cheap and were not the cost.** Static count over the generated source of the hot units (`m3-m4-static-counts.txt`; text, not executions, and
   m4's `strow` count includes the stores of the landing code, which run only on a throw): `local`: 235 ops, 44 escapes, reloads (`ldrow`) 165 in m3, 113 in m4;
   `mem`: 44 ops, 7 escapes, reloads 32 -> 26. A store or a reload of a row is four 32-byte vector moves to or from a cache-resident register file; removing a third of the
   reloads of a kernel in which an escape is a call into a handler that does a hundred times the work changes nothing measurable.
2. **The short-run losses are not an escape cost.** `warp` has **0 escapes** in its generated unit and loses 13% against the tier; `f64_fma` has 0 escapes and 0.2% of its
   instructions in regions and loses 10-17%; `shared` has 9% of its instructions in regions; `divergent` 2%. What they pay is what the JIT adds without helping:
   `tier_run_t<true>` instead of the plain loop for the sections that have a unit (about 10% on every op that is not in a compiled region, round 11), `jit_run_region` per
   region entry, and the cache lookup once the regions have run `VGPU_JIT_LOOKUP_AFTER` instructions (100 k with the threshold of these runs; `compiler_id` starts the compiler twice,
   2.5 ms in round 11). A run with `VGPU_JIT_THRESHOLD=10^12` (`m4a-harness-hook-only-10x5.txt`: the hook and the counting, no compile, and no lookup before 1 M region
   instructions, which `shared`, `f64_fma` and `divergent` never reach; `warp` reaches it once) is already 1.07x (`shared`), 1.19x (`warp`), 1.18x (`f64_fma`) and 1.03x
   (`divergent`) the tier. So the loss is in the hook and the counting, present before any lookup or compile. A fix for it has to go there (an adaptive rule: no `tier_run_t<true>`
   for a section whose regions are mostly escapes or tiny; not tried here).
3. **`mem`, `local`, `compute_virus` already gain from the JIT** (0.83x, 0.84x of the tier's time, 1.22x the tier's rate) from the arithmetic, and what is left is the loads
   themselves, which are in the tier's handler in every milestone.
4. **What M4b and M4c could and could not do** (reasoning, not measured). An inline LDG has to reach the same checked `MemoryManager` (not available to self-contained generated
   code), so it would be one more callback of the host with the same work as the escape's handler: it saves the store of the address row and the call chain `escape ->
   jit_escape -> handler`, and keeps the load's destination rows resident. M4a already keeps every row that is not an operand of the load resident across it, and that measured
   zero; what M4b would add to residence is two or three rows per load. The saving of the call chain is real but I did not measure it (the 8 ns per escape of round 11 is the
   figure of the whole generated-call overhead, not of the chain alone), and I did not build M4b to find out, per the stop rule.

## Not done / NOT MEASURED / for the parent

* **M4b and M4c: not started** (the stop rule). `jit_random.py` was not extended (nothing new is generated).
* NOT MEASURED: clang++; AVX-512 hosts; a full `VGPU_JIT_RANDOM_UNITS=1000` run of `test_jit_resid` (64 units run here; the default of CI is 4); a profile of where the short-run
  loss sits (the hook-only experiment locates it only to "present without any lookup or compile"); `perf` counters of any run.
* The verifier covers the forms that appeared in the programs listed; a form that never ran as an escape has no entry (unknown, as before).
* Decision for the parent: merge or drop the generator change (recommendation: drop); whether the short-run loss deserves its own milestone (an adaptive per-section rule) before any
  further generated form.
