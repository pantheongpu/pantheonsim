# Round 11: tier 2 (the JIT), milestone 1: the skeleton and the escape

Branch `jit-tier2` (on `224874ea` of `speedup`), 2026-10-22. Design: [06-tier2-jit.md](06-tier2-jit.md). This milestone generates **no arithmetic**: it proves the
machinery (discovery, the generated function's ABI, the escape into the tier's own handlers, the compile, the verified on-disk cache, `dlopen`, background
promotion, exceptions through generated frames, every failure falling back to the tier). The escape-only JIT is slower than the tier on purpose.

Rig: the round-11 c7i.4xlarge (Sapphire Rapids, AVX-512), my cores 4,5,6,7,12,13,14,15; timing runs pinned to CPU 4 with its sibling (12) idle. CI toolchain
image (gcc 13.3 in the CUDA 13.0 image). Raw output: `data/round11/`.

## What was built

| file | what |
| --- | --- |
| `src/sass/jit.hpp`, `src/sass/jit.cpp` | the Engine (generator, key, cache, compile, load; knows nothing of the executor), the Unit (one per code section), the Manager (worker thread, promotion, notices, statistics, exit handling) |
| `src/sass/tier.hpp` | `Program::jit`, the section's Unit |
| `src/sass/exec.cpp` | `jit_` flag (`VGPU_JIT=1` and the tier on), unit creation in `Code::tier_program()` |
| `src/sass/tier_ops.inc` | `jit_escape` (the generated code's callback), `jit_run_region` (the hook), `tier_run_t<bool Jit>` (the run loop; `Jit` adds the hook) and `tier_run` (picks the instance) |
| `tests/unit/test_jit.cpp` | 20 framework-free tests (`test_jit`, CMake-wired) |
| `nvidia/tests/e2e/run_sass_jit.sh` | the end-to-end differential, JIT against the tier (`e2e_sass_jit`, CMake-wired) |
| `docs/cli.md`, `SECURITY.md` | the `VGPU_JIT*` variables; the security statement |

Knobs: `VGPU_JIT=1`, `VGPU_JIT_THRESHOLD` (warp instructions a section's regions must run before it is compiled; default 10,000,000), `VGPU_JIT_CACHE`,
`VGPU_JIT_CXX`, `VGPU_JIT_FLAGS` (in the key), `VGPU_JIT_TIMEOUT` (s, 120), `VGPU_JIT_MEM_MB` (RLIMIT_AS of the compiler, 4096), `VGPU_JIT_MAX_REGION` (96),
`VGPU_JIT_SYNC=1` (compile on the thread that reaches the threshold: for tests), `VGPU_JIT_STATS=1` / `VGPU_JIT_STATS_FILE`. Off by default; off under
`VGPU_TIER=0`, `VGPU_FASTPATH=0`, `VGPU_SIMD=off`, `VGPU_SASS_TRACE`, and in ASan/TSan/MSan builds (compile-time).

### How a run goes

1. `Code::tier_program()` builds the tier's program, then `jit::make_unit` finds the **regions**: runs of consecutive ops with a handler (or `@!PT`), straight, not
   the last op of the section (the loop's own condition), and not a global store (`STG`, `ST`, form `Stg`). A region is at most 96 ops, at least 2, and a branch
   target (a `BRA`'s) starts a new one so that a warp that jumps to a loop top finds a region there.
2. A lookup job (background thread) asks the cache. Hit: the unit is Ready at once. Miss: the unit counts (`note(k)` at every region entry the tier makes) and,
   past the threshold, a compile job is queued; the launch goes on with the tier.
3. Ready: at a region's first op `jit_run_region` checks once what the loop checks per op (`jit::admissible`: `n + k <= 256`, `steps + k <= max_steps`, registers
   in the file) and calls the generated function `void r(Ctx*, group, glanes)`. If any check fails the ops go through the loop one at a time (so the budget, the
   chunk, the store gate, YIELD/BSYNC/spin rules, which live outside the stretch, are untouched).
4. The generated function, per op: computes the executing lanes from the guard (predicate rows through `Ctx::p`/`up`, the same formula as `tier_guard`),
   `++steps; ++instructions; thread_instructions += lanes; by_class[cls] += lanes`, then `escape(ctx, i, ex)`: the tier's own handler for the MOp, and the
   tier-statistics `native` count. Counting happens before the handler, as the loop does, so a throw leaves the counters where the loop leaves them.
5. An exception (`Fault`, `Error`) unwinds through the generated frames to the `try`/`catch` of `tier_run` that attaches the faulting instruction; the escape
   records the last op so the message names it.

### Where this differs from the design, and why

* **One object per code section (a "unit"), not per region.** One compiler start per section instead of per region (a start is most of the cost of a small
  region: below). The key is a hash of the generated source (which holds every instruction word and the counting facts), the compiler's identity
  (`--version` first line and `-dumpmachine`), the flags, the ISA level, the ABI and the generator version. The object carries the full word list and the facts
  as a metadata array and a load compares it word for word with what was asked (hash collisions cannot run the wrong code).
* **The cache is consulted at first sight, not only at the threshold.** A warm cache makes the unit Ready without counting (the lookup is on the worker thread,
  ~0.1 to 6 ms of CPU, plus 2 compiler starts for the identity once per process).
* **The store gate is not checked on entry.** The gate can only trip at an `STG`/`ST`, and no region contains one, so there is nothing to check; the design's
  "no store gate pending" test would only have kept the JIT away from every warp until its first store. Mutation m3 below shows what breaks if a store is let in.
* **`tier_run` is two instances** (`tier_run_t<false>`, inlined into `tier_run` as the base's loop was, and `<true>`, out of line, used only for a section that has regions with `VGPU_JIT=1`): the version of the hook that sat in the one loop cost the JIT-off path 1.5% to 10% (below).

## Tests

Run: `cmake --build build --target test_jit && build/test_jit` (add `VGPU_JIT_REPORT=1` to print the compile and cache-hit costs); `ctest -R 'test_jit|e2e_sass_jit'`; the e2e
needs a built tree (`VGPU_BUILD_DIR`), nvcc, and a C++ compiler on `PATH`, and takes about 4 minutes on 8 cores with the programs already built
(`VGPU_TIER_WORK=<dir>` keeps them; `VGPU_JIT_PROGRAMS="sass_tier.sm_86 sass_tier_steps.sm_86"` narrows it).

`build/test_jit` (20 tests, all pass; skip with a note without a compiler on `PATH`, and do nothing in a sanitizer build):
source is a function of the spec and has no byte that comes from input; key depends on a word, a counting fact, the architecture, the region count, the flags,
the ISA level, the compiler identity (a wrapper script reporting another version), and not on the cache directory; **generated regions against a model** of
the run loop's counting on random groups, predicates and uniform predicates (300 trials, 4 regions of 24 ops, stats on and off) including `@!P`, `@UP`, PT;
**exceptions** thrown at the 6th escape come out of the generated frame with the counters of the first six ops, compiled with and without `-fexceptions`;
second engine hits the cache without compiling; **damaged objects** (flipped bit, truncated, empty, garbage checksum, missing checksum, a good object made for
other words with a good checksum, zeroed) are never loaded and are replaced; objects of another ABI, generator or architecture word are rejected (the
metadata is edited, compiled and installed under the right key); a `Ctx` layout that differs fails the compile (the generated `static_assert`s name the host's
offsets); **no compiler** (nonexistent, empty `PATH`, a "compiler" that cannot run); a **failing compile** leaves no file behind; a compile that outlives
`VGPU_JIT_TIMEOUT` is killed in about the timeout; the compiler runs under `RLIMIT_AS` (the script prints `ulimit -v`); **unusable cache directories** (below a
file, writable by all, created `0700`); **an object that cannot be mapped** (`dlopen` hook returning "failed to map segment", as a `noexec` mount does);
**8 threads and 8 engines compile one key once; 6 processes compile one key once** (flock); admission boundaries (`n+k` 256/257, steps, registers);
discovery (stores, `@!PT`, branch targets, the last op, the cap, minimum length); promotion (40+40+40 against a threshold of 100 compiles once, a second program
with the same words is Ready at once from the cache, another architecture is not).

`nvidia/tests/e2e/run_sass_jit.sh` (52 checks in the final run; `data/round11/e2e-sass-jit-final.log`): builds the programs `run_sass_tier.sh` builds
(`sass_tier` for 7 architectures, `sass_tier_steps`, 44 others) and runs each with the tier and with the JIT forced on (`VGPU_JIT_THRESHOLD=1`,
`VGPU_JIT_SYNC=1`), one host thread, counters on: exit status, every byte of output (results and every counter of every launch) and the tier's own statistics
(`native` per mnemonic) must be identical. Also: four host threads (for every program whose tier output does not depend on the thread count, which `sass_tier` must be, the JIT's four equal the tier's one); on
`sass_tier` a warm-cache run that compiles nothing, the background compile (no `SYNC`), the chain off, and a requirement that generated regions ran at least 10%
of the warp instructions (`sass_tier` on sm_86: 1.44 M of 2.20 M warp instructions, 66%; `sass_tier_steps`: 557 M of 861 M, 65%); on `sass_tier_steps` the seven fault programs, tier
against JIT, with the generated code built with and without `-fexceptions`, requiring that at least 2 were thrown from inside a generated region (gather and
shared are; counted by the statistics line), a sweep of `VGPU_MAX_STEPS` across the 100 steps below each of those two faults, and regions capped at 255, 256
and 257 ops; and the **failure modes**, output identical to the tier and no region ever run: no compiler, a compiler that fails, a cache that cannot be
created, a cache others can write to, a compile that times out (`VGPU_JIT_TIMEOUT=1`).

Beyond CI, run by hand on the rig in the container (output in this document's measurements section): a cache on a **real `noexec` tmpfs** and on a **read-only
bind mount**: the notice, the tier carrying on, identical output.

The existing suites on this build: `test_tier` 8/8, `test_fastpath` 38/38, `e2e_sass_tier` (tier against the interpreter, 52 ok, PASS): the tier is unchanged.

### Mutation checks (each seen to fail)

Each is a one-line edit of the source on the rig, a rebuild, then `test_jit` and the e2e on `sass_tier.sm_86` + `sass_tier_steps.sm_86`
(`data/round11/mutants-run*.log`, the script's `sed` lines are in the table).

| | what the edit does | `test_jit` | e2e (`sass_tier.sm_86`, `sass_tier_steps.sm_86`) |
| --- | --- | --- | --- |
| m1 | the generator counts `thread_instructions += glanes` (the whole group) for every op instead of the executing lanes (**a wrong counter**) | 5 of 19 fail | FAIL: output differs (both programs) |
| m2 | `jit::admissible` allows `n + k <= 257` (**a wrong chunk limit**) | 1 of 20 fails (`jit_a_region_runs_in_one_go...`) | PASS |
| m3 | discovery lets `STG`/`ST` into regions (**a skipped store gate**) | 1 of 19 fails (discovery) | FAIL: output and the tier statistics differ |
| m4 | `admissible` drops `steps + k <= max_steps` | 1 of 20 fails | FAIL: the sweep, `gather` with `VGPU_MAX_STEPS=1856` (fault at step 1857) reports the fault where the tier reports the limit |
| m5 | the escape does not record the op it runs | 0 | FAIL: the `gather` and `shared` fault texts name another instruction |
| m6 | the generated code skips the escape when no lane executes, with the statistics on | 5 of 19 fail | FAIL: the tier's `native` statistics differ |
| m7 | the generator ignores `@!P` negation | 5 of 19 fail | FAIL: exit status, output, statistics |
| m8 | `tier_run` does not take the faulting op from the failed region | 0 | FAIL: the `gather` and `shared` fault texts differ |
| m9 | `admissible` drops the register-file check | 1 of 20 fails | PASS |

Notes. The mutants were run three times as the tests grew (`data/round11/mutants-run1.log`, `-run2.log`, `-run3.log`). In run 1 (the first version of the e2e
script and the code before `jit::admissible` and the helper functions were factored out) m1, m3, m6, m7 and m8 were caught, and **m2, m4 and m5 survived**:
m5 because the fault programs ran with `VGPU_QUIET=1`, which hides the text of the fault that names the instruction; m4 because nothing checked a step limit
that falls inside the region a fault is in; m2 because the chunk limit is invisible to our programs' output (below). Run 2 (the e2e without `VGPU_QUIET`) caught m5 and still
missed m2 and m4 (the first sweep of limits did not reach the faulting region). Run 3 is the final code and script: the table above (m1, m3, m6, m7 are the lines of
run 1, which did not change). **m2 and m9 are not visible in the output of any of our programs**: the chunk limit only changes
how a long straight run is cut into the scheduler's calls (observable by a program whose cross-warp timing depends on exactly where a body of 515 or 516
straight instructions is cut, which none is), and the register check only matters for a warp whose file is smaller than the code asks for (USETMAXREG shrinking it).
They are held by the unit test of the predicate the run loop calls. Each mutant was applied with `sed` to the rig's copy and the file restored afterwards; the
`sed` lines are in `data/round11/mutants.sh`.

## Measurements (rig, one CPU, sibling idle; medians)

**Compile time.** gcc 13.3, `-O1`, `-fexceptions`, 4 GB RLIMIT_AS, one compile at a time, nice 10:

| unit | compile | note |
| --- | --- | --- |
| 1 region of 60 ops (test) | 88 ms | a compiler start is most of it |
| 10 regions of 60 ops | 545 ms | 54 ms per region |
| 50 regions of 60 ops (3000 ops) | 2.6 s | 52 ms per region |
| `sass_tier`, 29 units, 430 to 545 regions of about 18 ops | 3.6 to 4.0 s in all | 135 ms per unit, 7 to 9 ms per region |
| `sass_tier_steps`, 26 units, 342 regions | 7.1 to 7.5 s in all | 280 ms per unit, 21 ms per region |

Peak compiler RSS over the 27 compiles of `sass_tier_steps`: largest 186 MB (3.6 s), mean 45 MB (`data/round11/compiler-rss-sass_tier_steps.txt`). Peak for one
full 96-op region alone: NOT MEASURED.

**Cache hit cost** (lookup = key, checksum of the file, `dlopen`, metadata compare, `dlsym`s; a unit test with `VGPU_JIT_REPORT=1`):

| unit | warm engine | new engine (adds the compiler identity: `--version` and `-dumpmachine`) |
| --- | --- | --- |
| 1 region, 60 ops | 0.14 ms | 2.5 ms |
| 10 regions | 1.2 ms | 3.6 ms |
| 50 regions (3000 ops) | 5.9 ms | 8.6 ms |

Most of the warm cost is generating and hashing the source for the key; the sync-mode process overhead measured on the harness's `startup` workload (one tiny
kernel) was 3.7 ms to 9.7 ms. In the default mode all of this is on the worker thread.

**The escape-only JIT against the tier**, `sass_tier_steps` (861 M warp instructions, 65% of them in generated regions), 5 alternating runs, cache warm:

| | wall s (5 runs) | median | Mi warp instr/s |
| --- | --- | --- | --- |
| tier | 11.39 11.12 11.12 10.91 10.95 | 11.12 | 77.5 |
| JIT, `THRESHOLD=1 SYNC=1` | 15.83 15.67 15.41 15.44 15.83 | 15.67 | 55.0 |
| JIT, default threshold (10 M), background | 15.45 15.23 15.35 15.31 15.96 | 15.35 | 56.1 |

So **the escape-only JIT is 1.41x (sync) and 1.38x (default) slower than the tier** where about two thirds of the instructions are in regions: 4.55 s more over 557 M instructions run in regions is
8.2 ns per op for the generated function's counting, the guard and the indirect call into the handler, on top of the handler. Expected, and what milestone 2 removes by generating the
arithmetic in place of the call. On the harness workloads (0.05 to 0.4 s each, 5 alternating passes of 3 runs, warm cache, sync) the JIT costs +0% to +16% of
wall (`api` 0.999; `wmma` 1.025 to `warp` 1.164), which is a fixed process cost of about 6 ms plus the per-op escape cost (`data/round11/harness-tier-vs-jit-warm-cache.txt`).

**The tier with the JIT off, build with these changes against the base build** (harness, alternating, CPU 4):

| version of the hook | divergent | shared | int_alu | local |
| --- | --- | --- | --- | --- |
| v1: hook, a `Ctx` and a host struct built in `tier_run` for every call (9 alternations; the full 16-workload table, 7 alternations, is in the data dir, all within 1.01 but `shared` 1.06 and `divergent` 1.10) | 1.103 | 1.067 | 1.001 | 1.018 |
| v2: hook moved into a `noinline` helper (spread ~1%) | 1.015 | 1.029 | 1.026 | 1.033 |
| v3: a second instance of the run loop for the JIT (a template parameter), both inlined into `tier_run` | 0.988 | 1.010 | 0.993 | 1.006 |
| **final: `tier_run_t<false>` always inlined as the base's loop was, `<true>` out of line** (7 alternations, all 16 workloads: below) | 0.995 | 1.006 | 0.975 | 0.998 |

**Final build against the base build, JIT off** (all 16 workloads, 7 alternating passes of 3 runs, `data/round11/jit-off-vs-base-all-workloads-final.txt`):
`jit/tier` between 0.949 and 1.006 for 15 workloads (the first six, `int_alu` to `mma_ptx`, come out 2.5% to 5% *faster* than the base build with spreads of about 5%:
code layout and noise, not a gain I claim), **`launch` 1.023** (spread 2.3%; 1.021 on a rerun of `launch`, `startup`, `api` with 9 alternations of 7 runs).
`launch` is thousands of one-warp launches: `perf stat` on it (3 alternations) shows +0.3% instructions (about 70 per launch) and +2.5% cycles against the base. A
control build (the base plus `jit.cpp` compiled in and never called) is equal to the base, and so is the split of `tier_run` (a build with only the
`jit_` flag, the call in `Code::tier_program()` and the `Program::jit` member shows the same +0.23% instructions), so the cost is in that plumbing, but I did not
find which line (NOT ISOLATED; `data/round11/launch-path-perf-stat.txt`). Offered as: a 2% cost on a pure launch microbenchmark, zero on every workload that runs
instructions, for an opt-in feature; the parent decides whether that is acceptable or I should keep hunting.

**Real failure modes in the container** (`noexec.sh` on the rig; output identical to the tier each time):

```
cache on a noexec tmpfs:   pantheonsim: JIT: cannot load an object just built in /nx/cache (is it mounted noexec?): ...: failed to map segment from shared object
cache on a read-only bind mount (dir to create): pantheonsim: JIT: cannot create /ro/cache: Read-only file system
cache that is the read-only mount's root:        pantheonsim: JIT: /ro is not owned by the user running the simulator
```

## Not measured, not done, and open questions

* **NOT MEASURED:** clang++ as the compiler (none on the rig image); non-x86 hosts; a sanitizer build of `test_jit` (the JIT is compiled out there, the test returns
  early); peak compiler memory for a single 96-op region; the cost of `-fexceptions` frames beyond "escape-only is 1.4x"; AVX-512 flags (they are passed and part
  of the key, but nothing in this milestone uses them); behaviour with `RLIMIT_AS` set by the user smaller than the compiler needs (it fails the compile, which is a
  fall-back, but I did not run it).
* **Mutants m2 and m4 are not visible in the output of the e2e programs** (below the table): the chunk limit changes only how the turn's `step_warp` calls are
  partitioned (visible to a program whose cross-warp timing depends on exactly where a body of 515 or 516 straight instructions is cut; none of ours is), and the
  step budget only whether ops after the limit ran (the error text is the same). They are held by the unit test of `jit::admissible`, which the run loop calls.
* Decisions for the parent: (1) per-unit objects against the design's per-region objects (this branch: per unit); (2) cache lookup at first sight (this branch: yes);
  (3) the default threshold (10 M warp instructions, not measured against any break-even: it is only meaningful once milestone 2 makes a compiled region faster);
  (4) the compiler flag set is `-O1`; milestone 2 will want `-O2` and the simulator's SIMD headers on the include path, which makes the key depend on them and the
  source no longer self-contained; (5) whether `VGPU_JIT_STATS` should join `VGPU_TIER_STATS`'s file output by default.
* **Milestone 2 was not started.** No arithmetic is generated.

# Milestone 2: the ALU subset (2026-10-22, second rig)

Approved by the user after milestone 1. Branch `jit-tier2`; checkpoints `e0c2ad1e` (WIP), `9e2902d7`, and the final commit. Raw output: `data/round11/m2-*`.

## What changed

* **Generated forms**: MOV, IADD3 (plain and with carries), IMAD (plain, with carry, WIDE, HI), LOP3 (with its predicate output), LEA (all variants), SHF, ISETP, FSETP (no |x|/-x), FSEL, FFMA/FADD/FMUL (plain and the ftz/sat `_x` kernels; a directed rounding mode stays an escape), PRMT (default mode), IMNMX, I2FP. Every other op of a region stays an escape call of the tier's handler (`src/sass/jit_gen.cpp`; each case names the handler it follows).
* **One source of truth for the arithmetic**: the generated code `#include`s `src/simd/kernels.inc`, the text the simulator's own kernels are compiled from. `kernels.inc` got `VGPU_LANES_ONLY` guards (leaving out half conversions, HMMA, binary64, which need other headers) and two attribute macros (`VGPU_KN`, `VGPU_KA`: empty in the simulator, `always_inline` in the generated code). `kernels.inc` and the new `jit_rt.h` are embedded in the binary by CMake (`jit_embed.cpp.in`), written into `<cache>/rt-<hash>/` at first use (rewritten if different, so the compiler reads what this binary embeds), and the hash of both is part of every key. Nothing of the source tree is needed at run time. The ISA level is the simulator's own (`simd::level()`: AVX2 or base; there is no AVX-512 kernel, so an AVX-512 host builds AVX2).
* **The simulator is unchanged by the refactor of `kernels.inc`**: `test_simd` 25/25, `test_fastpath` 38/38, `test_tier` 8/8, `bench.sh --check-counters` PASS (16 workloads), `e2e_sass_tier` PASS (52), and with the JIT forced on: `e2e_sass_archs` PASS (292 ok, every architecture the script covers), `run_sass_jit.sh` PASS (52).
* **Compile**: `-O2 -funroll-loops` (unrolling the 4-chunk kernel loops was worth 25% on `int_alu`), the kernels forced inline; each region body is compiled twice (a whole warp, group all ones, where every unguarded store is a plain row copy and the counters are constants; and any other group). Counters are accumulated in locals and written out before every escape call and at the end.
* **Only hot regions are compiled** (a compile costs by the size of the source: 15 s for an 844 KB unit, 1.0 to 2.1 s for the hot loops of `int_alu`, `compute_virus` and `int_virus`). Regions are counted per run while the section is `Idle`; the cache is asked once the regions have run `VGPU_JIT_LOOKUP_AFTER` (1 M) instructions; at the threshold the regions that ran at least 0.1% of the counted total (`VGPU_JIT_HOT_PERMILLE`), are at least half generated, and fit in 1500 ops (`VGPU_JIT_MAX_OPS`) are compiled as the unit; the selection is recorded in `<section key>.sel` so a later run finds it without counting. Regions under 4 ops stay on the tier (`VGPU_JIT_MIN_REGION`: a call of generated code has a cost). A section with no region that is at least half generated has no unit at all.
* The threshold default is now **100 M region instructions**. How it was chosen: the generated code saves about 5.7 ns per warp instruction on `int_alu` (87 to 150..207 Mi/s) and a hot-loop compile costs 1 to 2 s, so a compile pays back after 180 to 350 M instructions of that kind of work; waiting for 100 M (1.2 s of tier time) means a run that gets that far has, by then, shown that it is long. It is a knob.

## Tests and gates

* **(a) Special values, tier == JIT bit for bit, per form**: `sass_tier` (random inputs with zeros of both signs, extremes, NaNs with payloads, infinities, subnormals, shift counts past the width, carries at every word boundary; blocks of 128, 48 and 33 threads so the last warp is half empty; guards) on 7 architectures, now with an IMNMX kernel (`k_minmax`) added, runs identically with the JIT forced on (threshold 1, every region compiled). `e2e` shows 63 to 80% of its region instructions computed by generated code (`sass_tier_steps`: 99.9%). **Mutation check per form** (`-DVGPU_JIT_MUTANTS=ON`, `VGPU_JIT_MUTANT=<form>` flips one bit of lane 3 of the form's result, or of lane 2 of a predicate result; `maskless` makes every generated store ignore the executing lanes): all 20 mutants (19 forms and `maskless`) change the output (`m2-final-form-mutants.log`; `imnmx` survived until the `k_minmax` kernel was added to `sass_tier.cu`).
* **(b) Everything of milestone 1**: `test_jit` 21/21 (a test of the selection record and a changed promotion test added), `run_sass_jit.sh` PASS with 52 checks (counters, faults inside a region, 1 and 4 threads, threshold-1 differential across all of `run_sass_tier.sh`'s programs, failure modes). The e2e script's JIT environment now sets `VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2` so that every region is compiled.
* **(c)** `e2e_sass_tier` PASS; `e2e_sass_archs` with `VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1` and all regions selected: PASS, 292 ok.

## Measurements (rig, CPU 4, sibling idle)

**Harness workloads, tier against JIT** (5 alternating passes of 3 runs, warm cache, `VGPU_JIT_THRESHOLD=100000 VGPU_JIT_SYNC=1`, wall seconds; `data/round11/m2-harness-tier-vs-jit-sync-threshold100k.txt`):

| workload | tier | JIT | JIT/tier | share of warp instructions computed by generated code |
| --- | --- | --- | --- | --- |
| int_alu | 0.1466 | 0.0797 | **0.54 (1.84x faster)** | 90.5% |
| f32_fma | 0.0950 | 0.0695 | **0.73 (1.37x)** | 97.2% |
| local | 0.2111 | 0.1784 | 0.85 (1.18x) | 80.7% |
| mem | 0.3439 | 0.2968 | 0.86 (1.16x) | 65.7% |
| warp | 0.0940 | 0.0961 | 1.02 | 33.0% |
| divergent | 0.3601 | 0.3869 | 1.07 slower | 2.3% |
| shared | 0.0829 | 0.0923 | 1.11 slower | 7.2% |
| f64_fma | 0.0797 | 0.0934 | 1.17 slower | 0.2% |

The slowdowns are the cost of the infrastructure on a run too short and too escape-heavy to pay it: the loop that has the region hook (`tier_run_t<true>`) is 10% slower than the plain one for every op that is not in a compiled region, plus the cache lookup and compiler-identity queries (about 10 ms of CPU) done once a section has run 1 M region instructions. The hook is used only for sections with a region that is at least half generated, and a section whose compile finds nothing (or fails) goes back to the plain loop. The default-threshold mode (async) on these 0.1 to 0.4 s runs showed +4% to +16% on `divergent`, `shared`, `f64_fma` and `warp` (`m2ae`, not saved as a file; the number is from the run on 2026-10-22 17:50 UTC). **Cold**: a fresh cache pays the compile inside the run in sync mode (1.4 s for the 5-region `int_alu` unit); in the default async mode the compile runs on another thread while the tier goes on.

**Pantheon stress workloads** (prebuilt `build-vgpu-86` binaries from the WSL checkout, `bench.sh pantheon`, 10 s each, 3 passes, `data/round11/m2-pantheon-workloads.txt`; warm instruction rate per CPU second):

| workload | tier | JIT cold (default async mode, threshold 100 M) | JIT warm |
| --- | --- | --- | --- |
| int_virus | 60.4 Mi/s | 107.1 (**1.77x**) | 107.7 (1.78x) |
| compute_virus | 38.5 | 45.0 (1.17x) | 44.3 (1.15x) |
| fp64_virus | 37.7 | 37.3 (no unit: DFMA is an escape) | 37.1 |

Compile of the hot unit: int_virus 2.1 s, compute_virus 1.0 s (the cold int_virus run spent its first 2 to 3 s on the tier: 57% of its instructions ran in generated regions; warm: 93%). `pulse_virus` mostly sleeps; its rate means nothing.

**JIT off against the base build** (`m2-jit-off-vs-base-all-workloads.txt`, 16 workloads, 7 alternations): 0.97 to 1.01 on 15 workloads, `launch` 1.016. The earlier +2% on `launch` was not looked into further (no time-box left after the measurements; NOT FOUND).

**Compile time per section**: 0.9 to 2.1 s for the hot units above; 1.5 s mean (0.7 to 3.7 s) per unit when every region of `sass_tier` (29 units) or `mma_forms` is compiled. Parsing the embedded kernels costs 0.6 s of each. A precompiled header was tried and not used by the compiler (the generated file does not start with the include).

## Verdict

**Kept, as an opt-in feature that wins where it was meant to**: ALU-bound loops (`int_alu` 1.84x, `int_virus` 1.77x with the compile hidden in a background thread, `f32_fma` 1.37x, `compute_virus` 1.17x), with a cost of a few to 17% on short runs of kernels that are mostly escapes (`f64_fma`, `divergent`, `shared`, 0.1 to 0.4 s) and a lower bound of zero when `VGPU_JIT` is unset. It does nothing for fp64, divergent code, or memory-bound kernels (milestone 4's job), and nothing for the PyTorch tests (51 M warp instructions in 100 s).

## Not done / NOT MEASURED

* Register residence across ops (the 4 to 5x of study 1): not attempted; values go through the register file between ops (the compiler forwards stores to loads inside a region, but does not remove them).
* `e2e_sass_archs` was run with every region compiled, not with the default selection; `m2ae` (async short runs) not saved; clang++; AVX-512; the launch-path +1.6% unexplained.
* The default `VGPU_JIT_HOT_PERMILLE` (1) and `MIN_REGION` (4) are not tuned beyond the harness results above.
* Milestone 4 (memory forms) was not started.
