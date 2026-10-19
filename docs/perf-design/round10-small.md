# Round 10, workstream "small": interpreter leftovers and the profile-guided-optimisation number

Branch `speed10-small` (from `speed-round10`). Rig: AWS c7i.4xlarge (Sapphire Rapids, AVX-512), CPUs 6,7,14,15 of the shared box, timing
runs pinned to CPU 6 with its sibling idle, CI toolchain image (CUDA 13.0, Ubuntu 24.04, g++ 13.3 as the image's default `g++`).
Other workstreams ran builds and tests on their own cores at the same time; load average at the start of each run is in the raw files
(`docs/perf-design/data/round10/pgo/ab-*.start`) and was 1.4 to 17. Every A/B is `tools/bench/bench.sh --ab` (alternating, counters
compared, bits self-checked), medians with best-of and spread.

## Part 1: the leftovers

Result: **one change kept (PTX atomics, 1.71x on the PTX `atomics` workload)**, four dropped or not done, each with its measurement.

### a. PTX atomics: one lookup instead of two. KEPT

`exec_atom` did `load_routed` (-> `load_scalar`) then `store_routed` (-> `store_scalar`) for each lane: two table lookups, two
`scalar_location` calls. Now `MemoryManager::rmw_begin` (src/core/memory.cpp) answers the ordinary case with one lookup and returns the
old value, the interpreter computes the new one, and `rmw_commit` writes it (it re-reads the chunk pointer, as `store_scalar` did, and
keeps the "a store of the byte a uniform chunk already holds changes nothing" and "an untouched or uniform chunk becomes real" rules).
`rmw_begin` declines, having changed nothing, whenever the two-access path could behave differently, and the caller then runs the old
two calls: a fault armed on loads or stores (`load_pending`/`store_pending`) or a stuck cell, a host mapping in reach, an address not
naturally aligned, an address in no ordinary allocation, shared, local and parameter space (those never reach it), `VGPU_FASTPATH=0`.
So fault hooks, misalignment and out-of-range messages, the managed/pinned-memory host atomic and the stripe lock are unchanged; the
access counters (`stats_.atomics`, `atomic_bytes`) are bumped before, as before. The SASS engine does not use `exec_atom` and is not
touched.

A/B, `build-base` (speed-round10) against `build`, 15 runs, CPU 6, load average 1.4 at the end (17 at the start of the batch):

| workload | engine | A median Minst/s | B median | median ratio | best ratio | spread A / B | p | verdict |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: | --- |
| atomics | ptx | 2.038 | 3.482 | **1.709x** | 1.691x | 1.7% / 1.1% | 0.000 | faster |
| atomics | sass | 5.982 | 6.146 | 1.027x | 1.026x | 2.2% / 1.2% | 0.000 | faster (not attributed: SASS does not run this code; code placement) |

Earlier trees of the same change, same comparison: 9 runs 1.718x, 11 runs 1.743x, 15 runs 1.752x (PTX). The other 15 workloads on both engines,
same batch (`docs/perf-design/data/round10/ab-final-atomics-build-base-vs-build.txt`): all within 0.98x to 1.02x; four read outside the 2%
band with p = 0.000 and are layout noise, not attributed to a change: `mem` ptx 0.980x, `local` sass 1.022x, `launch` sass 1.026x,
`warp` sass 1.010x. Counters identical in every pair; `bench.sh --check-counters` passes on the new build (tier off and on).

Tests, `tests/unit/test_atom_rmw.cpp` (new, wired in CMakeLists.txt, framework-free, 2 s):

* `rmw_matches_load_then_store_on_every_state_of_memory`: 4000 random rounds, each on two identical managers, memory untouched /
  uniform / real / mixed per chunk, sizes 1 to 8 (3 included), aligned, misaligned, at a chunk's end, past the allocation; no fault, an
  armed-but-idle fault hook, a fault armed on loads, on stores, on both. The old sequence on one manager and the new on the other must
  give the same old value, the same error code and text, the same bytes afterwards, the same `resident_bytes()`, and call the fault hooks
  the same number of times.
* `ptx_global_atomics_match_the_two_access_path`: 13 atom/red forms (add, exch, min.s32, max.u32, and/xor/or with constants that leave a
  uniform chunk uniform, cas, inc, add.u64, exch.b64, add.f32, red) x 6 strides x 5 offsets (aligned, misaligned, near the end, running off
  it) x 4 memory states, 64 threads, `VGPU_FASTPATH=1` against `=0`: same returned values, same final memory, same resident bytes, same
  error text (about 1,500 kernels; some are errors, and the test requires more than 20 so the error text is compared too).
* **Seen to fail** on deliberately broken code (a temporary `MUT` switch, since removed): commit without the uniform "unchanged" rule
  (the PTX test fails: resident bytes), `rmw_begin` ignoring an armed store fault (the memory test fails: hook calls), `rmw_begin` without
  the alignment check (both fail), the interpreter returning the new value instead of the old (the PTX test fails).

### b. Texel reads, a texel's channels in one lookup. DROPPED

Built as `MemoryManager::load_fields` (one lookup for up to four channel fields, declining under the same conditions as above) plus a
per-fetch cache in `texel_channel_bits` (the first channel read of a texel with two or more channels reads them all; `texture_fetch`
clears it). Two tests with mutants that failed as expected. Measured:

* First version (all texels): harness `texture` **0.916x sass, 0.922x ptx** (slower: the harness `texture` workload reads one-channel
  float textures, so the lookup bought nothing and the cache cost). The `docs/performance.md` note "one `load_scalar` per
  channel" only holds for textures of several channels.
* Restricted to texels with a second channel: harness `texture` 1.007x sass, 1.005x ptx (same, as it should be).
* A new micro-workload with `float4` point, `float4` linear and `uchar4` normalised linear fetches
  (`docs/perf-design/data/round10/r10small_tex4.cu`, 11 alternating runs, CPU time): SASS 1.71 s -> 1.65 s (**1.036x**), PTX 1.58 s -> 1.525 s
  (1.036x), spreads about 1%. After the change `perf` still shows `tex_finish` 26%, `tex_footprint_linear` 15%, `fetch_texel` 13%,
  `texel_raw` 11%: the memory read was a sixth of it and the rest is the filter arithmetic.
* The cost: with the texel cache in `interpreter.cpp` the **SASS `shared` workload reads 0.967x** (p = 0.000, repeated three times, 15
  to 21 runs), SASS `shared` is code that never touches a texel. Building the same tree without the texel cache gives 0.993x; with only
  `interpreter.cpp` reverted to speed-round10 and the memory-manager changes kept, 0.995x; giving the cache `alignas(64)` changed
  nothing (0.967x). Two builds of one source in different directories are byte-identical (`cmp` of `libcudart.so.13`), so it is not path
  noise; it is the layout of the library after `interpreter.cpp` grows, the effect round 9 documented (3 to 6%). Moving the new
  memory-manager functions to the end of `memory.cpp` or marking them `noinline` did not matter.

A 3.6% gain on a four-channel texture micro-benchmark that the harness does not contain does not pay for a 3.3% loss on a workload it
does, so the change was dropped (code and tests removed; the micro-benchmark source is kept as the evidence). If a later tree changes the
layout anyway (the parent integrating other branches will), the idea is worth re-measuring on the integrated tree: the memory read is
about a sixth of a four-channel fetch.

### c. Two-entry `Warp::code_hint`. DROPPED, NOT MEASURED ON `bh`

Built (a second hint, swapped on a hit) and measured on a workload written for it (`r10small_divcall.cu`: a divergent warp whose lanes
alternate between the kernel's section and a `__noinline__` device function, 20,000 iterations): SASS 0.71 s against 0.71 s, PTX not
relevant. `perf` of that workload lists no `Module::code_at` and no `fetch` above 0.1%: a path stays in a section for the whole of a call,
so the hint misses twice per call and the time is elsewhere (`step_warp` 29%, `exec_control` 23%). The leftover in the notes came from
`bh`, a HeCBench program; HeCBench is not on the rig and the rules forbid builds and runs on the WSL machine, so **the 5% on `bh` is
NOT MEASURED**. Nothing is kept: a change with no measured win is not shipped. The parent can run `bh` (`~/.cache/HeCBench`) on a
machine that has it; the change is three lines (a second `const Code*` in `Warp`, tried second in `step_warp`, swapped on a hit).

### d. Bank-pass counting for warps not on a stride. DROPPED (mixed)

Tried a pre-pass for one-word-per-lane warps (SASS: the lane's bytes inside one word; PTX: span <= 4): OR one bit per lane's bank into a
mask; if the popcount equals the number of active lanes every lane is in a bank of its own and the answer is 1, without the table. The
general count moved into a `noinline` function. `docs/perf-design/data/round10/r10small_bankbench.cpp`, ns per call for one 32-lane
warp, widths and lane count passed at run time as the engines do, g++ 13 -O3, minimum of 7 runs, CPU 6, two repeats (they agree to 1 ns):

| shape | engine | before | after |
| --- | --- | ---: | ---: |
| permutation of 32 banks, one row | ptx / sass | 61 to 63 / 66 to 67 | 26 to 30 / 44 |
| permutation, random rows | ptx / sass | 62 / 66 | 36 / 44 |
| 8 lanes per 32-word stripe, conflict free (every eighth) | ptx / sass | 66 / 71 | 33 / 52 |
| two rows of a tile (already a closed form) | ptx / sass | 14.5 / 13.4 | 13.5 / 13.4 |
| random words, conflicts | ptx / sass | 118 / 134 | **146 / 197** |

Conflict-free warps off a stride get twice as fast; warps with conflicts off a stride get 25 to 45% slower (the failed pre-pass adds a
loop). Which of the two the programs of the leftover list (`blockScan`, `sortKV`, `fpdc`, `shmembench`) issue is not known without them,
and no harness workload has either shape (`shared` is all closed forms), so there is no measurement in which it wins. Dropped, with the
test for it (a 60,000-warp case in `test_memcount_forms`) removed with it. The bank count of a conflicted warp would need a different
algorithm (a sort-free per-bank dedup), not a pre-filter.

### e. Local-memory layout. NOT DONE (written up)

`Warp::local` is read and written in: `src/sass/exec.cpp` (`mem_read`/`mem_write`, any byte count at any offset, and `fingerprint`, which
hashes the bytes), `src/sass/exec_ops.inc` (the LDL/STL fast path and its fallback), `src/sass/tier_ops.inc` (the tier's `h_local`),
`src/exec/interpreter.cpp` (`lane_local`, a separate per-lane vector used by the PTX interpreter, plus the stack and frame addressing
built on it). Interleaving by words (byte `o` of lane `l` at `(o/4)*128 + l*4 + o%4`) is safe for observable behaviour as far as addresses
go (the program only sees per-lane offsets and the same bounds check), and `fingerprint` only compares equal states. What stops it:

* it helps only when the lanes of a warp use the same offset (static indices, spilled scalars: `base64e`); the harness `local` workload
  indexes dynamically, so its lanes use different offsets and the interleave would not change the cache lines it touches, so the harness
  cannot show a win and the only workload that could (`base64e`) is not on the rig;
* every access that is not a naturally aligned 1/2/4/8/16-byte lane access (`mem_read`/`mem_write` take any `n`, the PTX interpreter
  copies spans with `memcpy` for stack frames, `LDL.128` at 4-byte alignment) becomes a per-word loop;
* the PTX interpreter and the SASS executor would need the same change, each with its own tests (the e2e `sass_vector_memory.cu` moves
  1/2/4/8/16 bytes through local memory by lane-dependent indices and would be the check).

Estimated size: three files and two executors, a day of work, with the gain unmeasurable here. Not attempted.

### Checks run on the final tree

* `tools/bench/bench.sh --check-counters --build build` passes (16 workloads x sass + ptx), also with `--env VGPU_TIER=1`.
* ctest (inside `drx.sh`, -j4) on the final tree: `test_atom_rmw`, `test_fastpath`, `test_memfast`, `test_memcount_forms`,
  `test_sass_access_count`, `test_sass_code_at`, `test_e2e_parity`, `test_exec2`, `test_exec3`, `test_runtime`, `test_dsmem`,
  `test_dynpar`, `test_barriers`, `test_cublaslt_matmul`, `e2e_library_goldens`, `e2e_atomic_cas`, `e2e_host_atomics`, `e2e_textures`,
  `e2e_sass_fastpath`, `bench_counters`, `bench_counters_tier`: 21 of 21 pass. On the tree with the texel change also all texture and
  surface e2e tests (32 of 32, `e2e_packed_int_forms` skipped).
* A full `ctest -j4` of an earlier tree was started twice by mistake at the same time in the same build directory (the first background
  call looked empty and I launched it again); the two runs overwrote each other's files and `amd_memtest_patterns`,
  `test_cublaslt_matmul`, `e2e_library_goldens` failed in it. Run alone, on the same build and on `build-base`, all three pass.
  `lint_curl_retries` fails in the container (no git repository in `/w`), `amd_hipcc` needs `hipcc`, `e2e_cutlass_hopper` was killed
  compiling (memory). I stopped the two ctest containers (both mounted `~/small`, both mine) before they finished the last 47 tests, so
  **a full ctest of the final tree is NOT RUN**.

## Part 2: profile-guided optimisation

CMake already has `VGPU_PGO=generate|use` (CMakeLists.txt, GCC only, `-fprofile-generate/-use`, `-fprofile-update=prefer-atomic`,
`-fprofile-correction`, `-Wno-coverage-mismatch`). `scripts/pgo-train.sh` trains on pantheon's soak programs, which need a separate
checkout built with `make PLATFORM=CUDA`; they are not on the rig's tree in built form, so I trained on what the repository contains:
`docs/perf-design/data/round10/r10small-pgo-train.sh`. Default build untouched.

Two profiles, both from the tree at commit 567a28d1 (the atomics change plus the texel cache that was dropped afterwards; plain and PGO
builds are the same source):

* **harness-trained**: the 16 harness workloads at the quick size on the SASS engine (tier off and tier on) and on the PTX interpreter.
* **harness + PyTorch**: the same plus eleven groups of the PyTorch quick sweep (`amp attention fft graphs linalg losses ops optim rnn
  sdpa sparse`), i.e. **not `nn`** and not `compile`, so `nn` is out of sample for this profile as it is for the first.

### Cost (all targets, `-j4`, four CPUs of the shared box, from scratch)

| step | harness-trained | harness + PyTorch |
| --- | ---: | ---: |
| plain Release build | 642 s | (same) |
| instrumented build (`generate`) | 767 s | 767 s |
| training run (instrumented, 1 thread) | 13 s | 334 s |
| profile data (`build-pgo/pgo/*.gcda`) | 39 files, 1.6 MB | 155 files, 5.5 MB |
| `use` build | 645 s | 598 s |
| `libcudart.so.13` size | 7,668,240 -> 5,916,152 bytes (-23%) | (not recorded) |

So the PGO flow (instrumented build + training + `use` build) costs 1,425 s with the harness profile and 1,699 s with the PyTorch groups,
against 642 s for the plain build: 2.2x to 2.6x the compile time (every target and test is built each time; a build of the shared
libraries only would be less, NOT MEASURED).

### Speed, harness workloads (ratio of medians of Minst/s or units/s, B = PGO, A = plain; 9 runs, CPU 6; verdict of the harness)

Tier off (SASS and PTX), then tier on (SASS). "run 1" and "run 2" are the same pair of builds measured twice.


#### tier off

| workload | engine | PGO, harness-trained (run 1) | (run 2) | PGO, harness + PyTorch groups |
| --- | --- | ---: | ---: | ---: |
| int_alu | sass | 1.109x (faster) | 1.112x (faster) | 1.069x (faster) |
| int_alu | ptx | 0.967x (slower) | 0.965x (slower) | 0.978x (slower) |
| f32_fma | sass | 1.050x (faster) | 1.053x (faster) | 1.061x (faster) |
| f32_fma | ptx | 1.090x (faster) | 1.092x (faster) | 1.078x (faster) |
| f64_fma | sass | 1.087x (faster) | 1.088x (faster) | 1.086x (faster) |
| f64_fma | ptx | 1.134x (faster) | 1.138x (faster) | 1.102x (faster) |
| f64_subnormal | sass | 1.031x (faster) | 1.031x (faster) | 1.027x (faster) |
| f64_subnormal | ptx | 1.018x (same) | 1.019x (same) | 1.005x (same) |
| wmma | sass | 1.010x (same) | 1.008x (same) | 1.003x (same) |
| wmma | ptx | 1.000x (same) | 1.000x (same) | 0.937x (slower) |
| mma_ptx | sass | 1.012x (same) | 1.010x (same) | 1.004x (same) |
| mma_ptx | ptx | 1.259x (faster) | 1.267x (faster) | 1.234x (faster) |
| mem | sass | 1.178x (faster) | 1.179x (faster) | 1.163x (faster) |
| mem | ptx | 1.168x (faster) | 1.171x (faster) | 1.183x (faster) |
| shared | sass | 1.137x (faster) | 1.138x (faster) | 1.135x (faster) |
| shared | ptx | 1.293x (faster) | 1.288x (faster) | 1.255x (faster) |
| atomics | sass | 1.107x (faster) | 1.101x (faster) | 1.124x (faster) |
| atomics | ptx | 1.279x (faster) | 1.281x (faster) | 1.072x (faster) |
| texture | sass | 1.187x (faster) | 1.187x (faster) | 1.125x (faster) |
| texture | ptx | 1.221x (faster) | 1.222x (faster) | 1.216x (faster) |
| local | sass | 1.117x (faster) | 1.120x (faster) | 1.102x (faster) |
| local | ptx | 1.330x (faster) | 1.331x (faster) | 1.384x (faster) |
| divergent | sass | 1.042x (faster) | 1.041x (faster) | 1.035x (faster) |
| divergent | ptx | 1.184x (faster) | 1.185x (faster) | 1.088x (faster) |
| warp | sass | 1.079x (faster) | 1.082x (faster) | 1.048x (faster) |
| warp | ptx | 1.035x (faster) | 1.035x (faster) | 0.979x (same) |
| launch | sass | 0.977x (slower) | 0.973x (slower) | 0.884x (slower) |
| launch | ptx | 1.026x (faster) | 1.026x (faster) | 0.975x (slower) |
| api | sass | 1.034x (faster) | 1.035x (faster) | 1.005x (same) |
| api | ptx | 1.037x (faster) | 1.036x (faster) | 1.000x (same) |
| startup | sass | 1.019x (same) | 1.026x (faster) | 1.053x (faster) |
| startup | ptx | 1.113x (faster) | 1.115x (faster) | 0.984x (same) |

#### tier on

| workload | engine | PGO, harness-trained (run 1) | (run 2) | PGO, harness + PyTorch groups |
| --- | --- | ---: | ---: | ---: |
| int_alu | sass | 1.017x (same) | 1.021x (same) | 0.846x (slower) |
| f32_fma | sass | 1.052x (faster) | 1.058x (faster) | 0.918x (slower) |
| f64_fma | sass | 1.101x (faster) | 1.103x (faster) | 1.117x (faster) |
| f64_subnormal | sass | 1.027x (faster) | 1.027x (faster) | 1.024x (faster) |
| wmma | sass | 1.005x (same) | 1.005x (same) | 0.995x (same) |
| mma_ptx | sass | 1.003x (same) | 1.005x (same) | 0.998x (same) |
| mem | sass | 1.128x (faster) | 1.125x (faster) | 0.953x (slower) |
| shared | sass | 1.084x (faster) | 1.084x (faster) | 1.045x (faster) |
| atomics | sass | 1.091x (faster) | 1.083x (faster) | 1.092x (faster) |
| texture | sass | 1.183x (faster) | 1.182x (faster) | 1.089x (faster) |
| local | sass | 1.028x (faster) | 1.037x (faster) | 0.989x (same) |
| divergent | sass | 1.032x (faster) | 1.032x (faster) | 1.021x (faster) |
| warp | sass | 0.968x (slower) | 0.971x (slower) | 0.836x (slower) |
| launch | sass | 0.979x (slower) | 0.981x (same) | 0.888x (slower) |
| api | sass | 1.038x (faster) | 1.039x (faster) | 1.001x (same) |
| startup | sass | 1.027x (faster) | 1.023x (faster) | 1.054x (faster) |

Geometric mean of the ratio over the compute workloads (workload/engine rows; launch, api and startup left out): tier off, harness-trained **1.116x** (run 2: 1.117x;
range 0.965x to 1.331x), harness + PyTorch 1.092x (0.937x to 1.384x); tier on, harness-trained 1.054x (0.968x to 1.183x), harness +
PyTorch **0.990x** (0.836x to 1.117x). The harness-trained profile is in sample for these rows (it was trained on them), so these are an
upper bound; what the profile does to code it never ran is the next section.

The losses are the ones the CMake comment predicts: code the training did not weight is compiled for size. PTX `int_alu` 0.965x and SASS
`launch` 0.975x lose in every profile; with the PyTorch groups in the training mix the harness's own hot loops carry less weight and
tier-on `int_alu` 0.846x, `warp` 0.836x, `mem` 0.953x and `launch` 0.888x get worse.

### Speed, PyTorch `nn` group (quick tier, 79 checks, `OPENBLAS_NUM_THREADS=1`, `vgpu run --gpu nvidia/rtx5090 --preload python sweep.py`, CPU 6)

User + system CPU seconds of the whole process (`docs/perf-design/data/round10/r10small-nn-ab.sh`), alternating, 9 runs, first pair of
the earlier batch dropped as cold (52 s):

| build | median | best | spread | checks |
| --- | ---: | ---: | ---: | --- |
| plain (batch with the harness-trained profile) | 8.80 | 8.76 | 4.3% | 79 ok, 0 FAIL |
| PGO, harness-trained | 9.56 | 9.54 | 4.0% | 79 ok, 0 FAIL |
| plain (batch with the PyTorch profile) | 8.79 | 8.76 | 4.1% | 79 ok, 0 FAIL |
| PGO, harness + PyTorch groups | 7.85 | 7.82 | 4.7% | 79 ok, 0 FAIL |

Harness-trained PGO makes the PyTorch group **8% slower (0.92x)**; PGO trained with the other PyTorch groups makes it **12% faster
(1.12x)**. The same binary-versus-binary difference is stable in every pair; the occasional +0.3 s run is the same in all builds. So the
size of the win is the representativeness of the training run: a profile of synthetic kernels is a loss on the program users run.

### Same bits?

`bench.sh --check-counters` passes on both PGO builds, tier off and on (16 workloads x SASS + PTX). Every timed A/B run is self-checked
(a workload prints PASS only if its results match the host reference bit for bit; a run without it is a failure) and none failed; the
counters of every A/B pair are "identical". All 79 `nn` checks pass against the CPU on both PGO builds. **Not run on a PGO build:** the
full ctest, the e2e suite, the other PyTorch groups, CUPTI/trace/disassembly text comparisons. GCC may change floating-point
contraction or inlining with a profile; this build compiles with the project's flags unchanged, and the harness's bit-exact float
workloads (f32/f64 fma chains, wmma, mma) match, but the whole-suite check is NOT RUN.

### What a release process would need

* **Where the training run lives.** Something that runs the instrumented build over a representative mix. The repository has no such
  script that runs on a CI runner today: `scripts/pgo-train.sh` needs the pantheon workload binaries (separate checkout, `make
  PLATFORM=CUDA`). The harness script written here (`r10small-pgo-train.sh`) trains in 13 s; adding PyTorch groups made it 334 s and
  turned the PyTorch result from a loss into a win. A real release profile has to include PyTorch (and whatever else users run: cuBLAS
  paths, the pantheon programs, a ROCm program for the AMD engine, which was not trained here at all) and has to be weighted so that no
  family starves the others (the tier-on rows above got worse when PyTorch dominated the counts).
* **Compiler and path binding.** The `.gcda` files are named after the object's build path (`#w#build-pgo#CMakeFiles#vgpu.dir#...`) and
  are valid for one compiler version and one revision of each source file; `-Wno-coverage-mismatch` compiles a function whose source
  changed without its profile, silently, so a profile one commit old gives a partial win with no warning. The release job must build
  `generate`, train and build `use` in the same directory name of the same image, in one job, from the tagged source. Committing a
  profile is not an option (it goes stale at the next commit).
* **Reproducibility.** Two plain builds of one source in different directories are byte-identical in `libcudart.so.13` (`cmp`). Whether two
  PGO builds from two training runs are identical is **NOT MEASURED**; the multi-threaded runs of the sweep and the order of counter
  merges make it unlikely unless the profile itself is kept as a build input (then the release would archive the `.gcda` set, 1.6 to 5.5
  MB, with the tag).
* **Time and size.** 2.2x to 2.6x the build time (above) and a 23% smaller `libcudart.so`; the instrumented binary is slower (the CMake
  comment says about ten times; the harness training took 13 s, the PyTorch groups 334 s), so the training budget is minutes. The `use` build is the artifact; the `generate` build is thrown away.
* **Clang and other compilers.** `VGPU_PGO` is GCC-only (`FATAL_ERROR` otherwise). Clang's `-fprofile-instr-use` needs a different flow
  (`llvm-profdata merge`), not attempted. GCC versions: only 13.3 was used; a profile from 12 will not read with 13.
* **Risk.** A profile that does not match the workload makes things slower (the 8% on `nn` above), and the harness cannot see it: it
  needs a PyTorch group in the release A/B gate, as this measurement had.

Not done in this round, as asked: the default build is unchanged and nothing is wired to a release job.

## Raw data

`docs/perf-design/data/round10/`: `pgo/` (every A/B table, the `nn` runs, counter checks, build and training times, training logs),
`ab-final-atomics-build-base-vs-build.txt`, the micro-benchmarks (`r10small_tex4.cu`, `r10small_divcall.cu`, `r10small_bankbench.cpp`) and
the scripts (`ab.sh`, `r10small-nn-ab.sh`, `r10small-pgo-train.sh`, `r10small-pgo-run.sh`, `r10small-pgo-ab.sh`).
