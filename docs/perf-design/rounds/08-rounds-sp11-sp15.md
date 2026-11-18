## Round sp11: tensor-core kernels at the AVX-512 default, and the generic loads behind the wmma samples (branch `sp11-tensor`, base 6575b451 = `sp10-combined`)

### The AVX-512 default, per kernel (no change made)
`nvidia_tensor_bench` (SASS rows with `VGPU_SOURCE_DIR`), same binary, `VGPU_SIMD=avx2` against `avx512`, cpu 2 of server3 (Zen 4, 512-bit operations as two halves), 3 alternating repeats of 5, median of user cycles per warp instruction (B/A = avx512 / avx2):
* Faster at AVX-512: tcgen05 s8 0.59, wgmma s8 0.90, tcgen05 e4m3 0.91, tcgen05 tf32 0.92, tcgen05 f16 f32-D 0.95, tcgen05 mxf8f6f4 0.95, wgmma tf32 0.95, mma f64 m8n8k4 0.91 (1507 -> 1367 cycles), the other tcgen05 / wgmma f16 and bf16 rows 0.97 to 0.99.
* Slower at AVX-512 on this host: wgmma e4m3 1.04 (73.5 K -> 76.1 K cycles, the three repeats agree), SASS QMMA e4m3 1.04, SASS HMMA.1688.TF32 1.03, PTX mma s8 k32 1.03 and a few 1.01 to 1.03 rows inside the 3-repeat spread (about 3%). Every other row (mma f16/bf16/tf32, wmma, SASS HMMA/IMMA dense) is 0.97 to 1.02.
* `mma_avx2_chunked_f` (the f32-product kernel behind `mma.sync`/HMMA bf16, f16 and tf32 on the sm_86 sum) has no AVX-512 variant and `hmma_k` is the AVX2 one in both tables, so the level cannot move it: the two TensorCoreGemm samples at the two levels (perf stat, one thread, cpu 2) read bf16 1.686 T cycles (avx2) and tf32 2.097 T (avx2); with `VGPU_SIMD=avx512` (under `perf record`) the wall was 391.6 s against 377.5 s and 486.5 s against 499.4 s: the sign flips between the two samples, that is noise, not a level effect.
* Not changed: the rule was to change the default only for a kernel that is slower at AVX-512 on every host measured. The Sapphire Rapids numbers of round sp8 (geometric mean 0.916 over 43 forms; every row at or under 1.02 but a few small ones in the noise) do not show wgmma e4m3 or QMMA e4m3 slower there, so Zen 4 alone (a 4% loss on two rows of a rarely used type, 10 to 40% gains on the integer and tcgen05 rows) does not justify a change. Per-kernel tables above.

### Where the time of the two samples went, and the fix
`perf record` (cycles:u, 1 thread) of the bf16 TensorCoreGemm sample (390 s): `mma_avx2_chunked_f` 29%, then `exec_mem` (the interpreter's memory instructions) 17% + 9% + `mem_read` 6% + `shared_block` 4% + libc copies. `VGPU_TIER_STATS=2` named the instructions: `LD.E` (generic-address loads: 403 M of the 841 M warp instructions, 48%, the sample's `wmma::load_matrix_sync` from shared tiles through generic pointers), `LDGDEPBAR` (34 M, control) and `LDGSTS` (34 M, cp.async, already native in the next round's first commit below). Both `LD`/`ST` (generic) and `LDGSTS` had no tier form, so every execution went through `exec_mem`'s per-lane loop (a `classify`, a `mem_read` or `mem_write` and a `shared_block` per lane).
* `Form::Ldgsts` (`h_ldgsts`, src/sass/tier_ops.inc): a copy of 4, 8 or 16 bytes per lane without `.ZFILL`, with the predicate operand and the guard: the global reads are one `load_batch` (two 8-byte batches for a 16-byte lane), then the lanes' shared writes in lane order (lanes on one address leave the last lane's bytes; a lane whose predicate is false writes zeros), counted by `count_access` and `count_shared_access` of the same addresses as `exec_mem` takes. A batch the memory manager declines (fault, misaligned, host mapped), `.ZFILL`, and a shared access outside the block's array are `exec_mem`'s, nothing having been done yet.
* `Form::Ld` / `Form::St` (`h_generic`): the 64-bit generic address is told apart by window (`Runner::classify`): every active lane in the shared window moves as LDS/STS do (`with_lane_width` copies, the shared-window and array check first) and is counted as a shared access of the offsets; no lane in any window is plain global memory and runs `h_global` (LDG/STG's batches and counters); a mix, a local, parameter or constant address, an offset outside the array and a 32-byte access stay `exec_mem`'s (which counts a mixed warp by the last lane's window: unchanged).

### Results (same-bits: counters identical, outputs identical; `perf stat`, user cycles and instructions of the whole sample, 1 thread, cpu 2; A = `sp10-combined` build, B = this branch; A repeated twice agreed to 0.03%)
| sample | A cycles | B cycles | B/A | A instructions | B instructions |
|---|---:|---:|---:|---:|---:|
| bf16TensorCoreGemm (LDGSTS only, first commit) | 1.6908 T (x2) | 1.6234 T, 1.6126 T | 0.96, 0.95 | 7.700 T | 7.178 T |
| bf16TensorCoreGemm (LDGSTS + generic LD/ST) | 1.6908 T | 1.0218 T, 1.0161 T | 0.60 (1.66x) | 7.700 T | 4.286 T |
| tf32TensorCoreGemm (LDGSTS only) | 2.0594 T | 1.9489 T | 0.95 | 8.921 T | 8.399 T |
| tf32TensorCoreGemm (LDGSTS + generic LD/ST) | 2.0594 T | 1.3793 T | 0.67 (1.49x) | 8.921 T | 5.491 T |
The wall time of the bf16 run fell from 377 s (447 s in a loaded window) to 224 to 233 s, tf32 from 499 s (493 s) to 311 s. After the change `mma_avx2_chunked_f` is 50% of the bf16 sample (the kernel itself, which has no host-side excess left), `h_generic` 12%, `hmma_k` 10%, a libc copy 7%. Control (a TensorCoreGemm that uses neither form, `cudaTensorCoreGemm`, 5 alternating runs): 1.018x, p 0.69.
bench.py canaries (`wmma mma_ptx int_alu f32_fma mem shared atomics local divergent warp launch api`, sass and ptx, 9 runs, counters identical in all 24 rows): every row 0.99 to 1.01x, except SASS `wmma` 1.024x (p 0.003), SASS `mma_ptx` 1.037x (p 0.014), SASS `warp` 1.065x (p < 0.001) (all faster, the generic forms) and **PTX `mem` 0.965x** (p 0.04; re-run with 21 runs 0.961x, p 0.014; the A/A of the same builds 1.002x to 1.006x, p 0.3 to 0.6). The PTX engine and its memory manager calls are not touched and the instruction counts are identical: this is the placement effect of the lessons above (the new code sits between the interpreter and the PTX engine in the library). Moving the two handlers into `.text.unlikely` (`noinline`) took the loss off `mem` (0.998x) and put it on PTX `mma_ptx` (0.949x, p < 0.001, 21 runs), and `.text.unlikely` for `h_ldgsts` alone left both (mem 0.962x, `mma_ptx` 0.967x): the base's PTX layout is the lucky one and every shift costs one of the two rows 4 to 5%, so the plain form is kept (the worst row 0.961x, against 0.949x and 0.962x/0.967x). Same-bits at program level (`real.py`, 17 samples that use shared memory, atomics, warp primitives and the wmma/imma GEMMs, BEFORE `sp10-combined` against this branch, outputs and check lines): identical in all 17 (`cudaTensorCoreGemm`, `immaTensorCoreGemm`, `mergeSort`, `reduction`, `histogram`, `transpose`, ...); the two TensorCoreGemm samples under test print only a time.

### Tests (CMake-wired) and mutation checks
`test_sass_ldgsts`: hand-assembled LDGSTS (4, 8, 16 bytes; strides that are a run, a gather, one address and misaligned; shared strides that overlap; the predicate operand and the guard on either sense; two warps; `.ZFILL`), the shared memory read back through LDS, checked against the rule written in the test and against `VGPU_FASTPATH=0`, with every counter; and a copy past the shared array must fault with the interpreter's text. `test_sass_generic_mem`: LD.E / ST.E of 1, 2, 4, 8 and 16 bytes with sign and zero extension, in the shared window, in global memory and in both, with a guard on each instruction, checked against the rule, the interpreter and every counter, the global buffer included; and a generic access past the array must fault with the interpreter's text. `test_tier`: the form table (`ldgsts:O ld:O st:O`) and the form properties. Sanitizers: ASan+UBSan -O0 (`detect_leaks=1`, `halt_on_error=1`) of `test_sass_ldgsts`, `test_sass_generic_mem`, `test_tier` and `test_sass_dispatch`: clean (the new handlers move bytes by `memcpy` and `with_lane_width` copies, no atomic or threaded code: no TSan run). Mutants (a one-line edit of `tier_ops.inc`, the test run): LDGSTS shared write at the wrong address for two lanes (killed, 3 of 4 tests fail), a predicate-false lane writes nothing (killed by the overlapping-shared test), the high 16-byte half read from the low one (killed), shared count as a load (killed), shared offset dropped (killed), predicate operand ignored (killed), global counters of the wrong width (killed), global offset dropped (killed), no bounds fallback (killed by the fault test); generic: shared count as the opposite access (killed), no sign extension (killed), a store repeating the first word (killed), lane 0 skipped (killed), no bounds fallback (killed, segfault), and two survivors that are equivalent by construction (the mixed-window guard `in_shared != popcount`, and the window count, because a mixed warp is also refused by the array check: kept as the explicit rule). An LDGSTS 4-byte copy written as 8 bytes survives too: the extra bytes are zero and the next lane overwrites them (equivalent).

### Not done, and negative results
* An AVX-512 `mma_avx2_chunked_f`: the kernel is 50% of the two samples now, but Zen 4 runs 512-bit as two halves (the kernel is one 256-bit vector chain per row group, so the gain there would be the instruction count only), and the same bits would have to be proven against the f32-product rule on a second CPU: no Zen-4-only measurement can show it, and no Intel host was available this round.
* `.ZFILL` LDGSTS, local- and constant-window generic accesses and 32-byte `LD` stay `exec_mem`'s: the two samples do not use them.
## Round sp11: the SASS verdict file is on by default (branches `sp11-cache-tests`, `sp11-cache-default`, base a3859d74)
The verdict file of #438 (`src/sass/ok_cache.{hpp,cpp}`) shipped opt-in. Before it could be the default it was tried in every place it cannot work or must not be trusted; the answer of `sass::unsupported` is the same in each, nothing crashes or hangs, and the file never says "runs in full" for an image it was not told ran in full.
* Found and fixed (each by a new test, which fails on the code of #438): a relative `XDG_CACHE_HOME` or `HOME` was used as is, so the cache landed in the directory the process happened to run in (both are ignored unless absolute, the XDG rule);
  a FIFO at the cache path hung the process in `open` (opened `O_NONBLOCK`); a short write (a full disk, a size limit) left half a key that shut the file to every later process (it is cut back, when nothing was appended meanwhile);
  a process that started before another had made the file never recorded its own verdicts (it reads the file again and appends).
* Held already, now tested: no `HOME` and no `XDG_CACHE_HOME`, empty or relative (no file, nothing next to the process); the root a file, missing or read-only, and a read-only directory with a warm file (it still answers, never appended to);
  `pantheonsim` or the file a symlink, a dangling symlink, the root itself a symlink (works); files and directories of another user (the test makes the process claim another uid); the file of another build or format version is neither read nor touched
  and the name is `sass-ok-v1-` and the executable's GNU build id; a file cut after 0, 1, 7, 15, 16, 17, 31, 32, 33 bytes; twelve forked first runs released together (eight rounds): every record is the key of an image that ran, none lost, no refusal recorded, no torn record, every verdict right;
  65,536 keys: a full file is read and not appended to, 65,535 takes one more, a longer file is read up to the cap only.
* The default: `VGPU_SASS_CACHE=0` turns it off, anything else or unset is on (a build without a GNU build id, or with no absolute `XDG_CACHE_HOME` / `HOME`, has no file as before). Files of old builds stay in `~/.cache/pantheonsim` (at most 1 MB each, a few KB in practice) and can be deleted any time.
* Not testable on hosted CI: a real `ENOSPC` (the test uses `RLIMIT_FSIZE`, which gives the same short write and then `EFBIG`) and a file owned by another user (the test interposes `geteuid`).

## Round sp12: shared-memory loads and stores of the tier (branch `sp12-shared`, base 2c68075d = `sp11-tensor-pr`)

Where the time went (perf `cycles:u`, one host thread, server3 Zen 4, A = 2c68075d; shares of the program's samples): `h_shared` is 24% of Samples convolutionSeparable (28% of that on one
instruction: the per-lane 4-byte load loop), 25% of sortingNetworks (`rep stos` 5 to 10% of those: the 256-byte `addrs[]` zeroing of every partial-warp access), 13% of Rodinia lud, 8.6% of
transpose. `step_warp` is 17% of transpose, 8% of convolutionSeparable, but spread over the dispatch of the tier loop (the largest single instruction is 6%): nothing to fix there.

### What changed
* `h_shared` works out the 32 offsets as 32-bit words for every lane mask (a lane that does not run reads its register and is left out of the checks) and never zeroes the 64-bit address array: it
  is filled only where it is used, after the contiguous and one-word cases have returned (`tier_widen_offsets`: the active lanes' offsets, zero elsewhere, as before).
* The window check is `exec::shared_window_inside_warp` (include/vgpu/exec/window_check.hpp): rank bits and `size - bytes - offset` negative-or-not, ORed over the active lanes, in one SSE2 pass.
  SSE2 has no unsigned 32-bit maximum, so the earlier `add_warp` maximum was emulated with ten instructions a vector. Same answer as `SharedWindowCheck` (compared in test_memcount_forms).
* The contiguous / one-word test of a full warp of words is 32-bit (it vectors) instead of 64-bit scalar.
* A full warp of 4-byte accesses that is neither one run of 32 words nor one word (`tier_shared_words`): each group of eight lanes whose offsets are eight consecutive words moves as 32 bytes
  (two vector copies), the other groups lane by lane. Tile rows of 8, 16 and 32 words, two rows in one warp. The groups run in order, so two lanes on one word still end with the later lane's value.

### Results (A = 2c68075d, B = branch; same bits: outputs identical, counters identical in every canary row)
Zen 5 (server2, Ryzen 9 9950X, cpu 16, `real.py`, 7 runs, median wall s; A/A of the same window 0.97x to 1.06x, the 1.06x being convolutionSeparable, the other rows 0.99x to 1.01x):
| program | A s | B s | A/B |
|---|---:|---:|---:|
| Samples convolutionSeparable | 11.70 | 10.06 | 1.16x |
| Samples sortingNetworks | 17.23 | 14.77 | 1.17x |
| Samples transpose | 7.77 | 6.79 | 1.14x |
| Rodinia lud | 1.87 | 1.70 | 1.10x |
| Samples mergeSort | 7.84 | 7.25 | 1.08x |
| Samples dct8x8 | 0.86 | 0.80 | 1.08x |
| Rodinia hotspot | 1.15 | 1.09 | 1.05x |
| Samples histogram, reduction | 15.36 / 8.10 | 15.43 / 8.14 | 1.00x, 1.00x |
(The first commit alone, before the group copy: convolutionSeparable 1.07x, sortingNetworks 1.12x, lud 1.07x, transpose 1.04x, mergeSort 1.08x.)

Zen 4 (server3, cpu 2, `perf stat` user cycles and instructions of the whole program, 3 runs each, the box at load 14 to 35; instructions are exact):
| program | A cycles | B cycles | B/A | A instructions | B instructions |
|---|---:|---:|---:|---:|---:|
| convolutionSeparable | 74.9 / 71.8 / 72.7 G | 65.6 / 65.0 / 64.9 G | 0.90 | 322.8 G | 295.0 G |
| sortingNetworks | 93.5 / 92.5 / 92.7 G | 83.9 / 83.3 / 84.9 G | 0.905 | 320.0 G | 312.0 G |
| lud | 16.7 / 16.8 / 16.6 G | 15.7 / 15.7 / 15.7 G | 0.94 | 39.84 G | 36.91 G |
| transpose | 47.3 / 46.0 / 46.4 G | 44.1 / 44.4 / 44.4 G | 0.955 | 190.5 G | 184.5 G |

bench.py canaries (12 workloads, SASS and PTX, 15 runs, Zen 5; counters identical in all 24 rows; A/A in the same window all within 0.99x to 1.01x): `shared` SASS 1.08x, `mem` SASS 1.03x to 1.06x,
others 0.99x to 1.02x, except `api` 0.98x (both engines; the api path does not touch the changed code, so this is code placement after the relink; it did not move with the next variant, and
`launch` is 0.99x to 1.01x). Placement lesson again: the first version, with the group copy inlined into `h_shared`, cost `atomics` SASS 0.914x (p<0.001) and `mem` PTX 0.959x, `mma_ptx` SASS 0.966x
at equal instructions; making the group copy a `noinline` function brought them to 1.00x. Making `tier_widen_offsets` `noinline` too halved the `shared` gain (1.08x to 1.04x) and was dropped.

### Tests (CMake-wired) and the mutations that fail them
* `test_memcount_forms`: `shared_window_inside_warp` against the per-lane window test over 400 000 random warps and masks, and at its edges (last byte in and out, sizes under the access and over the
  clamp, rank bits, one lane, all but one lane). Mutations: limit one too high; the active mask ignored. Both fail `window_checks_equal_the_per_lane_tests`.
* `test_sass_dispatch`: the whole-warp shared shapes now run in warps of 32, 64, 33, 5 and 19 threads (a partial warp), with four more shapes (rows of eight words with a gap; the same two bytes into
  a word; runs of four words only; four rows on the same eight words), and every kernel reads the words back in order (a store and a load that agree on a wrong place hid a wrong group copy). All
  60 shared-memory digests were recorded from the base tree (2c68075d) with only the test changed. Mutations: only the first half of a group checked; the second half of a group loaded from the wrong
  place; the second half stored wrong; the widened addresses taken from the high halves. Each fails `shared_memory_of_a_whole_warp_...`.
* ASan+UBSan (-O0, detect_leaks=1, halt_on_error=1) of test_sass_dispatch, test_memcount_forms and test_bank_memo: clean. TSan not run: no threaded or atomic code was touched.

### Not done, and negative results
* An AVX2 gather for the strided 4-byte loads: the per-lane loop is 2 loads and 1 store per lane (about 32 cycles a warp) and a Zen 4 `vpgatherdd` is about 9 cycles per eight lanes: no gain expected, not tried.
* `step_warp` / `fetch`: flat (largest instruction 6% of the function); nothing structural. `sass_bank_passes_memo` (12 to 22% of convolutionSeparable and sortingNetworks) is the next cost there.
* 8- and 16-byte whole-warp accesses still go lane by lane (`LDS.64/.128` are rare in these programs).
## Round sp12: the AVX-512 float-product tensor kernel (branch `sp12-avx512`, base `sp11-tensor-pr`)

### What was missing
`mma_avx2_chunked_f` (the f32-product kernel behind `mma.sync`, HMMA and `wmma` on the sm_86 sum for f16, bf16 and tf32) had no 512-bit form, so the AVX-512 default of sp8 did not touch it (sp11 measured the two TensorCoreGemm samples as equal at both levels on a Zen 4, where 512-bit operations run as two halves). Measured on a Zen 5 (Ryzen 9 9950X, full-width 512-bit), the kernel itself is the cost of those samples, so it got a 16-lane form.

### The kernel (`mma_avx512_chunked_f`, include/vgpu/exec/tc_ampere.hpp)
Same arithmetic, sixteen lanes, mask registers in place of compare vectors (`shift_term_avx512`, `sum_to_f32_avx512`, `sum_to_f16_avx512`, the cold `mma_f_fix512`). A 16-lane unit is one row and 16 columns when N is a multiple of 16 (A broadcast as one number), or two rows and 8 columns when N == 8 (lane 2j + r = row r of the pair, column j: the A side is a 64-bit broadcast of the pair (row 2p, row 2p+1) at the same k, built once per call with unpack/permute; the B side has each column doubled; the accumulators are interleaved on the way in and out). Other shapes (N = 24, odd M with N == 8) stay with the 256-bit kernel. The decode and the range checks that send a call to the integer kernel are copies of the 256-bit ones, so both kernels take and decline the same calls (the test checks it, and that a declined call writes nothing). Used only at `Level::Avx512` (`mma_avx2_acc`); `VGPU_SIMD=avx2` keeps the old path. Four units at a time (`VGPU_TC86_ROWS_Z`; 2, 6 and 8 measured slower or equal on a loop of calls: 0.79, 0.80, 0.69 against 0.67 on f16 16x8x16).

### Results (server2, Zen 5, cpu 9 alone, same binary pair A = `sp11-tensor-pr`, B = this branch, `VGPU_THREADS=1`)
Kernel alone, ns per call of `mma_avx512_chunked_f` over `mma_avx2_chunked_f` (inputs random normal, best of 5 x 20000 calls): f16 16x8x16 300 -> 206 (0.68), 16x8x8 171 -> 122 (0.72), bf16 0.66 / 0.72, tf32 16x8x8 0.61, 16x8x4 0.85, 16x16x16 557 -> 322 (0.58), 32x8x16 0.63, 8x32x16 0.61; values 0/1/2 as in the samples 0.55 to 0.72.
`bench.py` (9 runs, sass and ptx, counters identical): wmma 1.36x / 1.67x, mma_ptx 1.33x / 1.31x (p < 0.001). Canaries int_alu f32_fma mem shared atomics local divergent warp launch api: 0.97 to 1.02, counters identical in all rows; the ones below 0.98 (ptx warp 0.975 / 0.986 in two runs, api 0.975 to 0.985, sass launch 0.965 in one of two runs) do not touch the new code and moved with a noinline on the kernel; A/A of the same build gave 0.99 to 1.01. (This is the code-placement effect of earlier rounds; two to three per cent on api / launch.)
CUDA Samples on the simulator (wall seconds, one thread, outputs identical): cudaTensorCoreGemm 22.9 -> 18.6 (0.81), tf32TensorCoreGemm 231 -> 207 (0.90), bf16TensorCoreGemm 197 / 200 -> 153 / 158 (0.78; a first pair measured while the box was busy read 210 / 216 and is not used), immaTensorCoreGemm 10.8 -> 10.4 (IMMA does not use the kernel: the noise of the box).
`nvidia_tensor_bench` rows for these instructions (ns per warp instruction, SASS and PTX) are dominated by the interpreter for a few dozen calls, and the bf16 m16n8k16 PTX row declines the kernel in about half its calls (operands outside the range rule); it is not a good meter for this change.
Defaults: unchanged. The kernel is faster at AVX-512 on the only full-width host measured; the Zen 4 check below found no regression (the rows that use the kernel are faster there too); no Intel host was measured.
Zen 4 check (server3, Ryzen 7 PRO 8840U, 512-bit operations as two halves, cpu 2, box load 12 to 38 from other sessions; A = `sp11-tensor-pr`, B = this branch, `VGPU_THREADS=1`): `bench.py` 21 runs, B/A of the median Minst/s, sass / ptx: wmma 1.030 / 1.203, mma_ptx 1.028 / 1.029 (counters identical; A/A of the same build 0.998 to 1.002, p > 0.48 in all four rows). `nvidia_tensor_bench` (5 repeats of the whole table, 3 alternating A / B rounds, median cycles per warp instruction, B/A at `VGPU_SIMD=avx512`): mma f16 m16n8k8 0.83, wmma f16 m16n16k16 0.87, wmma tf32 0.86, SASS HMMA.1688.F32 0.88, wmma m32n8k16 / m8n32k16 0.95 / 0.93, SASS HMMA bf16 0.93; bf16 rows (a first pass read 1.07 to 1.12 on bf16 m16n8k16 and wmma bf16, a second pass of 7 rounds of 20 repeats on the bf16 rows read mma bf16 m16n8k8 0.85, m16n8k16 0.72, wmma bf16 0.76: the first was noise from the loaded box). Rows that do not use the kernel (wgmma, tcgen05, IMMA, f64, s8) sit at 0.97 to 1.04 with B2/A2 (both at `VGPU_SIMD=avx2`) at 0.93 to 1.09, so the box noise is about 5 per cent; wgmma / tcgen05 / QMMA e4m3 read 1.04 to 1.085 at both SIMD levels (code placement, not the kernel). B at avx512 over B at avx2: the same rows 0.79 to 0.95. No shape is slower than A on Zen 4 beyond noise, so no CPU-family guard is needed.

### Tests (CMake-wired)
`test_tc_ampere_fast` now also (a) calls `mma_avx512_chunked_f` and `mma_avx2_chunked_f` directly on every random and adversarial round of the existing test (94,737 calls, 24,159 taken, shapes 16x8, 16x16, 32x8, 8x32, 12x16, 6x8, 3x16, tails K = 4 and 12, f32 and f16 accumulators, chunks 8 and 4) and requires the same taken/declined answer, nothing written when declined, and the frozen form's bits; (b) repeats the edge-exponent, float-subnormal and tf32-tail checks on the 512-bit kernel; (c) checks `sum_to_f32_avx512` and `sum_to_f16_avx512` against the scalar conversions over the sums of the 256-bit checks (4.5 M and 2.5 M sums, 16 lanes). It runs three times (default, `VGPU_SIMD=avx512`, `VGPU_SIMD=avx2`; skipped parts on a CPU without AVX-512). Mutation checks (header copy mutated, test rebuilt; all killed): fix-up lane row / column, B column doubling, both interleave permutes, pair order (unpack, permute2x128, tail), unsigned compare in the f16 rounding, the tie condition, the f32 range bound, shift sign, accumulator sign, `live` threshold, `badg` removed, fix-up cases removed, `redo` removed, round-up removed, C load / D store offsets, row remainders, fix-up row. One survivor is equivalent: the `badg` bound 254 -> 253 only sends more lanes to the exact scalar rule, which gives the same bits.
Also run: test_tc_ampere, test_mma_fast (both levels), test_tile_mma, test_simd, test_sass_mma_tiles, test_tier, test_fastpath. Not run: ASan / TSan (no box with the memory; no threaded code was touched), an Intel check of the new kernel.
## Round sp12: AMD matrix instructions after #439 (branch `sp12-amd`, base `sp11-tensor-pr`)

The question: did the AMD engine slow down after #439, and is it real or layout. Builds: A0 = a6e80555 (before #439), C = 873dbc69 (#439), B = `sp11-tensor-pr`
(2c68075d), E = B plus this change. `tools/bench-amd/bench.py`, one host thread, device-phase wave-instructions per CPU second, counters identical in every row.

### What was measured

* **Zen 4 (server3, cpu 4, 11 alternating runs, A0 against B)**: no regression. Every canary within 0.987 to 1.057 (`mfma_i8` 1.057, `mfma_f64_4b` 1.026 faster; `wave`
  0.987, `lds` 0.993 the lowest). `perf stat` user cycles of the whole process, best of 5: `mfma` A0 932 M, B 947 M, E 921 M; `mfma_bf16` 854, 863, 846 M.
* **Zen 5 (server2, Ryzen 9 9950X, cpu 24, 15 alternating runs, A0 against B)**: reproduces, so it is not a Zen 4 front-end effect. `mfma_bf16` 0.889 to 0.912, `mfma`
  0.961 to 0.975, `smfmac` 0.977 to 0.987 (p < 0.01), `flat` 0.97; `valu_*`, `lds`, `divergent`, `mfma_f64`, `mfma_f64_4b`, `mfma_i8` within 3 %. The step A0 to C
  (#439 alone) is about 3 % of that on `mfma_bf16`; C to B is the rest.
* It is placement, not work: instructions retired are equal (B has 0.3 % fewer), cycles are higher, and every function of the matrix path got slower with identical
  code (the tile kernel `mfma_tile_f_avx2` +7 %, the bf16 decode +40 % of its small share). Stack position (environment padding 0 to 4096 in 128-byte steps) and heap position
  (an `LD_PRELOAD` allocation of 0 to 4096 bytes before start) were swept on Zen 5: B is 5 to 10 % behind A0 in every position, so the cause is the code, not the aliasing of the
  tiles. (The heap sweep also found a cliff: with 2112 to 3136 bytes allocated first, `mfma_bf16` takes 1.8x as long for A0 and B alike, 0.25 s against 0.14 s. The tiles' stagger
  (`TileBuf`) does not cover whatever aliases there; not investigated.)
* The hot inner loops sit at different offsets in a 64-byte line in A0 and in B (the tile kernel's 116-byte loop at +0x20 and +0x30), because everything before them in the
  library moved.

### What changed

`-falign-loops=32` for `amd/src/gcn_exec.cpp`, and `optimize("align-loops=64")` on `mfma_tile_f_avx2` and `matrix_decode_wide_avx2`. The loops of the matrix path now start at the same offset
in a line whatever is linked ahead of them.

| Zen 5, 15 runs, mi300x | E / A0 | E / B |
| --- | ---: | ---: |
| `mfma` | 0.996 (p 0.53) | 1.026 |
| `mfma_bf16` | 0.988 (p 0.18) | 1.061 |
| `mfma_f64`, `smfmac`, `valu_*`, `lds`, `divergent`, `launch`, `api` | 0.985 to 1.023, all within the A/A spread | |
| `mfma_f64_4b` | 0.971 (p 0.02) | 0.959 |
| `mfma_i8` | 0.965 (p < 0.001) | 0.938 |

Zen 4 A0 against E (11 runs, load 5 from other sessions): every canary within 0.947 to 1.056, no row slower at p < 0.05 (`smem` 1.051 and `lds` 0.986 are the only two below 0.05); best-of-5 cycles lower than A0 and B on all four matrix rows.

### Not done, and negative results

* `mfma_i8` (-4 to -6 % against A0 and B on Zen 5, +5 % on Zen 4) and `mfma_f64_4b` (-3 %) stay behind on Zen 5. Aligning every loop to 64 bytes (`-falign-loops=64`) was no better than 32;
  moving the `std::vector<int>::resize` calls of the integer path behind a size test (`fit_size`) took the bf16 and `mfma` gains away again (0.937, 0.961): the layout changes with
  any edit, which is why this is a compile flag and not another function. A deterministic fix is to put the tile kernels in their own `-fno-lto` unit, as `gcn_tier_ds.cpp` is; not done.
* The effect is a property of one host (Zen 5) and of the library's layout at one commit: re-run `mfma`, `mfma_bf16`, `mfma_i8` on a Zen 5 host after any large AMD change.
* No test can fail for an alignment change (the arithmetic is unchanged): the guards `test_amd_mfma_math`, `test_amd_matrix_fast`, `test_amd_nan_order`, `test_amd_gcn_same_bits`,
  `test_amd_gcn_packed_same_bits`, `test_amd_gcn_memory_same_bits` and `test_amd_codeobject_same_bits` pass.
## Round sp12: small launches stay off the worker pool (branch `sp12-pool`, base 2c68075d = `sp11-tensor-pr`)
The scaling study (multi-core, never measured before this round) found a launch loop that gets slower with every thread added. `launch64` (a 64-block, one-warp kernel launched 300 k times) took 4.48 s at 1 thread, and 18 s (0.24x) at 2 threads and 20 s (0.22x) at 16. Cause, in `src/exec/worker_pool.cpp`: a worker polls for `kSpin` (40 us) only while its jobs keep coming less than `kSpin` after it finished the one before ("hot"). A 64-block launch is about 15 us of work; on 8 threads the launch period (post, wake, run, wait) is about 53 us, so the workers fall out of the hot state and sleep, and every later launch pays a futex wake per worker (about 13 voluntary context switches per launch). The state keeps itself: the workers are cold because the launches are slow, and the launches are slow because the workers are cold.
### The fix, in two parts
* **A launch is kept to the threads its work can pay for** (`LaunchWidth`, `cap_threads`, `blend_unit_cost` in `vgpu/exec/worker_pool.hpp`). Each kernel (the address of its `CubinKernel` / `EntryFn`) remembers what one block cost, in nanoseconds, from its last launches: the wall time of the block-running section times the threads it ran on, divided by the blocks, replacing the old value when the launch ran on one thread (exact) and averaged into it otherwise (an over-estimate, which errs towards more threads). A launch of `n` blocks then gets at most `n * cost / 100 us` threads (at least 1, never more than asked: `VGPU_POOL_MIN_WORK_US`, 100 by default, 0 turns the cap off). The first launch of a kernel has no cost and runs as before; a request of one thread, or of one block, is not timed at all and takes the path it always took. A launch capped to one thread runs the engine's own one-thread path on the calling thread (no pool, no hand-off). The per-block cost is kept per kernel, so a grid of another size is judged by what its blocks cost, and a kernel whose blocks get dearer is found out by its next launch (at one thread the figure is exact). The 256-slot table is racy by design: a collision or a torn pair gives one wrong estimate, which moves time only. Which thread runs a block changes nothing it computes, so results, outputs and counters are the same at every thread count (the SM number a block reads, `SR_VIRTUALSMID`, is the thread's index, as it already was at any thread count).
* **No lock per block** (SASS engine, `Runner::run`, the multi-thread loop). The scaling run on 16384 one-warp blocks per launch also showed no speed-up at 8 threads (1.61 s against 1.44 s on one): each block took `fail_mu` to look for a failure and `stats_mu_` to add its counters. Each thread now sums its blocks' counters (integers: their order changes nothing) and adds them to the launch's once when it leaves, and a failure is announced by an atomic flag read without the lock. The one-thread loop is untouched (it needs no lock and its code is exactly what it was). Not done: running body 0 on the calling thread. The pool's contract says the caller only waits (the thread that holds the API lock and owns the CUDA call's `thread_local` state is never a worker), and with the cap the small launches no longer go through the pool, so it would buy one hand-off in the middle-sized launches only.
### Results (server2, Ryzen 9 9950X, A = `sp11-tensor-pr`, B = this branch, `scale.cu`/`sz.cu` probes in the sp12-scaling logs, median wall of 5 to 7 alternating runs, pinned to free cpus of 0-15, `VGPU_THREADS` = t; outputs identical in every row)
| launch64, 90 k launches of 64 blocks | A | B | B/A |
|---|---:|---:|---:|
| t = 1 | 1.33 s | 1.32 s | 1.01x |
| t = 2 | 5.38 s (10.4 s cpu) | 1.35 s (1.35 s cpu) | 3.99x |
| t = 4 | 5.34 s | 1.34 s | 3.99x |
| t = 8 | 4.53 s | 1.39 s | 3.26x |
| t = 16 | 6.06 s (26.0 s cpu) | 1.36 s (1.37 s cpu) | 4.46x |
`launch` (one block per launch, 600 k launches) was on one thread already and is 1.00x to 1.02x at every t. Launches of other sizes (one-warp blocks, 8 threads, wall including 0.05 s of process start; at 1 thread the two builds agree within 0.93x to 1.00x): 16 blocks 0.90 s -> 0.09 s (10x), 64 blocks 0.50 -> 0.15 (3.3x), 256 blocks 0.26 -> 0.24 (1.08x), 1024 blocks 0.27 -> 0.19 (1.4x), 4096 blocks 0.34 -> 0.10 (3.4x), 16384 blocks 1.61 -> 0.33 (4.9x, the lock per block).
Large grids, a regression check (compute, mem, divergent, atomics, tensor of the scaling probe, 5 to 7 runs, t = 1, 4 and 16; A/A of the two build trees, 7 runs: 0.96x to 1.09x, so the noise is about 4 percent): compute 1.01 / 1.00 / 1.03, mem 1.00 / 1.00 / 1.00, divergent 1.01 / 1.02 / 1.02, atomics 0.98 / 0.99 / 1.00, tensor 0.98 / 0.96 / 0.95 (t = 1 / 4 / 16 in each row; the tensor row is the wmma kernel and follows the placement loss below).
The 12-workload canary (`bench.py --ab`, `--threads 1 --cpus 7`, 9 runs, sass + ptx, counters identical in all 24 rows): int_alu, f32_fma, shared, divergent, launch, api, mem sass, atomics sass, local sass, warp ptx, mma_ptx ptx 0.985x to 1.015x; **SASS wmma 0.948x, SASS mma_ptx 0.962x, SASS warp 0.941x, PTX mem 0.968x, PTX atomics 0.962x, PTX local 0.954x** (p < 0.001). These six rows are the placement effect of earlier rounds and not this change: the one-thread path executes the same source as before (the SASS one-thread loop is the unchanged lambda, `LaunchWidth` is only constructed with more than one thread, and the instruction counts are identical), and the same rows swing in both directions between builds that differ only in where the new code sits. Over six builds of this tree the SASS wmma row read 0.93x, 1.11x, 0.995x, 0.945x, 0.96x and 0.948x, SASS mma_ptx 0.944x, 1.106x, 0.991x, 0.945x, 0.955x and 0.962x, PTX local 0.926x, 1.003x, 0.998x, 0.939x, 0.967x and 0.954x. Moving the new members out of line (`noinline`), `cold`, and constructing `LaunchWidth` only for more than one thread removed the one real one-thread cost found (SASS `launch` 0.965x to 0.978x while the call was made at one thread; 1.00x since) and left the lottery. Nothing structural remains to change in these rows without a layout tool (the A side is one lucky layout of those six rows).
### Tests (CMake-wired) and mutation checks
* `test_worker_pool`: `the_cap_keeps_small_launches_on_few_threads` (the policy table: unknown cost, request of one, cap off, a 15 us launch, the one, two and four thread boundaries, saturation) and `the_unit_cost_is_exact_on_one_thread_and_averaged_otherwise`.
* `test_sched`: `small_launches_in_a_loop_stay_off_the_pool_and_give_the_same_results` (a small kernel launched 200 times at each of 2, 8 and 16 threads: bytes and counters equal to the one-thread launch, and `worker_pool_stats().calls` does not move once the kernel's cost is known; with the cap off, or a 1 us threshold, the same loop does call the pool, so the check is not vacuous). `WorkerPoolStats` has a new `calls` field for it.
* `e2e_pool_cap` (`nvidia/tests/e2e/pool_cap.cu`, `run_pool_cap.sh`): 1500 launches of a 64-block kernel, other grid sizes of the same kernels, atomics, a heavy 256-block grid and small ones again, with the performance counters on, at VGPU_THREADS 1, 2, 8 and 16, the cap on and off (`VGPU_POOL_MIN_WORK_US=0`), SASS and PTX: every run passes and prints the same bytes and the same per-launch counters.
* Mutants (server2, tests run): `cap_threads` returning the request (killed: the policy test and the stay-off-the-pool test fail); the per-thread counters never added to the launch's (killed: `e2e_pool_cap` fails at every thread count above one, 6 of 8 SASS runs). The unit-cost blend ignoring the thread count is killed by its table (`6625` for 8 threads), by reading, not run.
* Also run and passing: `test_exec`, `test_exec3`, `test_launch_env`, `test_ptx_block_reuse`, `test_sass_block_reuse`, `test_config`, `test_ptx_paths`, `test_sass_ldgsts`, `test_sass_s2r_fill`, `e2e_lockstep_blocks`, and `e2e_sass_tier` on sm_86 with `block_semaphore`, `cooperative_grid`, `smid`, `dsmem_cluster`, `cooperative_cluster`, `vector_add`. Sanitizers were not run (server2 has 10 GB free): the new shared state is relaxed atomics, a mutex-guarded merge and a lock-free table; the CI TSan leg covers `test_worker_pool` and `test_sched`.
### Not done
* Running body 0 on the caller (see above) and a per-launch thread count that follows the grid's size as well as the cost: the cap already gives the right order of magnitude, and a threshold of 100 us of work per thread was chosen, not tuned (50 us would give 2 threads to a 100 us launch).
* The PTX engine's multi-thread loop keeps its own per-thread statistics and a chunked counter, and has no lock per block.

## Round sp13: why the AVX-512 float-product kernel gained little on Zen 4, and what was not the multiply (branch `sp13-zen4-wmma`, base `sp12-combined`)

The question: after sp12 the `wmma` / `mma_ptx` rows gained 1.36-1.7x on a Zen 5 but only 1.03-1.20x on a Zen 4 (Ryzen 7 PRO 8840U). Is the 512-bit kernel selected there, and where does the time go besides the product?

### Findings (server3, Zen 4, `bench.sh profile` and `perf`, before any change)
* **It is selected.** `kAvx512Default` is true on a CPU with AVX-512 F/BW/DQ/VL and VNNI, which Zen 4 has; the profile of `wmma` and `mma_ptx` is 61-70% in `mma_avx512_chunked_f<8, false>`. Run as two 256-bit halves, the kernel is 0.77-0.92 of the 256-bit one's time on Zen 4 as sp12 left it (Zen 5: 0.67), so about 1.1x is all the width can give there.
* **The product is not all of it.** For `wmma` (SASS) the kernel was 69.5%, `hmma_k` 12.5%, libc 12% (all `__memcpy_chk`), `step_warp` 3.5%. Two host-side costs sat outside the multiply:
  1. **`__memcpy_chk`.** glibc's fortification (Ubuntu's default) turns a `std::memcpy` to a computed address of a local array into a call to `__memcpy_chk` even at a fixed 16 or 32 bytes (the compiler cannot prove the object size). `hmma_k` made nine such calls per warp instruction (fragment gather of A, the C interleave), about 25 cycles each. The copies in `src/simd/kernels.inc` are now `__builtin_memcpy` (same bytes; the file's comment says why).
  2. **The kernel's own operand pass.** `perf` on a loop of calls (16x8x16 f16): 37% of the kernel's time was the code before the first row (decode of A and B with 8-lane vectors, the store-and-scalar `max_element` reductions, the row-pair interleave from a separate buffer, the doubling of B's columns), at 5,950 instructions per call (4,510 after the first form of the change below, before the pair fusion). It is now 16 lanes at a time (`decode_v` / `decode16`: a dense A as 16-float runs across rows, a dense B (N = 8) as two rows a run, anything else a row or a 16-column block, lanes past the operand read as zero through a masked load and are counted by no check), the largest and smallest exponent fields reduce with `_mm512_reduce_*`, an infinity or NaN shows as a largest field of 255 (one test in place of two compares and a mask), and for N == 8 with a dense A of 8 or 16 columns the two rows of a pair are interleaved as they are read and the pairs are what is decoded and stored (`a_paired`: no row-wise copy and no second pass). A full-vector store when all 16 lanes are live (a masked store does not forward to the next pass's load). The calls the kernel takes and declines are the same as before, and the arrays it fills are the same bytes (the test checks the answers against the 256-bit kernel and the frozen form).
* **What is left on Zen 4:** the rows (chunk loops, `top`, the accumulator frame, `sum_to_f32_avx512`) are about three quarters of the kernel (staged returns on Zen 5 before the pair fusion, 16x8x16: A decode 18 ns, B decode 9, checks 2, pair and doubling 15, rows 123 of 168). 512-bit operations are two half-width ones on Zen 4, so the operation count is the cost, and there is no host-side copy left to remove. `wmma` on the PTX engine (N = 16: the non-pair path, whose B decode costs 16 vectors) gains only 2% on Zen 4 and nothing on Zen 5; the kernel is 65-68% of its time and the rest is spread over `fast_wmma_mma`, `gather_words`, `f16x16_to_float`, each under 3%.

### Results (`bench.sh --ab`, 9 runs alternating, counters identical in every row; B/A of the median, A = `sp12-combined`)
| row | Zen 4 (server3, cpu 6) | Zen 5 (server2, cpu 12) |
|---|---|---|
| wmma sass | 1.286 | 1.471 |
| wmma ptx | 1.024 | 0.987 (p 0.019; A/A 0.995 to 1.002) |
| mma_ptx sass | 1.286 | 1.446 |
| mma_ptx ptx | 1.219 | 1.217 |

A/A (same build twice) on the 24 rows: 0.993 to 1.013 on Zen 4, 0.994 to 1.005 on Zen 5 (one median 1.028 on Zen 5 `mma_ptx` sass with best-of 0.999). The other ten canary workloads (int_alu f32_fma mem shared atomics local divergent warp launch api, both engines) read 0.931 to 1.046 on Zen 4 and 0.978 to 1.019 on Zen 5. The ones under 0.985 are placement: Zen 4 `mem` sass 0.971 (best-of 0.984), `mem` ptx 0.931 (p 0.16; best-of 0.975), `atomics` sass 0.983, `local` sass 0.984, `launch` 0.98 / 0.981, `int_alu` 0.988 / 0.986, `f32_fma` sass 0.983; Zen 5 `atomics` sass 0.978, `mem` sass 0.987. `divergent` sass reads 1.046 on Zen 4. None of them runs the changed code (`kernels.inc` only changes `hmma_k` and the other copies' call form).

Kernel alone, min ns per call over 8 alternating runs, `mma_avx512_chunked_f` before -> after (the 256-bit kernel is unchanged: Zen 4 410 / 250 / 771 ns on 16x8x16 / 16x8x8 / 16x16x16):
* Zen 5: f16 16x8x16 202.6 -> 154.4, 16x8x8 118.8 -> 86.4, 32x8x16 369.7 -> 283.7, 16x16x16 313.3 -> 295.0, 8x32x16 316.7 -> 302.2, bf16 16x8x16 203.2 -> 154.2, tf32 16x8x8 147.2 -> 114.8, 16x8x4 151.1 -> 85.3.
* Zen 4: f16 16x8x16 369.3 -> 337.1, 16x8x8 222.6 -> 183.1, 32x8x16 722.6 -> 618.0, 16x16x16 654.0 -> 644.6, 8x32x16 641.6 -> 641.6, bf16 16x8x16 387.4 -> 334.1, tf32 16x8x8 280.4 -> 240.9, 16x8x4 235.6 -> 159.7. The 256-bit kernel takes 1.2-1.4x the 512-bit one's time on the pair shapes now.
* 512-bit over 256-bit on Zen 4, whole rows (B at `VGPU_SIMD=avx512` over `avx2`, same build): wmma sass 1.19, ptx 1.26; mma_ptx sass 1.19, ptx 1.16.

### Tests
`test_tc_ampere_fast` (CMake-wired, runs at the default, `VGPU_SIMD=avx512` and `avx2`) gained:
* the 512-bit kernel repeated on every call with A and B at the end of a mapped page (the next page unmapped), dense and with padded rows (strides past the widths, infinities between the rows): the answer is the dense call's, nothing is read past the last float or from the padding (`k512_edge` is held to twice the call count);
* `float_product_kernel_declines_on_infinities_and_nans_beside_small_numbers`: six special values placed at random in A or B with every other element small (exponent fields that pass the range rule), 6000 rounds over the shape list; both kernels decline and write nothing.
Mutants (run against the rebuilt test; killed unless listed): the live mask, the bias of each of the seven decode call sites, `lo` / `hi` / `in8` / `dup0` / `dup1` permutes, the term shift and `emin` offset, `subn` dropped, the largest-field-255 test of A and of B dropped (the second test above killed these two; the first test did not), the masked load turned into a plain one (page-edge test), the dense-A and dense-B conditions, the B-doubling store offset, the fallback pair interleave, the largest-field tracking, the pair count. Survivors, all equivalent: `dead` bound 254 -> 255 (an `e == 255` lane is then live in the arrays, but the call is declined for it), `field_min` bound 254 -> 255 (the same 255), the smallest-field accumulator's start value (all-zero operands read 255 either way), and the full-width store for a partial run (it writes zero lanes inside the same array).
Also run and passing at default, avx2 and avx512: `test_simd`, `test_sass_mma_tiles`, `test_tier`, `test_fastpath`, `test_tc_ampere`, `test_wmma_generic_fast`, `test_wgmma_fast`, `test_tcgen05_tiles`, `test_simd_dispatch`, `test_mma_tiles`, `test_mma_fast`, `test_tile_mma`. Not run: ASan / UBSan / TSan (no threaded code was touched), an Intel CPU.

### Not done
* The rows of the kernel (about three quarters of it): fewer vector operations per chunk would help both CPUs, but every one of them is part of the bit-exact chunk rule. The N = 16 B decode (16 vectors) could be fused with the load as A's now is; it is 6% of the 16x16x16 call.
* The `__memcpy_chk` pattern is not limited to `kernels.inc`: 111 call sites remain in the shim (`h_imma` has six, `h_global` seven, the PTX `exec_*` functions more). Their cost per instruction was not measured.
## Round sp13: shared-memory bank counting keyed by 32-bit offsets (branch `sp13-bank`, base `sp12-combined` b6545db2)

Where the time went (perf `cycles:u`, one host thread, server3 Zen 4, base b6545db2): `sass_bank_passes_memo` 12% of Samples convolutionSeparable and 21% of sortingNetworks, with
`count_shared_access` (thread_local table, 2.7 to 3.2%), `bank_passes_progression` (1.5 to 2%) and, for sortingNetworks, `sass_bank_passes_table` (5%) after it: 16% and 31% of those programs
for the counters of one kind of instruction. The memo itself is flat (no instruction over 8%): it builds a 64-bit key from 32 widened addresses (the tier handler had just widened them from its
32-bit offsets, 16 stores), subtracts the base, hashes eight of them and compares 256 bytes.

### What changed
* `h_shared` no longer widens its offsets. `count_shared_offsets` -> `memcount::sass_bank_passes_offsets` (include/vgpu/exec/mem_counters.hpp) keys a warp by its 32-bit offsets less the first
  active lane's word (a lane that does not run is taken at that word, as in the 64-bit key): 128 bytes to build and compare, in SSE2 vectors.
* Every shape goes through the table, the ones on a stride too (a hit is cheaper than finding the stride), and the entry that answered last is compared first, without the hash (a loop asks for
  one shape two or three times running: load, store, load of the same indices). A miss is counted as before (`sass_bank_passes` on the widened offsets: closed forms, then the table of words) and stored.
* Validity of the 32-bit key: no active offset may have bit 31 set (else the call is counted, never remembered). Then every true difference between two lanes is under 2^31, so two warps with
  equal keys differ by a move of whole words, and the passes (distinct words per bank) cannot differ. Without the test two lanes 4 GB apart in truth and one word apart modulo 2^32 are equal keys
  (test below).
* Same window and back-off as the 64-bit table (under a third hits in 64 lookups: 1024 calls uncounted by the table). `BankMemo` grows by a second table of 32 x 136 bytes, the same thread_local object.
* Tried and dropped: counting runs of lanes (`mask = 2^n - 1`, which a sorting network asks for by the thousand, 1.1 M distinct masks in sortingNetworks) the old way, before the table: 2% to 3%
  slower than putting everything through the table (instructions 290.4 G against 282.9 G, cycles 71.9 G against 68.8 G). A table of 128 entries: hit rate 86.4% -> 87.4% only (the misses are
  masks that never repeat, not capacity), dropped. `sass_bank_passes_widened` inline: worse (mem 0.954x on both CPUs).

### Results (A = b6545db2, B = branch; outputs and bank-conflict counters identical per kernel in the programs below, checked with VGPU_COUNTERS=1)
Zen 4 (server3, Ryzen 7 PRO 8840U, cpu 3, `perf stat` user cycles / instructions of the whole program, box load 1 to 2, 3 to 4 runs; A/A of the same runs 0.997 to 1.003):
| program | A cycles | B cycles | B/A | A instructions | B instructions |
|---|---:|---:|---:|---:|---:|
| convolutionSeparable | 64.8 / 64.9 / 64.9 G | 59.4 / 59.6 / 59.9 G | 0.918 | 295.0 G | 268.2 G |
| sortingNetworks | 84.6 / 84.5 / 84.6 G | 69.0 / 69.9 / 69.1 G | 0.817 | 311.95 G | 284.7 G |
| transpose | 44.2 / 44.1 / 44.6 / 44.2 G | 44.5 / 44.2 / 44.2 / 44.3 G | 1.00 | 184.5 G | 182.5 G |

Zen 5 (server2, Ryzen 9 9950X, cpu 30, wall s, 5 alternating runs, box load 7 to 9 from other sessions):
| program | A s | B s | A/B |
|---|---:|---:|---:|
| sortingNetworks | 15.16 to 15.27 | 11.19 to 11.47 | 1.35x |
| convolutionSeparable | 10.33 to 10.43 | 9.16 to 9.75 | 1.13x |
| transpose | 6.93 to 7.04 | 7.14 to 7.26 | 0.97x |
(transpose: its shared accesses are contiguous or one stride, which the base counted in closed form; on Zen 4 it is level at 1.00x cycles with 1.1% fewer instructions, so the 0.97x on Zen 5 is a
cost of the table path on that core or placement; not separated.) `real.py` on server3 (5 runs, cpu 3, load 1 to 20, wall): sortingNetworks 1.23x, convolutionSeparable 1.10x, dct8x8 1.05x,
hotspot 1.08x, lud 0.99x, mergeSort 1.00x, transpose 1.01x (outputs identical); bh timed out at 150 s on both sides.

bench.py canaries (12 workloads, SASS and PTX, 15 runs, counters identical in all 24 rows on both CPUs). Final tree: A/A on Zen 4 all within 0.985x to 1.007x. A/B on Zen 4: `divergent` SASS
1.067x, `wmma` SASS 1.017x, `mma_ptx` 1.011x/1.016x (placement gains, not goals), `mem` SASS 0.976x (p<0.001; 0.985x and 0.981x in two other windows) and `warp` PTX 0.967x (p<0.001) are the
rows below 0.98x. A/A on Zen 5 within 0.96x to 1.02x (0.98x to 1.01x apart from `launch` PTX 0.962x, p 0.77); A/B on Zen 5: `wmma` PTX 0.945x (0.935x to 0.941x in two other runs, p<0.001),
`warp` PTX 0.977x/0.975x, `api` 0.980x to 0.987x, `mem` SASS 0.982x, `atomics` SASS 0.984x. None of these runs any code this round touches (the PTX interpreter, the API path), instructions are equal,
and they moved from variant to variant of the same change (see below): the placement effect of earlier rounds (sp9, sp12 shared), not a measured cause. The variants measured (same source, only
`noinline`): plain inline `sass_bank_passes_offsets`: Zen 4 `mem` PTX 0.978x, `divergent` PTX 0.974x, `launch` SASS 0.969x (all reproduced with 21 runs); `noinline` on it (kept): those three at 0.998x,
0.996x, 1.000x, the losses move to the rows above; also `noinline`-less `sass_bank_passes_widened`: `mem` 0.954x to 0.977x on both CPUs, dropped. Zen 5 `wmma` PTX read 0.92x to 0.94x in all three
variants, 1.00x on Zen 4 (it is the 16-lane tensor kernel's host, which sp12 measured as layout-sensitive on Zen 5), so it is reported, not fixed.

### Tests (CMake-wired, `test_bank_memo`) and the mutations that fail them
* `offsets_are_counted_as_the_definition_counts_them`: 600 000 random warps (7 kinds of shape, widths 1 to 64, full, partial and sparse masks, bases at every byte, a cluster rank above the offset,
  across bit 31 and past 2^32, idle lanes holding anything) against the definition (sorted vectors of the words of each bank), with one table throughout (hits, collisions, the last-entry compare, the back-off).
* `offsets_that_differ_in_one_lane_one_byte_or_the_mask...`, `offsets_on_a_stride_and_ones_that_never_repeat...`, `offsets_every_pair_of_lanes_and_every_lane_left_out...` (all 496 pairs and 32
  masks of all lanes but one, twice, idle lanes holding real offsets), `offsets_a_lane_that_joins_the_mask_changes_the_answer...` (lanes c and a+16, then a joins: 1 then 2 passes, for all c, a), `offsets_across_bit_31_are_not_taken_for_the_same_shape_in_low_memory` (offsets 4096/4094 then 0/0xfffffffe, 1 then 2 passes).
* Mutations (header mutated, test rebuilt; all killed): no blend of idle lanes, bit-31 test removed, width left out of the key, base not rounded to a word, last-entry hit without the
  key compare, hash-entry hit without the key compare, lane-select vector with lanes 0 and 1 swapped, mask ignored in the key. The first version of the suite let the bit-31 and the swapped-lane mutants
  survive; the last two tests above were written for them.
* Also run and passing (Release, server3): test_memcount_forms, test_sass_dispatch, test_tier, test_memfast, test_sass_access_count, test_sass_mem_rows.

### Not done
* The remaining bank cost in sortingNetworks is `sass_bank_passes_table` (7% of the program) on the masks that never repeat (merge steps); a closed form for two runs of consecutive words, or a per-shape
  word-group table counted for any mask, would take it. `thread_local` access (`__tls_get_addr`, 0.8%) not changed (initial-exec TLS in a shim library loaded by dlopen is a risk, a pointer in `Block` moves heap blocks).
* The PTX interpreter's shared-memory banks (`ptx_bank_passes`), the exec path (`count_shared_access`) and the other tier handlers keep the 64-bit table.
## Round sp13: the AMD matrix kernels in a unit of their own (branch `sp13-amd-lto`, base `sp12-combined`)

The sp12 fix for the AMD matrix path was a pair of compile flags (`-falign-loops=32` on `gcn_exec.cpp`, an `align-loops=64` attribute on two functions): the layout of a
few-instruction loop still depended on what the linker put ahead of it, and `mfma_i8` and `mfma_f64_4b` stayed behind. The question: take the matrix kernels out of the
library's link-time optimization and the layout is theirs alone.

### What changed

`amd/src/gcn_matrix_kernels.cpp` (new) holds `mfma_tile_f_avx2`, `mfma_tile_fma_hw`, `mfma_tile_fma_avx2`, `mfma_tile_int_avx2` (moved out of `mfma_math.hpp`, which
declares them) and `matrix_decode_wide_avx2<T, Run>` (moved out of `gcn_exec.cpp`, eight instances, `gcn_matrix_kernels.hpp`). The unit is built `-fno-lto -falign-functions=64
-falign-loops=32` (CMakeLists.txt), as `gcn_tier_ds.cpp` is. Two things the move needed: the decoder's AVX2 target has to come from a `#pragma GCC target` region (on a function
template's definition the target did not reach the explicit instantiations and the always-inline lambdas in them), and the K = 4 fused kernel got its own pair of loops (out of line,
with the K test inside the column loop, it ran 5 % slower than when it was inlined into a caller that knew K). Same arithmetic, same bits.

### Layout stability (server2, Zen 5, cpu 24, `tools/bench-amd/bench.py`, one host thread, 21 alternating runs a row, cpu seconds)

The test: add 1 (P1) or 4 (P4) unrelated `noinline` functions to `src/core/error.cpp`, which is linked ahead of everything, and compare each build with the same tree before the edit.
Medians of with-the-edit / without, noise (A/A, same code, two copies of the build) 0.97 to 1.01. (The branch columns were timed with the unit at `-falign-loops=64`, one step before the final 32; for the final build the check is the identical machine code below.)

| row | sp12 tree, +1 function | sp12 tree, +4 | this branch, +1 | this branch, +4 |
| --- | ---: | ---: | ---: | ---: |
| `mfma` | 0.982 | 0.953 | 0.999 | 0.997 |
| `mfma_bf16` | 0.981 | 0.977 | 1.008 | 1.010 |
| `mfma_i8` | 0.969 | 0.973 | 1.019 | 1.000 |
| `mfma_f64` | 0.985 | 1.001 | 0.998 | 1.003 |
| `mfma_f64_4b` | 1.021 | 1.005 | 0.981 | 0.983 |
| `smfmac` | 0.985 | 0.981 | 1.004 | 1.002 |

With the integer kernel still inline the same edit cost `mfma_i8` 7 % (0.929), which is why `mfma_tile_int_avx2` moved too. The machine code and the offset modulo 64 of all twelve
functions of the unit are identical with and without the edit (with the edit the library's text moved by some hundred bytes; `nm` + `objdump -s` of each symbol): the layout cannot move
any more, short of an edit of the unit itself.

### Speed against sp12-combined (same bench, median of three 21-run A/Bs, same host, load 7 to 12 from other sessions; A/A over 21 runs: 0.97 to 1.01)

| row | B / A |
| --- | ---: |
| `mfma_i8` | 1.103, 1.060, 1.060 |
| `mfma` | 0.988, 0.986, 0.957 |
| `mfma_bf16` | 0.977, 0.970, 0.959 |
| `smfmac` | 0.980, 0.981, 0.981 |
| `mfma_f64` | 1.008, 0.972, 0.979 |
| `mfma_f64_4b` | 1.062, 0.994, 0.997 |
| `valu_int`, `valu_f32`, `lds`, `divergent`, `wave` | 0.99 to 1.03 |
| `launch` | 1.054, 1.043, 1.042 |

So the integer rows gain 6 to 10 %, and `mfma`, `mfma_bf16`, `smfmac` are 1 to 4 % below the sp12 tree, whose layout was a good one for them: the same sp12 tree with one unrelated function
added is 2 to 5 % below itself on those rows (table above). Counters are identical (`--check-counters`).

### Zen 4 (server3, Ryzen 7 PRO 8840U, cpu 3, load 14 to 67 from other sessions, 15 alternating runs, two A/Bs and one A/A)

The box was too loaded for anything under about 5 % (the A/A over 15 runs spreads 0.91 to 1.11 on the noisiest rows, 0.99 to 1.04 on the matrix rows). B / A: `mfma_f64` 1.064, 1.059
(p 0.02, 0.005), `mfma_f64_4b` 1.137, 1.068 (p 0.001, 0.000), `smfmac` 1.041, 1.085, `mfma` 1.096 (p 0.48), 0.984, `mfma_bf16` 0.985, 0.971 (p 0.09, 0.34), `mfma_i8` 0.974, 1.017 (p 0.03, 0.5);
the other canary rows 0.95 to 1.04, none slower at p < 0.05.

### Side finding: the heap position cliff of `mfma_bf16` does not reproduce

An `LD_PRELOAD` library that allocates N bytes before `main` (N = 64 to 4096 in steps of 64), each point timed 5 times between two pad-0 references: the largest ratio of a point to its
neighbouring references is 1.04 on this branch and 1.02 on the sp12 tree, no cliff. What did move was the box: the same pad measured 0.149 s and 0.216 s a few minutes apart while other
sessions ran jobs on the sibling thread, which is what a contiguous "cliff" at one pad in an unpaired sweep looks like (the sp12 sweep was unpaired). If it comes back, pair every point with
a reference.

### Test

`test_amd_matrix_kernels_layout`: each of the four kernels and the eight decoder instances is a function of its own that starts a 64-byte line. Mutant: the unit compiled with
`-falign-functions=1` fails both tests (`mfma_tile_f_avx2` at 16 modulo 64, the decoder likewise). The arithmetic is held by `test_amd_mfma_math`, `test_amd_matrix_fast`,
`test_amd_nan_order`, `test_amd_gcn_same_bits`, `test_amd_gcn_packed_same_bits`, `test_amd_gcn_memory_same_bits`, `test_amd_codeobject_same_bits` (all pass).

### Not done

* `perf` does not run on server2, so the 1 to 4 % left on `mfma`/`mfma_bf16` is not attributed (the call into the unit and the lost constant propagation into the decoders are the suspects).
## Round sp13: device atomics without the stripe locks (branch `sp13-atomics`, base `sp12-combined`)
The scaling study (round sp12, `scale.cu`) found atomics going backwards: 0.77x at 16 threads, 1.04x at 4. Cause: every global atomic of the SASS engine (and the PTX engine, with its own 251-lock array) took a `std::mutex` out of a striped array (64 stripes keyed by cache line in `atomic_lock`, `src/sass/exec.cpp`); a warp's atomics took up to ~25 of them in turn, so every thread reached for the same few mutex lines.
### What changed
* **The CPU's own compare-and-swap on each word instead of the locks** (`sassatom::global_lockfree`, `include/vgpu/exec/sass_atomics.hpp`; `MemoryManager::atomic_word_ptrs`, which gives the bytes of a warp's 4/8-byte words and declines, having done nothing, wherever the locked form could differ: a fault armed, a host mapping in reach, an unaligned word, a word outside the one allocation). Each distinct word of the warp gets one swap loop whose step runs that word's lanes in lane order from the value the swap replaces (a retry runs them again from the new value), so a lane finds what the lanes before it left and the old values are the ones the locked form gave. An integer add is one `fetch_add` of the word's lanes' sum (lane l's old value is the word's plus what the lanes before it added): no retry loop, which is why a contended counter scales. An untouched or uniform chunk goes through `atomic_rmw` (the chunk is made real only where the locked form made it). Used by the SASS warp batch, the SASS lane-by-lane path (vector forms, the shapes the batch declines), TMA reductions, and in the PTX engine by the warp batch (`rmw_batch` with `lock_free`), the lane loop, the vector atomics and `cp.reduce`.
* **Only from four host threads** (`kLockFreeAtomicsFromThreads`, `vgpu/exec/host_atomic.hpp`). Measured on Zen 5, a lock pair on a cache line costs about 1.8 ns and a locked instruction per word about 4 ns, and a warp's 32 words lie on about 14 lines: with one or two threads the locks are 1.25x / 1.3x faster than the swaps (the first version of this change used the swaps at every thread count: `atomics` 3.5 -> 4.7 s at one thread). From four threads the stripe lines bounce and the swaps win. The choice is the process's, from the configured thread count (`VGPU_THREADS`, else the CPUs it may use), made once, not per launch, so two overlapping launches never meet on one word under different roads. At one or two threads the code is the stripe-lock code it was.
* **What is guaranteed.** Each word's update is atomic. Integer add, and, or, xor, min, max leave the serial value and give each lane an old value consistent with some order at any thread count; the old values of a fetch-add are a permutation (`e2e_atomics_threads`). Exchange, compare-and-swap, inc, dec and float add take the order in which the threads reach the word (a float sum varies with that order, as on the card, and as it did before: the locks never fixed an order across blocks). At one thread (and in every single-thread run) the results are bit-identical to before. A warp's atomics are no longer one step to another host thread (the locked batch held every stripe of the warp at once); CUDA promises no such step. Words the lock road kept: `atom.b128`, misaligned words and host-mapped words (the CPU's atomic). A 2-byte atomic (PTX scalar f16/b16, `cp.reduce`, TMA reductions) on the swap road swaps the aligned 32-bit word that holds it (`exec::atomic_rmw16`, `vgpu/exec/host_atomic.hpp`), so it is atomic against the 4-byte atomics on that word; it costs one 32-bit swap, the same as a 4-byte atomic, and 1 or 2 threads still use the locks (which key a 2-byte and a 4-byte atomic on the same word to one lock).
### Results (server2, Ryzen 9 9950X, `scale.cu` as in the sp12 scaling study, A = `sp12-combined`, B = this branch, median wall of 5 alternating runs, cpus 12-15 / 7,9-15 / 0-15, outputs identical (the sums are self-checked))
| SASS engine, wall seconds (`atomics`: 536 M atomics spread over 4096 words; `atomics1`: 50 M on one counter) | A | B | A/B |
|---|---:|---:|---:|
| `atomics` t = 1 | 3.50 | 3.56 | 0.98 (locks both: layout) |
| `atomics` t = 2 | 2.22 | 2.17 | 1.02 (locks both) |
| `atomics` t = 4 | 2.75 | 1.15 | 2.4x |
| `atomics` t = 8 | 3.32 | 1.75 (0.99 to 2.30: other jobs on the box) | 1.9x |
| `atomics` t = 16 | 3.79 | 0.73 | 5.2x |
| `atomics1` t = 1 | 3.27 | 3.41 | 0.96 (locks both) |
| `atomics1` t = 2 | 1.94 | 2.09 | 0.93 (locks both) |
| `atomics1` t = 4 | 2.00 | 0.91 | 2.2x |
| `atomics1` t = 8 | 2.22 | 1.00 | 2.2x |
| `atomics1` t = 16 | 4.59 | 0.60 | 7.6x |

Scaling against the same build at one thread: A is 0.92x at 16 threads in this run (0.77x in the study), B is 4.9x (`atomics`) and 5.7x (`atomics1`). The PTX engine (`VGPU_SASS=0`), same probes: `atomics` t = 1 2.89 -> 2.96, t = 4 3.05 -> 0.94 (3.2x), t = 8 4.50 -> 1.22 (3.7x, noisy), t = 16 3.03 -> 0.68 (4.5x); `atomics1` t = 1 4.31 -> 4.54, t = 4 33.1 -> 2.43 (13.6x: the PTX lane loop took a lock per lane on one word), t = 8 56.7 -> 3.24 (17x), t = 16 72.6 -> 3.0 to 20 (the box was shared). The sums are checked in every run (`out=` of `scale`): no run of B lost an update. Rows at t = 1 and 2 run the same code as A, so their differences (0.93x to 1.02x) are the noise of two builds (the A/A of the canary below is 0.99x to 1.01x per row, the layout of two builds is a few percent more).
Zen 4 (server3, Ryzen 7 PRO 8840U, 8 cores, shared box, load 6 to 28 during the t = 1 and 2 rows, about 6 for t = 4 and 8; medians of 3, sums checked): `atomics` t = 1 5.82 -> 5.66, t = 2 4.75 -> 4.73 (locks both), t = 4 6.64 -> 2.56 (2.6x), t = 8 6.97 -> 1.66 (4.2x); `atomics1` t = 1 5.46 -> 5.59, t = 2 4.52 -> 4.95 (the t = 2 row ran at load 8 to 10), t = 4 5.67 -> 2.51 (2.3x), t = 8 6.05 -> 1.58 (3.8x).
The 12-workload canary (`bench.py --ab`, `--threads 1`, 9 runs, sass + ptx, **counters identical in all 24 rows** on both hosts). Zen 5 (`--cpus 13`, load about 6; A/A of the base against itself 0.99x to 1.01x in every row, on a loaded box): int_alu, f32_fma, mem ptx, atomics, local, divergent, warp, launch, api 0.98x to 1.02x; **SASS wmma 0.952x, PTX wmma 0.966x, SASS mma_ptx 0.964x, PTX mma_ptx 0.968x, SASS mem 0.967x, SASS shared 0.975x, PTX shared 0.975x** (p < 0.01). Zen 4 (`--cpus 6`, A/A 0.94x to 1.01x, the one low row a load swing): every row 0.98x to 1.05x except **PTX mem 0.947x**; SASS wmma 1.022x and mma_ptx 1.013x. These are the placement losses of the earlier rounds moving again, not work: the one-thread path runs the same atomics code as before behind one branch on a process-wide flag, the instruction counts are identical, and the same rows swung between 0.92x and 1.11x across builds in round sp12. A first build of this change, with `global_lockfree` inlined into `exec_mem` once per step, read SASS wmma 0.935x, PTX wmma 0.925x, SASS mma_ptx 0.924x and SASS atomics 0.975x on Zen 5; making it `noinline` took the atomics row and half of the rest back. Nothing structural is left to change here without a layout tool (the `-fno-lto` unit of the `amd-lto` idea would apply to these as well).
### Tests (CMake-wired) and mutation checks
* `test_sass_lockfree` (new, unit): a warp's lock-free atomics against the lane-by-lane rule (`sassatom::apply`, held against the frozen executor lambda by `test_sass_atomics`) over every operation and type, compare-and-swap, partial masks, lanes sharing a word, words in a never-written, a uniform and a real chunk: old values, the words left, the changed flag; declines with nothing touched (misaligned, past the end, before the start, freed); from six threads, a fetch-add hands out each old value once (u32, u64, f32) and add / and / or / xor / min / max, spread and contended, leave the serial result.
* `two_byte_atomics_are_atomic_against_four_byte_ones_on_the_same_word` (in `test_sass_lockfree`): three threads count the low half and three the high half with `atomic_rmw16` while three add 0x10000 to the whole word; both counts exact. Mutant (a plain 16-bit load and store): killed. `e2e_atomics_threads` has the same mix as a kernel (a 16-bit `atomicCAS` beside `atomicAdd` on one word); the mutant fails it on the PTX engine at 4 and 8 threads (the SASS engine compiles the 16-bit CAS to 32-bit operations, so it passes either way).
* `e2e_atomics_threads` (new, `nvidia/tests/e2e/atomics_threads.cu`): contended and spread atomics, tickets (every one handed out once), a CAS-built sum and `atomicInc` wrapping, at `VGPU_THREADS` 1, 2, 4 and 8 on the SASS and PTX engines (so the lock road and the swap road).
* Mutants (server2, run): the swap loop's step reporting nothing changed (killed: `a_warps_lock_free_atomics_are_the_lane_by_lane_ones`); each lane given the value after its own step as its old value (killed: lane-by-lane test, and the fetch-add test); the result not cut to the type's width (killed: lane-by-lane); the fetch-add prefix dropped (killed: lane-by-lane and fetch-add tests) and its changed flag dropped (killed: lane-by-lane); plain load and store instead of the swap (`lock_free = false`: killed, 2 of 4 tests, lost updates); in the engines, the SASS fetch-add replaced by load, add, store, and the PTX batch run with `lock_free = false` plus the PTX lane path off (killed by `e2e_atomics_threads` at 4 and 8 threads on both engines, passing at 1 and 2, where the lock road runs, as it must).
* Also run and passing on server2: `test_sass_atomics`, `test_memrmw_lockfree`.
### Sanitizers
Final tree (server3, Zen 4): TSan (`-O1`, `setarch -R`) clean on `test_sass_atomics`, `test_hostmap_cache`, `test_memrmw_lockfree`, `test_sass_lockfree`, `e2e_host_atomics` and `e2e_atomics_threads` (SASS and PTX, 1, 2, 4, 8 threads); ASan+UBSan `-O0` clean on `test_sass_lockfree` and `test_sass_atomics`. One-thread canary of the final tree on Zen 5 (load about 8): SASS wmma 0.939x, SASS mma_ptx 0.940x, PTX wmma 0.978x, PTX mma_ptx 0.965x, SASS mem 0.972x, SASS atomics 1.038x, every other row 0.98x to 1.02x, counters identical.
### Not done
* A single `fetch_or`/`fetch_and`/`fetch_xor` for the bitwise steps (they use the swap loop; a contended counter was the case that needed `fetch_add`) and a closed form for min / max.
* `atom.b128`, misaligned words and host-mapped words keep the lock (or the CPU's atomic, for host-mapped).
* The AMD engine already used `atomic_rmw` (its own lock-free flag); not touched.
## Round sp12: f32 <-> f64 conversions as SIMD kernels (branch `sp12-f2f`, base 4cc7247d)
`F2F.F64.F32`, `F2F.F32.F64`, `I2F.F64.U32/.S32`, `FCHK` and `MUFU.RCP64H` are 0.76% of the SASS instructions but sit in hotspot, JACOBI1D, backprop, atomicCost, jacobiCudaGraphs and simpleCudaGraphs; the per-lane `cv_f2f` of `fast_float` was 12% of hotspot's samples and 8% of JACOBI1D's. An earlier tier form that only routed them through `tier_float_op` cut instructions 1-4% and made cycles worse, because the per-lane conversion stayed.
* Three kernels in the `simd::Kernels` table (`f2f_widen`, `f2f_narrow`, `i2d`; `src/simd/kernels.inc`, one baseline and one AVX2 form, the AVX-512 table inherits the AVX2 ones). f32 to f64 and u32/s32 to f64 are exact (the unsigned form adds two non-negative parts, so a zero is +0 in every rounding mode); f64 to f32 runs in the host's rounding mode, which the callers set as `dop_m` does (`RoundingScope`). `|x|` then `-x` are applied to everything but a NaN, which keeps its sign and payload as the register holds it (the conversion shifts the payload and quiets it), as `cv_f2f` does. The baseline forms keep the conversion of a NaN lane (an `asm volatile` empty statement), so the sticky exception flags are the scalar code's too (the test holds them equal).
* `Runner::fast_float` calls them for F2F f32 to f64, f64 to f32 and I2F 32-bit to f64 (every other size keeps `cv_f2f` / `i2f_wide_loop`); the tier has the form `Cvt64` (kernel: the same three calls, a pair of rows on the f64 side, R254's partner RZ, partial warps blended) and `FltInterp` (the rest of F2F and I2F, and MUFU.RCP64H/RSQ64H: `tier_float_op`, so the dispatch of the interpreter is skipped). FCHK already had its form. JIT: both forms are left to the handlers.
### Results (server3, cpu 6, one thread, A = 4cc7247d, B = branch, `perf stat` user cycles and instructions, 15 alternating runs, Mann-Whitney p; outputs identical, rc 0)
| program | cycles B/A | p | instructions B/A | A/A cycles |
|---|---:|---:|---:|---:|
| rodinia hotspot | 0.802 | <0.001 | 0.854 | 1.000 (p 0.46) |
| polybench JACOBI1D | 0.860 | <0.001 | 0.889 | 1.000 |
| rodinia backprop | 0.880 | <0.001 | 0.906 | 1.002 |
| samples jacobiCudaGraphs | 0.902 | <0.001 | 0.936 | 0.999 |
| samples simpleCudaGraphs | 0.829 | <0.001 | 0.785 | 0.997 |
| no-change rows: GEMM, 2DCONV, lud, gaussian, pathfinder, nn | 0.994, 0.995, 0.989, 0.970, 0.995, 0.993 | 0.11, 0.02, 0.21, 0.05, 0.84, 0.68 | 1.0000 (nn 1.0001) | GEMM 0.998, lud 0.996 |
(A first A/A of hotspot read 1.215 with a 20% spread while another job ran on the box; run again alone it reads 1.0002.) bench.py canaries (13 workloads x SASS and PTX, 15 runs, counters identical in all 26 rows, load 1 to 13): 25 rows within 0.984x to 1.015x of A. One row is below 0.98x with identical instructions: PTX `mem`, 0.968x (15 runs) and 0.972x (31 runs, p<0.001; its A/A over 31 runs is 1.002x). The PTX engine does not run any code of this change, so this is where the linker put its code after simd.cpp grew; SASS `mem` reads 0.984x. Not fixed: see the layout notes of rounds sp4 to sp10.
### Tests (CMake)
* `test_simd`: `f2f_widen_matches_the_rules_at_every_level`, `f2f_narrow_matches_the_rules_in_every_mode_at_every_level`, `i2d_matches_the_conversion_in_every_mode_at_every_level`, `f2f_and_i2d_may_write_over_their_sources`, and `f2f_widen_every_f32_when_asked` (VGPU_TEST_EXHAUSTIVE=1: all 2^32 f32 patterns at every level, 12 s on 8 threads). The references are written from the card's rules, not from the kernels: every sign/exponent with the mantissas that matter, the f32 overflow and subnormal edges, random words, each of the four `|x|`/`-x` combinations and rounding modes, and the MXCSR exception flags per block of 32 lanes.
* `test_sass_alu_runs`: `the_conversions_between_f32_and_f64_give_the_general_rules_registers_and_counters`: 40 seeds x 8 parts of sm_86 SASS (F2F.F64.F32 with `|x|`/`-x`, F2F.F32.F64 in the four rounding modes, I2F.F64.U32/.S32, results as sources, a destination that is a source, guards for half a warp, MUFU.RCP64H/RSQ64H) on classes of values (zeros, subnormals, infinities and NaNs of every payload, the f32 edges, ties), 32 / 100 / 33 / 5 threads, registers and counters of the tier against `VGPU_FASTPATH=0` (the unchanged general rule). Also run with `VGPU_TIER=0` (`test_sass_alu_runs_tier0`: the interpreter's `fast_float` alone) and `VGPU_SIMD=base` (`test_sass_alu_runs_simd_base`).
* `test_tier`: the form table lists `cvt64:K fltinterp:I`; the handler guard reads both handlers (no scalar float code in them).
* Mutations that fail them (each rebuilt and run): a NaN with `|x|` applied in widen (AVX2) and in narrow (test_simd); unsigned i2d by sign flip, which gives -0 when the host rounds down (test_simd); `|x|` and `-x` swapped in the tier's analysis, the tier handler without the rounding scope, the tier handler's high-word row off by one (test_sass_alu_runs); `fast_float` without the rounding scope (only `VGPU_TIER=0` run fails: the reason for that ctest entry).
* ASan+UBSan (-O0) of test_simd, test_tier and test_sass_alu_runs: pass. Not run here: TSan (no threads in the change), `e2e_sass_tier` / `e2e_sass_archs` / the AMD guards (no nvcc on server3; CI).

## Round sp14: the engines outside link-time optimization, on fixed alignment (branch `sp14-placement`, base `sp13-combined`)

Rounds 12 and 13 left one-thread differences at equal counters on rows that run unchanged code (Zen 5: SASS `mem` 0.96x, PTX `wmma` 0.98x, `launch64` 6 % slower; faster on Zen 4). The question: structural, or the lottery of where the code lands?

### What was measured (server2 Zen 5 9950X, pinned cpu, alternating runs of the benchmark program, one thread; server3 Zen 4 8840U, `perf stat`)

* Same source, one unrelated edit (two `noinline` functions added to `src/core/error.cpp`, or three to `src/runtime/runtime.cpp`): the library's text moves by some hundred bytes, instructions retired are equal to the instruction, and on Zen 5 the rows move: sp12-combined (A) and its two edited copies, medians of 40 runs, A = 1.00: SASS `mem` 1.000 / 1.002 / 1.041, PTX `wmma` 1.000 / 0.956 / 0.973, PTX `mma_ptx` 1.000 / 1.067 / 1.078, PTX `launch` 1.000 / 0.978 / 0.994, SASS `atomics` 1.000 / 0.998 / 1.015. sp13-combined (B) and its two copies: SASS `mem` 0.986 / 1.016 / 1.014, PTX `wmma` 1.035 / 1.031 / 0.991, SASS `atomics` 0.960 / 0.982 / 0.991 (Zen 4 retires 0.5 % more instructions there, so part of that is real), SASS `wmma` 1.47 / 1.48 / 1.49. The same copies on Zen 4: cycles within 1 to 2 % (equal instructions): the lottery is a Zen 5 front-end affair. So the three rows were not a property of round 13's code. A build of sp12-combined in another directory ran `launch64` at 5.0 s where round 12 had measured 4.67 s: 8 % between two builds of one source.
* Not a matter of function alignment alone: every function start was already on 16 bytes; the tensor kernel `mma_avx512_chunked_f` is `aligned(64)`. A whole-library `-falign-functions=64` build gave SASS `wmma` 1.34x in one copy and 1.52x in the same copy after the unrelated edit (the kernel is 17.6 KB: its position modulo 4096 and what neighbours it move).
* The link-time optimizer re-rolls the inlining of the engines' hot paths as well: in one build `run_micro_batch` of the PTX interpreter was a function of its own instead of inlined (`launch` PTX 0.945x, 2.8 % more instructions on Zen 4), and `setup_block`, `step_block` changed size, with `interpreter.cpp` itself unchanged, only the units around it in the LTO partition.

### What changed

`CMakeLists.txt`: `VGPU_HOT_UNIT_FLAGS` = `-fno-lto -falign-functions=64 -falign-loops=32`, on `src/sass/exec.cpp` (the SASS engine and its tier handlers), `src/exec/interpreter.cpp` (the PTX interpreter; keeps `-Werror=psabi`), `src/core/memory.cpp` (already outside LTO, now aligned) and `src/simd/simd.cpp`, `src/simd/lanes.cpp` (the SIMD kernels both engines call, `hmma_k` and the others). As for `gcn_tier_ds.cpp` and `gcn_matrix_kernels.cpp`, a unit outside LTO is compiled once from its own source, so an edit elsewhere no longer re-orders it or changes what is inlined in it. Text grows 3 % (6.78 -> 6.97 MB).

Why all five and not fewer (builds of the same base, Zen 5, medians of 60 to 100 interleaved runs, A = 1.00; "edits" = the two unrelated edits above):
| configuration | what happened |
|---|---|
| exec.cpp + memory.cpp only (PTX interpreter left in LTO) | the PTX side re-rolled: `run_micro_batch` split off, `launch` PTX 0.945x |
| + interpreter.cpp, SIMD kernels left in LTO | SASS tensor rows 1.52x and stable under edits, but `mem` SASS 0.988 to 1.018 and `launch` PTX 0.949 to 0.975 across edits |
| + simd.cpp, lanes.cpp (kept) | for the five-unit tree (built without `-Werror=psabi`, a diagnostic), a copy of it and the two edits: SASS `wmma` 1.526 / 1.521 / 1.522 / 1.517, `mma_ptx` 1.484 / 1.488 / 1.487 / 1.478, `shared` 1.027 / 1.027 / 1.026 / 1.027; `mem` 0.978 / 0.979 / 0.976 / 0.974; `launch` SASS 1.019 / 1.022 / 1.020 / 1.011 |
| functions only on 64 (no `-falign-loops=32`) on the three big units | PTX `wmma` 0.936x and 0.953x (the tensor kernel's loops) |
| same, but `simd.cpp` and `lanes.cpp` on functions only | SASS `wmma` 1.35x (1.52x with the loops), PTX `mma_ptx` 1.28x (1.42x): the loops of the SIMD kernels matter too |

The spread of one row over the edits (max - min, in points of A): before, A: `mem` SASS 4.0, PTX `mma_ptx` 7.8, PTX `wmma` 4.4; B: `mem` 3.0, `atomics` 3.1, PTX `wmma` 4.4; after, the kept flags: `mem` SASS 0.5, PTX `mma_ptx` 0.9, PTX `wmma` 0.8, `atomics` SASS 1.0, `launch` SASS 1.1, `shared` 0.1, tensor SASS rows 0.9 to 1.0.

### Results (A = sp12-combined, W = this branch, counters identical in all 26 rows on both CPUs)

`bench.py --ab`, 13 workloads x SASS and PTX, 15 runs, one thread. Zen 5 (cpu 30, load 8 to 9 from other sessions): A/A 0.99x to 1.01x in every row; A -> W: SASS `wmma` 1.726x, PTX `wmma` 1.041x, SASS `mma_ptx` 1.534x, PTX `mma_ptx` 1.442x, `shared` SASS 1.034x, `atomics` SASS 1.022x (p 0.87), `launch` SASS 1.023x, `f32_fma` SASS 1.024x; `mem` SASS 0.975x (0.962x best-of), `f32_fma` PTX 0.984x, `launch` PTX 0.989x; every other row 0.99x to 1.02x. Repeated with the builds swapped in another window (load 6 to 19, W over A): `wmma` SASS 1.60x, `mma_ptx` 1.44x / 1.46x, `mem` SASS 0.976x, `atomics` SASS 0.976x, `launch` PTX 0.979x, `shared` 1.02x, the rest 0.97x to 1.05x. W against B (interleaved medians, below): the tensor rows 1.52x / 1.49x against 1.48x / 1.43x of A, everything else within 2 %.
Interleaved medians of 60 to 120 runs (one pinned core, builds alternating; the kept flags built without `-Werror=psabi`): SASS `mem` A 1.000 / B 0.995 / W 0.979 in two windows and 1.001 in a third, `atomics` SASS A 1.000 / B 0.966 / W 0.980, PTX `launch` A 1.000 / B 1.005 / W 0.981: this is the part of the three named rows that stays: SASS `mem` about 2 %, PTX `launch` about 2 % and SASS `atomics` 2 % (the last one was 3.4 % in B).
`launch64` (scale.cu, 300 k launches of 64 blocks): 1 thread A 5.01 s, B 4.75, W 4.78; 4 threads A 5.07, B 4.80, W 4.83. (A built here is the slow end of the lottery; round 12's own A build ran 4.67.) 16 threads, `scale atomics` (wall, 5 runs): A 3.2 to 4.2 s, B 0.68, W 0.60. `sortingNetworks` (CUDA sample, 3 runs): 1 thread A 14.1 s, B 11.2, W 11.3; 16 threads A 1.6 to 2.2, B 1.35 to 1.64, W 1.61 to 1.65.
Zen 4 (server3, cpu 3, load 2 to 4, `bench.py --ab`, 15 runs; A/A 0.98x to 1.01x in all rows but PTX `local`, 0.84x at p 0.84 in one window and 1.00x in the other): A -> W, two windows: SASS `wmma` 1.278x / 1.281x, `mma_ptx` SASS 1.266x / 1.271x, PTX 1.233x / 1.247x, `shared` 1.028x / 1.018x, `divergent` PTX 1.039x / 1.052x; **PTX `wmma` 0.959x / 0.952x** (p < 0.01 both), `f64_fma` SASS 0.971x / 0.962x (p 0.016 and 0.000; not explained), `atomics` 0.99x, `mem` 0.99x to 1.00x, `launch` PTX 0.983x; the rest 0.98x to 1.03x. B against A in the same window: PTX `wmma` 1.035x.

### Tests (CMake-wired) and mutations
* `lint_hot_unit_layout` (`tests/lint/check_hot_unit_layout.py`, reads the five objects under `CMakeFiles/vgpu.dir`): none carries LTO bytecode, every executable section but the cold and startup ones is aligned to 64, and each holds code. Mutations, each rebuilt: `VGPU_HOT_UNIT_FLAGS` with `-falign-functions=16` fails (exec.cpp.o `.text` at 32, the comdat sections at 16); without `-fno-lto` fails (`.gnu.lto_.odr_types` in exec.cpp.o, interpreter.cpp.o, memory.cpp.o ...); the sp13-combined objects fail (LTO bytecode, 16). The `-falign-loops=32` half is not read back by the test (loop alignment leaves no trace in the object's headers); it is held by the flag list in the same line and measured above.
* Passing on server2 (Release): `test_bank_memo`, `test_exec*`, `test_jit*`, `test_launch_*`, `test_mem*`, `test_mma_*` and their `_simd_` twins, `test_ptx*`, `test_sass_*` (46 incl. `test_sass_lockfree`, `test_sass_atomics`, `test_sass_ok_cache`), `test_sched`, `test_simd*`, `test_tc_ampere*`, `test_tcgen05_*`, `test_tier`, `test_wmma_generic_fast*`, `test_worker_pool`: 73 of 73. Not run here: the e2e programs (CI), sanitizer legs (the change adds no code; `-fno-lto` and alignment flags only).

### Tried and not kept
* A dedicated unit for the chunked tensor kernels (`extern template` in `tc_ampere.hpp`, explicit instantiation in `simd.cpp`, then in a `tc86_kernels.cpp` of its own) with the lambdas of `mma_avx512_chunked_f` `always_inline`: it makes the kernel one copy (as inline templates each unit carries one and the linker keeps the first; outside LTO that copy has its lambdas as functions of their own, 3.6 % more instructions in PTX `wmma`, which is the Zen 4 loss above) and Zen 4 PTX `wmma` was at A, SASS tensor rows 1.44x to 1.51x. But the PTX interpreter, with the copies gone from its object, landed elsewhere: PTX `mem` 0.95x to 0.97x, `local` 0.925x to 0.93x, `divergent` 0.96x, SASS `launch` 0.956x in one window. The loss moved from one place to another; dropped, no flag list would hold it.
* `always_inline` lambdas alone (copies kept): PTX `mma_ptx` 1.25x against 1.40x. `--param=inline-unit-growth=400 --param=large-unit-insns=1000000` on the hot units (the big translation units inline less than an LTO partition does): text +13 %, SASS `atomics` 1.08x and `launch` 1.04x on Zen 5, instructions -3 % on those rows on Zen 4, PTX `shared` 0.987x and PTX `wmma` 1.00x; not kept (not run through the canary, and the size), but the best single lead left.
* Whole-library alignment, the PTX interpreter left in LTO: above.

### Not done
* The 4 to 5 % of PTX `wmma` on Zen 4, and the 2 % of SASS `mem` and PTX `launch` on Zen 5 (rows that still sit 2 % under sp12-combined, though level with sp13-combined in the same windows). `perf` does not run on server2, so the Zen 5 rows are read by wall time and by instructions on Zen 4 only.
* Per-function hot/cold ordering (`.text.hot`) inside the engines' units; the layout of `interpreter.cpp` and `exec.cpp` still depends on their own source (any edit inside them re-rolls it, as an edit inside `gcn_matrix_kernels.cpp` does).
## Round sp14: fixed-size copies that were calls to `__memcpy_chk` (branch `sp14-memcpy-chk`, base `sp13-combined` 870a8903)

Round sp13 found that glibc's fortification (`_FORTIFY_SOURCE=3` is Ubuntu's compiler default; the CMake files set nothing) turns a `std::memcpy` to a computed address in a local object into a call to `__memcpy_chk` even at a fixed 4 to 32 bytes, and fixed `src/simd/kernels.inc`. This round counts which of the sites left are executed.

### Measuring it
A preloaded `__memcpy_chk` that counts calls by return address (two builds of the shim, one with the fix; the tool is a 60-line C file, not kept) on the 14 `bench.py` workloads on both engines, 51 real programs (PolyBench, Rodinia and part of HeCBench and Samples, `tools/bench/real`, runs cut at 15 to 45 s), `nvidia_tensor_bench` (43 rows) and the AMD bench (gfx942). The cost of one call, a micro-benchmark of a 4-byte store into a local array at a run-time offset: 1.4 to 2.8 ns more than the inline store on Zen 4 (about 10 cycles).
* **Executed and hot** (calls on the base): `exec_st`'s run buffer in the PTX interpreter (`pack_run`): 4,194,304 calls in the `mem` workload (6 to 10% of its cycles); `h_global<Store>` in the SASS tier (`tier_put` into the run buffer): 6.6 M in HeCBench aop (1.5 to 2% of it), 3.8 M in PolyBench 2MM, 1.1 M in Rodinia particlefilter, 1.5 M in GRAMSCHM; `MemoryManager::store_run` (`run_in`, a copy of `unit` bytes with `unit` a run-time 1, 2, 4 or 8): 8.2 M in HeCBench base64e, 2.0 M in GRAMSCHM, 0.5 M in LU; `micro_ld_param` (every PTX launch reads its parameters: 40,000 calls in the `launch` workload); `h_cndmask_body` in the AMD tier (3.3 M calls, about 7% of the AMD `flat` workload).
* **Executed, not worth a change:** the SASS rows of `bench.py` make none but `f64_fma` (8,192 calls, 0.09%); `nvidia_tensor_bench` makes 15,922 calls over all 43 rows; `h_imma`'s six sites and the PTX `frag_*_dense` ones are not reached by any row.
* **Not executed anywhere in these runs:** most of the rest. Static call sites in the shim: 92 -> 69 (`libcudart`), 20 -> 16 (`libamdhip64`).

### What changed
* `pack_run` (`src/exec/interpreter.cpp`): `__builtin_memcpy` (a fixed `Size`), and `[[gnu::noinline]]`. Inline, its loops grew `exec_st` and moved the inlining of its callees: PTX `shared` and `warp` read 0.97 to 0.98 at equal instructions in three builds; out of line they are 0.996 to 1.002.
* `micro_ld_param`: `load_bytes` (its switch on 1, 2, 4, 8) in place of `memcpy(&v, p, size)`. The two `ld.param` paths in `exec_ld` keep the `memcpy` (not executed by any row; touching `exec_ld` moved the placement of the PTX loop).
* `MemoryManager::run_in` / `run_out` (`src/core/memory.cpp`): `word_to` / `word_from`, a switch on the unit with a constant size in each case.
* `tier_put` / `tier_get` and `h_imma`'s operand loads (`src/sass/tier_ops.inc`): `__builtin_memcpy`.
* `h_cndmask_body` (`amd/src/gcn_tier.cpp`): the store into the local row `r` is `__builtin_memcpy`.
* `Interpreter::gather_words` / `scatter_words`: `[[gnu::aligned(64)]]`. Not part of the copy work: with the changes above the linker put `scatter_words` 96 bytes before a 4 KB page boundary (so it crossed the page) in the build examined, and the `mma_ptx` workload on the PTX engine, which calls it after each mma, read 0.977 on Zen 4 and 0.935 to 0.953 on Zen 5 with the same instructions; `perf` showed the extra cycles in `scatter_words` (6.7 -> 10.5% of the row) and L1i misses +6%. With both on a 64-byte boundary Zen 4 reads 1.003 and Zen 5 0.969.

### Results (cycles:u of a pinned process on Zen 4 (server3, cpu 4, `perf stat`, median of 9 alternating runs), CPU time (user+sys) on Zen 5 (server2, cpu 26, median of 15; no perf there). A = `sp13-combined`, B = this branch; counters and instruction counts equal in every row (Zen 4: `perf stat` instructions equal to within 0.01% except where the fix removes the calls: `mem` ptx -5.7%, `launch` ptx -0.1%).)
| row | Zen 4 B/A | Zen 5 B/A |
|---|---:|---:|
| `mem` ptx (4.2 M calls) | 1.061 | 1.052 |
| `launch` ptx | 1.012 | 1.009 |
| AMD `flat` (mi300x) | 1.050 | 1.016 |
| HeCBench aop | 1.017 | 1.020 |
| PolyBench 3MM (2MM and GRAMSCHM run past the timeout) | 1.003 | 1.011 |
| Rodinia needle | 1.013 | 1.002 |
| PolyBench 3DCONV | 1.002 | 1.023 |
Earlier builds of the same change gave `mem` ptx 1.056 to 1.106 on Zen 4 and 1.044 to 1.088 on Zen 5 (the placement of the rest of the library moves it by that much). The gain on a real program is the share of the calls: 1 to 3%. Rows that make no call also moved with the layout: `mem` sass 1.027 / 1.057 (Zen 4 / Zen 5) and `wmma` / `mma_ptx` sass 1.006 / 1.024 and 1.006 / 1.035 are placement, not this change.

The 12-workload canary (`int_alu f32_fma f64_fma wmma mma_ptx mem shared atomics local divergent warp launch api`, both engines) and `startup`, B/A of the medians; A/A (the base twice) is 0.990 to 1.006 on Zen 4 (`startup` sass 0.951, a 5 ms row; AMD `mfma` 1.030) and 0.991 to 1.011 on Zen 5 (AMD `launch` 0.985):
* Zen 4: 0.982 (`wmma` ptx: the same instructions; 0.988 to 0.998 in other builds) to 1.018, with `mem` sass 1.027 and `mem` ptx 1.061 above; `startup` sass 0.955 (its A/A is 0.951).
* Zen 5: 0.969 (`mma_ptx` ptx), 0.974 (`local` sass), 0.982 (`shared` sass) to 1.035, with `mem` sass / ptx 1.057 / 1.052 above (placement; `mma_ptx` ptx was 0.935 to 0.953 before the alignment).
* AMD (mi300x, nine workloads): 0.957 to 1.050 except `atomics` on Zen 4, 0.969 at equal instructions in four builds (`MemoryManager::rmw_batch_fn` 10.6 -> 13.1% of the row: placement in `memory.cpp`).
The round 12 and 13 gains are intact on the rows they were measured on: `wmma` sass 1.006 / 1.024 and `mma_ptx` sass 1.006 / 1.035 (Zen 4 / Zen 5), `shared` 1.000 / 0.982 (sass), `atomics` sass 1.016 / 1.026; `wmma` and `mma_ptx` on the PTX engine are the 0.982 / 0.989 and 1.003 / 0.969 above. The 16-thread scaling was not re-measured (no code of the scaling paths changed).

### Project-wide `-U_FORTIFY_SOURCE`
Built (the branch with `-DCMAKE_CXX_FLAGS=-U_FORTIFY_SOURCE`, so every call in the shim is plain) and measured on the same rows: C over A, geometric mean of the 25 rows that make no call in the base: 0.9995 (Zen 4) and 1.0012 (Zen 5), against 0.9985 and 0.9986 for the targeted change; every row it moved by more than 2% is also a row where the targeted change differs by the same amount (placement). So the targeted fixes take what a global switch would. Not adopted: it removes glibc's checks (`memcpy`, `strcpy`, `sprintf`, `read`, `__printf_chk` abort with "buffer overflow detected" instead of corrupting memory) from a library that is loaded into other people's processes, only where the compiler adds them by default (Ubuntu, Fedora; conda-forge passes `-D_FORTIFY_SOURCE=2` in its flags, which a CMake `-U` would override), and a sanitizer build is a `-O0`/`-O1` build, where glibc's fortification is off or the sanitizer's checks cover the same overflows. If it is wanted anyway: `option(VGPU_FORTIFY ... ON)` and `add_compile_options(-U_FORTIFY_SOURCE)`; the JIT's generated code (`src/sass/jit_rt.h`, seven fixed-size `std::memcpy`) is compiled at run time with the user's compiler flags and is not changed by it.

### Tests (CMake-wired)
* `test_memcpy_chk` (tests/unit): defines `__memcpy_chk` in the test, so the statically linked library's calls land in it, counts the calls of up to 256 bytes, and requires none where the compiler fortifies (an optimised build, `_FORTIFY_SOURCE` set; not built under a sanitizer). It runs: `store_run` / `load_run` at units 1, 2, 4, 8 against a model (400 random runs each, every phase against the 8-byte boundary); a SASS kernel storing 1, 2, 4 and 8 bytes from 32 consecutive lanes (`STG`, hand-assembled as in `test_sass_dispatch`), the bytes checked; a PTX kernel storing u8, u16, u32, u64 the same way (this one also reads its parameter, `micro_ld_param`); and the AMD tier's `v_cndmask_b32` over the corpus shapes with the tier on and off (equal digests).
  Mutation check: the test built against the unchanged base library fails all four cases (`store_run / load_run, unit 4: 429 calls`; `SASS STG of 2 bytes: 32 calls`; `PTX st.global.u8: 33 calls` (32 from `pack_run`, one from `micro_ld_param`); `v_cndmask_b32 in the tier: 224 calls`) and passes on the branch, also at `VGPU_SIMD=base`, `avx2` and `avx512`.
* `lint_ptx_word_moves_aligned` (tests/lint/check_ptx_word_moves_aligned.sh): reads `nm` of the built `libcudart` and requires `gather_words` and `scatter_words` on a 64-byte boundary. The base fails it (`scatter_words` at 0x2361b0, 48 bytes into a line) and the branch passes.
* Also run, passing: `test_memfast`, `test_memalloc`, `test_sass_dispatch`, `test_tier`, `test_sass_access_count`, `test_sass_mem_rows`, `test_memcount_forms`, `test_sass_lanes`, `test_amd_gcn_tier_same_bits` (default and `VGPU_SIMD=base`). Compiled with g++-12 `-Wall -Wextra -Werror` (and `-fsanitize=address,undefined`, compile only): the test, `memory.cpp`, `gcn_tier.cpp`, `interpreter.cpp`, `sass/exec.cpp`. Not run: ASan / UBSan / TSan executions (no threaded code touched; the test is not built under a sanitizer).

### Not done
* The 69 remaining call sites in `libcudart` are not executed by any row measured here; a program that reaches `exec_ld`'s variable-size paths, `Runner::exec_mem` or the TMA copies would need the same measurement.
* `src/sass/jit_rt.h` (the generated code's seven copies) belongs to the JIT and was not touched.
* `mma_ptx` ptx on Zen 5 is 0.969: the remaining loss is placement of the PTX tensor path (no further alignment was tried; the `sp14-placement` work is the place for it).

## Round sp14: the two changes together (branch `sp14-combined` = `sp13-combined` + `sp14-placement` + `sp14-memcpy-chk`)

Merged as they were, the two changes interacted: the PTX interpreter's `exec_st` shrank (`pack_run` out of line) inside a unit that is now built outside LTO, which moved the rest of the unit and the units linked after it, and at equal counters PTX `int_alu` read 0.903x to 0.910x (Zen 5, 15 runs, A/A 0.997x; each branch alone 0.996x and 1.002x), PTX `wmma` 0.948x on Zen 5 and 0.919x to 0.948x on Zen 4 (alone 0.996x and 1.002x). Which of these rows loses moves with the offset of the interpreter modulo a page (a sweep of 24 offsets from 0 to 3840 bytes: PTX `wmma` 0.84x to 1.007x, `mma_ptx` 0.915x to 1.09x, `launch` 0.95x to 0.99x), so the fix is to pin it: the five hot units begin with an `asm` line that raises their `.text` alignment to 4096 (where each starts modulo a page no longer depends on what is linked ahead of it), and the interpreter's starts 2816 bytes in, the offset at which no PTX row of the sweep lost more than 2 % (0.989x to 1.07x). `lint_hot_unit_layout` also requires the 4096 alignment (mutation: the `sp14-placement` and `sp14-combined` objects without the line fail it). Counters are identical in every row on both CPUs; this is a layout choice measured on these two machines, and an edit inside `interpreter.cpp` re-rolls it as before.

Results, `bench.py --ab`, 13 workloads x SASS and PTX, one thread, pinned cpu, load 4 to 16 from other sessions; Zen 5 (server2, 15 runs), Zen 4 (server3, 9 runs). W = this branch; A/A (`sp13-combined` against a copy of its libraries) on Zen 5: 0.99x to 1.01x in every row except SASS `atomics` 0.955x (p 0.000; a row that reads 0.96x to 1.03x between identical builds), on Zen 4 0.99x to 1.04x.
| row | Zen 5 W/sp13 | Zen 5 W/sp12 | Zen 4 W/sp13 | Zen 4 W/sp12 |
|---|---:|---:|---:|---:|
| `mem` SASS / PTX | 0.995 / 1.036 | 0.992 / 1.172 (1.04 best-of) | 0.976 / 1.045 | 0.999 / 1.064 |
| `atomics` SASS / PTX | 1.014 / 1.006 | 0.987 / 1.023 | 1.030 / 0.971 | 1.001 / 1.010 |
| `launch` SASS / PTX | 1.023 / 0.979 | 1.036 / 0.967 | 1.028 / 0.994 | 1.014 / 0.983 |
| `wmma` SASS / PTX | 1.020 / 0.996 | 1.505 / 1.035 | 1.013 / 0.999 | 1.471 / 1.024 |
| `mma_ptx` SASS / PTX | 1.026 / 1.083 | 1.468 / 1.437 | 1.018 / 1.017 | 1.486 / 1.347 |
| `shared` SASS / PTX | 1.014 / 1.004 | 1.050 / 1.036 | 1.010 / 0.995 | 1.022 / 1.036 |
| `local` SASS / PTX | 1.005 / 1.000 | 1.009 / 0.995 | 1.008 / 1.030 | 1.023 / 1.027 |
| `f64_fma` SASS / PTX | 1.007 / 1.015 | 1.014 / 1.011 | 0.998 / 1.010 | 0.998 / 0.994 |
Every other row 0.97x to 1.02x on both CPUs (Zen 4 W/sp13 `warp` PTX 0.970, `api` SASS 0.978). AMD (`bench-amd`, mi300x, one thread, 15 runs, Zen 5; A/A 0.96x to 1.00x): W/sp13 `atomics` 0.988, `valu_int` 1.015, `mfma` 1.007, `divergent` 1.013, but `mem` 0.946 to 0.958, `flat` 0.926 to 0.929, `mfma_i8` 0.96 and `p_sbuf` 0.93 (W/sp12: `mem` 0.945, `flat` 0.898, `mfma_i8` 1.074, `launch` 0.951), at equal counters; `p_cmp` 1.19 is the `h_cndmask_body` fix. `sp14-placement` alone already read `flat` 0.882 and `mem` 0.990, so these AMD rows are moved by the engines leaving the LTO partition (the AMD code stays inside it). Zen 4 AMD: 0.98x to 1.03x in every row (A/A 0.90x to 1.00x). Not fixed: taking `gcn_tier.cpp` out of LTO on the same flags and anchors (a tree that was not kept) gave `mem` 1.015, `flat` 1.10, `atomics` 1.026, `p_sbuf` 0.997 but `mfma` 0.965, `mfma_i8` 0.94, `valu_f32` 0.975, at every one of eight offsets of `gcn_matrix_kernels.cpp`; one set of losses for another.
`launch64` (scale.cu, 300 k launches of 64 blocks, medians of 5): Zen 5 1 thread sp12 5.17 s, sp13 4.86, W 4.54; 4 threads 5.07 / 4.85 / 4.58 (Zen 4: 15.5 s for all three at 1 thread, bimodal 8 or 15 s at 4 threads, no difference). `scale atomics` at 16 threads: Zen 5 sp12 3.46 s, sp13 0.72, W 0.61; Zen 4 8.08 / 1.28 / 1.36. `sortingNetworks` (3 runs): 1 thread Zen 5 sp12 15.7 s, sp13 11.3, W 11.5; 16 threads 2.24 / 1.59 / 1.87 (the 16-thread rows spread 1.45 to 2.28 s within one build); Zen 4 1 thread 35.1 / 30.2 / 28.1, 16 threads 5.16 / 4.28 / 4.25.
Unit tests, Release, passing: `test_memcpy_chk`, `lint_hot_unit_layout`, `lint_ptx_word_moves_aligned`, `test_sass_*` (21), `test_ptx*` (5), `test_tc_ampere*` (4), `test_tier`, `test_memfast`, `test_amd_gcn_tier_same_bits` (and `_simd_base`): 34 of 34.

## Round sp15: the thread cap costs a block from the work, not the wall time (branch `sp15-scaling`, base `sp14-combined` d9e63ab3)

The question: why do multi-block programs not scale from 4 to 16 threads (CUDA sample `sortingNetworks` 1.6 to 1.9 s at 16 threads against 11.5 s at one on Zen 5, `launch64` bimodal 8 s / 15 s at 4 threads on Zen 4)? Measured on server2 (Zen 5, 9950X, load 4 to 20 from other sessions; cpus 0-15 for 16 threads, one physical core each). Zen 4 (server3) could not be measured: its load stayed at 60 to 110 for the whole round, so the build never finished and nothing there is claimed.

### Where the time goes (no change needed)
* `sortingNetworks` (290 launches: 70 of 1024 blocks x 512 threads, 220 of 2048 x 256), launch-level trace of `Runner::run` and `run_parallel` (temporary instrumentation, not kept): inside the pool launches the threads were busy 92 % of the launch wall time at 4 threads and 90 % at 16 (big blocks 0.97 / 0.93, the 6.5 us blocks 0.76 / 0.81). Launch wall time falls 13.2 s (1 thread) -> 4.1 s (4) -> 0.89 s (16), 14x on 16 cores. A 2048-block launch of 1.2 ms spends 135 us (average) to 1 ms (the last of 16) waking its workers, but the first workers have taken every block by then, so the launch ends when the work does (`end_min` 1003 us, `end_max` 1013 us in the trace); the join adds 5 to 12 us. Nothing between `Runner::run` and `run_parallel` costs anything (gap 5 to 12 us per launch).
* What is left of the 1.4 s at 16 threads is **0.50 s of host code between the launches** (the sample's own validation of the sorted arrays, at its build's optimization level: gaps of 232, 116, 59, 30 and 16 ms after the first five launches, 0.45 s of the 0.50 s are gaps over 10 ms, 1.7 ms are gaps under 1 ms). It is the same 0.50 s at 4 threads, so it is the Amdahl floor of this program, not a simulator cost: 36 % of the wall time at 16 threads, 12 % at 4. Process start and exit take 0.02 s here.
* The synthetic probe (`scale.cu`, medians of 5 to 7, t = 1 / 4 / 16, quiet to half-busy box): compute 4.9 / 1.4 / 0.42 s, divergent 10.3 / 2.6 / 0.82, tensor 6.4 / 1.8 / 0.43, mem 6.8 / 2.6 / 0.81, atomics (4096 shared counters) 4.9 / 1.6 / 0.66. mem and atomics scale less (memory bandwidth, true sharing of the counters' lines), the rest 11 to 15x.
* `next_unit_` (the block hand-out counter) shares a cache line with fields every block reads. Putting it and `units_` on lines of their own (`alignas(64)`, the one idea of the held `sp4-threads` branch that main still lacks; its other two changes are in since round 12) changed nothing: `sortingNetworks` 1.40 / 1.40 / 1.40 s (A / W / with the alignment) at 16 threads, 4.30 / 4.28 / 4.26 at 4; compute, mem, atomics, divergent, tensor at 16 threads within 1 to 3 % in both directions. Dropped; `sp4-threads` has nothing left worth merging.

### The bug: a launch that the pool made slow stayed on the pool (found with a pinned CPU)
Round 12 keeps a launch to the threads its work can pay for, and learns what a block costs from the last launches of the kernel. A launch on one thread measures it exactly. A launch on several threads used its wall time: `wall x threads / blocks`. That figure holds the wake-up of the workers and the wait for the slowest, which are the cost of the pool and not of the blocks. 64 blocks of 0.23 us (15 us of work, so one thread) launched on 2 threads with a 100 us hand-off give 100 x 2 / 64 = 3.1 us a block, which is above the 1.56 us a block that two threads need (the threshold is 100 us of work per thread), so the next launch is given two threads again, is slow again, and the cost stays high: the state keeps itself, whatever the kernel does. Any launch that takes more than 78 us on two threads keeps a 64-block launch there. How often a program lands in it depends on how slow the wake-ups are (a loaded host, a CPU that is crowded, a slow or power-managed core), which is the bimodal 8 s / 15 s of `launch64` on the loaded Zen 4 box.
Reproduced on Zen 5 by making the hand-off slow in the way a crowded host does: `scale launch64 5` (15,000 launches of 64 one-warp blocks) at `VGPU_THREADS=4` pinned to **one** CPU (`taskset -c 12`): A (`sp14-combined`) 2.18 s, 15,000 involuntary context switches, every launch on the pool; with the cap off (`VGPU_POOL_MIN_WORK_US=1000000`) or at one thread 0.22 s. On two CPUs (`12,28`) both builds run 0.21 s.
### The fix
`run_parallel_raw` times each body (`Worker::busy_ns`, from the start of the body to its end on whichever thread ran it, the caller's own inline bodies too) and adds the sum to a per-thread tally (`t_busy_ns`) when it returns. `LaunchWidth` clears the tally when it is made and, when it ends, takes the cost of a block from it (`unit_cost_from_busy`: busy time / blocks, never zero) if the launch used more than one thread and the pool ran bodies; a launch on one thread, or one whose bodies did not go through the pool, still uses its wall time as before. The cost no longer contains the wake-up or the wait for others, so a slow hand-off cannot raise it. (A first version put the timer in the two engines' worker bodies. It worked, but it edited `src/sass/exec.cpp` and `src/exec/interpreter.cpp` and SASS `wmma` / `mma_ptx` read 0.92x in `bench.py --ab` at equal counters: the layout lottery of rounds 12 to 14. The pool is a unit of its own, and the engines' sources are untouched in this version.) A body's own setup (a pooled block, an interpreter) counts as busy time; at a few microseconds per thread it cannot hold a launch on the pool (that would need 100 us of setup per thread).
Same `launch64` run, pinned to one CPU at 4 threads: A 2.18 s -> W 0.22 s (10x, 3 runs each, 0.22 s in every one), pool calls 15,000 -> 1; on two CPUs 0.21 s -> 0.22 s.
### Results (server2, `scale.cu` and `sortingNetworks`, wall medians, A = `sp14-combined`, W = this branch, 5 to 7 interleaved runs, outputs identical)
| row | t = 4 A / W | t = 16 A / W |
|---|---:|---:|
| launch64 (20 x 3000 launches) | 0.89 / 0.89 s | 0.87 / 0.82 s |
| compute | 1.18 / 1.16 | 0.33 / 0.33 |
| mem | 1.70 / 1.69 | 0.61 / 0.61 |
| atomics | 1.18 / 1.16 | 0.61 / 0.62 |
| divergent | 1.84 / 1.85 | 0.64 / 0.65 |
| tensor | 1.12 / 1.18 | 0.40 / 0.39 |
| sortingNetworks | 3.46 / 3.61 (max 7.19 in one run) | 1.44 / 1.44 |
Large grids and the program that motivated the study are unchanged, as expected: the change only matters when the pool is slow.
The 13-workload canary (`bench.py --ab`, one thread, sass + ptx, 15 runs, cpu 8, counters identical in all 30 rows): every row 0.99x to 1.02x (SASS `wmma` 1.003, `mma_ptx` 0.998, `mem` 1.018, `launch` 1.049 / 1.034, which an A/A of two copies of A also showed as 1.029 / 1.015). A/A (A against a copy of its library): 0.999x to 1.012x in every row except `launch` 1.029x / 1.015x (p 0.000: a row that sits in a different place in a copy of the same build). Zen 4: not measured (see above).
### Tests (CMake-wired) and mutants
* `test_worker_pool`: `the_cost_from_busy_time_is_per_block_and_never_zero`; `a_slow_wake_up_does_not_keep_a_small_launch_on_the_pool` (a 300 us wake-up on every pool launch of 64 blocks that cost 15 us in all: the second launch and every one after it get one thread; the wall-time estimate fails it, the first launch keeps its threads); `heavy_blocks_keep_their_threads`.
* `test_sched` `a_pool_slowed_by_a_crowded_cpu_does_not_keep_small_launches_on_it` (PTX engine) and `test_sass_block_reuse` `a_crowded_cpu_does_not_keep_a_small_sass_launch_on_the_pool` (SASS engine): a forked child pinned to one CPU launches an 8-block kernel 400 times at `VGPU_THREADS=4`; at most 40 of them may reach the pool (A: 400 of 400). Skipped under ThreadSanitizer (a fork child cannot start threads there).
* Mutants (server2, tests run): the cost taken from the wall time again (`used_ > 1 && busy > 0` replaced by `false`): the two unit tests fail; the tally never filled (`t_busy_ns += busy` removed from `run_parallel_raw`): the PTX and the SASS test fail (the unit tests pass, as they feed the tally themselves); in the first version, the timer removed from the PTX body: the PTX test failed with 400 of 400 launches on the pool, the SASS timer removed: the SASS test failed.
* Also run, passing: `test_worker_pool` (16), `test_sched` (9), `test_sass_block_reuse` (7), `lint_hot_unit_layout`. Sanitizers: not run (server2 forbids them, server3 was unusable); the new shared state is one plain field written by a worker before its `done` store and read by the caller after the matching load, and a `thread_local`; the CI TSan leg covers `test_worker_pool` and `test_sched`.
### Not done
* Zen 4 numbers (server3 load 60 to 110 all round). The `launch64` 8 s / 15 s on Zen 4 is explained by the mechanism above but was not reproduced there; if the pool really is faster than one thread for 64 blocks of 0.8 us on that core (the 8 s mode), the 100 us threshold is too high for it, and the new estimate will keep that kernel on one thread every time.
* The wake-up latency itself (135 us on average, up to 1 ms for the last of 16 workers in a 1.2 ms launch). It costs nothing while the first workers finish the grid, and a loaded host decides most of it.
* `e2e_pool_cap` could not run here (no nvcc on server2); it is unchanged and runs in CI.
## Round sp15: the AMD tier outside link-time optimization (branch `sp15-amd-layout`, base `sp14-combined` d9e63ab3)

PR #458 left four AMD rows under `sp13-combined` at equal counters (Zen 5, mi300x, one thread): `mem` 0.95x, `flat` 0.93 to 0.96x, `mfma_i8` 0.96x, `p_sbuf` 0.915x. The question: can the AMD code be made independent of the layout lottery as the engines were in round 14?

### What was found

* The loss is inlining and placement inside the link-time-optimized library, not the source (the AMD sources are unchanged between the two bases). Comparing the two libraries symbol by symbol: `Machine::scalar_load` is inlined in `sp13-combined` and a function of its own in `sp14-combined`, `Machine::at` is a call inside `memory_run` in one and inlined in the other (`memory_run` is 3.8 KB against 2.4 KB), and a hundred more interpreter functions (the ALU lambdas of `vector_alu`, `more_alu`, `cdna5_alu`; `matrix_by_type`; `divide_step`) differ in size by 5 to 40 %. `tier_global` and `global_access` call the same functions in both, so those two rows are placement.
* Taking the interpreter out of link-time optimization (`gcn_exec.cpp` with `VGPU_HOT_UNIT_FLAGS`, or with `-fno-lto` alone, or together with the tier) is not an answer: the ALU rows fall 5 to 10 % (`valu_int` 0.93x, `valu_f32` 0.94x, `p_pkf16` 0.88x, `p_lane` 0.90x, `smem` 0.93x) while `mem`, `flat`, `p_sbuf` stay where they are. The unit is 14 k lines whose code the optimizer lays out better with the rest of the library in sight.
* Taking only `gcn_tier.cpp` out (the unit the previous attempt moved) recovers `mem`, `flat`, `p_sbuf` and `mfma_i8`; what then moves is the rows that run through the interpreter (`mfma`, `mfma_bf16`, `p_lane`), by where the link-time-optimized code lands relative to a page: sweeping the unit's start offset (`.skip N` after the page alignment) at 0 / 1024 / 1536 / 2048 / 2560 / 3072 bytes, `mfma` read 0.958 / 0.965 / 0.999 / 1.012 / 0.996 / within 0.98 to 1.03 of `sp13-combined` and `p_lane` 0.96 / 0.91 / 0.996 / 0.97 / 0.93 / 0.92 (single windows, core 9). The `-fno-lto`-only variant (no 64-byte alignment) read the same `valu_f32` loss, so the loss is the unit's code inlined differently outside LTO, not alignment.

### What changed

`gcn_tier.cpp` joins the hot units: `VGPU_HOT_UNIT_FLAGS` (`-fno-lto`, functions on 64, loops on 32) and the page-alignment line, followed by a 1536-byte skip: the offset at which, in the sweep above, no row of the 18-row AMD set was more than 3 % under `sp13-combined` except `valu_f32` (0.972 to 0.976 at every offset). `lint_hot_unit_layout` reads the sixth object. An edit inside `gcn_exec.cpp` still re-rolls everything the interpreter does (the offset is a measurement of this tree, as the interpreter's 2816 is for the PTX side).

### Results (`bench-amd`, mi300x, one thread, counters identical in all 44 rows in every run; W = this branch, A = `sp14-combined`, S = `sp13-combined`)

The measurement floor is high and file-dependent: identical copies of one library (`cp`) differ by up to 7 % on the scalar-memory rows (`smem` 0.928, `p_smem_wide` 0.962 between two copies of W in one window) and by 8 to 24 % on single ALU rows in other windows (`valu_f32` 0.76 and `valu_int` 1.09 between two copies of W in one window; the same pair read 1.00 in the next). The same pair read `p_sbuf` 0.87 in three windows in a row and 0.98 to 0.995 in earlier ones. So the table below is the median-of-three pairs of different copies (A_i against W_i, S_i against W_i), 15 runs, one window, core 9, and a row is called only when the three pairs agree.
| row | Zen 5 W/A (3 pairs) | Zen 5 W/S (3 pairs) | Zen 4 W/A, W/S |
|---|---|---|---|
| `mem` | 1.044 / 1.054 / 1.051 | 1.001 / 1.006 / 1.003 | 1.037, 1.030 (A/A 0.98) |
| `flat` | 1.152 / 1.161 / 1.148 | 1.092 / 1.094 / 1.094 | 1.073, 1.076 (A/A 0.99) |
| `mfma_i8` | 1.004 / 1.004 / 1.005 | 1.018 / 0.971 / 0.967 | 1.006, 1.007 |
| `valu_f32` | 0.992 / 0.998 / 0.995 | 0.975 / 0.977 / 0.974 | 0.986, 1.002 |
| `valu_int` | 1.030 / 1.012 / 1.011 | 1.022 / 1.020 / 1.029 | 1.058, 1.046 |
| `mfma_bf16` | 1.055 / 1.027 / 1.024 | 0.996 / 0.987 / 1.010 | 1.016, 0.985 |
| `mfma` | 1.004 / 0.978 / 0.970 | 0.991 / 0.977 / 1.002 | 1.028, 1.009 |
| `p_sbuf`, `p_lane` | 0.98 / 1.13 / 1.02 and 0.98 / 1.03 / 0.94 | 0.92 / 1.04 / 1.03 and 0.92 / 0.97 / 0.92 | within the A/A floor |
Result: `mem` and `flat` are recovered and then some (`flat` +15 % over `sp14-combined`, +9 % over `sp13-combined`; `mem` +4.5 % over `sp14-combined`, level with `sp13-combined`), on both CPUs. `mfma_i8` is level with `sp14-combined` and with `sp13-combined` within the noise (0.97 to 1.02). `p_sbuf` and `p_lane` swing by ±10 % between copies and windows, with the sign of the change not reproducible: not recovered, not shown lost. `valu_f32` stays 2.5 % under `sp13-combined` (it is level with `sp14-combined`), which is the cost of this unit's code inlined differently outside LTO. The full 44-row run, W against A on Zen 5 (15 runs, one window): every row 0.94x to 1.19x; the ones outside 0.98x to 1.02x were `launch_wide` 1.19, `divergent` 1.14, `flat` 1.15, `launch_mid` 1.14, `mem` 1.06, `lds` 1.06, `ldsx` 1.05, `p_perm` 1.06, `mfma_bf16` 1.03, `mfma_fp8` 1.03 against `valu_int` 0.955, `p_lane` 0.94 (both windows later contradicted), `mfma_f64_4b` 0.977, `api` 0.978. Zen 4 (15 runs, cpu 3, A/A 0.97x to 1.05x): `valu_int` 1.06, `atomics` 1.05, `wave` 1.08, `divergent` 1.06, `smem` 1.11, `launch_mid` 1.11, `p_cmp` 1.07, `p_i64` 1.11, `p_sdwa` 1.09; `p_dpp` 0.951 and `p_dpp_arch` 0.927 (p 0.000) are the losses there.

### Tests (CMake-wired) and mutations
* `lint_hot_unit_layout` now includes `amd/src/gcn_tier.cpp.o`. Mutations, each compiled from the unit's own command line: without `-fno-lto` fails (LTO bytecode); `-falign-functions=16` fails (the comdat sections at 16 and 32); without the anchor line fails (`.text` at 64, not 4096).
* Passing on server2 (Release): `test_amd_gcn_tier_same_bits`, `test_amd_gcn_same_bits`, `test_amd_matrix_fast`, `test_amd_matrix_kernels_layout`, `test_amd_mfma_math`, their `_simd_base` twins, `lint_hot_unit_layout`: 10 of 10. The counters of the benchmark's 44 rows are identical to `sp14-combined`'s in every run.

### Not done / not kept
* The interpreter (`gcn_exec.cpp`) outside LTO, in three flag sets, and the same with the tier: ALU rows 5 to 10 % under (above). Not kept.
* A hot/cold split inside the tier: its cold paths (`fix_nan32`, `fix_nan64`, `post32`, `post64`, the int/f64 general paths) are already `noinline`, and the loss that remains (`valu_f32`) is inlining, not cold code in the way. Forcing the inlining decisions of `sp13-combined` in the interpreter (`always_inline` on `scalar_load`, `at`) was not tried; the interpreter's thousands of other differing decisions make it a whole-file exercise.
* The `p_sbuf` / `p_lane` floor needs a different method (pinned physical pages, or many more copies); a single A/B of two files is not a measurement on these rows.
## Round sp15: fewer instructions in the memory handlers (branch `sp15-fewer-instrs`, base `sp14-combined` d9e63ab3)

The question: compute loops already run at IPC 4 to 5 on Zen 4, so which instructions can the SASS engine stop executing, without changing a result or a counter?

### Where the instructions were (server3, Zen 4, `perf record -e instructions:u -c 100000`, one thread, SASS, base build)
* The 12 canary workloads: `Runner::step_warp` (the tier's run loop is inlined in it) is 16 % of all instructions (47 % of `divergent`, 25 % of `warp`), then the tensor kernel, `dop`, `tier_issue`, the SIMD kernels. A straight-line integer or float instruction costs 4.3 to 4.4 instructions of host code per warp instruction, so nothing is left there; the dearer kinds per warp instruction are `texture` 90, `atomics` 36, `divergent` 22, `shared` 19, `mem` 13.
* A set of real programs (PolyBench/GPU, the Rodinia apps of `tools/bench/real`, seven CUDA Samples; `perf stat` complete runs, 35 to 2,200 G instructions each): `step_warp` 12 %, `h_global<false>` 11.6 %, `MemoryManager::load_batch` 7.8 %, `load_run` 5.5 %, `h_global<true>` 4 %, `sass_bank_passes_table` 3.9 % (histogram 34 %), `sectors_fast` 3.5 %, `S2R` fill (`tier_sreg_fill`) 5 to 7 % of FDTD-2D and LU. In SYRK, SYR2K, CORR and COVAR (the programs that read or write a column) a quarter of `h_global`'s own instructions were two loops over the 32 lanes: the global-address pass (`full_warp_addresses`: load, shift, or, add, store, and a compare plus conditional move for the constant-window test, about 10 instructions a lane) and the store-data widening.
* `__popcountdi2` in `int_alu` is the benchmark program's own host-side `__popc`, not the simulator.

### What changed (the sources are in `include/vgpu/exec/` so that each has a unit test that needs no GPU)
1. `sass_mem_rows.hpp` `full_warp_addresses`: two lanes to a vector (SSE2): the low and high words interleaved, `add` added, the pair stored; the constant-window question ("is some lane in the 4 GiB from `kConstWindow`") is a compare of the high word with the window's, because the window starts on a multiple of 4 GiB (any other window keeps the scalar loop), ORed over the warp and looked at once. About 4.5 instructions a lane instead of 10. `tier_global_addresses` (the full-warp case) uses it too. The return value is now 0 when a lane is in the window and all ones when none is; the callers only compare it with 2^32, as before.
2. `sass_mem_rows.hpp` `full_warp_store_values`: a full warp's store data (1, 2, 4 or 8 bytes) widened to the 32 64-bit values the batch store takes, four lanes to a vector (it was a lane loop that decoded the width for every lane).
3. `sass_lanes.hpp` `fill_lanes` / `lane_ids`, used by `Runner::tier_sreg_fill`: S2R of a value the same for every lane (`ctaid`, `ntid`) or of `SR_LANEID` was a lane search and a store a lane (5 instructions a lane) even for the whole warp; the whole warp is now a broadcast / a constant row.
4. `sass_lea.hpp`: LEA without a carry in (the low word of every 64-bit address calculation) skips the carry-in operations (eight of about twenty-five per vector); `lea_t<Hi, Cin>`.

### Results (instructions retired and cycles from `perf stat`, user mode, pinned cpu 5, A = `sp14-combined`, B = this branch; results and every counter identical)
Real programs, Zen 4 (server3), instructions B/A, and cycles B/A as the median of three alternating runs (load 1 to 3 from other sessions) where three were made, one run otherwise (marked *):
| program | instr B/A | cycles B/A |
|---|---:|---:|
| SYR2K | 0.914 | 0.919* |
| SYRK | 0.917 | 0.933 |
| LU | 0.938 | 0.941 |
| 3MM | 0.948 | 0.965* |
| COVAR | 0.951 | 0.974* |
| CORR | 0.952 | 0.959 |
| FDTD-2D | 0.968 | 0.950 |
| transpose | 0.970 | 0.986 |
| lud | 0.973 | 0.988* |
| particlefilter | 0.964 | 0.977* |
| convolutionSeparable | 0.976 | 0.969* |
| mergeSort | 0.998 | 0.981* |
| sortingNetworks | 0.997 | 0.996* |
| histogram, reduction | 1.000 | 0.992*, 1.005 |
Geometric mean of the 15: 0.964 in instructions. The programs that read a column (a gather: every lane its own address) gain most; programs whose loads are coalesced or shared-memory bound are unchanged.

Canary workloads, SASS, Zen 4 instructions B/A: `mem` 0.985, `shared` 0.999, `atomics` 0.998, `int_alu` 1.003 (not understood: the code it runs is unchanged; layout of `step_warp`), every other row 1.000 to within 0.01 %. `bench.py --ab`, counters identical in all 32 rows:
| row (SASS) | Zen 5, 15 runs (A/A 0.99x to 1.02x) | Zen 4, 9 runs (A/A 0.99x to 1.02x; the PTX `wmma` 1.08x reads the same between two copies of A) |
|---|---:|---:|
| `mem` | 1.067 (p 0.000) | 1.020 (p 0.004) |
| `atomics` | 1.018 | 1.030 (p 0.006) |
| `shared` | 1.018 | 1.005 |
| `divergent` | 1.009 | 1.004 |
| `warp`, `launch` | 1.013, 1.015 | 1.001, 0.992 |
| `f32_fma` | **0.982** (p 0.000; 0.992 and 0.985 in two earlier windows) | 0.961 (p 0.11) |
| `local` PTX | 0.985 (0.986, 0.989 earlier) | 0.997 |
Every other row 0.99x to 1.01x on both CPUs. The two rows that lose, SASS `f32_fma` and PTX `local` on Zen 5, run unchanged code at equal instruction counts: they are the layout lottery of Round sp14 (an edit inside `exec.cpp` re-rolls the engine's unit, which is pinned to 4096 but not to a particular offset of its hot loops); `lint_hot_unit_layout` passes.

### Tests (CMake-wired, no GPU) and mutations
* `test_sass_mem_rows`: the vector pass against the lane loops for pairs of rows, one row and none, carries from the low word into the high one, a lane in the window at every lane number, on the window's edges and one byte short of it, found by its high word alone, and a window not on a 4 GiB boundary (the scalar path); `full_warp_store_values` against the lane rule for 1, 2, 4, 8 bytes and RZ. `test_sass_lanes`: `fill_lanes` / `lane_ids` for random masks and the whole warp, the other lanes untouched. `test_sass_lea` (existing: carry in all ones, none, random) holds the new specialization.
* Mutants, each built and run (server2), all killed: window high word + 1; the low dwords tested instead of the high ones; `add` not applied; the high row ignored; the 4 GiB test dropped (vector pass for any window); 2-byte store mask 0xff; the high half stored at +1; the 8-byte high row read from +31; RZ stores one lane short; whole-warp fill one lane short; lane ids off by one; whole-warp fill for a partial mask; LEA with the carry-in specialization never taken.
* ASan + UBSan builds of the three unit tests (server3): clean. Release `ctest` on server2: `lint_hot_unit_layout`, `lint_ptx_word_moves_aligned`, `test_sass_*`, `test_tier`, `test_mem*`, `test_memcpy_chk`, `test_ptx_mem_runs`, `test_fastpath`, `test_simd*`, `test_tc_ampere*`, `test_jit*`, `test_exec*`, `test_launch*`, `test_amd_gcn_tier_same_bits` (and `_simd_base`): all passing. Not run: TSan (no threaded code touched), the e2e programs (CI).

### Tried and not kept
* `run_out` / `run_in` (`MemoryManager::load_run` / `store_run`, `src/core/memory.cpp`) four words an iteration: it removes 2 % to 5 % of the instructions of the programs that load coalesced rows (CORR 0.938 against 0.952 without it, `mem` 0.970 against 0.985, `texture` 0.981) and was verified by `test_memcpy_chk` with four mutants, but the cycles do not follow: against the rest of the branch CORR +1.2 %, FDTD-2D +1.8 %, `reduction` +6.9 % (0.5 % fewer instructions), LU -1.7 %, SYRK -0.6 %, transpose -2.1 %, three runs each. The eight-byte stores into the register row land closer together, ahead of the AVX2 kernels' 32-byte loads of the same row (store-to-load forwarding does not work across the pieces), which is the likeliest reason. Kept on the local branch `sp15-fewer-instrs-memrun` (commit a64865f0), not in this branch.
* `tier_dyn` (a constant-bank or uniform source broadcast to 32 lanes, a call of about 37 instructions): inlining its Bank0 case would save half of 0.7 % of the real programs' instructions; not worth the code size.
* `h_local`'s address pass is already vectorized by the compiler (the hot spot there is the lane gather); `iadd3x_k`, `BankTable::worst`, `load_batch`'s lane loop and `batch_addresses_ok` (6 to 8 % of CORR: 9 instructions a lane, no 64-bit compare below SSE4.2) were looked at and left.

### Not done
* `step_warp` is the biggest line item (12 % of the real programs, 47 % of `divergent`) but is spread over the pass bookkeeping (the 32 program counters written at every branch, `popc(group)`, the counters); `divergent` also loses cycles to a store-forwarding stall in `wait()` (byte stores to `wait_kind` read back by a 16-byte load in `waiting_on` at the next BSYNC: the cycle profile puts 12 to 15 % of its samples on those few instructions; not measured further): that belongs with the divwarp work.
* Zen 5 has no `perf` (paranoid level 4, no sudo), so its numbers are wall time only.

## Round sp15: the three changes together (branch `sp15-speed-pr` = `main` feef61a0 + pool cost from busy time + AMD tier as a hot unit + fewer instructions in the memory handlers)

Each change was measured alone against `sp14-combined`; the layout lottery of rounds 12 to 14 means the merged tree needs its own canary. A = `main` (feef61a0), W = this branch, both built Release on the host that ran them (server2 Zen 5 9950X, server3 Zen 4 8840U, load below the gate of 8 / 20 for every timed window, pinned cpu 9 / 5, one thread). `bench.py --ab`, 14 workloads x SASS and PTX (28 rows, `api` and `startup` left out), 15 runs on Zen 5 and 9 on Zen 4, two windows each, with an A/A of A against a copy of its libraries in the same window; `bench-amd`, mi300x, 44 rows, 15 runs, three windows on Zen 5 and one on Zen 4 (gfx942 benchmark binaries copied from server2, a stub `hipcc` that only reports its version). Counters identical in every row of every window (all NVIDIA windows and all AMD windows).

* NVIDIA canary, W/A, window 1 / window 2: Zen 5 `mem` SASS 1.082 / 1.071, `atomics` SASS 1.041 / 1.023, `shared` SASS 1.022 / 1.006, `divergent` SASS 1.011 / 1.015, `launch` 1.008 / 1.002; losses `f32_fma` SASS 0.978 / 0.985 (the row of the previous round that sits in the layout lottery; A/A 0.995 / 1.003) and `local` PTX 0.988 / 0.986 (A/A 1.004 / 1.004); every other row 0.99x to 1.01x. Zen 4: `mem` SASS 1.054 / 1.045, `mma_ptx` PTX 1.039 / 1.057 (A/A 1.048 / 1.055: a placement effect between copies), `atomics` 1.019 / 1.020, `divergent` SASS 1.014 / 1.018, `wmma` SASS 1.012 / 1.011; `local` PTX 0.996 / 0.987; every other row 0.99x to 1.02x. No NVIDIA row is below 0.97 at equal counters. PTX `wmma` on Zen 4 reads 1.014 / 1.014 W/A but 0.950 / 0.948 in the two A/A controls: copies of one library differ by 5 % on that row.
* AMD mi300x, one thread, W/A (Zen 5, three windows; Zen 4 one window): `flat` 1.167 / 1.170 (Zen 4 1.063), `launch_wide` 1.184 / 1.182, `launch_mid` 1.179 / 1.212, `narrow` 1.098 / 1.099, `mfma_bf16` 1.094 / 1.086, `mem` 1.071 / 1.070 (Zen 4 1.038), `p_sdwa` 1.060 / 1.072, `p_dpp_arch` 1.056 / 1.056, `atomics` 1.040 / 1.048, `ldsx` 1.033 / 1.036, `valu_int` 1.030 / 1.028, `divergent` 1.032 / 1.031; Zen 4 `smfmac` 1.085. Rows under 0.97 at equal counters: Zen 5 `mfma` 0.959 / 0.957 / 0.970, `mfma_f64` 0.972 / 0.961 / 0.975, `mfma_f64_4b` 0.962 / 0.986 / 0.970, `p_smem_wide` 0.957 / 0.953 / 0.942, `api` 0.974 / 0.973 / 0.974; Zen 4 `p_dpp` 0.955, `p_dpp_arch` 0.967. The controls say how much of that is the floor: an A/A of two copies of A read `mfma` 0.975, `api` 0.973, `valu_int` 0.945, `p_perm` 0.959, and a copy of W against W read `p_smem_wide` 0.959 and `p_lane` 0.941, so `api` and `p_smem_wide` are not changes. `mfma` and `mfma_f64` on Zen 5 are 1.5 to 3 % under the floor in every window and stay there: that is the cost the previous round's AMD tier change already showed on these rows (`mfma` 1.004 / 0.978 / 0.970 against `sp14-combined` there), not an interaction of the three changes.
* A layout fix was tried and not kept: the offset of the tier unit (`.skip` after the page alignment in `gcn_tier.cpp`) at 2048 and 2560 bytes instead of 1536 (Zen 5, one window each): `mfma` 0.966 / 0.970, `mfma_f64` 0.963 / 0.979 (no better), while `mem` fell from 1.07 to 1.045, `flat` from 1.17 to 1.11 / 1.145, `mfma_bf16` from 1.09 to 1.00 / 1.01, `launch_mid` 1.18 to 1.11 and at 2560 `launch` 0.937 and `p_sbuf` 0.945. 1536 stays.
* Scale (wall medians, interleaved A, W, A copy; Zen 5 7 reps, Zen 4 5 reps): `launch64` at `VGPU_THREADS=4` pinned to one CPU 2.18 s -> 0.21 s (Zen 5) and 2.25 s -> 0.28 s (Zen 4), outputs identical; unpinned (four CPUs) 0.217 / 0.214 s and 0.303 / 0.286 s; `atomics` at 16 threads 0.626 / 0.638 s (Zen 5; A copy 0.623) and 1.086 / 1.071 s (Zen 4); `sortingNetworks` at 16 threads 1.62 / 1.55 s (A copy 1.58) and 4.22 / 4.21 s, at 4 threads 3.57 / 3.57 s and 8.21 / 8.20 s. The Zen 4 box is a low-power 8-core laptop part (8840U), so its 16-thread numbers are 2.5 to 3 times slower than the Zen 5 ones and 16 threads mean 8 cores with SMT.
* Tests, built and run on server2 in the merged tree: `lint_hot_unit_layout`, `test_worker_pool`, `test_sched`, `test_sass_block_reuse`, `test_sass_mem_rows`, `test_sass_lanes`, `test_sass_lea`, `test_amd_gcn_tier_same_bits` (and `_simd_base`), `test_amd_matrix_fast` (and `_simd_base`), `test_amd_matrix_kernels_layout`: 12 of 12 pass.
