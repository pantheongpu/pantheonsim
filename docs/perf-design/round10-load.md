# Round 10, workstream "load": the cost of a kernel's first launch

Rig: AWS c7i.4xlarge (the round-10 rig), nothing of mine on it besides these runs, timing runs pinned to CPU 4 with its hyper-thread
sibling (12) idle, `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1`, `nvidia/rtx5090`, PyTorch 2.14.1+cu130, Release builds in the CI toolchain image.
Base = `speed-round10` unmodified (`~/loadbase/build`); candidate = this branch (`~/load/build`). `uptime` load average during the A/B runs was
about 11 (other workstreams run on their own cores; the pinned core is mine). Raw output: `data/round10/load-*.txt`.

## 1. What happens on a kernel's first launch (read from the code, then confirmed)

`cudaLaunchKernel` (nvidia/src/runtime_api.cpp) -> `module_on_current` -> `registered_ptx` (first use of a registered fatbin) ->
`vgpu::cuda::pick_cubin` (nvidia/src/fatbin_sass.cpp):

1. `extract_elf`: walks the fatbin and **decompresses (zstd or LZ4) the cubin of every architecture** in it (a PyTorch fatbin has sm_75 .. sm_120).
2. picks the best cubin for the device;
3. `sass::unsupported(cubin)` (src/sass/module.cpp): `parse_cubin` (copies every section), then **decodes every instruction of every `.text.` section of
   the cubin** (`sass::decode`, `runs_instr`) to find out whether the executor runs all of it; if not, the whole module goes to the fatbin's PTX. The
   decoded instructions are thrown away. A string for the refusal message was also formatted (`snprintf`) for every instruction, used or not.
4. the cubin goes to `sass::load` (a second `parse_cubin`; one `Code` per `.text.` section, nothing decoded);
5. `Code::instr` (the executor) decodes **one section (one function)** with `std::call_once` the first time any instruction of it is fetched.

So the unit of work is: a whole **fatbin** (one translation unit of the library, 0.7 to 11 MB of cubin, 45 to 700 thousand instructions) for steps 1 to 3,
and one function section for step 5. A new kernel from a fatbin already seen costs only step 5.

Counters (an instrumented build, `mini2.py`, 7 fatbins loaded, base code):

| | count | time |
| --- | ---: | ---: |
| `pick_cubin` | 7 | 0.779 s |
| of which `extract_elf` (decompress all archs) | 7 | 0.128 s |
| of which `unsupported` | 7 | 0.627 s |
| instructions decoded by `unsupported` | 1,702,216 | |
| `sass::load` (of which `parse_cubin` 0.014) | 7 | 0.016 s |
| instructions decoded for execution (`Code::instr`) | 6,768 | 0.002 s |

Execution decodes 0.4% as many instructions as the pre-check does, and the pre-check is 80% of the first-launch cost: **the cost is the whole-cubin
check, not the decoding of the code that runs**. (The 0.05 to 0.2 s per "new kernel" of step 0 are new fatbins, not new kernels.)

Of the 1,702,216 words the check looked at, 252,511 are distinct (per architecture); only 174 distinct 12-bit opcodes occur. A PyTorch process
re-checks the same words over and over, since the libraries' fatbins are built from the same templates.

## 2. Changes kept (each with its A/B)

Micro-benchmarks are in `nvidia/tests/unit/test_decode_same_bits.cpp` (`VGPU_DECODE_BENCH=1`: `sass::decode` over the 41,896 words of the disassembler
corpora, best of 5; `VGPU_UNSUPPORTED_BENCH=<dir of cubins>`: `sass::unsupported` over the 7 cubins of `mini2.py`, dumped from the process, in one process
in order). Spread over repeated runs was 0.2 to 1% on these.

### 2.1 `sass::unsupported` remembers the words it has seen (src/sass/module.cpp)

A per-thread exact set of (architecture, 128-bit word) that decoded and ran. A word in the set is skipped. Only passes are stored, so the first refusal
is found by the ordinary path and its text (section, offset, message) is unchanged. `VGPU_SASS_REFUSE` set turns the set off. The table's
memory is prefetched 16 words ahead; slots are 16 bytes, one table per architecture, at most 2^21 slots (32 MB) each.

| step (unsupported bench, 7 cubins, ms) | total |
| --- | ---: |
| base | 515 to 519 (5 runs) |
| + set of seen words | 113 |
| + prefetch ahead, 16-byte slots, snprintf only on a refusal | 103.5 |
| + direct opcode tables (2.3) | 101 |
| + operand vectors reserved (2.4) | 94 |
| + mods reserved | 91 |
| final, 5 alternating runs | 88.9 to 89.4 |

With every word already in the set (same cubins again in one process) the pass costs 23 ms (12 of them `parse_cubin`): the remaining 66 ms are the 252 thousand
first-time decodes.

### 2.2 `extract_elf_for`: decompress only the cubins the device can run (nvidia/src/fatbin.cpp)

`pick_cubin` can only use a cubin with the same major and an architecture no newer than the device's, so the others are skipped
before decompression. The result of `pick_cubin` is the same; a damaged entry for another architecture was already ignored (it was dropped after failing to
decompress). Instrumented build, same `mini2.py`, 7 picks: `extract_elf` 0.128 s -> 0.029 s (the decompression of the one cubin that is used remains).

### 2.3 Direct opcode tables (src/sass/decode.cpp)

The three `unordered_map`s of decoders are kept as the one list, and `decode` reads an array built from them once instead of hashing. Decode benchmark:
228 -> 215 ns per word (3 runs each: 228, 228, 228 against 214, 215, 215).

### 2.4 Operand vectors reserved (src/sass/decode.cpp)

`dst` (2), `src` (4) and `mods` (4) are reserved when decoding starts: one allocation each instead of three growth steps with the moves of a 120-byte `Operand`.
Decode benchmark: 215 -> 180 ns (dst/src), -> 175 ns (mods). Decode with the final code, 3 runs: 171, 171, 172 ns per word against the base's 225, 225, 227.
`decode + to_text` 512 -> 452 ns.

### 2.5 End to end

`mini2.py`, seconds, 5 alternating runs each (medians; the spread over the 5 was <= 0.003 s on every line):

| step | base | candidate |
| --- | ---: | ---: |
| add (first elementwise kernel) | 0.148 | 0.027 |
| add again | 0.002 | 0.002 |
| mul (new kernel) | 0.093 | 0.016 |
| relu | 0.126 | 0.022 |
| sum reduction | 0.219 | 0.040 |
| softmax | 0.170 | 0.068 |
| layer_norm | 0.076 | 0.043 |
| layer_norm again | 0.024 | 0.024 |

The `nn` sweep group (`VGPU_SWEEP_GROUPS=nn`, `VGPU_SWEEP_TIER=quick`, pinned to CPU 4): base wall 8.76 to 9.53 s (median 8.80) over 5 alternating runs;
candidate 3.49 to 4.27 s. The candidate has two modes (3.5 s in 3 of 11 runs, 4.2 to 4.3 s in the others) that differ in wall time, not in CPU time (`time`: user 2.45 + sys 0.68 against
user 2.80 + sys 0.67): there is a wait in the process of about 0.8 s that happens in one run in two; it is also in the base (user 7.64 + sys 0.74 = 8.4 s CPU against 9.5 s wall)
and I did not chase it. CPU time, base -> candidate: 8.4 s -> 3.1 to 3.5 s. Peak RSS 1,746 MB -> 1,622 MB (cause not investigated).
The checked output lines are identical (same md5 after stripping each line's own timing).

The whole PyTorch quick sweep (`run_pytorch_sweep.sh quick`, 4 cores, sum of the groups' reported seconds, `compile` group failing the same way in both
for want of an Inductor toolchain): base 101 s and 96 s; candidate 53 s and 55 s (alternating, two runs each). Every group has the same ok / known-failure count.

## 3. Candidates dropped (the data does not support them)

* **Lazy decoding per kernel or per instruction.** Execution already decodes lazily, per section (step 5): 6,768 instructions, 0.002 s. The big decode is the
  all-instruction check, whose answer (does the whole cubin run?) decides the PTX fallback of the whole module; making it lazy would change which module falls back.
  Kept as is; made cheaper instead (2.1).
* **Text built only when asked.** After 2.1 `float_text` and `signed_hex` are below 1% of the profile (0.24% in the mini2 profile, 252 thousand decodes). Most of
  `Operand::suffix`/`mnemonic`/`mods` is also read by the executor and tier (`ins.mnemonic == "BAR"`, `o.suffix == ".H0_H0"`), so removing it is a large change for a few percent. Not done.
* **Sharing decoded code across modules.** Only 6,768 instructions are decoded for execution in `mini2.py`; there is nothing to share.
* **Transparent huge pages for the set's table** (mmap + `MADV_HUGEPAGE`): about 90 -> 89 ms in the micro-benchmark and no change in `nn` wall (3.51 s, 4.26, 5.77, 3.58, 3.61 against 3.55 to 3.65 before); dropped.
* **A 4096-entry front cache before the table:** 90 -> 96 ms (worse); dropped.
* **Prefetch distance 16 against 40 words:** 89.1 against 90.4 ms; 16 kept (12 was the first value tried, 103.5 ms in the table of 2.1 before the other changes).
* Not done, small: `parse_cubin` runs twice per fatbin (`unsupported` and `load`, 12 + 14 ms for the 7 cubins of mini2, about 4% of what remains).

## 4. Tests (all in CMake, framework-free, run on hosted CI)

* `test_decode_same_bits` (nvidia/tests/unit): every `Instr` field and every user-visible text (`to_text`, mnemonic, refusal message, `runs_instr`, the `annotate_instr`
  facts) of the 41,896 corpus words on 8 architectures, of 120,000 words with 1 to 3 flipped bits on 5 architectures, and of 40,000 random words on 4, as three 128-bit
  digests recorded from the **unmodified** decoder (80 + 133 + 13 MB of text). `VGPU_DECODE_DUMP=<dir>` writes the full dumps to diff two builds. It passes on the base build too.
  Seen to fail on a deliberate break: the opcode table shifted by one (`k + 1`) fails the flipped and the random pass (and the corpora pass).
* `test_unsupported_memo` (tests/unit): the answer and the exact refusal text (section, offset) with the table warm, for words that run, an unknown opcode, an instruction
  the executor refuses, a word that runs on one architecture and not another, words one bit apart in either half, and `VGPU_SASS_REFUSE` after the words were seen. Each test runs
  on a fresh thread. It passes on the base build too. Seen to fail on: the architecture ignored by the table; the high half left out of hash and comparison; refusals remembered as
  passes; `VGPU_SASS_REFUSE` not turning the set off.
* `test_pick_cubin` (extended): `extract_elf_for` against `extract_elf` filtered by the rule, for 12 compute capabilities; a damaged compressed entry for another architecture
  is left alone. Seen to fail on a filter that dropped the device's own architecture (`<` for `<=`): 5 of 7 tests fail.
* Existing tests that touch this code, all passing on the rig (ctest -j4 on cores 4,5,12,13): test_sass_decode, test_sass_code_at, test_sass_access_count, test_tier,
  test_fastpath, test_pick_cubin, e2e_sass_path, e2e_sass_fastpath, e2e_sass_tier (it runs `VGPU_TIER=0` against `1` itself), e2e_sass_archs, e2e_sass_vector_memory,
  e2e_nvjitlink_sass, bench_counters (`run_bench_check.sh`, the `--check-counters` check) and bench_counters_tier (`VGPU_TIER=1`), e2e_nvrtc_paths, nvrtc_builtins,
  e2e_driver_gaps, e2e_nvjitlink_paths, e2e_nvfatbin_paths, e2e_managed_module, e2e_arch_specific_cubins, e2e_cupti_trace_driver, e2e_cupti_driver_params,
  e2e_graph_driver, e2e_pytorch_sass_sm75. `lint_curl_retries` fails on the base build too (not mine).

## 5. Design only: a persistent cache across processes (not implemented)

What it would store: the verdict of `unsupported` ("every instruction of this cubin runs"), which is a function of the cubin bytes, the architecture, and the decoder/executor code. Decoded
`Instr` are not worth storing: execution decodes 0.4% as many instructions as the check (section 1), and an `Instr` carries heap strings and vectors, so a serialized form would be larger than the cubin it came from (not measured).

* **Key:** a 128-bit or better hash of the cubin image (or of the compressed fatbin entry, which is 8 to 10 times smaller to hash) plus the architecture plus a build id of the
  decoder/executor (any change to `decode.cpp`, `executes()` or `ops.inc` must miss; a version number in a header that those files include, checked by a test that fails when the tables change).
* **Value:** one bit (ok). Refusals are not stored (they are rare and carry text).
* **Where:** `$XDG_CACHE_HOME/pantheonsim/sass-ok-<build id>` as an append-only file of 16-byte keys, read into a hash set at the first `unsupported`; appends are one `write` of a whole key;
  a size cap (say 64 MB) after which the file is not appended to.
* **Invalidation:** by build id (a different build never reads the file); `VGPU_SASS_REFUSE`, `VGPU_SASS=0/1` set: bypass. A user can delete the file at any time.
* **Risk:** (1) a wrong "ok" sends an unsupported cubin to the executor instead of PTX, which fails at launch rather than falling back, so a collision or a damaged or planted file matters:
  use a 256-bit hash, refuse a file not owned by the user or writable by others, and treat any short read as empty. (2) Concurrent processes: an append of <= PIPE_BUF bytes with `O_APPEND` is atomic enough; a torn
  tail is ignored. (3) Hashing costs: in this rig SHA-256 (with SHA-NI, from python's hashlib) runs at 1.8 GB/s: the 43.6 MB of the 7 cubins is 24 ms; the product has no dependencies, so it would need its own hash, and
  portable C++ SHA-256 is several times slower than that (NOT MEASURED); a non-cryptographic 128-bit hash is much faster (NOT MEASURED) but is not collision-safe against a planted file.
* **Expected gain from these numbers:** `unsupported` is 89 ms of the 216 ms the six first launches of `mini2.py` take now (decompression 29 ms, `parse_cubin` 12 + 14 ms, the rest execution).
  With a 24 ms hash, 65 ms (30% of what remains); with a ~5 ms hash, about 85 ms (40%). A second process of the same program would then spend about 130 to 150 ms where the first spends 216, and the base
  spent 830. It does not remove decompression: the cubin is needed at the module load right after the check. NOT MEASURED for `nn` or the sweep.
