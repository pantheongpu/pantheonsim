# Study 2: tiering and caching

**Question.** When does a compile pay for itself, what should key a cache, and what invalidates it?

**Recommendation.** **DO** tier-1 (closure compilation, study 1a) with a launch-time heuristic that is almost always
"compile at the first launch"; **DO LATER, opt-in** tier-2 (generated C++, study 1b) with an on-disk cache keyed on the
relocated instruction words, never on launch arguments. All numbers are from the r7-design-proto prototype
(`040114fc`) and `data/`; they are prototype numbers, not production ones.

## 1. Break-even

Definitions: `T_c` is the one-time cost of building a tier, `s` the time saved per warp instruction relative to the
tier below, so break-even is `T_c / s` warp instructions executed on that tier (summed over blocks and launches).

| tier | `T_c` | `s` vs interpreter | break-even |
| --- | --- | --- | --- |
| 0 -> 1 (decode, already paid: 0.36-0.41 us per instruction, once per section) | ~50 us for 100 instructions | | |
| 1 closure | 40-165 us per kernel of 32-136 instructions (prototype, measured: `compile` in the harness output; 0.4-1.5 us per instruction, unoptimised) | int_virus 41 -> 8.2 ns SSE2 (33 ns), ffma_loop ~0 on SSE2 / 73 -> 8.7 ns with FMA, int_loop 64 -> 25 ns, vecadd 385 -> 123 ns | **~5,000 warp instructions** in the worst case measured (165 us / 33 ns), i.e. about 0.2 ms of interpretation. A kernel that runs 40 warps of 136 instructions once already pays |
| 2 generated C++, `-O2 -mavx2` | 0.10-0.50 s for the real kernels measured (process start about 0.06 s; synthetic 1,000 ops 0.21 s, 5,000 ops 0.85 s, 20,000 ops 4-5 s) | 2.2 ns vs closure's 4.1 ns on int_virus (1.9 ns saved); 2.1 vs 8.7 ns on ffma_loop (6.6 ns); 2.0 vs 10.8 ns on int_loop (8.8 ns) | against the interpreter 41 -> 2.2 ns on int_virus: 0.14 s / 39 ns = **3.6 M warp instructions (~0.15 s of interpretation)**. Against tier 1: 0.14 s / 1.9 ns = **74 M warp instructions (~0.3 s of tier-1 execution)** on int_virus, 0.2 s / 6.6 ns = 30 M on ffma_loop |

So tier 1 should be unconditional (the heuristic below is only about huge cold code), and tier 2 is worth building
only for kernels that will execute hundreds of millions of warp instructions. That is the full-size pantheon runs:
TODO.md:1457-1463 records 72-195 s at full size for int_virus, memory_bank_thrash, pulse_virus, omni_virus and
memory_retention_bake, and the 360 s watchdog for fp64_virus (mma_virus 186 s). At 40 ns per warp instruction those are billions
of warp instructions; a 0.3 s compile is noise for them and for nothing shorter than a second.

Counting executed instructions is free: the block header micro-op already adds `n_total` to a per-warp counter
(`c.steps`); a per-kernel atomic accumulating those at block end would drive promotion. A cap on `T_c` is needed for
huge kernels: omni_virus has 5,447 non-NOP instructions (about 1.1 s of `-O2` at 0.2 ms per op) and a 20,000-op region
needs 877 MB of compiler RSS, so regions should be split (100-op regions needed 257 MB for the same 20,000 ops) and
the machine's memory rule (shared 23 GB VM that has been OOM-rebooted) respected: run the compiler under
`ulimit -v`, one at a time, in a background thread while the interpreter keeps going.

Promotion must not change anything observable. A kernel that is promoted while a launch is running would have to
switch at a block boundary (blocks are independent and the counters are per block, so this is safe) and never
inside a warp's turn. Simplest rule: decide at launch; the compile happens in the background for the *next*
launch, so a single huge launch is not helped (NOT MEASURED how often that matters).

## 2. Cache keys

For tier 1 there is nothing to cache across processes: building takes tens of microseconds and the structure is
per launch only because it folds launch constants. It could be built once per (module, kernel) and bound per launch
(the prototype keeps bank-0 words as "launch constant" rows to make this possible; for tier 1 the cost of rebuilding
is below the cost of a hash lookup).

For tier 2 the key is the **generated object's inputs and nothing else**:

1. The kernel's code **after relocation**: `sass::rebind` re-applies relocations in "a constant bank, a variable's
   initialiser or the code" (sass/exec.hpp comment), so the instruction words can change between loads of the same
   cubin. Hash the words, not the file.
2. Reachable callee sections (`sass::reachable`), because a CALL target's code is compiled with it.
3. `sm` (decode differs by architecture: `Instr::sm`).
4. Generator version and options: region splitting parameters, the set of specialised ops.
5. The compiler: `g++ --version` text and the exact flag string, including the ISA flags used (`-mavx2 -mfma`). Cache
   entries compiled with `-mavx2` must not be loaded on a host without AVX2; store the required CPU features in
   the entry and check them with `__builtin_cpu_supports` before `dlopen`.
6. The region function ABI version.

**Not** in the key: the grid, block shape, arguments, bank-0 contents. Measured: passing bank-0 words (kernel
arguments, `blockDim`, `gridDim`) as a `const uint32_t* K` parameter instead of baking them as literals costs nothing
measurable (int_virus 17.0x over base with literals, 16.8x with parameters; ffma_loop 32.5x vs 29.4x, a 10% difference that is the size of
the load-induced noise on the base column). Without that, an argument that changes every launch (a pointer from a fresh allocation, a
loop count) would defeat the cache.

Operationally: the entry is `<hash>.so` written to a temp name and `rename`d (atomic), in `$VGPU_CACHE_DIR` or
`$XDG_CACHE_HOME/pantheonsim` (hypothetical names), a lock file so that concurrent processes compile once, a size cap
with least-recently-used eviction, and a "compilation failed" negative entry so a broken toolchain is not retried
on every launch. On CI the cache directory must be restored with the build cache the Action already keeps
(action.yml caches the simulator build against a hash of the source and toolkit version): the same step can
restore it.

## 3. Invalidation

* Any key component changing is automatic invalidation (the hash differs); there is no in-place invalidation.
* A simulator source change that alters generated code must change the generator version string; so the version
  should be the hash of `codegen.cpp` and the template header at build time.
* A compiler upgrade is covered by key 5.
* Corruption: verify the entry on load (a recorded size and a content hash of the `.so`); on failure delete it and
  fall back to tier 1. Never trust a cache directory writable by other users (the loaded object runs with
  the process's privileges); require the directory to be owned by the user with mode 0700.
* Security: SECURITY.md says the simulator is not a sandbox, which does not make a shared writable cache of
  loadable code acceptable.

## 4. Where tier 2 cannot run

The published image has no compiler (docker/Dockerfile:3-4); read-only filesystems and `noexec` mounts stop
`dlopen` of a generated file; unknown for the pantheonsim.com playground. The tier must therefore be strictly
optional: absent compiler, failed compile, missing AVX2, or any error is silent fallback to tier 1 and one
stderr line at most (the project's rule is that an unimplemented thing says so once, never a wrong answer).

## 5. What I did not measure

Promotion heuristics on real workloads (no end-to-end integration exists, only the harness); cache hit rates
across a CI matrix (profiles differ in `sm`, so a matrix of GPUs compiles a kernel once per architecture, not per
profile); the cost of the first launch of a 5,000-instruction kernel through tier 1 (the prototype's compile is
not exercised on one that large because no such kernel is wholly inside the subset).
