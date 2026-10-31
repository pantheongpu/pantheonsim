# Study 6: tier 2, generated C++ for hot straight-line regions (a JIT with the host compiler)

Status: design, 2026-10-22. Nothing here is built. It is route (b) of [study 1](01-whole-kernel-translation.md) and the "tier 2" of
[study 2](02-tiering-and-caching.md), written against the tier as #403 leaves it (on by default, `src/sass/tier*`, `tier_ops.inc`).

**What "JIT" means here.** Not LLVM (the product is zero-dependency) and not an x86 template JIT (study 1: no faster than the closure tier it would
replace, needs `mmap(PROT_EXEC)`). At run time, for a hot code section, generate C++ for its straight-line regions, compile it with the host's
`g++`/`clang++` into a shared object, `dlopen` it, and let `Runner::tier_run` call it in place of the per-instruction loop. Opt-in
(`VGPU_JIT=1`), never needed, and anything it cannot do goes back to the tier, which goes back to the interpreter.

## 1. What it is worth (from the repo's own measurements; nothing new has been measured for this document)

* Study 1's prototype, generated AVX2 code against the AVX2 closure tier (the tier of today): int_virus 1.85x, ffma_loop 4.2x, int_loop 5.5x,
  i.e. 450 to 510 M warp instructions/s against 24 M for the interpreter of the time. Memory-bound kernels stop at about 3x whatever the executor.
* Round 10 (rig, nothing else running): the tier alone gives 1.07x to 3.3x on the harness workloads, and nothing on PyTorch (51 M warp
  instructions in 100 s). The pantheon stress workloads gain 1.14x to 1.92x from the tier and run for 72 to 195 s at full size, which is the
  population a JIT can help. Loop-body coverage of the 14-op subset was, statically, int_virus 100%, pulse_virus 97%, memory_retention_bake 82%,
  scheduler_virus 81%, sfu_stress 69%, memory_bank_thrash 66%, compute_virus 60%, the rest below 56% (dynamic weights NOT MEASURED).
* So: a large gain on ALU-bound stress loops, a small one elsewhere (Amdahl over the covered fraction), none on ML test suites. It is a stress-test
  accelerator, not a general one. That is why it is opt-in.

## 2. Where it hooks in

`Runner::tier_run` (tier_ops.inc) executes the straight-line MOps of a warp's group one at a time: store-gate check, `++w.steps` against
`max_steps_`, the counters (`++st.instructions; st.thread_instructions += lanes; st.inst_by_class[cls] += lanes`, `lanes` = popcount of the
executing mask: the group, or the group narrowed by the guard), then `op.fn(&c, op, ex)`. A region replaces the inside of that loop for a
stretch `[idx, idx+k)` of consecutive straight MOps that all have a handler (`fn != null`) and none of which is a `STG`/`ST` (the store gate).

On entry the loop checks once what it checks per instruction today: `n + k <= 256` (the chunk the scheduler gives a turn), `w.steps + k <= max_steps_`,
`regs_ok` (the program's registers fit the warp's file), no store gate pending. If any fails it runs the stretch the old way, so the budget, the
chunking and the YIELD/BSYNC/spin rules, which live outside the stretch, are untouched.

**Counters must come out identical** (they are part of the same-bits rule): a region knows statically `k`, the class of each op, and which ops are
unguarded. It adds `k` to `instructions`, and `Σ lanes` to `thread_instructions` / `inst_by_class`: `lanes = glanes` for an unguarded op,
`popcount(group & guard mask)` for a guarded one (computed in the generated code from the same predicate rows). `@!PT` ops (`never`) count but do
nothing. Tier statistics (`VGPU_TIER_STATS`) get the same per-op `native` counts.

## 3. The generated code

* One function per region, a plain C ABI: `void region(JitCtx* c, uint32_t group, uint32_t glanes)`. `JitCtx` is a small struct of pointers: the
  warp's register file `R`, predicate and uniform-register storage, the launch's constant bank 0 (pointer and size), the program's pool rows, the
  counters, and an `escape` function pointer. Its layout is versioned; a mismatch makes the library refuse to load.
* **Operands are literals.** The MOp already resolves every source to a register-row offset or a pool row, so the generated statements contain
  `R[7*32 + l]` and the pool row's constant bytes. The only run-time inputs are the bank-0 words (`Dyn::Bank0`): they are read through `c->b0`,
  not baked in (study 1: no measurable cost), so a region depends on instruction words only and one object serves every launch.
* **Native forms** are generated as straight C++ over 32 lanes using the same SIMD kernels the tier and the interpreter use
  (`include/vgpu/simd/simd.hpp`), so integer results cannot differ. The rule of tier.hpp ("a handler is held to the bits of the interpreter's path")
  carries over: the generated source for a form calls the kernel or the shared function that the form's `Impl` table entry names, never a
  re-derivation. Floating point is compiled with `-ffp-contract=off` and explicit `fma` calls, the flags the kernels are validated with.
* **Escape.** A MOp the generator does not cover (but that has a handler) becomes `c->escape(c, &op_i, ex)`, which calls the MOp's existing tier
  handler: total coverage from day one, speed-up by Amdahl over the covered fraction, forms added in order of measured weight (study 1's "region
  granular with a per-instruction escape"). The one thing study 1 could not show is the escape under the real `Runner`; here it is a function
  pointer to the handler the tier already has, so that is the first thing milestone 1 proves.
* Faults: memory accesses stay calls into the checked `MemoryManager` through the escape/handler (never inlined raw loads), so fault text,
  armed-fault hooks and race reports are the tier's. An exception thrown inside the region unwinds through `dlopen`ed frames (compile with
  `-fexceptions`; milestone 1 tests it) and is caught by the same `try`/`catch` in `tier_run` that attaches the faulting instruction.

## 4. The compile, the cache and the promotion

* **Key.** Hash of the region's relocated instruction words, the architecture, the JIT's generator version, the compiler's identity and version
  string, the flag set, and the `JitCtx` layout version. Not the launch arguments (study 2). The hash is 128 bits made of two independent 64-bit
  hashes, and the object also stores the full word list so a load compares the words (a hash collision must never run the wrong code).
* **Where.** `$VGPU_JIT_CACHE`, else `$XDG_CACHE_HOME/pantheonsim/jit/<simulator build id>/`. Written to a temp name and renamed (atomic), a lock
  file per key so two processes do not compile the same region; a corrupt or truncated object is deleted and rebuilt. A read-only or `noexec` cache
  directory turns the JIT off with one notice.
* **When.** Counting is free (the region entry already knows `k`): a section is promoted once its regions have run N warp instructions (study 2's
  break-even: 4 to 74 M for tier 2). Compilation runs in a background thread under `ulimit -v`/`RLIMIT_AS` and a wall timeout, one at a time, never
  blocking a launch; the launch that triggered it keeps running on the tier and the next one picks up the object. A single huge launch is not helped
  (study 2 said so). Regions over ~100 ops are split (compiler memory: 257 MB for 20,000 ops in 100-op pieces against 877 MB in one).
* **Compiler.** `$VGPU_JIT_CXX`, else `c++`, `g++`, `clang++` on `PATH`. None found, or a compile that fails: log once, keep the tier, mark the
  region as not compilable. The published Docker image has no compiler by design, so the JIT is simply off there.
* **ISA.** The object is compiled for the host (`-march=native` is NOT used: the cache may be shared); it is compiled for the level the running CPU
  reports (`-mavx2 -mfma`, and AVX-512 where the kernels have it) and that level is part of the key.

## 5. Risks, and what answers each

| risk | answer |
| --- | --- |
| Wrong bits. | Native forms call the shared kernels; a differential test runs the whole SASS test programs with `VGPU_JIT=1` against the tier (as `run_sass_tier.sh` does for the tier), results and every counter; mutation checks for each generated form; the corpus of study 1's gate (special values, half-empty warps, masked writes). |
| New attack surface: a compiler is run on text derived from a cubin, the result is loaded. | The source contains only numeric literals and fixed templates, never a string from the cubin (study 1). The cache directory is created `0700`, owned by the user; objects are verified (words, version) before `dlopen`. SECURITY.md already says the simulator is not a sandbox; this is stated there too. |
| Compile cost on a cold CI job. | Opt-in; the cache directory can be a CI cache; promotion threshold keeps short runs on the tier. |
| Memory and CPU of the compiler on shared machines (the 23 GB VM has been OOM-rebooted). | One compile at a time, `RLIMIT_AS`, regions capped, `nice`. |
| Hosts without `dlopen`/a compiler/exec permission. | Everything degrades to the tier. |
| Sanitizer builds. | A sanitized simulator would `dlopen` unsanitized objects: the JIT is off under ASan/TSan builds (as for the tier's `VGPU_SIMD`), and the differential test runs with it on in the normal build only. |

## 6. Milestones

1. **Skeleton and the escape, no codegen of arithmetic.** `VGPU_JIT=1`, region discovery in `Code::tier_program`, the generator writing a region
   that calls `c->escape` for every op (so it is slower than the tier on purpose), compile/cache/`dlopen`/promotion, exception unwinding, counters
   identical. Proves the ABI, the counters, the faults, the cache and the failure modes before any arithmetic is generated. Gate: the whole tier
   differential suite passes with `VGPU_JIT=1`.
2. **The ALU subset**: MOV, IADD3 (plain and with carries), IMAD (plain, WIDE, HI), LOP3, LEA, SHF, ISETP, FFMA/FADD/FMUL, SEL, PRMT, IMNMX, I2FP,
   S2R from the tier's `Form`s, guards and masks included. Gate: study 1's special-value test per form, SASS == tier == JIT.
3. **Measure**: harness workloads and the pantheon stress workloads, tier against JIT, with the dynamic coverage (VGPU_TIER_STATS-style) that study 1
   did not have. Keep the JIT only if it wins outside the noise on something that matters; otherwise write the negative result down.
4. **Memory forms and more** in order of measured weight (LDG/STG 32-bit through the checked calls, shared, FSETP/FSEL, MUFU, packed half), each
   with its test.

## 7. What this study does not settle

* Dynamic weight of the covered forms in the stress workloads (static loop-body coverage only).
* Whether a compile in a background thread starts early enough to help launches of a few seconds.
* The cost of `-fexceptions` frames through generated code, and AVX-512 benefit (the part study 1 measured on has none).
* How #403's tier and this layer interact if #403's handlers change after review; this study is written against #403's `224874ea`.
