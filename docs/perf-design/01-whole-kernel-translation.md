# Study 1: whole-kernel translation without a dependency

**Question.** ARCHITECTURE.md (D1, lines 91-95) names `PTX -> internal IR -> LLVM JIT` as the upgrade path
and TODO.md:1555 still lists it first. The project is zero-dependency C++20 (ARCHITECTURE.md:65), so LLVM is out for
the product build. What zero-dependency routes exist, how fast could each go, what do they cost to cover the
instruction set, and what in the engine's semantics limits them?

**Recommendation.**

| route | verdict |
| --- | --- |
| (a) closure compilation ("decode once, run an array of pre-bound micro-ops") | **DO**, as a tier in front of the existing interpreter, convergent warps only, with a per-instruction fallback to the existing code |
| (b) generate C++ and compile it with the host compiler, cached on disk | **DO LATER**, opt-in, after (a); only where a compiler exists |
| (c) x86-64 template JIT | **DO NOT** |

Everything below is measured on the prototype in branch `r7-design-proto` (head `040114fc`, `tools/r7proto/`), a
throwaway that is not for merging, unless marked NOT MEASURED. Raw output is in `data/`.

## 1. What was measured, and how

The harness loads a cubin onto a simulated RTX 3060, then runs the same launch through four executors that each
get their own copy of identical device memory, alternating (base, proto, genr, genf, base, ...) for 9 repetitions
(`data/prototype-sweep.txt`, host `uptime` load 10-12 on a 20-vCPU i9-12900K shared with other agents; no
AVX-512 on this part):

* `base`: the real engine, `sass::launch` from `build/libvgpu.a` of origin/main `443e79c0` (Release, `-O3 -flto`,
  baseline x86-64, i.e. SSE2), `VGPU_THREADS=1`.
* `proto`: route (a). `engine.cpp` decodes the kernel's `Instr`s once into 72-byte micro-ops
  (`sizeof(MOp)`; `sizeof(Instr)` is 176 bytes plus two vectors of 96-byte `Operand`s with strings), each a function
  pointer plus row indices into a per-warp array of 32-lane rows. Constants (immediates and the launch's bank-0
  words) are pre-broadcast rows, so every source is "a row". Basic blocks start with a header micro-op that adds
  the block's precomputed counters. It was compiled twice: baseline SSE2 flags, and `-mavx2 -mfma` (the "AVX2" rows).
* `genf` / `genr`: routes (b) and (c)-bound. `codegen.cpp` turns straight-line runs of ALU micro-ops into C++ and
  `g++ -O2 [-mavx2 -mfma] -ffp-contract=off` builds a shared object that is `dlopen`ed. `genf` emits one 32-lane
  loop per run with values in registers (what a source-level generator would emit). `genr` emits one loop per
  instruction with the row array round-tripped through memory and a compiler barrier between them: it is the
  best case for a template JIT that has no register allocation.

Same-bits gate. After the timed launches the harness compares (i) every output buffer byte for byte and (ii)
**every counter word of `LaunchStats`** (`kCounterWords`: instructions, thread_instructions, the class mix, global
loads/stores/bytes/sectors/requests...) against `base`. Results:

* int_virus (a real pantheon kernel, extracted from a binary built from the public pantheon sources), ffma_loop, int_loop,
  vecadd, saxpy: all four executors identical in memory and counters.
* `ops_int` (16 integer expressions, including negated addends), `ops_wide` (IMAD.WIDE signed and unsigned),
  `ops_shift` (funnel shifts, shift counts past the width, 64-bit shifts), `ops_float` (FFMA/FADD/FMUL with abs and
  negation), `ops_cvt` (I2FP): inputs are 36 special values (0, INT_MIN, INT_MAX, UINT_MAX, signed zeros, subnormals,
  infinities, quiet and signalling NaNs with payloads, 2^23 and 2^31 boundaries) mixed with random bits, run with 48-thread blocks so the
  second warp is half empty (masked writes). All identical.
* The gate can fail: with a deliberately wrong FFMA (`a*b+c`, two roundings; `-DR7_BREAK_FFMA`) `ops_float` reports
  `same bits NO`.

Not covered by that gate (NOT MEASURED): divergent control flow (a prototype branch that would split a warp throws;
`ops_pred`, a plain `if` with stores, compiles to BSSY/BSYNC and was refused), barriers, shared/local memory,
atomics, 64-bit or vector memory forms, FP modes (rounding, ftz, sat: refused), half precision, tensor cores.

## 2. Where the time goes today

`perf record -e cpu-clock` works here (hardware counters do not; the perf binary is the extracted linux-tools
copy in `~/.cache/perf-tools`). Baseline, int_virus (`data/perf-baseline-intvirus.txt`):
`Runner::fast_alu` 60.5% self + 13.1% (its operand-staging lambda for modified sources) + 2.5% + 2.1% = 78%,
`execute` 6.9%, `step_warp` 4.9%, `straight` 2.0%, `read32_slow` 1.7%, libgcc `__popcountdi2` 1.9% (baseline x86-64 has no
POPCNT). vecadd (`data/perf-baseline-vecadd.txt`): flat, 25 functions: `MemoryManager::resolve` 14.7%,
`write64` 6.4%, `exec_int` 5.5%, `count_access` 4.7%, `scalar_location` 4.3%, `exec_mem` 4.3%+2.4%, `execute` 3.8%, ...

Per warp instruction (base, `data/prototype-sweep.txt`, AVX2-binary section, load ~11):

| kernel | base | closure, SSE2 | closure, AVX2 | generated, AVX2 | generated, per-op round trip (template-JIT bound), AVX2 |
| --- | --- | --- | --- | --- | --- |
| int_virus (100-instruction loop body) | 24.1 M/s = 41 ns | 122 M/s (5.0x) | 244 M/s (10.1x) | 450 M/s (18.7x) | 177 M/s (7.4x) |
| ffma_loop (8 FFMA chains) | 13.7 M/s = 73 ns | 13.6 M/s (1.6x) | 116 M/s (8.4x) | 484 M/s (35.4x) | 210 M/s (15.4x) |
| int_loop (LCG + xorshift + rotate) | 15.7 M/s = 64 ns | 40 M/s (3.3x) | 93 M/s (5.9x) | 508 M/s (32.3x) | 197 M/s (12.5x) |
| vecadd (2 loads, add, store) | 2.6 M/s = 385 ns | 8.1 M/s (3.0x) | 8.3 M/s (3.2x) | 8.4 M/s (3.3x) | 8.3 M/s (3.2x) |
| saxpy (grid-stride) | 2.5 M/s = 400 ns | 6.1 M/s (2.4x) | 6.3 M/s (2.5x) | 6.5 M/s (2.6x) | 6.5 M/s (2.6x) |
| vecadd, contiguous warp accesses through `load_run`/`store_run` | | 12.4x | 15.1x | 17.3x | 15.8x |
| saxpy, same | | 12.2x | 17.0x | 21.2x | 18.7x |

(Spread: min-max per row is in the raw file; the baseline column varies about 10% run to run with the other
agents' load; the speed-ups are per-repetition ratios of adjacent runs. The SSE2 section of the sweep has its own
base column and gives similar ratios.)

Reading the table:

* The three ALU kernels are bound by interpretation overhead, not by lane arithmetic: removing the per-instruction
  decode of operand kinds, the staging copies, the `step_warp` group search and the counter bookkeeping gives 1.6-5x
  *with the same ISA*. The 1.6x on ffma_loop is because both executors call libm's `fma` per lane on a baseline
  x86-64 build (TODO.md:1431 records the same trap for the PTX path; the SASS FFMA path in `fast_alu`,
  exec_ops.inc:1252, calls `std::fma`, which links to libm's `fmaf`: `nm -u` on the harness shows `fmaf@GLIBC`): with `-mfma` the closure tier jumps to 8.4x.
* SSE2 has no 32-bit vector multiply (`pmulld` is SSE4.1), so on a baseline build even generated code is no faster
  than the closure tier on IMAD-heavy int_virus (122 vs 122 M/s). A shipped tier needs runtime ISA dispatch as
  `wmma` already has, or an x86-64-v3 build. On aarch64 NEON has the multiply (NOT MEASURED).
* Memory-bound kernels stop at about 3x whatever the executor, because most of their time is the per-lane scalar
  access path (vecadd in the closure tier: 32.3 ms, of which 24 ms disappears when the same accesses go through the bulk calls) (32 `load_scalar` calls per warp access, each a chunk lookup and a relaxed atomic) and the exact
  coalescing counters (`count_access` sorts two heap vectors per access, exec_ops.inc:1782-1828). The prototype's
  `count_global` produces identical counters with a contiguous-warp shortcut. The last two rows show what
  `MemoryManager::load_run`/`store_run` would give: 12-21x. **Caveat:** `run_allocation` declines when any access
  fault hook is attached (memory.cpp:1075) and `Device::install_fault_hook` attaches one to every device
  (runtime.cpp:1019-1024), so in production those calls return false. Making them check "armed" instead of
  "attached" is an r7-memory item and is a precondition for the last two rows; my harness runs with no hook, for
  both sides of every comparison.
* For the ALU rows the generated code costs 1.5-2.5 ns per warp instruction, i.e. it is at the lane-arithmetic floor
  for AVX2 (4 ymm operations per 32-lane op, ~0.5-1 ns at 5 GHz, plus loop and mask overhead). That is the ceiling
  of any route here for these kernels; ffma_loop's generated code runs at 22.6 GFLOP/s on one core
  (65.5 MFLOP in 2.9 ms), against 0.63 GFLOP/s today. The host's nominal single-core AVX2 FMA peak is about
  150 GFLOP/s (arithmetic from the part's specification, NOT MEASURED).

## 3. Route (a): closure compilation

**Shape.** `compile()` walks the reachable code from the entry, splits it into basic blocks, and emits one micro-op per
instruction with handler chosen from the op's decoded form (e.g. `IADD3` without modifiers gets `a+b+c`, with them
the mask form; `LOP3` with LUT 0x96 gets a three-way xor; a source that is a bank-0 word or an immediate becomes a
constant row). Execution is `for (p = ops; p; p = p->fn(ctx, *p))`. Guarded instructions run through one
trampoline that sets the executing mask to `ex & guard` and counts that popcount.

**Cost to build.** 40-165 us per kernel for 32-136 instructions in the unoptimised prototype (std::map/set
constant tables); the decoder alone costs 0.36-0.41 us per instruction (`dec` measurement, once per section,
lazily, `Code::instr`, sass/exec.hpp). Break-even is therefore about 5,000 warp instructions executed, i.e. 0.2
ms of baseline interpretation (see study 2).

**Cost to cover the ISA.** `include/vgpu/sass/ops.inc` has 156 ops. The fully general interpreter is
`exec_ops.inc` 4,979 lines plus 3,944 lines of `decode.cpp`. The prototype covers about 14 ops (IMAD incl. WIDE,
IADD3, LOP3, SHF, LEA, ISETP, FFMA, FADD, FMUL, I2FP, MOV, S2R/S2UR/ULDC for CTAID/TID/bank-0, LDG/STG 32-bit
`.E`, BRA, EXIT, NOP) in 1,006 lines of handlers and compiler plus 323 of C++ generator. Coverage of a real corpus, the
46 sm_86 cubins of the 47 pantheon workload binaries (144 kernels, 27,512 instructions, `data/coverage-pantheon-workloads.txt`):

* whole-kernel, all-or-nothing: **3 of 144 kernels (2.1%)** compile. This is the number that matters for the
  design: one unsupported instruction anywhere (a divergent `if`, a SEL, a vote) rules out a kernel.
* per instruction: **59.5%** of non-NOP static instructions and 57.7% of the instructions inside loops
  (backward-branch ranges, 71.6% of all code) are inside the 14-op subset.
* per workload, loop-body instructions inside the subset: int_virus 100%, pulse_virus 97%, memory_retention_bake
  82%, scheduler_virus 81%, sfu_stress 69%, memory_bank_thrash 66%, compute_virus 60%, atomic_virus 56%,
  tensor_virus 50%, omni_virus 45%, compute_virus_agg 33%, transformer_virus 19%, mma_virus 6%, fp64_virus 5%.
  Static, not dynamic; dynamic weights were NOT MEASURED.
* what is missing, by instruction count: HFMA2 1,970 (7%), IMAD/IADD3/ISETP/LOP3 forms the subset refuses
  (carry-out predicates, `.X`, neg/not on multiplicands, 64-bit compares: about 2,300 together), UIADD3-as-NOP 712,
  HMMA 480, STL 290, SEL 286, LDG/STG forms 523, BSSY+BSYNC 504, POPC 204, FSEL 180, VOTEU 160, MUFU 154, F2I 143, FSETP 140,
  I2F 139, CALL 137, DFMA 128, RED 124. A long tail, as the corpus in `nvidia/tests/data/sass/*.txt` (41,888 lines
  over 11 targets, built to test the decoder) also shows.

**Consequence: the tier must be region-granular, with a per-instruction escape to the existing code.** A micro-op
for an unspecialised instruction would hold the `Instr*` and call `Runner::execute` for that one instruction on
the warp's own state. Then coverage is total from day one, speed-up follows Amdahl over the specialised fraction,
and ops are specialised in order of measured weight. This needs the compiled tier to use `Warp`'s own storage
(`Warp::r` is already `r[reg*32+lane]` rows, exec.cpp:65; the prototype's layout is the same plus constant
rows), and the escape must run under the same `try/catch` that builds the fault text (exec_ops.inc:208-220).
The prototype could not show the escape (the `Runner` is private to exec.cpp); this is the main thing
a production implementation has to prove.

## 4. Route (b): generated C++, host compiler

* Speed: 1.85x (int_virus), 4.2x (ffma_loop), 5.5x (int_loop) faster than the AVX2 closure tier; ISA matters: with
  SSE2 only, generated code is 1.0x the closure tier on int_virus, 1.6x on ffma_loop, 4.3x on int_loop. `-O2` is
  required: int_virus at `-O0` 2.2x, `-O1` 4.6x, `-O2` 17.8x, `-O3` 17.2x over base.
* Compile cost (`data/gcc-compile-times-synthetic.txt`, synthetic regions of the same statement shapes, wall seconds,
  `-O2 -mavx2`): ~0.07 s floor for 100 ops (process start and link), 0.21 s for 1,000 ops in one region (0.30 s as ten
  regions), 0.85 s for 5,000, 4.9 s for 20,000 (877 MB peak RSS in one region; 4.1 s and 257 MB split into
  100-op regions). About 0.2 ms per generated op at `-O2`. The real kernels above took 0.10-0.50 s each.
* Launch constants do not need to be baked in: passing bank-0 words as a `K[]` argument cost nothing measurable
  (int_virus 17.0x with literals, 16.8x with parameters), so the generated object depends on the instruction words
  only and is cacheable (study 2).
* Availability: the GitHub Action installs `g++` because it builds the simulator (action.yml:98). The published
  Docker image has none by design (docker/Dockerfile:3-4, docs/docker.md:30). The playground container is
  unknown to me. A cold CI job pays the compile every time unless the cache directory persists.
* Risk: code generated from an untrusted cubin is compiled and loaded into the process. Mitigation: the source
  contains only numeric literals and fixed templates (no cubin string reaches the compiler). SECURITY.md already
  says the simulator is not a sandbox, but this is new attack surface (a compiler invocation, a writable cache,
  `dlopen`), and read-only root filesystems or `noexec` temp dirs disable it. Memory accesses must stay calls into
  the checked `MemoryManager`, as the prototype's do.

## 5. Route (c): x86-64 template JIT

* The per-op round-trip generated code (`genr`) is the upper bound for a template JIT that does not allocate
  registers: 0.73x, 1.8x and 2.1x of the AVX2 closure tier on the three ALU kernels. It is not clearly faster than
  route (a), and 2.3-2.6x slower than route (b) on the three loops.
* Requires `mmap(PROT_EXEC)` or `mprotect` flips (W^X: write, then flip to RX; blocked by SELinux `execmem` denials, some
  hardened container profiles and macOS-like policies), an encoder for every op and form, x86-64 only (the project
  also targets hosts that are not x86: ARCHITECTURE.md assumes little-endian 64-bit, not x86), and its own debugging story
  (no symbols for perf, no source text to read). There are no `PROT_EXEC` mappings in the tree today (grep of `src/`,
  `include/`, `nvidia/src/`).
* Maintenance: each op exists three times (interpreter, template, and the test oracle).
* Verdict: **do not**. Route (a) gets most of the benefit portably; route (b) gets the rest with a real optimiser.

## 6. Semantics that constrain every route

All of these were read from code on origin/main `443e79c0`; the compiled tier must preserve them or stay out of the way.

1. **Independent thread scheduling.** Each lane has its own pc (`Warp::pc[32]`, exec.cpp:69); `step_warp`
   (exec.cpp:1198) runs the group of runnable lanes at the lowest pc, with `slice`, `give_way`, `ws_held` adjustments
   for YIELD, spin and WARPSYNC (exec.cpp:1200-1290). A compiled run is valid only when the group is the whole
   runnable set, `slice == 0`, `give_way == false`, `ws_held == 0`, and nothing waits. At a run's exit every lane's pc
   must be set (`w.pc[l] = pc`, as exec.cpp:1305-1307 does). Divergent warps stay in the interpreter; in the
   real kernels that is any region after a BSSY until the matching BSYNC releases.
2. **BSSY/BSYNC/BREAK/WARPSYNC/BAR/YIELD/NANOSLEEP and everything that waits or wakes others** end a run
   (`straight()`, exec.cpp:1314-1345, lists them, plus SHFL/VOTE/MATCH/REDUX/ATOMS/SYNCS). They have complex state
   (`b[16]`, `bssy_stamp`, `wait_kind`) and the standoff breakers (`break_bsync_standoff` exec.cpp:957,
   `break_warpsync_standoff` :989, `break_spin_wait` :1166) read it from outside the warp. They stay interpreted.
3. **Spin breaking** is driven by `spin_yields`, `give_way`, `w.spins >= 64` (exec_ops.inc:3239) and the
   4096-step warp turn in `run_block` (exec.cpp:1070-1095). Warp interleaving inside a block is therefore a
   deterministic function of step counts: one `step_warp` call is up to 256 straight instructions plus one more
   (exec.cpp:1300), and a turn is up to 4096 such calls. A compiled run must be chopped at the same 256-instruction
   boundaries and report the same number of "steps" for kernels whose result depends on that interleaving (races,
   flag spins). For race-free kernels the order is unobservable. The prototype does not chop (NOT MEASURED).
4. **Step counts are observable**: `S2R SR_CLOCK*` returns `w.steps` (exec.cpp:632-637) and `max_steps` aborts with
   a message naming the count (exec.cpp:1297). The prototype adds each block's instruction count at its header and
   checks the cap there; it would stop up to one block early or late on an infinite loop, and a clock read inside a
   block would need the exact count (not implemented).
5. **Counters** (`execute`, exec_ops.inc:133-148) add per instruction: instructions, active-lane counts per class,
   tensor, barriers, divergent branches; memory counters come from the addresses (`count_access`). The prototype's
   precomputed per-block counts reproduce all of them exactly for the kernels above, including predicated-off lanes;
   guarded instructions count their own popcount.
6. **Diagnostics.** A fault reports kernel, warp, the first lane of the group, `pc` and the disassembled instruction
   (exec.cpp:`fault`, exec_ops.inc:208-220). A compiled op needs its `pc` (the prototype stores it) and a catch
   around the dispatcher that rebuilds the same message; the lane reported is `countr_zero(group)`, not the lane that
   faulted, so it is cheap to reproduce.
7. **Debug and trace modes** (`VGPU_SASS_TRACE`, exec_ops.inc:223, the `vgpu debug` single-thread mode, `--race`)
   must select the old path; the tier should be keyed off those at launch.
8. **Threads.** `VGPU_THREADS=1` keeps the strict block order (launch.hpp:288-294; exec.cpp:879). A compiled tier is
   per launch and shared read-only by workers; block order is unchanged. Thread scaling already works well
   (int_virus base: 1/2/4/8/16 threads = 0.40/0.20/0.10/0.079/0.052 s wall at load ~14), so single-core speed multiplies.
9. **Fault injection and memory hooks.** Loads and stores go through `MemoryManager::load_scalar/store_scalar`, which
   consult the access-fault hook. Compiled code must keep calling them (or the guarded bulk forms). ALU fault injection
   is PTX-only today (`alu_fault_armed`, interpreter.cpp:3730, 4488...; no use in src/sass), so SASS is unaffected.
10. **FP exactness.** `-ffp-contract=off`, explicit `fmaf`, the engine's NaN canonicalisation (`0x7fffffff`),
    abs/neg as bit operations. No FTZ/DAZ (TODO.md:1496 records the rejected attempt).

## 7. Staged plan

1. Tier-1 closure compilation of straight-line runs inside convergent warps, escape to `execute` for the rest;
   specialise by measured weight (static order above: IMAD, ISETP, LOP3, IADD3, FFMA, BRA, SHF, UIADD3, LEA, MOV, S2R,
   LDG/STG, FMUL, SEL...). Gate: `VGPU_TIER=0` (name illustrative) keeps the old path; a unit test in the style of
   test_fastpath that runs every form through both paths over arithmetic-breaking inputs *and* compares all counters;
   the daily `sass-ptx` digest compare and the card-oracle suites stay as they are.
2. Make `load_run`/`store_run` usable with the fault hook attached, add contiguous-warp memory ops and the cheaper
   exact `count_access` (the memory agent's area): that is the only way memory-bound kernels leave the 3x ceiling.
3. Runtime ISA dispatch for the lane kernels (AVX2/FMA, same-bits tested against SSE2), POPCNT.
4. Only then consider route (b) as `VGPU_JIT=cc` for long ALU-bound runs (study 2).

**Risks.** Escape-hatch state coherence (the biggest); divergence-heavy kernels get nothing; a second implementation of
every specialised op doubles the places arithmetic can be wrong (the gate above is what contains this); memory
growth is small (72 bytes per micro-op against 176+ for `Instr`).

**Not done / not measured.** The PTX engine (VGPU_SASS=0, the fallback for PTX-only modules; its measured rates are in
TODO.md:1437-1446) has the same structure but was not prototyped. aarch64, AVX-512. Dynamic instruction mix. Divergent
and barrier kernels. The escape path. Any claim about speed-ups on real full-size pantheon runs: only the ratios above.
