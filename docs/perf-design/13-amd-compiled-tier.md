# Study 13: a compiled tier for AMD kernels

Status: design, 2026-11-14, with its first slice built: **blocks** (`VGPU_AMD_BLOCKS=1`, off by default; section 9). Measurements, tests and
mutants are in *Round sp72* ([rounds/11-rounds-sp36-sp41.md](rounds/11-rounds-sp36-sp41.md)); scripts and data in `data/sp72/`. Everything was
measured on server3 (Ryzen 7 PRO 8840U, 8 cores / 16 threads, g++ 13.3, Release with LTO), one host thread unless a line says otherwise, on a
machine shared with other sessions.

**What "compiled tier" means here.** The AMD engine has two levels today: the interpreter (`amd/src/gcn_exec.cpp`) and the pre-decoded tier
(`amd/src/gcn_tier.cpp`, on by default: each instruction's handler and operand classes are worked out when it is decoded, and a handler works on
whole register rows with AVX2 lane kernels). The tier still dispatches one instruction at a time: `run_round` calls `step`, which finds the
instruction (`fetch_next`), asks the wave's state, calls the handler through a pointer, and counts. A compiled tier runs a stretch of a kernel
without coming back to that loop. Two ways to make one were compared:

* **(A) a closure tier**: pre-decoded straight-line runs whose handlers are called back to back, operands bound when the run is built, counters
  added once. In-process, nothing but the simulator's own code runs, no compiler, no executable memory. Slice 1 (blocks) is its first layer.
* **(B) generated host code**: C++ written per run, compiled by the host's compiler in a forked process, loaded with `dlopen`, cached on disk.
  This is what the NVIDIA side's tier 2 does (`VGPU_JIT=1`, [study 6](06-tier2-jit.md), rounds 11 to 15 and sp44 to sp64).

The short answer: build (A) first and all the way, because it is what a program of many short processes can use (inq's test suite is 244
processes of 3 to 60 s), it can become the default, and it is the frame (B) needs anyway. Then build (B) as an opt-in layer that replaces the
*fusable runs inside a block*, for programs that run long enough to pay for a compile: that is where the large factor is (section 2).

## 1. Where the time goes today (the ceiling)

`perf` is not usable on the host (`perf_event_paranoid` 4, no sudo), so host instruction counts could not be taken. What was measured instead:
(1) nanoseconds per wave-instruction from the bench's device CPU time and the exact instruction counts; (2) the share of each kind of host
work, from SIGPROF samples of the instruction pointer classified by function and, inside the tier's handlers and lane kernels, by the host
instruction at the sampled address (`data/sp72/scripts/classify.py`, the method of round sp48 without IBS: the function classes are solid, the
split inside handlers is approximate because a sample names the instruction after the slow one); (3) the same kernels' arithmetic as plain host
code (`native_ceiling.cpp`), which is what is left when nothing of the simulator is.

The classes:

* **loop**: `run_round`, `step`, `tier_step`, `fetch_next` (all inlined into `run_groups` by LTO): finding the instruction, the checks before
  it, the call, the four counters after it.
* **decode**: scalar host instructions of the handlers: reading EXEC, `lanes_of`, the `switch` on each operand's kind, the calls of the lane
  kernels, the scans for NaNs' control flow.
* **rows**: vector moves with a memory operand in handlers and kernels: a register row loaded or stored, a constant spread into a temporary
  row, the merge of a partial EXEC, the 64-bit pack and unpack.
* **lanes**: the rest of the lane kernels and every other vector instruction of the handlers: the guest instruction's arithmetic (with the
  kernel's own loop control and any load folded into an arithmetic instruction).
* **mem**: the Machine's memory instructions and the memory manager.

AMD bench rows, simulated MI300X, base build (`origin/sp69-hotspot-start` 77a8b8c8), percent of the samples inside the simulator's library:

| row | ns per wave-instruction | loop | decode | rows | lanes | mem | wave-instructions a loop iteration | native ns an iteration (64 lanes) | native ns per wave-instruction | ceiling over today |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| valu_int | 14.3 | 32.7 | 18.7 | 14.3 | 34.2 | 0.1 | 16 | 17.2 | 1.08 | 13x |
| valu_f32 | 15.8 | 18.0 | 29.3 | 14.7 | 37.6 | 0.1 | 11 | 7.9 | 0.72 | 22x |
| valu_f64 | 16.7 | 19.6 | 23.1 | 10.1 | 47.1 | 0.0 | 8 | 7.6 | 0.95 | 18x |
| p_i64 | 18.8 | 22.7 | 16.6 | 18.8 | 41.7 | 0.0 | | | | |
| divergent | 11.7 | 29.9 | 29.6 | 15.0 | 25.1 | 0.1 | | | | |
| lds | 31.9 | 34.9 | 31.9 | 1.9 | 6.9 | 18.2 | | | | |

inq's unit tests on the simulated MI300X (the workload of round sp70; one host thread, base build; the last column is the share of the
process's samples that are not in the simulator at all: inq's host code, the run-time compile of VkFFT's kernels, libc):

| test | loop | decode | rows | lanes | mem | interp | other | outside the library |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eigensolvers::steepest_descent | 25.4 | 29.1 | 8.9 | 21.2 | 6.2 | 3.8 | 5.2 | 8 |
| ground_state::initial_guess | 23.6 | 27.8 | 9.1 | 20.5 | 10.4 | 6.5 | 2.3 | 34 |
| solvers::poisson | 22.8 | 22.8 | 5.6 | 17.3 | 16.4 | 6.0 | 9.2 | 42 |
| hamiltonian::ks_hamiltonian | 16.1 | 18.1 | 6.0 | 17.3 | 26.2 | 7.0 | 9.3 | 59 |

(The LDS handlers are tier handlers and fall in decode, rows and lanes here; `h_ds` is 2 to 6 % of the library's samples. *interp* is
the interpreter's own code for instructions with no handler, *other* the launch path, work-group set-up and the HIP API.)

Reading it:

* **Dispatch is 18 to 35 % of an ALU-bound kernel**, operand decode another 17 to 32 %. Together, the part a closure tier can remove, they are
  45 to 60 %: the bound of route (A) on an ALU-bound loop is about 1.8x to 2.5x.
* **The arithmetic itself is 25 to 47 %**, far more than on the NVIDIA engine (3 to 13 % in round sp48): a wave is 64 lanes, so a row is 256
  bytes and an instruction is eight vector operations, where a warp's is four. That raises the floor under route (A).
* **The native column is the other end**: with the lanes outermost (eight lanes in one `ymm` through every instruction of the loop body, rows
  loaded once and stored once an iteration) the three ALU loops cost 0.7 to 1.1 ns per wave-instruction, 13 to 22 times less than today and
  4 to 7 times less than the *lanes* share alone. The lane kernels' cost is not the arithmetic: it is one call, one loop, three row loads and
  a row store per guest instruction. Only code generated for the run (or a fused kernel written for it) removes that.
* The native numbers are a ceiling, not a forecast. They leave out what generated code must still do: test EXEC, keep the results staged
  until the run is known to have no NaN that the interpreter must decide (section 5), and the scalar instructions of the run.

## 2. The two routes, compared

| | (A) closure tier | (B) generated host code |
|---|---|---|
| removes | loop (layer 1, built), decode (layer 2), the loop between blocks and around memory instructions (layer 3) | all of that inside a fused run, plus rows and most of lanes |
| measured | layer 1: section 9 | the native ceiling above: 13x to 22x on ALU loops |
| bound on an ALU-bound loop | 1.8x to 2.5x (loop + decode gone) | 13x to 22x (the native ceiling); 5x to 10x is a fair guess once EXEC, NaN staging and scalar instructions are in |
| cost of the first run | microseconds a block (a walk over decoded instructions) | a compiler run: 2.3 s a unit on the NVIDIA side (copy-and-patch study, section 1), 0.4 s of it the header |
| needs | nothing | a C++ compiler on `PATH`, a writable and executable cache directory, a forked child |
| can be the default | yes, once it has soaked | no (the NVIDIA JIT is off by default for exactly these reasons; round sp57 measured the flip and did not make it) |
| short processes (inq: 244 of them, kernels made at run time by VkFFT) | gain from the first block | lose unless the cache is warm; a cold CI runner pays every compile |
| real programs on the NVIDIA side | the tier: 1.2x to 2.8x | the JIT over the tier: 1.10x geometric mean warm (round 15), 1.22x *slower* cold |

Why the NVIDIA JIT's 1.10x is not the forecast for (B) here, and why it is still a warning:

* Its generated code is instruction-major (one pass over the 32 lanes per op, rows in memory). Round sp48 measured that form's ceiling at 6 to
  15 % of a program and built lane-major *fused runs* for eight forms: 1.01x to 1.11x, because compares, selects and guarded ops ended the runs.
  The lesson for this design: generated code is only worth its cost if it is lane-major **and** covers compares, selects and masks from the
  start (on AMD: `v_cmp_*` into VCC or a scalar pair, `v_cndmask_b32`, a partial EXEC).
* What bounded the NVIDIA JIT in real programs was not the straight-line code but the run loop between regions (24 to 40 % of a compute-bound
  program with the JIT on, a region left every 8 to 14 instructions) and memory instructions. The AMD numbers have the same shape: inq's blocks
  are 8 instructions long on average (section 9), and memory instructions are 11 to 53 % of its execute time. Route (A)'s layer 3 (blocks
  that go on across branches and memory instructions) attacks exactly that part, and (B) inherits it.
* The first run. A compile on 16 busy threads never finished in time for 3 to 10 s programs (rounds sp63 and sp64). inq's tests are that
  population. Route (B) is for the long SCF runs and the stress workloads, behind a threshold, never for a test suite's cold run.

So the routes are not alternatives but layers, and the order is by what each needs: (A) needs nothing and is measured to pay now; (B) needs
(A)'s blocks as its unit and entry conditions, and its pay-off is large but only on long ALU-bound runs.

## 3. The unit of compilation

**A block** (`gcn::TierBlock`, built): a run of consecutive instructions that the tier has handlers for and that are not the Machine's
(`Op::mach`: global loads and stores, scalar memory, barriers, `s_endpgm`), ending *with* the first branch or *before* the first instruction
that is not the tier's. It is found lazily, the first time a wave is at its first instruction, by walking the decode cache forward; it is
immutable once published (compare-and-swap on `Inst::block`), shared by every thread, and owned by its first instruction, so it lives as
long as the decode cache. A wave that enters a run in the middle (a branch target, the instruction after one that was handed back) gets a block
of its own from there: blocks may overlap. A block is 3 to 256 instructions and does not start with a wait for loads or a plain store,
which can end the wave's turn before the block has run anything.

Not a whole kernel and not a region with internal control flow, for three reasons. (1) Where a wave is between blocks is where every
scheduling decision is taken today (the end of a turn, a barrier, a yield before a store): keeping blocks straight-line keeps those decisions
in `run_round`, unchanged. (2) A block has one entry, so its entry conditions are checked once. (3) Kernels made at run time (VkFFT's) are
many and short-lived; a lazily built block costs a walk over the instructions that run and nothing for those that do not.

Layer 3 widens the unit without changing its nature: a *chain* is a block whose branch, when taken to a block already built, goes on into it
inside the runner (the hint `Inst::next` already says where the last wave went), and a *memory member* is a Machine instruction called from
inside the runner. Both keep the per-instruction rules of sections 5 and 6.

**A fused run** (layers 2b and 4, not built): inside a block, a maximal run of *lane-local, register-only* instructions (lane l of every
result depends on lane l of the sources only: integer, float and double ALU, compares, selects, conversions; not DPP, `v_readlane`, permutes,
the matrix instructions, LDS, memory), between two instructions that write EXEC. It is the unit generated code or a fused kernel replaces.

## 4. Wave state layout

Nothing moves. The compiled tier reads and writes the wave where the interpreter keeps it, so a fallback at any instruction boundary finds the
whole state with no restore step:

* **VGPRs**: `Wave::vgpr`, rows of 64 `uint32_t` (256 bytes, 64-byte aligned, one file of up to 256 rows from the per-thread pool). A 64-bit value
  is the pair of rows r (low words) and r+1 (high words). Layer 2 binds a row as a byte offset from the file (`256 * r`), computed when the
  block is built; the file's address is read once per block run (`gcn::tier::Regs::vgpr`, valid for a round: `Wave::tier.ok`).
* **SGPRs, VCC, EXEC, SCC, the program counter**: `Wave::sgpr[106]`, `vcc`, `exec` (64-bit lane masks; a wave32 wave uses the low half), `scc`,
  `pc`, reached through the same `Regs`.
* **In a fused run** a guest register is one `ymm` for the eight lanes being worked (a pair for a double: lo and hi rows are combined with
  `vpunpck`/`vpermd` once at first use, split once at the last). Rows a run writes are stored when the block of eight lanes is done, into a
  staging area of the run when a NaN could still hand the run back (section 5), else into the file.
* **Constants** of a block are spread once, when the block is built, into rows of the block's own pool; a scalar register operand is spread
  when the run starts (it cannot change inside a fused run: fused runs hold no scalar writes to their own scalar sources; where one does, the
  run ends).

## 5. What stays exactly as observable today

The contract is the tier's: results, counters, errors and the order of effects are the interpreter's, bit for bit, with the knob on or off.

* **Divergence (EXEC).** EXEC is data. A handler reads it per instruction today and writes only the lanes it names. In a block that does not
  change. In layers 2 and 4 the test is hoisted: at the start of an *EXEC-stable stretch* (no instruction in it writes EXEC, which is known when
  the block is built from the destinations) the runner looks at EXEC once: all lanes on, the all-lanes forms run (kernel straight into the
  destination row); none on, the stretch's vector instructions are skipped and counted; otherwise the merging forms run (in a fused run: one
  blend per written register per eight lanes, against the mask spread once). `s_and_saveexec`, `s_or_b64 exec` and friends are ordinary
  scalar instructions that end a stretch. Branches on EXEC or VCC end a block.
* **NaN hand-back.** A handler that meets a NaN whose payload the interpreter must decide returns 0 having changed nothing, and the
  interpreter runs the instruction. In a block: the run stops there, the instructions before it are counted, the program counter is on the
  instruction, and the plain step runs it (the handler is called a second time and hands it back again, which notes it once in
  `VGPU_AMD_TIER_STATS` and in the bail counter, as today). In a fused run the lanes are outermost, so "nothing changed" needs staging: the run
  writes its rows to a staging area, accumulates the NaN test of every instruction whose rule is hand-back, and commits the rows only when the
  test is clear; otherwise the staged rows are dropped and the same instructions run one handler at a time. (The tier's `h_f64r` handlers
  already work this way for one instruction.)
* **Barriers** (`s_barrier`, gfx1250's named and cluster barriers) are the Machine's or the interpreter's: never inside a block. A block ends
  before one; `at_barrier`, `nb_wait`, `cl_wait` and the release at the end of `run_round` are untouched.
* **LDS.** `ds_read` and `ds_write` handlers are block members, called as `tier_step` calls them. A plain store's first of a round yields to the
  group's other waves (`Wave::store_gate`): the handler returns 3, the block run ends there with nothing of the instruction done and the turn
  given up, exactly as `tier_step` turns 3 into `step() == false`. An access outside the LDS is handed back (0) and the interpreter throws its
  error. Every other DS form runs through the handler it has today, or through the interpreter where it has none.
* **Atomics** on global memory are the interpreter's (`global_atomic_*` has no tier handler): the lock stripes, the lock-free path and
  `LoneRmw` are not reached from a block.
* **Memory faults.** In slice 1 no instruction of a block touches device memory (a block ends before a Machine instruction) and no tier
  handler throws (it hands the instruction back instead), so a fault is thrown where it is thrown today, by a plain step. From layer 3 on,
  a memory member is called through `tier_mach` with the program counter advanced as `tier_step` advances it, lanes in lane order; the
  runner then needs a `catch` that counts the instructions before the one that threw and gives its address to `run_round`'s
  `at_instruction`, and a test that faults in the middle of a block (the address in the error's text, the counters, the lanes written
  before the fault). A first form of slice 1 had that `catch`; no test could reach it, so it was taken out.
* **Counters.** `DispatchStats::instructions`, the unit counters (`valu`, `salu`, `branch`, `lds`), `mix[]` and `mops[]` are sums over the
  instructions that ran, known when the block is built: added once when the whole block ran, added for the prefix otherwise. They are flushed
  before the runner returns, so an instruction that reads the instruction count (`s_getreg_b32` of the perf registers, `s_memtime`), which is
  the interpreter's and therefore outside any block, reads what it read before.
* **Traces, the debugger, PC sampling, `VGPU_TRACE_WAVE`.** `tier_on` is false under any of them, and blocks run only where the tier runs
  (`blocks_on = tier_on && ...`): those runs are the interpreter's, one instruction at a time, as today. `VGPU_AMD_TIER=0` turns blocks off
  with the tier.
* **`VGPU_AMD_TIER_STATS`.** The same lines: an instruction with no handler is noted by the plain step; a handed-back one is noted once.
* **Turns and lockstep.** `run_round` gives a wave a turn of `slice` instructions (4,194,304 for a group's first turn, 32,768 later, 64 in
  lockstep) and stops after the instruction that completes it. A block runs at most as many instructions as can complete before that
  instruction (`room`), and that instruction is always a plain step, so a turn ends on the instruction it ends on today and
  `Wave::round_steps` has the same value. `run_groups`' admission, the order groups take turns in, `waiting()` and the lockstep scheduler read
  only what `run_round` returns and the waves' flags: unchanged. The same-bits test digests `round_steps`, the number of rounds and every
  wave's flags.
* **Abandoning a dispatch** (`g_abandoned`, asked every 4,096 instructions of a turn) is asked whenever a block run crosses a multiple of 4,096.
* **Host threads.** A block is immutable and shared; a wave is one thread's. Nothing else is shared.

## 6. What falls back, and to what

To the plain step (the tier one instruction at a time, then the interpreter), always at an instruction boundary with the full state in the
wave:

* a wave with VGPR indexing on, VGPR MSBs set, VSKIP set, or MODE not at its default (round or denormal mode): no block starts (the
  conditions `tier_step` asks, asked once);
* an instruction with no tier handler, or a Machine instruction (until layer 3): the block ends before it;
* a handler's 0 (a NaN the interpreter decides, a register past the file, an LDS access outside the LDS): the block stops at it;
* the last instruction of a turn; a run shorter than three instructions; a run's first instruction when it is a wait or a plain store; an
  instruction that is not at a word of the code object;
* the tier being off for the dispatch (tracing, the debugger, a sampler).

For generated code (layer 4), also: no compiler, a cache directory that is read-only or `noexec`, a sanitizer build, a block that has not
reached the promotion threshold, a compile in flight (the closure tier runs meanwhile, never a wait).

## 7. Cache and keying

Layers 1 to 3: none. A block is built in memory per decode cache (per loaded code object) in microseconds and dies with it.

Layer 4 reuses the NVIDIA JIT's machinery where it fits (`src/sass/jit.cpp`: the forked compile with a time and address-space limit, the
directory, the lock file per key, the precompiled runtime header `VGPU_JIT_PCH`, the cap on the cache's size, the history that makes
`VGPU_JIT=auto` start a compile early in a second run). Key: the instruction words of the fused run with their relocations applied, the target
(gfx9 family, gfx10, gfx11, gfx12, wave32 or wave64), the generator's version, the compiler's identity and flags, the ISA level. Not the
launch arguments and not the code object's address: VkFFT's kernels are compiled again in every process, and 360 compiles in four inq tests made
218 distinct code objects, so a key on the words is what makes a second process (or a second plan of the same size) hit. The object stores the
words and a load compares them. What it does not reuse: the SASS generator (`jit_gen.cpp`), regions and linking, which are that engine's.

## 8. Test plan

* **`test_amd_gcn_blocks_same_bits`** (built, CI-wired): programs of 6 to 40 instructions made of the ISA corpora's shapes (four targets) with
  branches forward and backward between their own instructions and now and then any shape at all, run by the executor's own `run_round` with
  blocks off and on, on nine kinds of state (small integers, floats with and without NaNs, half the lanes off, any bits, the integers' edges,
  one or two waves with loads in flight) and eight turn lengths (so a turn ends inside a block, on its last instruction and just after it).
  The digest holds every wave's registers, masks, flags, program counter and `round_steps`, LDS, scratch, a memory region, every counter, the
  number of rounds and the error's text with its address. It also checks that it tested something (most instructions ran in blocks, some
  were handed back, some runs threw, some ended). Each later layer extends it: bound forms and EXEC stretches (layer 2), chains and memory
  members with real addresses and faults in the middle of a block (layer 3), fused runs against the same digest (layers 2b and 4).
* **`bench_amd_counters_blocks`** (built, CI-wired): the 82 workload/target pairs of the AMD bench with `VGPU_AMD_BLOCKS=1`: each workload
  checks its own results against a host reference, and the instruction counts must be those of the expected file.
* **The existing corpus with the knob on**: `test_amd_gcn_tier_same_bits` (every handler shape) is unaffected by blocks (one instruction);
  the AMD ctest set and the e2e tests (hipcc programs, rocBLAS, PyTorch, hipBLASLt's Stream-K GEMMs, the lockstep tests) run once more with
  the knob on before it becomes a default. For inq: the eight tests' outputs byte for byte, knob off and on.
* **Mutants** for each layer (`data/sp72/scripts/mutants.py` for layer 1): every rule of section 5 broken one at a time must fail a test.
* **Before a default flip**: ASan and TSan runs of the same-bits test and of the threads tests with the knob on; the AMD bench A/B with a
  no-op control (the one-thread rows move up to 3.5 % from code placement alone, round sp70).

## 9. Slice 1, built: blocks (`VGPU_AMD_BLOCKS=1`)

`Machine::run_block`, `block_at`, `make_block`, `step_at` and `run_wave_blocks` in `amd/src/gcn_exec.cpp` (about 170 lines), `gcn::TierBlock`
and `Inst::block` in `amd/include/vgpu/amd_gcn.hpp`, the knob in `amd/src/gcn_tier.cpp`. With the knob off the only code that runs differently
is one test of `m.blocks_on` per wave turn in `run_round`.

What a block run is: the wave's state is asked once; for each instruction the branch target is put where the handler reads it, the program
counter is advanced and the handler is called, as `tier_step` does; a result other than "done" ends the run as section 5 says; at the end the
block's counters are added. What is no longer done per instruction: `fetch_next` (an atomic load, a compare, a store), `step`'s and
`tier_step`'s tests (six), four counter increments through three indirections, `run_round`'s turn test and abandon test.

Measured (round sp72; one host thread, best of 9 runs a side, two passes; both sides are the same library, so nothing but the knob differs;
the host was shared, and the rows that run no kernel range from 0.89 to 1.07, which is one row's noise):

| row | MI300X, on / off (two passes) | RX 6800, on / off | instructions in blocks | instructions a block run |
|---|---:|---:|---:|---:|
| valu_int | 1.26, 1.06 | 1.20, 1.19 | 100 % | 16 |
| valu_f32 | 1.13, 1.15 | 1.20, 1.25 | 100 % | 11 |
| valu_f64 | 1.10, 1.32 | 1.12, 1.19 | 100 % | 8 |
| p_i64 | 1.19, 1.20 | 1.25, 1.30 | 100 % | 43 |
| p_cmp | 1.15, 1.20 | 1.32, 1.49 | 100 % | 53 |
| wave | 1.30, 1.32 | 1.41, 1.43 | 100 % | 31 |
| divergent | 1.20, 1.19 | 1.28, 1.26 | 91 % | 9 |
| smem | 1.34, 1.33 | 1.37, 1.59 | 91 % | 10 |
| mem | 0.95, 0.87 | 0.96, 1.05 | 44 % | 5.2 |
| narrow | 0.99, 0.98 | 0.96, 0.98 | 43 % | 4.4 |
| lds | 0.96, 0.94 | 0.94, 0.95 | 26 % | 2.6 (the blocks are 6.7 long) |

* All 82 rows: geometric mean 1.07 and 1.08; 35 rows are at 1.05 or more in both passes; the instruction counters of all 62 counted rows are
  the same with the knob off and on.
* **The rows that are arithmetic gain 6 to 59 %.** That is most of the *loop* share of section 1 (valu_int: 33 % of the time was the loop,
  1/(1 - 0.33) = 1.49 would be all of it).
* **The memory rows do not gain, and the LDS row loses 4 to 6 %.** Their blocks are short and, in `lds`, mostly not run to the end: a run is
  2.6 instructions of blocks 6.7 long, because a wave's first store of a round and its waits end the turn inside the block, and a block that
  stops early costs an entry, a walk over what ran to count it, and a second block for the rest. That is what layer 3 is for (and a block
  that ends before a store whose gate is still armed, which was not tried).
* inq (one host thread, best of two passes): CPU time 0.84 of the knob-off run for eigensolvers::steepest_descent, 0.90 for
  hamiltonian::ks_hamiltonian, 0.96 for ground_state::initial_guess, 1.00 for solvers::poisson; 82 to 92 % of their instructions run in
  blocks, 8 to 15 instructions a run. The outputs have the same checksum.
* With the knob off the bench's geometric mean against the base is 0.993 to 0.996 over two passes, and a control (the base with one function
  nothing calls added) is 0.993 to 1.003: the same within what this host could resolve (single rows scatter from 0.77 to 1.31 in both).

What changed inside the slice, by argument and not by measurement (four passes over 12 rows could not tell the forms apart on this host): a
block of two instructions was allowed at first (entering and leaving a block costs about one plain step, so the minimum is three); a block
could start at a wait or a plain store (it would be left at once); the block's counters were three lists (now one list of places and
amounts). A `catch` around the handlers was taken out because nothing can reach it until memory instructions are members.

`VGPU_AMD_BLOCKS_STATS=1` (with the knob on) prints at exit how many instructions ran in blocks, the mean run length and the blocks that ran
most, with their listings (`=n` for the top n): the tool for choosing what layers 2 to 4 cover first.

## 10. Milestones

Expected gains are over the layer before, on one host thread, from the class shares of section 1 and slice 1's measurement; "ALU rows" are the
bench's valu, probe and control rows, "inq" the eight tests of round sp70.

| # | what | removes | expected gain | gate |
|---|---|---|---|---|
| 1 | **blocks** (built) | loop, inside straight-line runs | measured: section 9 | done: same-bits test, mutants, bench counters |
| 2 | **bound operations**: a block holds an array of `{kernel, row offsets, constant rows}` beside the instruction list; EXEC looked at once per EXEC-stable stretch; the all-lanes forms call the lane kernel on the destination row | decode: 17 to 32 % of today's time on ALU rows, most of it reachable for the handler families `h_int`, `h_intv`, `h_f32`, `h_f64r`, the compares and `v_cndmask` | 1.15x to 1.35x on ALU rows; inq 1.05x to 1.10x | same-bits test with every operand kind in every stretch shape; a handler family is bound only when its shapes pass |
| 3 | **longer blocks**: a taken branch goes on into the block at its target inside the runner (chains); global loads and stores, scalar memory and `s_waitcnt` are members called through `tier_mach` | the loop between blocks: inq's blocks are 8 instructions long, so the per-block entry (fetch, block lookup, the counter flush) is paid every 8 instructions; and the whole loop share of memory-bound kernels | 1.05x to 1.15x on inq and on the memory rows; little on the valu rows, which are one block already | faults in the middle of a block (address, counters, lane order), the store gate inside a block, lockstep and Stream-K tests with the knob on |
| 2b | **fused pairs** written by hand for the hottest adjacent pairs that `VGPU_AMD_BLOCKS_STATS` shows (compare + select, 64-bit shift + or, multiply + add of doubles): lanes outermost for two instructions | one row store and load and one call per pair | 1.05x to 1.15x where the pairs are; only if layer 4 is not started | per pair, against the two handlers |
| 4 | **generated fused runs** (`VGPU_AMD_JIT`, opt-in): C++ per fused run, lanes outermost, EXEC as a blend, NaN staging, compiled by the host compiler with the NVIDIA JIT's compile, cache and promotion machinery; the closure tier runs until the object is there | rows and most of lanes inside fused runs | 3x to 8x on ALU rows warm (ceiling 13x to 22x); inq warm 1.2x to 1.5x, cold a loss | a generated run against the same digest as the handlers; mutants per generated form; cold and warm timings at 1 and 8 threads; never on by default |
| 5 | the default flip of layers 1 to 3 | | | the test plan's last line |

Layers 2 and 3 are independent; 3 matters more for inq (its execute time is 11 to 53 % memory instructions and its blocks are short), 2 more
for the ALU rows. Layer 4 is worth starting only after 2 and 3, because its entry conditions, its fallbacks and its unit are theirs.

## 11. What this study does not settle

* Host instruction counts and IPC (no `perf` on the host). The class shares are from instruction-pointer samples.
* How much of *decode* layer 2 really removes: the estimate assumes the handler's operand `switch`, the EXEC tests and the temporary rows go,
  and the kernel call stays.
* Layer 4's numbers are a ceiling from hand-written loops plus the NVIDIA side's experience, not a prototype of generated code for AMD
  instructions. The first thing layer 4 must measure is one real block (inq's 84-instruction block of section 9) generated by hand.
* Thread scaling with blocks on (a block is shared read-only, so no new sharing is expected; measured at 1 and 8 threads only on the rows of section 9).
* gfx1250 (named barriers, clusters, VGPR MSBs) runs through the fallbacks; its kernels were not timed.
