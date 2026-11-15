# Study 13: a compiled tier for AMD kernels

Status: design, 2026-11-14, with its closure layers partly built behind `VGPU_AMD_BLOCKS=1` (off by default): **blocks** (section 9,
round sp72), then **memory instructions and waits inside blocks** and **bound integer runs** (section 12, round sp73); and, in the tier
itself and so with the knob off too, **row handlers for the 64-bit forms** (section 13, round sp74). Measurements, tests and
mutants are in *Round sp72*, *Round sp73* and *Round sp74* ([rounds/11-rounds-sp36-sp41.md](rounds/11-rounds-sp36-sp41.md)); scripts and data in
`data/sp72/`, `data/sp73/` and `data/sp74/`. Everything was measured on server3 (Ryzen 7 PRO 8840U, 8 cores / 16 threads, g++ 13.3, Release with LTO), one
host thread unless a line says otherwise. Sections 1 to 11 are the design as written in round sp72, on a host shared with other sessions;
section 12 says what round sp73 built, measured on a quiet host, and where the design's estimates were wrong; section 13 what round sp74
did to the tier's handlers for inq, what is left by class, and what that says about milestone 4.

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
is one test of `m.blocks_on` per wave turn in `run_round`. (**Corrected in round sp73**: that was the intent, not the result. The plain loop
was slower with the knob off, valu_f64 by 13 and 23 %; section 12.)

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
  (**Corrected in round sp73**: on a quiet host the scatter is 2 %, not 20 %, and valu_f64 was 13 and 23 % slower with the knob off in this
  slice. The 0.77 above was that row, read as noise.)

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
| 1 | **blocks** (built, round sp72) | loop, inside straight-line runs | measured: 1.06 to 1.36 on arithmetic rows (section 12's table, column *blocks*) | done: same-bits test, mutants, bench counters |
| 2 | **bound operations**: a block holds an array of `{kernel, row offsets, constant rows}` beside the instruction list; EXEC looked at once per run; the all-lanes forms call the lane kernel on the destination row | decode | **built for the 32-bit integer family (round sp73)**: valu_int 1.14 and 1.11 over blocks alone, p_sdwa 1.09 and 1.10, p_i64 1.07 and 1.04; nothing on inq. Estimated here at 1.15x to 1.35x on arithmetic rows and 1.05x to 1.10x on inq for all families; the float, double, compare and select families are not bound, and section 12 says why binding is the wrong tool for them | same-bits test with every operand kind; a handler family is bound only when its shapes pass |
| 3 | **longer blocks**: global loads and stores, scalar memory, waits and plain stores as members; a wave goes on in the block it left | the loop around memory instructions | **built (round sp73)**: the `lds` and `mem` rows no longer lose with the knob on (0.99 to 1.02 against the base), smem and p_smem_wide 1.18 to 1.32; inq 1 to 4 % less CPU with the knob on (estimated 1.05x to 1.15x). A taken branch going on into the next block inside the runner was not built: the hint `Inst::next` already makes that step a load and a compare | faults in the middle of a block, the store gate inside a block, turns that end inside a block |
| 2c | **row handlers for the forms a real program runs most** (the tier's own work, with the knob off too) | decode, rows and lanes of one handler at a time | **built for the 64-bit forms and SDWA (round sp74, section 13)**: p_f64x 2.4x and 2.1x, p_sdwa 1.8x and 1.5x, the bench's 62 rows 1.06 with the knob off; inq's steepest_descent 0.82 of its CPU, initial_guess 0.88, poisson 0.95. Bound binary64 runs (milestone 2 for doubles): measured, nothing, not kept. No handler above 7 % of the library's samples is left | the tier's same-bits test over the forms, mutants |
| 2b | **fused pairs** written by hand for the hottest adjacent pairs that `VGPU_AMD_BLOCKS_STATS` shows (compare + select, 64-bit shift + or, multiply + add of doubles): lanes outermost for two instructions | one row store and load and one call per pair | 1.05x to 1.15x where the pairs are; only if layer 4 is not started | per pair, against the two handlers |
| 4 | **generated fused runs** (`VGPU_AMD_JIT`, opt-in; **the next step for kernels that are arithmetic**, section 13): C++ per fused run, lanes outermost, EXEC as a blend, NaN staging, compiled by the host compiler with the NVIDIA JIT's compile, cache and promotion machinery; the closure tier runs until the object is there | rows and most of lanes inside fused runs | 3x to 8x on ALU rows warm (ceiling 13x to 22x); inq warm 1.2x to 1.5x, cold a loss | a generated run against the same digest as the handlers; mutants per generated form; cold and warm timings at 1 and 8 threads; never on by default |
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

## 12. Phase 2 as built (round sp73), and what it corrects

**The knob off.** Round sp72's slice made the interpreter slower with the knob off, and its measurement (on a loaded host) could not see
it: valu_f64 ran 13 % (MI300X) and 23 % (RX 6800) slower, because `run_round`'s `pc` was passed to the blocks loop by reference and the
plain loop therefore wrote it to the stack after every instruction. Three smaller costs of the same kind followed in round sp73 and were
removed one at a time against a no-op control: `tier_mach` out of line once it had a second caller, the machine's hot members moved by
the members blocks added among them, and the plain loop re-laid by a flag test and a second loop inside `run_round`. The rule that came out
of it: **with the knob off the plain path is the text it was** (`run_round` untouched and always in line, `run_groups<false>` the old
function, blocks in a twin `run_round_blocks` and their own instantiation `run_groups<true>`), and what blocks keep in the machine comes
after every other member. With that the knob-off build is at 1.001 of the base (geometric mean of 62 rows; the control 1.001), every row
within 3.5 %, as the control's rows are. (That is the build before the frame pointers. The committed library, with them, is at 0.993
with three rows 4 to 5 % low, smem on the RX 6800 the lowest at 0.946; a variant with the blocks' loop in its own text section moved the
loss to mem and narrow. A link of this library places two to eight rows 3 to 5 % below the base's, different ones each time: round sp73,
section 1.)

**Longer blocks (milestone 3).** The instructions the tier hands to the Machine are block members, called through `tier_mach` with the
stretch before them counted and the error's address on them. A block is *stretches* of plain instructions, each counted once, separated by
the instructions that are the Machine's or can end the wave's turn. A wave that leaves a block before its end goes on in it at the
instruction it stopped at (`Wave::blk`), so a gated store or a wait costs one exit and one entry and no second block. Two trials keep
blocks from running where they do not pay, neither of which changes what any instruction does: a block whose runs are left for a turn's
end more often than every 8 instructions over its first 64 runs is turned off; a machine whose turns average under 16 instructions over
256 turns goes back to the plain loop for the rest of the dispatch. Without the second, `lds` (a wait or a first store every two
instructions in work-groups of several waves) still lost 7 to 10 % with the knob on: what it paid was not the blocks but coming into
the blocks' loop for a turn of two instructions.

**Bound integer runs (milestone 2, in part).** Three or more consecutive 32-bit integer instructions whose sources are vector registers and
constants are a *bound run*: with every lane of the wave on, the runner calls their lane kernels on the rows one after the other
(`gcn::tier::run_bound_ints`), with no handler, no EXEC test and no constant spread between them (a constant's row is made once, when the
block is). Lanes off, or a turn that ends inside the run: the handlers, one at a time. `VGPU_AMD_BLOCKS_BOUND=0` turns binding off.

Measured (round sp73; CPU seconds of the device phase, the base's over each build's, best of 7 alternating runs pinned to one core on a
quiet host; *control* is the base plus a function nothing calls; full table in `data/sp73/steps.md`):

| row | MI300X: control | knob off | blocks | blocks + bound | RX 6800: control | knob off | blocks | blocks + bound |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| valu_int | 0.995 | 0.997 | 1.161 | 1.326 | 0.997 | 0.998 | 1.206 | 1.341 |
| valu_f32 | 0.979 | 0.980 | 1.119 | 1.087 | 0.997 | 1.004 | 1.202 | 1.204 |
| valu_f64 | 0.996 | 1.000 | 1.115 | 1.101 | 0.994 | 0.987 | 1.085 | 1.088 |
| p_i64 | 0.980 | 0.974 | 1.162 | 1.242 | 1.005 | 1.003 | 1.268 | 1.323 |
| p_sdwa | 0.996 | 0.994 | 1.099 | 1.199 | 0.998 | 0.989 | 1.180 | 1.296 |
| p_cmp | 1.007 | 0.998 | 1.193 | 1.203 | 0.992 | 0.965 | 1.263 | 1.282 |
| divergent | 1.007 | 1.016 | 1.156 | 1.158 | 1.021 | 1.018 | 1.240 | 1.243 |
| wave | 0.996 | 1.002 | 1.224 | 1.233 | 1.004 | 0.990 | 1.281 | 1.289 |
| smem | 1.023 | 0.988 | 1.324 | 1.323 | 1.016 | 0.990 | 1.303 | 1.306 |
| mem | 0.984 | 1.002 | 1.006 | 1.016 | 1.003 | 1.012 | 1.018 | 1.014 |
| lds | 1.001 | 1.010 | 1.011 | 1.007 | 0.997 | 0.995 | 1.005 | 0.990 |
| narrow | 1.003 | 1.004 | 1.006 | 1.000 | 1.005 | 1.036 | 1.024 | 1.031 |
| buf | 0.998 | 0.982 | 0.984 | 0.982 | 1.001 | 0.984 | 0.984 | 0.986 |
| flat | 1.013 | 1.001 | 1.010 | 1.014 | 0.990 | 0.993 | 1.010 | 1.005 |

Geometric mean of the 62 kernel rows: control 1.001, knob off 1.001, blocks 1.099, blocks with bound runs 1.107; no row below 0.97 of the
base with the knob on. The committed library (frame pointers in the blocks' loop): control 0.999, knob off 0.993, knob on 1.103, again no
row below 0.97 of the base with the knob on.

**Where the design's estimates were wrong.**

* *Bound operations on inq* (estimated 1.05x to 1.10x): nothing. inq's decode share (18 to 29 % of its time in the library) is not the
  operand `switch` binding removes. It is the general binary64 handler (`h_f64`: `v_max_f64` with `|x|` modifiers and the other forms that
  have no row kernel go through 64-bit arrays packed from and unpacked into the register pairs; 13 to 15 % of the library's samples in
  steepest_descent and initial_guess), the SDWA and 64-bit compare and shift handlers, and LDS. The lever there is row kernels for those
  forms in the tier itself (as `f64_rows` has for add, multiply and fused multiply-add), not a block mechanism.
* *Bound operations for the float families*: their handlers scan the result for NaNs, apply the rule or hand back, and merge; binding
  cannot drop any of that, only the operand reads, which are a small part. Not built.
* *Chains* (a taken branch continuing inside the runner): the step between two blocks is `fetch_next`'s hint (a load and a compare) and the
  block lookup (a load); a chain would save the lookup. Not built.
* *The trials* were not in the design: it assumed a block always pays once entered. A block entered for one or two instructions costs
  more than the plain steps; the turn-length trial is what makes the knob safe on barrier- and store-heavy kernels.

## 13. Round sp74: the tier's handlers for inq, and what is left

**What was wrong with section 12's reading.** It named `h_f64`, the general binary64 handler, at 13 to 15 % of two tests, and guessed at
several forms. A build that times each handler by instruction form (round sp74, `data/sp74/forms.md`) found one: `v_max_f64`, 331
time-stamp ticks a call against 83 for `v_add_f64`, 41 million calls in steepest_descent. The general handler packs each register pair
into 64-bit lanes, tests for two zeros of opposite sign a lane at a time, runs the kernel, scans for NaNs and unpacks.

**What was built** (`amd/src/gcn_tier.cpp`, its last section; AVX2 hosts): row handlers for `v_min_f64` and `v_max_f64`, the one-input
functions, every 64-bit comparison (a comparison of doubles is the host's one comparison for its test; of integers, the two rows of words
eight lanes at a time), `v_lshlrev_b64`, `v_lshrrev_b64` and `v_lshl_add_u64`, the conversions between doubles and 32-bit values,
`v_frexp_*` and `v_ldexp_f64`, and the SDWA forms of the integer instructions on whole rows; and `v_add_f64`, `v_mul_f64`, `v_fma_f64`
run only the groups of eight lanes that hold a lane of EXEC where some are off. The rule that keeps the bits: **a row handler does only
the lanes whose answer is one host instruction's; where a lane that is on is anything else (a NaN, two zeros of opposite sign, a denormal
in frexp, a result of ldexp that is not a normal number) it writes nothing and calls the handler it replaced.**

| | base | round sp74, knob off | knob on |
|---|---:|---:|---:|
| the bench's 62 kernel rows (geometric mean; control 1.004) | 1 | 1.060 | 1.184 (the base with the knob on: 1.115) |
| p_f64x, MI300X / RX 6800 | 1 | 2.40 / 2.08 | 2.68 / 2.43 |
| p_sdwa | 1 | 1.83 / 1.51 | 2.54 / 2.21 |
| steepest_descent, one host thread, CPU s | 26.0 | 21.2 | 19.9 |
| initial_guess | 65.4 | 57.2 | 55.3 |
| poisson | 116.2 | 110.8 | 107.1 |
| ks_hamiltonian | 26.4 | 25.9 | 25.5 |

**Measured and not kept.** Bound binary64 runs inside blocks (milestone 2 for doubles): valu_f64 1.09 before and after, steepest_descent
the same; what binding removes of `h_f64r` is the operand kinds, and the kernel, its NaN mask and the copy of two rows are its time. A
pair's lanes kept in the order the words interleave, without the moves across the halves of a vector: 4 % slower in the kernel of add,
multiply and fma, for a reason not found. The groups-of-lanes rule with every lane on: 3 % slower on valu_f64.

**What is left** (samples of the final library, knob off, shares of the whole process; `data/sp74/profile.md`):

| test | outside the library | loop | decode | rows | lanes | mem | interp | other |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| steepest_descent | 7 % | 29 % | 24 % | 8 % | 14 % | 7 % | 4 % | 5 % |
| initial_guess | 37 % | 18 % | 16 % | 5 % | 10 % | 7 % | 5 % | 2 % |
| poisson | 43 % | 14 % | 14 % | 3 % | 8 % | 10 % | 4 % | 5 % |
| ks_hamiltonian | 61 % | 6 % | 8 % | 2 % | 6 % | 11 % | 3 % | 4 % |

* No handler is above 7 % of the library's samples any more (`h_ds` 7.0 %, `h_dminmax_rows` 6.1 %, `h_cmp64_rows` 3.9 % in
  steepest_descent; `h_f64r` 7.3 % in poisson). More closure-tier work is worth 1 to 3 % of a test a handler.
* The loop is the largest class, and blocks take a fifth to a third of it (1.3 of 6.0 s in steepest_descent, 4.6 of 15.1 s in poisson):
  a block still calls a handler through a pointer for each instruction, 4.0 ns an instruction.
* Loop, decode and rows, which a generated fused run removes inside the run, are 61 % of steepest_descent and 39, 30 and 16 % of the
  others; the guest's arithmetic is 14, 10, 8 and 6 %. With the share of instructions that run in blocks today (76, 19, 72, 80 %) and
  half of the three classes gone there, the tests are 1.3, 1.04, 1.12 and 1.07 times as fast; all of it gone, 1.9, 1.08, 1.28 and 1.15.

**So milestone 4 is next, and what it is good for is narrower than section 10 says.** Generated fused runs are the lever for kernels
that are arithmetic: steepest_descent's kind and the bench's ALU rows. They should start with the hottest blocks only and need a cache
that outlives the process (inq's tests run 20 to 110 s with 860 to 3,650 blocks each). For initial_guess the lever is blocks for
work-groups with barriers (a fifth of its instructions run in blocks); for poisson and ks_hamiltonian it is the memory path, and 43 and
61 % of their CPU is not the simulator's. Section 10's "inq warm 1.2x to 1.5x" holds for one test of the four measured.

