# Host-GPU mode: the size rules, sampling and residency

Status: experimental, opt-in (`VGPU_HOST_GPU=exact`). Read [09-host-gpu-spike.md](09-host-gpu-spike.md) first. This document describes the rules that
decide which launches of an exact-mode run go to the real card and which stay in the simulator, and the machinery behind them: what a module may hold, the
sample of a launch's first blocks, the residency of device memory, the driver start and very short launches. Tracking issue: pantheonsim #587.

The section numbers are kept from the longer document this one was cut from, so that the round logs and the data directories that point at a section still
find it; sections 1, 2, 4, 5, 15, 16 and 18 to 23 described a second, relaxed mode that is no longer part of this repository and are gone. The measurements of sections 8
to 13 were taken in the modes as they were on the dates given; where a figure was measured in that relaxed mode (it sent kernels that exact mode keeps in the
simulator: approximate math, tensor-core instructions, float atomics) the text says so. They are kept as the record of why the rules are what they are.

## 3. What a launch and a module must pass

Whatever the size rule says, a launch is kept in the simulator when its launch shape is one the card path does not build (cooperative and cluster launches,
warp size, a zero grid or block dimension), when an argument points into mapped host memory, when device memory exists outside the allocation table, when a
kernel needs of its module what the card cannot be given (section 10: its variables, which the card is given at the simulator's addresses, and the external
functions it calls, none of which is admitted), when a diagnostic that only the simulator gives is on (`VGPU_HOST_GPU_MAX_MIB` and the list of switches that
turn the mode off are in [cli.md](../cli.md)). The size rule and its start cost, the first-launch rule, the sample of a launch's first blocks (section 8), the
bans and the 64 KiB residency protocol apply after that.

* **Kernels that do not return.** `cuCtxSynchronize` has no deadline, a running kernel cannot be cancelled, and while it runs the card's memory cannot be read
  back, so there is no fallback. The card computes the simulator's bits, so a kernel loops on the card exactly when it would loop in the simulator.
  `VGPU_HOST_GPU_TIMEOUT` (seconds, none unless set) ends the process with status 124 when a launch has been on the card longer.
* **The ski-rental account and the bans** count time, not opcodes. The simulator's time for a kernel is learnt by two simulated launches, or one long one.

Two bugs of exact mode's scan were found on the way:

* A kernel refused for its module (variables, external calls) was reported and counted as "not in the exact set" on every launch after its
  first, because the cached verdict kept no reason. The reason is cached now; such launches are counted as unsupported launch shapes every time.
* `exact_mods_allowed` looks for the modifiers `F32`, `F64`, `F16` on a `RED`, but the decoder prints a float reduction's type as one
  modifier, `F32.FTZ.RN`. So exact mode admitted float reductions on sm_75 to sm_89 (on sm_90 and later the mnemonic is `REDG`, which exact
  mode does not name). Exact mode now refuses them, which is the one change to exact mode's set, and it is a change towards what the set was written to be.
  Whether a card run had ever offloaded such a kernel in exact mode was not checked; sp42's verify runs found no exact-set kernel that differed.

## 6. The benchmark harness

`tools/bench/hostgpu/run.py` runs the prebuilt programs of `tools/bench/real` (PolyBench/GPU, Rodinia, the reduced HeCBench list; the same
layout, arguments and scratch directories) as sim and exact, `--runs` times each, interleaved, and optionally once as exact with
`VGPU_HOST_GPU_VERIFY=all`. It writes JSON and a markdown table: wall time, speed-up over sim, launches on the card of all launches, seconds
on the card and in the simulator, launches kept in the simulator by reason, verify's counts, and the program's output against the sim
run's: identical, numeric (the same words, every number with a fraction or exponent within a small tolerance; integers and hashes
must be equal), or different, with the largest numeric difference. A run past `--timeout` is killed (its own process group) and recorded.
The form `exactdiv` is exact with `VGPU_HOST_GPU_DIV=1`; `exactr0` is exact with `VGPU_HOST_GPU_START_REPEATS=0`
(section 11), `exactl0` with `VGPU_HOST_GPU_START_LAUNCHES=0`, `exactb0` with `VGPU_HOST_GPU_SAMPLE_BLOCK=0` (section 12); a program slower than `--long-secs` in
the simulator gets `--long-runs` runs of each form. `--dry-run` is a self-test that needs no card; it is the ctest `hostgpu_bench_selftest`.

```
python3 tools/bench/hostgpu/run.py --build build --corp /path/to/corp --gpu nvidia/a10g --runs 3 \
    --forms sim,exact,verify --timeout 600 --json hostgpu.json --md hostgpu.md
```

## 7. Tests

| test | where it runs | what it pins |
|---|---|---|
| `test_host_gpu_logic` | every CI job | the mode names; exact's set against a frozen copy and every decoder mnemonic; the sampling rule (how many blocks, what a process can afford, the estimate, what repays the card, when not to sample); the rule for a launch while the card holds the memory (both sides by the formula, the margin and the floor found by bisection, the Costs it reads, when it does not act); the bans' reference while the card holds memory; the zero-grid check; the kinds of external functions, the texture instructions, the card image (each relocation kind applied, the tables shortened, what it refuses) and the code relocation writer (Round sp66); the bounded blocks (Round sp75: the limit of a sampled grid and of a small one, the cap, what a stopped block's time sends, the least time a sample sends, which small grids are sampled and their floor, the reasons a launch is not sampled) |
| `e2e_host_gpu_stub` | CI jobs with nvcc (a stand-in driver, no card) | the lines for scripts; unreadable values; verify against a card that computes nothing; the timeout; a contended reduction under verify; kernels with module variables, the image the card gets (no relocation of a variable left), the constant bank looked up and filled before the launch, a live texture object, a kernel that calls `malloc` (Round sp66, `host_gpu_vars.cu`, steps 30 to 36); a short launch that follows a long one to a card that keeps memory and computes nothing while the card holds the memory, and stays with `VGPU_HOST_GPU_STREAK=0` or when the program read the memory back (steps 24 to 26); a zero grid while the card holds the memory gets the simulator's error (step 27); a verify run pushes again only the chunks that differed (Round sp67, step 28); the square root idiom only with `VGPU_HOST_GPU_SQRT=1` (Round sp68, step 29); 400 launches of one kernel, each a hundredth of the start cost, start the driver after the 200th, and not with `VGPU_HOST_GPU_START_REPEATS=0` nor when there are 150 (Round sp69, step 37); a launch whose every block outlasts the wait is sent on a block stopped at the limit, the card given the memory as it was, and with `VGPU_HOST_GPU_SAMPLE_BLOCK=0` its copy does not answer; a grid of 96 blocks with no followed loop is sampled by bounded blocks, and not with the switch off (Round sp75, steps 38 to 40) |
| `hostgpu_bench_selftest` | every CI job | the harness: parsing, output classes, the table, a hang that is killed |
| `e2e_host_gpu`, `tools/card-ci/run-card-ci.sh` | a machine with a card only | module variables and a live texture object on the card, the output the mode-off output to the byte (Round sp66) |


## 8. The first blocks of a launch, timed (Round sp60b; both modes)

Round sp59 found that the size rule, not the instruction set, is what keeps time in the simulator: a kernel's first launch went to the card only
when a static estimate said so, and that estimate counts every instruction of the code once a warp. It is low by the trip count of the kernel's
loops, and programs that launch each kernel once never get a second chance. Nothing static says how long a thread runs; the launch does.

**The rule.** A launch that the history rule and the static first-launch rule leave to the simulator for want of an estimate, of an admitted kernel
that has a loop, on a grid of at least 256 blocks, in a process whose driver has not been started, is *sampled* before the simulator runs it:

1. The process is forked. The copy runs up to 32 blocks of the launch on its one thread, spread evenly over the grid (the blocks of a grid need
   not be alike). It runs each block twice and times the second run: the first pays what a launch pays once (decode, set-up, the chunks of
   memory the launch is the first to write). It takes no further block once the timed runs have taken 2 ms, writes the times to a pipe and
   ends with `_exit`. The process waits for the answer, 50 ms and 0.2 ms more per MiB of live memory at most (the fork copies the
   page tables); a copy that has not answered is ended and the launch is the simulator's. (Since Round sp75 a block's run in the copy is
   bounded, so a long block no longer keeps the copy from answering: section 12.)
2. The timed runs give a per-block time, and the launch is extrapolated from it:
   per-block time x blocks / host threads. Host threads do not scale perfectly, so this errs low, towards the simulator.
3. If that estimate is more than twice the card path (the driver start, the live memory each way, the launch, and a card kernel time assumed to
   be 1/8 of the estimate, the least that [07](07-host-gpu.md) measured), the card runs the launch. Otherwise the simulator does, all of it.
   The first-launch rule's cap of 256 MiB of live memory is not applied here: the copies are charged against a measured time.

The driver start is charged in full against this one launch. The account of simulator time already spent (the ski-rental rule of sp43) is not
consulted here: it is what let short programs pay for a start they could not use.

**Why it is sound.** The sample changes nothing in the process that goes on. A forked copy has its own memory (copy-on-write): the device
memory the sampled blocks wrote, the simulator's caches and counters, everything, ends with the copy. So there is nothing to undo and nothing
that can be undone wrongly: after the sample the process decides between two things it could already do, run the launch in the simulator or run
it on the card from the memory as it is, and each is what it was before this round. The card computes the simulator's bits for the
launch, as for any launch it takes.

What the copy can do outside its memory, and why it does not:

* *Output.* The process's buffered output is flushed before the fork, and the copy ends with `_exit`: no buffer is written twice, no exit handler
  or destructor of the program runs. A kernel that prints is not admitted to the card in either mode (its module has an external call), so it is
  never sampled.
* *The real driver.* A process whose driver start has begun is not sampled: the copy never holds a driver context. (After the start, the rules of
  sp43 and the static first-launch rule find the start paid and decide as before.) (Since Round sp76 a launch is sampled with the driver up
  too, by a copy that cannot call the driver: section 13. This paragraph is the rule of `VGPU_HOST_GPU_SAMPLE_STARTED=0`.)
* *Files and shared memory.* Device memory shared with another process, or mapped by `cuMemMap`, keeps the launch off the card and so is not sampled;
  pinned and managed host memory is the process's own (copy-on-write in the copy) unless the program mapped it shared itself. The diagnostics that
  write files per launch (counters, traces, digests) switch the mode off.
* *Other threads.* The copy has the calling thread only. The sampled blocks run on that thread (the engine's one-thread loop; a cooperative,
  clustered or lockstep launch runs no block in a part and so gives no estimate). If another thread held a lock the copy needs when the process
  was forked, the copy waits for ever, the process stops waiting at the limit and ends it: a lost sample, not a wrong one.
* *The caches on disk.* The copy queues no compile for the code cache and adds no key to the cache of modules, so it leaves no file, whole or
  partial, when it ends or is ended. (Both caches write a temporary file and rename it, or append one 16-byte key, so a copy that did write
  could not have left a broken one; it now writes nothing.)
* *The program's own handling of children.* The copy is reaped by the process (`waitpid` on its pid): at once if it has ended when it
  answers, else at the next sample or by the mode's exit handler, which waits for it, so no child is left when the process ends. A program
  that waits for any child (`wait`, a `SIGCHLD` handler that reaps) can see a child it did not make: `VGPU_HOST_GPU_SAMPLE=0` is for such a
  program.

**What a sample costs.** The fork, the copy's decode and set-up and the sampled blocks: 1 to 12 ms waited, whatever the launch turns out to
be. And more afterwards: the fork write-protects the process's pages, and every page the process writes later takes a fault (Rodinia backprop
with 36 MiB live: 6 ms waited, 19 ms more over the rest of the run, on the A10G rig). Three things keep a short program from paying for samples
it cannot use:

* *Only a kernel with a loop is sampled*: a cycle in its control flow that the entry reaches. Without one a warp runs each instruction once at
  most, so the static count bounds the work, and such kernels took 0.4 to 1.2 times their static estimate over the corpus. (The older test, "a
  branch to a lower address", is true of every kernel: compilers end each one with a branch to itself after its `EXIT`.)
* *Its static estimate is at least 30 ms*, ten times the cost of a small sample, and at least three times what the faults afterwards are
  expected to cost (0.6 ms per MiB of live memory).
* *Or the process can afford it*: what its samples have cost so far, plus this one, is within 2% of the time the process has run. Most programs
  launch within milliseconds of their start, so in practice the estimate decides.

The floor is a rule of thumb fitted to this corpus, not a bound: the static estimate of a kernel with a loop was 0.2 to 80 times off.
`VGPU_HOST_GPU_SAMPLE=0` switches sampling off.

**What the card said** (one A10G, [Round sp60b](rounds/11-rounds-sp36-sp41.md#round-sp60b-host-gpu-mode-the-first-blocks-of-a-launch-timed);
against the same build with `VGPU_HOST_GPU_SAMPLE=0`): PolyBench SYRK 5.16 s to 1.30 s and SYR2K 9.81 s to 2.37 s in both modes, 2MM 172 s to
122 s (what is left is its CPU reference). Geometric mean over 61 programs, against the simulator: 1.12x to 1.18x in exact mode. Outputs are the
simulator's. No program with a sampled launch was more than 1% slower; the estimate was 1.5 to 1.6 times under the simulator's real time, and
the card's kernel time 270 to 770 times under the estimate, not 8.

**What it did not reach, and progress inside a block (Round sp61).** A grid of fewer than 256 blocks has too few blocks to time: a copy would
have to run one of them to its end, and on such a grid a block is a large share of the launch. (Round sp75 bounds the run of a block, and samples
such a grid by whole blocks when its kernel has no loop this section's sample can follow: section 12.) PolyBench COVAR and CORR spend their time in a
kernel launched on 8 blocks of 256 threads (each thread loops over the whole matrix: 9.5 s and 10.2 s in the simulator on server3). Such a launch is
now sampled *inside* its blocks, when its kernel has a loop the copy can follow:

1. *The loops followed* are found in the code once per kernel (`find_probe_loops`, vgpu/host_gpu.hpp): top-level loops (a back edge the entry
   reaches, in no other loop's range) that are decided by a conditional branch whose predicate the loop last sets with an integer compare, one side
   of it the loop's induction register (the one register the loop writes exactly once, by adding a constant to itself, or a register it computes
   once from that one by adding a constant) and the other an immediate, a kernel parameter or a register the loop never writes. In COVAR's
   `covar_kernel` that is the loop over `j2`: `IADD3 R9, R9, 0x1` and `ISETP.GE.AND P1, PT, R9, c[0x0][0x160]` before `@!P1 BRA` back.
2. *The copy* runs each block it takes (all of them up to 32, spread over the grid) round by round until every warp stands in a followed loop
   (or has ended, or 1 ms has passed), and reads per lane how far the compared value is from the bound, in steps: the turns the lane has left.
   A warp has the most of its lanes; a warp in no followed loop adds none. In the first block it then runs the warp with the most turns left alone
   for 4 ms: the turns its induction register moves and the steps it takes give the steps of one turn and the engine's time per step. It takes no
   further block once 12 ms have gone on blocks.
3. *The estimate* is, per block read, the steps taken plus the turns left at the measured cost of a turn; the mean block over the grid, shared among
   the host threads, and never less than the longest block read, at the measured time per step. The same rule as for whole blocks then decides
   (over twice the card path with the driver start and the copies, a card kernel time of 1/8 of the estimate).
4. *The floor* is asked of the static count times the turns the followed loop has if it runs from 0 to the bound that the launch's own arguments give
   (an immediate, or the parameter's value): `covar_kernel` is counted at 87 us, which no floor admits, and its loop is bounded by the parameter
   `m` = 2048, which makes 0.18 s. This only decides whether the sample is worth its cost; the launch goes on the sample.

What it reads is the value the compare will read, so a loop whose bound is data, or whose counter the loop changes in more than one place, is not
followed, and its launch is not sampled. The error it can make is in the cost of a turn: the measured warp's turns stand for every warp's, and a loop
whose later turns cost more than its first (a triangular nest whose inner loop grows) is estimated low, one whose later turns cost less, high.
COVAR on server3: 66,560 turns left read in 64 warps (exactly the sum of 2048 - j1 over the threads' first lanes), 14,000 steps a turn, 41 ns a step:
9.61 s estimated against 9.53 s taken, for a sample of 5 ms. A loop that nvcc unrolls and counts through a register it adds to the counter (`IADD3 R7,
R8, UR4` compared with 0, R8 the bound's complement) is not followed: its remainder loop is, but the warps are not in it while the unrolled part runs, so
such a launch gets no estimate and stays in the simulator (the stand-in test's kernel had this shape and was given `#pragma unroll 1`).

**What the card said** (one A10G, [Round sp61](rounds/11-rounds-sp36-sp41.md#round-sp61-host-gpu-mode-progress-inside-a-block); medians of 3 runs, against
`VGPU_HOST_GPU_SAMPLE=whole`, the rule of sp60b): COVAR 64.7 s to 41.8 s and CORR 64.8 s to 42.0 s (1.54x; what is left is the programs' CPU reference). Each long kernel's one launch, 24.5 s in the simulator, took 1.67 s on the card
path, driver start and copies included; the sample cost 10 ms (COVAR) and 14 ms (CORR). In exact mode both kernels are admitted, and the outputs are the
simulator's in every form. Everything sp60b kept is kept: the fork, the copy's no-cache rule, the age and cost
limits, no sample once the driver's start has begun, the wait for copies at exit; a grid the whole-block sample takes is sampled as before.

**What was tried first and removed** (Round sp60b has the numbers): moving a launch while the simulator runs it (a watcher thread, a copy of the
whole live memory before the launch, the simulator stopped between two blocks) worked against the stand-in driver, and the copy cost too much
for launches that turn out short; sampling the first blocks in the process itself, with every page they touch saved through the card gate and
put back, was exact (every byte compared) and made a sampled block 14 to 26 times slower than a block of the launch, which is both the cost and
the error of the estimate.

## 9. A launch while the card holds the memory (Round sp65; both modes)

With residency (09, section 7), a launch on the card leaves every live 64 KiB chunk the card's alone, and a launch the simulator runs first pulls every such
chunk back and leaves every chunk the simulator's, so the next card launch pushes them all again. The size rule weighed a launch's simulator time alone
against the card path. A short launch between two card launches (PolyBench GRAMSCHM's two small kernels between its long one, LU's `lu_kernel1` between
its `lu_kernel2`, Rodinia gaussian's `Fan1` between its `Fan2`) was therefore kept in the simulator, which then pulled the whole live memory back for it,
and the card launch after it pushed the whole memory again: a streak of card launches broke into ping-pong, and the card path of the long kernel was
mostly that push.

**The rule** (`streak_offload`, vgpu/host_gpu.hpp). When the size rule keeps a launch of a kernel that has an estimate, the driver is up and the card
holds chunks the simulator does not, the launch is priced with what each side really moves:

* keeping it costs its simulator time and the pull of every chunk only the card holds, at the size rule's 10 GB/s;
* sending it costs the launch (7 us), the kernel's time on the card (its last time there; 100 us before it has run there) and the push of the chunks
  only the simulator holds; nothing else moves.

It goes when keeping it costs over twice as much as sending it, and at least 200 us: the margin and the floor of the size rule, on both sides' real
costs. The pushes that the next card launch would make after a simulator launch are not counted (no card launch may follow), so the rule errs towards
the simulator. The bans are kept, and weigh the card path of any launch made while the card holds such chunks, whichever rule sent it, against the
same sum (its simulator time and the pull), since that is what the simulator would have cost: weighed against the simulator time alone, one slow
launch of GRAMSCHM's 1-block kernel1 (0.15 ms on the card, 0.29 ms estimated, sent by the size rule itself) ended the kernel's offloading and broke
the streak for the rest of the run (Round sp65, D). The traffic ban and the first-launch, sampling and start-cost rules are unchanged. `VGPU_HOST_GPU_STREAK=0` keeps the rule of
Round sp61; the summary's `host_gpu_streak: sent=.. card_only_mib=.. rule=..` counts the launches it sent and the bytes only the card held at them.

**Why it is sound.** It changes only which side runs a launch that either side could already run: the launch is admitted by the same checks (code,
launch shape, memory), and the card path is the one every resident launch takes. The card computes the simulator's bits for an admitted
kernel, so outputs cannot change. It never starts the driver (a
program whose launches are all short never pays the start, as before) and never acts when the card holds nothing (the previous launch that touched the
memory ran in the simulator, or the host has read everything back).

What it can cost: a launch it sends whose kernel takes longer on the card than its simulator time plus the pull (the bans end that kernel after one such
launch, as for any kernel); and a launch it sends whose pull happens anyway, because the launch after it is one the card cannot take (a refused kernel
then pulls everything): the launch's card path against its simulator time, a few microseconds.

**What the card said** (one A10G, [Round sp65](rounds/11-rounds-sp36-sp41.md#round-sp65-host-gpu-mode-a-launch-while-the-card-holds-the-memory); medians of 3
runs against `VGPU_HOST_GPU_STREAK=0`): PolyBench LU 11.6 s to 1.5 s in exact mode with `VGPU_HOST_GPU_DIV=1` (22 GB moved each way
to 16 MiB), Rodinia gaussian 0.99 s to 0.53 s, HeCBench axpby 1.11x and 1.14x.
In exact mode without the division LU, gaussian and GRAMSCHM do not change: their short kernel is refused for its code, so the simulator runs it and
the card holds nothing when the next launch comes. Outputs are the simulator's; no program was more than 3 % slower after ten runs of each reading
over it. The run also found a launch with a grid of 0 blocks reaching the card (LU's last `lu_kernel1`): such a launch now stays in the simulator,
which reports its own error. The card's own time for a short launch was 0.014 to 0.15 ms, and for a kernel on one block only twice under the simulator's estimate.

## 10. What a module holds besides code (Round sp66; both modes)

Until Round sp66 a kernel went to the card only if its *module* (a translation unit's cubin) had no `__device__` or `__constant__` variable, no
external function and no device-side launch, and only if the device had no live texture object. One `printf` anywhere in a file, one `__constant__`
table, one texture object anywhere in the program, kept every kernel of it in the simulator, whatever the kernel's own code. The round's census
([data/sp66/census.md](data/sp66/census.md)) measured what that cost; this section says what replaced each rule and why it is sound.

**Module variables.** The simulator gives each variable section of a module (`.nv.global`, `.nv.global.init`, the constant bank `.nv.constant3`)
device memory of its own, in the window the card mirrors, and resolves the module's relocations to it. The real driver, loading the same cubin,
would give the variables copies of its own, elsewhere, and resolve the relocations to those: two copies of one variable, of which only the
simulator's is the one `cudaMemcpyToSymbol`, `cudaMemcpyFromSymbol`, `cudaGetSymbolAddress` and the simulator's kernels see. So the card is not given
the module's own cubin but a copy in which every relocation whose symbol lies in a variable section has been applied with the simulator's address,
as the simulator's loader applies it (the same code writer, `include/vgpu/sass/reloc_patch.hpp`), and taken out of the relocation table, so that
the driver leaves it as it is (`include/vgpu/host_gpu_image.hpp`). Every other relocation (code addresses, function descriptors, `vprintf`) is the
driver's. The kernel on the card then reads and writes the simulator's copy of each variable, which is device memory like any other: pushed when
the simulator wrote it, pulled when the program reads what the card wrote, by the residency protocol of section 3. A pointer the host took with
`cudaGetSymbolAddress` and passed to a kernel is the same address the kernel names the variable by, on both sides. The driver's own copies of
the variables are still made, and nothing addresses them.

What no relocation names is the constant bank as the kernel reads it, `c[0x3][offset]`: the card reads its own bank, which the driver fills from
the cubin at load. So before every launch of such a module on the card, after the launch's memory has been pushed (the window then holds the
simulator's bytes), each constant bank that holds a variable is copied on the card from the window into the card's bank (`cuMemcpyDtoD`). Where the
card's bank is comes from `cuModuleGetGlobal` on the bank's variables, less each one's offset in the bank (all must agree; a bank none of whose
variables the driver names keeps the module in the simulator). Kernels cannot write a constant bank, so nothing comes back. A `__managed__` variable
lives in host memory the window does not hold: a module with one keeps all its kernels in the simulator, as before. The image is made once per
module, with no card; a relocation of a variable it cannot apply (a kind the loader does not apply to variables, an offset past the end of its
section) keeps the module in the simulator with the reason. This is sound because the kernel reads the bytes it
would read in the simulator, at the addresses it would read them.

**External functions, by the kernel.** The external functions a kernel can reach are its call graph's and its own code's call relocations
(`sass::reachable`), and only those decide, by kind (`extern_kind`, `include/vgpu/host_gpu.hpp`): a kernel that calls `malloc` or `free` stays in
the simulator (the card's heap is the card's own, outside the window: a pointer from it kept in memory means nothing to the simulator), so does one
that calls `assert` (a failed assert on the card loses the context, and with it data only the card holds), one that launches kernels (the child
grids, their code and their addresses would be the card's, and the scan has not seen them), and one that calls anything else not named. A kernel
of the same module that calls none of them is judged by its own code. A kernel that calls `vprintf` stays in the simulator: its promise is about memory, not about a second printer, and the card's lines would come from the card's own formatting.

**Texture objects.** A kernel reads a texture or surface only through a texture or surface instruction (`TEX`, `TLD`, `TLD4`, `TXD`, `TXQ`, `TMML`,
`SULD`, `SUST`, `SUATOM`, `SURED`: `texture_mnemonic`), which the exact set does not admit; the handle itself is a number. So a
launch is no longer refused because the device has a texture object: a kernel that fetches is refused by its code (class `never`), and every other
kernel is judged as if no texture existed. Sound for both modes.

**Graphs.** Every kernel launched by `cudaGraphLaunch` is given the graph's conditional handles, and until Round sp66 that refused every such
launch. A kernel reaches the handles only by calling `cudaGraphSetConditional` (or by launching a graph from the device), and a kernel that
calls either is kept in the simulator by what it calls; so a graph's other kernels are judged as any kernel is. Sound for both modes.

**What was not lifted, and why.** The device heap and device-side launches (above); memory mapped with `cuMemMap` or shared with another
process (the window mirrors the allocation table: an alias of one physical page at two addresses, or another process's memory, cannot be made the
same on the card without mirroring the mappings themselves); arguments that point into mapped or managed host memory; cooperative and cluster launches and warp size (launch shapes the card path does not build). `VGPU_HOST_GPU_WIDER=0` restores the safety rules of before the round, for A/B runs (the benchmark's form `exactn`).

## 11. The driver start of a program of many short launches (Round sp69; both modes)

The driver start (0.15 to 0.25 s) is bought by a launch that repays it alone, or, once the program has simulated as long as the start costs
(the account of sp43), by a launch that saves a share of it by itself: the start over 50 launches, 5 ms (the integration of sp64 to sp68, for
four short CUDA Samples that paid a start they did not repay). That share kept the start from Rodinia hotspot, whose 1,000 launches of about
1 ms save four times the start between them: 1.00x with the share (2.27x without it, Round sp66).

The rule (`start_share_launches`, `include/vgpu/host_gpu.hpp`): **the launch must repay the start within 50 launches like it, or, once the
simulator has run its kernel 200 times, within as many launches as it has run it, at most 500. A kernel launched that often is taken to be
launched as many times again, so the start is bought when the launches already made would have repaid it.**

* *Why a count of the kernel's launches.* The four Samples' launches save more each than hotspot's (3.5 ms against 0.9 ms), so no bar on a
  launch tells them apart; what does is that their kernels are launched about 100 times and hotspot's 1,000. Under 200 launches nothing
  changes.
* *Why at most 500.* The launch must then save 0.5 ms at the least. The card path's estimate has a fixed part of 0.107 ms that nothing has
  measured for the kernel before its first card launch, and LargeKernelParameter (two kernels, 1,001 launches each of 0.2 to 0.3 ms) saved
  0.1 to 0.2 ms a launch by it and repaid none of the start on the card.
* *What it does not touch.* The account is asked first, as before; a first launch judged by its static estimate keeps its share of 12.5 ms; a
  launch that repays the start alone, a sampled launch and every launch with the driver up are decided as before. The rule only adds
  launches to those the share of 50 sends, and none that the account alone (`VGPU_HOST_GPU_START_LAUNCHES=0`) would not send.
* *Exact mode.* The rule is the same in both modes and decides only when the driver starts: which kernels may run on the card, and so the
  bits of an exact run, are not its business.

On an A10G hotspot's run (in the relaxed mode that has since been removed) goes from 1.06 s (1.00x) to 0.57 s (1.85x; 2.05x under the account alone, which buys the start 48 launches
sooner); the four Samples send nothing and stay at the simulator's time; Rodinia sc_gpu (launches of 0.6 ms, under the floor) is not helped;
no program of the integration's keep list is more than 3 % slower, and every output is the one it was.

`VGPU_HOST_GPU_START_REPEATS=0` is the rule before. Numbers, the per-kernel table and the tests:
[Round sp69](rounds/11-rounds-sp36-sp41.md#round-sp69-host-gpu-the-driver-start-of-a-program-of-many-short-launches).

## 12. A block that outlasts the wait (Round sp75; both modes)

The summary run of 2026-11-13 left eight programs whose long launches the card never got (pantheonsim #656, #657): the CUDA Samples' three
tensor-core GEMMs (16 to 208 s, one launch each), HeCBench aidw (365 s in the relaxed mode that has since been removed, two of its four launches simulated), and four short
programs. A logged run of each against the stand-in driver ([data/sp75/](data/sp75/README.md)) read what kept every launch, which the mode had
only ever called "size rule". There were three causes, and this round removes two.

| what kept the launch | launches |
|---|---|
| it was sampled, and the copy did not answer: one block takes longer than the whole wait | aidw's `AIDW_Kernel` (400 blocks of 0.58 s each on server3; 78 s kept) |
| no sample was possible: a grid under 256 blocks, and no loop the sample inside the blocks follows | the tensor GEMMs (80 or 160 blocks; their one loop counts by the grid's size and ends on a mask), Rodinia lavaMD (216 blocks) |
| no sample was taken: the static estimate is under the floor and the process too young to afford one | `initRNG` of MC_SingleAsianOptionP and MC_EstimatePiInlineP (1.6 s of each program's 1.8 s; 9.5 ms counted), two kernels of warpAggregatedAtomicsCG, lavaMD again (3.4 ms counted) |
| it was sampled and the estimate does not repay the driver start | the other launches of the two Monte Carlo programs and of warpAggregatedAtomicsCG: kept rightly |
| no sample is possible once the driver is up (until Round sp76: section 13) | aidw's second kernel, on the card (below) |

**The rule: a sampled block is bounded.** The copy ran each block to its end, and the process waited 50 ms for it (`sample_timeout_seconds`). A
block that has not ended after *t* seconds takes more than *t*. So the copy now stops the first run of a block once it has lasted a limit, runs
the block again from its start for as many rounds as the first run made, times that second run, as it times the second run of a whole block,
and answers. The block counts for that time, which is under what it takes; the estimate (per-block time x blocks / host threads, unchanged) is
then under what whole blocks would have given, and the same margin against the same card path decides.

* *A grid the sample of whole blocks takes (256 blocks and up).* The limit is 15 ms (`kSampleBlockSeconds`): both runs fit the 50 ms that were
  already given, with 20 ms to spare for a busy machine, so no sample costs more than it did, and every block shorter than the limit is run and timed exactly as before. What changes
  is the launch whose copy did not answer: it cost the whole wait and said nothing; now it costs less and says "at least this much". aidw's
  first launch: the copy answers after 30 to 34 ms, 14 ms timed for the stopped block, 1.9 s for the launch at least (it takes 78 s), sent.
* *A grid of 2 to 255 blocks whose kernel has no followed loop.* Round sp60b left such grids out because a copy would have to run a block to
  its end. Bounded, it does not, so they are sampled by whole blocks too. A stopped block of a small grid says little unless it was watched
  long enough, so here the limit is what one block must last for the whole launch to repay the card path (`sample_need_seconds`: the least
  simulator time `sampled_offload` sends, with the driver start and the live memory each way), shared among the launch's host threads, with a
  slack of 1.4 (`exec::part_block_limit`), 15 ms at least. When that is over 60 ms (`kSampleBlockCapSeconds`) no stopped block could show
  enough, and no block is run. The wait is 120 ms longer for such a sample.
* *The floor of a small grid.* A sample is taken when the kernel's static estimate reaches a floor or the process is old enough to afford it
  (section 8). The floor of a small grid sampled this way is 30 ms whatever the live memory (`sample_floor`). The term that grows with the
  live memory (three times 0.6 ms a MiB, for the page faults after the fork) asked 0.36 to 1.45 s of the tensor GEMMs, whose one loop's body
  the count sees once (46 to 116 ms); and a grid of under 256 blocks that counts 30 ms has blocks of thousands of instructions, the kernels
  written to loop over their work inside each block. In the corpus the small grids with no followed loop that count 30 ms are the four
  tensor GEMMs of the CUDA Samples (immaTensorCoreGemm with the three above: 80 blocks, 0.95 s here) and one sort pass of radixSortThrust
  (161 blocks, 83 ms counted), whose blocks take turns, so that the copy runs none of them and the sample is the fork alone (1.3 ms in a
  program of 14 s); the next counts 19 ms (HeCBench aop), the 89 others under 10 ms, and none of those is sampled by the floor. By its age a process affords such a sample only when it has run 50 times two runs of a block at the
  cap, 6 s (`sample_expected_seconds`): before that was counted, HeCBench bilateral's third kernel (128 blocks, 6 ms counted, 0.2 s into the
  process) was sampled for 13 ms, 2.5 % of the program, and kept.

**Why it is sound.** Nothing about which side may run a launch changes, nor what the card is sent: the copy is the forked copy of section 8,
with its own memory, and the decision is still between two things the process could already do. A stopped block is a lower bound for that
block: a block runs on one host thread in the launch as in the copy, the rounds are the launch's own (`run_block` is the same loop of
`block_round` without the clock; a bounded run takes turns of 64 steps a warp, so that the clock is read every few milliseconds even in a
block of 32 warps), and the timed run pays none of what a launch pays once. From there on it is the estimate of Round sp60b, with that rule's
one assumption, that the sampled blocks stand for the grid: a launch whose first block is its only long one is sent on it, as it was when
that block ran whole, and costs one card path. Exact mode's bits and the plain simulator's are untouched: the engine's change is behind
`LaunchPart`, which only a sample's copy has.

**What it can cost.** For a sampled grid, nothing more than before. For a small grid, two runs of a block (each up to the limit: 30 to
120 ms in all) and the page faults after the fork, on a launch that then stays (it took under the 0.7 to 1 s the card path needs). That is
the most a launch can lose, and it is not small: immaTensorCoreGemm's sample costs 67 ms here, 5 % of the program when the launch stays, and
on server3 it is at the bar (0.77 s estimated against 0.755 s needed, 0.95 s taken: sent in every run made, by 2 %; on the rig, whose
simulator is 2.5 times slower, the launch is over 2 s). Only kernels that count 30 ms on such a grid, or a process 6 s old, pay it,
and in the corpus those are the four GEMMs, which do not stay.

**What it does not reach** (measured, not built):

* *aidw's second kernel.* Once the first launch has gone to the card the driver is up, and a process that holds the driver is not forked
  (section 8), so `AIDW_Kernel_Tiled`'s first launch has no sample and the static first-launch rule refuses its grid (400 blocks, under 8 a
  multiprocessor): it is simulated, and its second launch goes by its history. Expected on the card: 3 of 4 launches, about half of the
  365 s left (measured: 3 of 4, 193 s of 360 s). What would take it: a trial in the simulator that the card can undo. With the driver up the card holds the memory before the
  launch once the simulator's chunks are pushed, so a launch the simulator has run for a while can be given up and run on the card from that
  copy. Not built. (Round sp76 took it another way: the launch is sampled with the driver up, section 13.)
* *`initRNG` and the other launches under the floor in a young process.* The first launch comes 5 ms into the process and counts 9.5 ms; a
  sample costs 3 to 7 ms and the faults. Nothing known before the launch tells it from HeCBench silu's or s3d's kernels (5 to 14 ms counted,
  under 4 ms taken), and a sample for each of those is 5 % of such a program. Its loops have immediate bounds (32 x 5 turns), which a count
  that weighs a loop's body by its turns would see; a note in the round's text.
* *lavaMD* (216 blocks, 3.4 ms counted, 0.62 s on server3): under the floor of a small grid, as s3d's fourteen kernels on 108 blocks are.
* *A small grid whose kernel has a followed loop that its warps are not in* (a loop nvcc unrolled: section 8) is still sampled inside its
  blocks, where it gets no estimate.

`VGPU_HOST_GPU_SAMPLE_BLOCK=0` is the rule before: the copy runs whole blocks however long, and a small grid with no followed loop is not
sampled (the benchmark's form `exactb0`). `VGPU_HOST_GPU_SAMPLE=whole` keeps the bound and samples no small grid. The
`host_gpu_sample:` line of a bounded sample ends with `small=`, `unfinished=` and `block_limit_s=`, and with `VGPU_HOST_GPU_LOG` the mode now
says why each launch stayed: the `host_gpu_kept` reasons name the rule (`size rule: no estimate, not sampled: ...`, `size rule: the sampled
estimate does not repay the card path and the driver start`, `size rule: the estimate does not beat the card path`, ...), and a
`host_gpu_nosample:` line gives the numbers for a kernel's first two launches, and `host_gpu_overhead: kept_launches=.. decide_s=.. scan_s=..
after_s=.. kernels=..` what the mode's own work cost for the launches that stayed (deciding them, of which scanning each kernel's code once,
and learning from them afterwards).

**On a card** (2026-11-15, one A10G, the summary run's rig and build: [data/sp75/card.md](data/sp75/card.md)). In the relaxed mode that has since been removed (it sent kernels that exact mode keeps in the simulator), tf32TensorCoreGemm goes from 209 s to 3.0 s, bf16TensorCoreGemm from 157 s to 3.8 s, cudaTensorCoreGemm from 16.1 s to 1.3 s and
immaTensorCoreGemm from 3.5 s to 1.2 s, each launch sent on a block stopped at its limit (37 to 54 ms) with an estimate 2.4 times what the
card path needs, and aidw from 360 s to 193 s with 3 of 4 launches on the card. Verify runs found no word of the four GEMMs' results that
differs between the card and the simulator. The keep list, the small grids and the rest of the corpus (115 programs) show no program
slower by the rule; Rodinia particlefilter_float, whose time depends on its clock seed, is not settled (0.947 s against 0.818 s, medians
of 30 runs, with the same decisions in both forms).

**On server3** the round had no card. There, against the stand-in driver, the three GEMMs and aidw's first launch are sent at their
first launch (their estimates: 0.95, 1.37, 1.42 and 1.9 s at the least, against 5.3, 83, 57 and 78 s taken; the card path needs 0.81, 1.14,
1.15 and 0.5 s of them), so is immaTensorCoreGemm's (which the summary run did not hold), and the stand-in test's steps 38 to 40 pin it. What the card made of them is above ([data/sp75/card-run.sh](data/sp75/card-run.sh) is the run); the numbers, the census over the corpus
and the tests are in [Round sp75](rounds/11-rounds-sp36-sp41.md#round-sp75-host-gpu-a-block-that-outlasts-the-wait).

## 13. A kernel's first launch once the driver is up (Round sp76; both modes)

Round sp75's card run left one launch of HeCBench aidw in the simulator: the first launch of its second kernel, `AIDW_Kernel_Tiled`, 149 s of
the program's 193 s. The mode's line for it was `host_gpu_nosample: ... reason=the driver has been started`. The same reason keeps the first
one or two launches of every kernel a program first launches after its driver start (pantheonsim #657, #659: segmentationTreeThrust 34
launches, histogram 4, mergeSort 2; over the corpus on server3, 68 launches in 8 programs).

**Where the refusal came from.** It is Round sp60b's, and deliberate: a sample is taken in a forked copy of the process, and the round had
never forked a process that held the real driver's context, so it did not ("the copy never holds a driver context", section 8; its text lists
"the fork itself against a real driver" under what no test covers). Three things stood behind that caution, and this is what each is worth
now:

| what could go wrong | what holds |
|---|---|
| the copy calls the driver. A forked copy of a CUDA process must not: the driver's threads are not in it, and its context is the process's | the copy runs blocks of the launch in the simulator and ends with `_exit`; the one way from there into the driver is the gate, when a block reads a chunk only the card holds (section 3). Before the fork the process pulls every such chunk; the copy then makes every chunk its own, in its own memory, before it runs a block, so that no access of its blocks reaches the gate; and the gate's pull refuses to run in a copy at all |
| the copy lives long, holding the driver's files and mappings, while the process goes on to launch on the card | since Round sp75 a block's run in the copy is bounded: a copy lives for the wait at most (50 ms and 0.2 ms per MiB of live memory, 120 ms more for a small grid), with or without an answer, as before the start, and the thread that launched waits for it |
| the fork itself harms the process that holds the driver | nothing in the mode can show this without a card. A forked child that never calls the driver is what every program makes that calls `system()` or `popen()` after its first CUDA call; the copy shares the driver's open files, closes its own descriptors when it ends (which releases nothing while the process holds them) and maps nothing new. **Measured on an A10G, 2026-11-15** ([data/sp76/card.md](data/sp76/card.md)): 20 of 20 runs as expected (`sampled=2 sent=2 timeouts=0 after_driver=1 copies_left=0`), no new `NVRM` or `Xid` line, no process left; the fork costs 0.13 ms before the driver's start and 5 ms with the driver loaded |

So the reason no longer holds for the first two, and the third was confirmed on an A10G (above); other cards and drivers were not run.

**The rule** (`sample_allowed`, `sample_card_path_seconds`, `sample_short_before_seconds`, `sample_floor_estimate`, vgpu/host_gpu.hpp). A launch nothing is known of
is sampled with the driver up as it is before the start: the same candidates (an admitted kernel with a loop; a grid of 256 blocks and up, or
a small one sampled inside its blocks or by bounded blocks), the same floor (a static estimate of 30 ms, more with much live memory) or a
process that can afford it (its samples within 2 % of its age), the same copy, the same estimate. What differs is what the launch is held
against: with the driver up it has no start to repay, only the card path's own cost. `sampled_offload` already priced a sample that way
(`driver_started`); the round gives the same card path to the two places that assumed the start: the least a block of a small grid must
show (`sample_need_seconds`), and the bar under which an earlier whole launch of the kernel says "short, do not sample again".

That second bar needed a rule of its own. Before the start it is twice the card path with the start, 0.5 s: over any floor. With the driver
up it is twice 0.1 ms for a program with little memory, and a kernel whose first launch took a millisecond would be sampled at its second
launch for three (the history rule has its two launches at the third whatever the sample says). So with the driver up a kernel is sampled
again only if its whole launch reached the floor: what is asked of a static count, that it be ten samples long, is asked of the measured
launch instead (`sample_short_before_seconds`).

And the other way round: once the simulator has run a kernel's whole launch with the driver up, that launch says more than the static
count does, so where it is the larger it is what the floor is asked of (`sample_floor_estimate`). The CUDA Samples' `histogram256Kernel`
(240 blocks, 2.8 ms counted, 64 MiB live, 4 s into the process) takes 0.19 s here: its first launch is refused by the floor, as before the
round it was refused for the driver's start, and its second launch, which the history rule would also give to the simulator, is now
sampled on the strength of the first. Before the start nothing of this applies: a whole launch long enough to matter there (0.25 s) gives
its kernel an estimate by itself, and the launch is not a candidate for a sample.

**Before the fork.** With the driver up the card may hold chunks the simulator does not (every launch on the card leaves all of them so).
`sample_plan` pulls them, under the lock that guards the card, and leaves them shared (`Where::Both`): it is the pull the simulator makes
before it runs the launch itself, made a moment earlier, and because the card's copy stays valid a launch the sample then sends pushes
nothing it would not have pushed. What it can waste is that pull, when the launch is sent after all: the live memory once, at 10 GB/s, which
the estimate already charges (`sampled_estimate` counts the whole live memory coming back).

**Why it is sound.** As in section 8, the copy changes nothing in the process that goes on, and the decision is between two things the
process could already do. The copy's memory is the launch's real input (pulled before the fork), so its blocks time what the simulator would
run. The copy cannot reach the driver: `sample_copy_begin` runs in it before any block, takes no lock (a lock another thread held at the
fork is held for ever in the copy), ends the copy if a chunk is the card's alone (another thread of the program sent a launch between the
pull and the fork) or if the gate's table was being changed at the fork (the lock's generation is odd), and otherwise makes every chunk the
simulator's; and `pull_chunks` throws in a copy. A copy that ends that way gives no answer and the launch is the simulator's: a lost sample,
not a wrong one. Exact mode's bits are the simulator's for any admitted kernel wherever it runs, so more launches on the card change no
output; the plain simulator has no part in any of it.

**What it costs.** A sample, as before the start: the fork, the copy's decode and set-up, the sampled blocks (1 to 12 ms for short blocks,
30 ms for a block stopped at its limit, up to 120 ms for a small grid), and the page faults afterwards. The floor and the 2 % of the process's
age bound it as they do before the start, and a process whose driver is up has run for a quarter of a second at the least. What is new: the
pull before the fork when the launch is then sent (above), and whatever a fork costs a process with the driver loaded (the driver's own
memory is copied on write like any other): the `host_gpu_sample:` line's `fork_s` of a sample taken with the driver up, against one taken
before the start, is the card run's to read (0.5 ms against 0.5 ms with the stand-in driver, which holds nothing).

**What it decides** (server3, the stand-in driver; [data/sp76/](data/sp76/README.md)). HeCBench aidw against a stand-in card that keeps memory
and computes nothing: `AIDW_Kernel` is sampled and sent as in Round sp75, and `AIDW_Kernel_Tiled`, which was `not sampled: the driver has
been started`, is sampled with the driver up: 2.3 MiB pulled in 0.4 ms, the first block stopped at 15 ms, 2.0 s at the least against a card
path of 0.25 s (its assumed card time, an eighth of the estimate; no start), sent; the sample cost 32 ms. All four launches go, where three
did. Over the corpus the rule touches eight programs
([data/sp76/census.md](data/sp76/census.md): 43 programs on the final build, the 29 that start the driver among them): of 49 lines of
`not sampled: the driver has been started`, 17 launches are now sampled, 12 sent and 5 kept. alignedTypes sends ten first launches (94 to
592 ms estimated), histogram the second launch of `histogram256Kernel` (0.11 s), mergeSort one (85 ms); the rest stay for the reasons the
rule keeps (no loop; under the floor of a process with 290 MiB live; an earlier launch that was short). With the switch off against on,
22 programs that start the driver are within 0.990 to 1.017 of each other in wall time (the stand-in simulates what is sent: this is what
the samples cost).

**What it does not reach.** A kernel with no loop, a launch under the floor in a process too young to afford a sample, a grid of one block:
the reasons of sections 8 and 12 stand with the driver up as before it. A kernel whose first launch after the start is short and under the
floor is simulated twice, as ever, and goes by its history at the third launch. And the rule changes nothing for a program that never starts
the driver: the four programs of pantheonsim #656 that are under the floor in a young process (`initRNG` of the two Monte Carlo Samples,
warpAggregatedAtomicsCG, Rodinia lavaMD) are where they were ([Round sp76](rounds/11-rounds-sp36-sp41.md#round-sp76-host-gpu-a-kernels-first-launch-once-the-driver-is-up)
says what a safe rule for each would need).

`VGPU_HOST_GPU_SAMPLE_STARTED=0` is the rule before: no launch is sampled once the driver's start has begun (the benchmark's form `exactd0`). The `host_gpu_sampled:` line's `after_driver=` counts the samples taken with the driver up, the decision line of such a
sample reads `sampled with the driver up: ... against a card path of ... ms without a start`, and with `VGPU_HOST_GPU_LOG` the pull before a
sample is said (`... MiB only the card held pulled before a sample`).

**On a card** (2026-11-15, one A10G, driver 595.112.07, CUDA 13.0, every stage of `data/sp76/card-run.sh`; [data/sp76/card.md](data/sp76/card.md); `d0` is `VGPU_HOST_GPU_SAMPLE_STARTED=0`). Measured:

* *The fork.* The card test passes with step 13 (the first fork of a process that holds the real driver). Twenty of twenty runs as expected, no new `NVRM` or `Xid` line, no process left. The fork costs 0.13 ms before the start and 5 ms with the driver loaded (what the stand-in could not show: it was 0.5 ms there).
* *Programs changed* (the removed relaxed mode against its `d0` form, median of 10 runs; aidw one run, atomicAggregate 3): aidw 181.57 s to 44.41 s (4 of 4 launches on the card, 3 before); mergeSort 3.18 to 2.12 s (34 of 49 launches on the card to 42); histogram 3.27 to 2.30 s (58 of 68 to 62); segmentationTreeThrust 6.52 to 5.25 s (241 of 514 to 243); atomicAggregate 1.48 to 0.89 s (4 of 6 to 5). Outputs identical (aidw's was not compared).
* *Programs kept* (18 programs, 3 runs): every one within 3 % of its `d0` form but particlefilter_float (1.07 s against 0.88 s), which takes no sample with the driver up and is the program's own race (below).
* *The rest of the corpus* (113 programs, one run): no program that takes a sample with the driver up is slower. Seven programs over 0.3 s read 3 to 11 % slower with the same decisions in both forms; five are within 2 % in the `keep` stage. samples/interval (1.09) and samples/reduction (1.07) were **not rerun**, so the single-run reading stands as unconfirmed: a rerun is `data/sp76/rerun-interval-reduction.sh`.

What the card did not show and nothing here measured: other cards, drivers and toolkits; a program with hundreds of MiB live on the rig (the page faults after the fork). If the fork does harm on another card, `sample_started` in `src/runtime/host_gpu.cpp` is the default to turn off.

**Two things the round measured and did not change** (numbers in the round's text):

* *Rodinia particlefilter_float*, which Round sp75's card run left unsettled (in the removed relaxed mode 5 to 8 % slower than with
  `VGPU_HOST_GPU_SAMPLE_BLOCK=0`). The rule of section 12 does nothing to it: the two forms take the same decisions launch by launch and differ
  in the text of a reason. What moves its time is the program: `normalize_weights_kernel` has a race between blocks (block 0 writes `u[0]`,
  every other block reads it, nothing orders them), so with several host threads the program computes one of several answers per run, clock
  seed fixed or not, and the next kernel's first launch takes 7 to 44 ms or 82 to 235 ms by which it was (server3, four host threads, the
  clock fixed: 90 runs). With one host thread and the clock fixed the two forms are within 1 % of each other at six seeds.
* *The driver start bought late* (pantheonsim #662, HeCBench silu: the start bought at the 31st of 50 launches, with 0.20 s of simulation
  left). Replayed over the launches of the corpus, every rule that keeps the start from silu without knowing what follows (the account at
  1.5 or 2 times the start; an account of what the card would have saved; this kernel's launches so far as the bound on what remains)
  takes more from programs that repay the start than silu loses. Section 11's rule stands.


## 14. What is next

A start-cost account that weighs the gain for programs of many short launches (they lose 0.1 to 0.2 s or gain nothing; section 11 is a first step: it counts
the launches a kernel has had, not yet what the card saved on them); other cards and toolkits.

From Round sp75: a first launch while the driver is up (a trial in the simulator that the card's copy undoes: aidw's second kernel, and every
kernel a program first launches after its start was bought); a static count that weighs a loop's body by its turns where the bound is an
immediate (the floor of section 8 for `initRNG`); the bounded blocks as the fallback of a sample inside the blocks that finds no warp in a
followed loop.

From Round sp76 (which took the first of those another way: the launch is sampled with the driver up, section 13): a rerun of
samples/interval and samples/reduction on a card (`data/sp76/rerun-interval-reduction.sh`); a static count that weighs a loop's body by its
turns, which needs the flow scan to follow a loop that is left through a call to the kernel's `EXIT` and the census of every kernel it would
newly have sampled (design with numbers: [data/turbo-left/README.md](data/turbo-left/README.md)); what the page faults after a fork cost a program with
hundreds of MiB live on the rig (the floor of warpAggregatedAtomicsCG's two kernels); whether the size of a process's environment tilts
particlefilter's race (the card run's `pf` stage ran, but `data/sp76/card.md` records only that the program's own race moves its time: its group counts per form are not kept).

## 17. Very short launches (#660; both modes; merged after sections 15 and 16 were written, so numbered after them)

A10G, 2026-11-17 (`data/turbo-left/card-short-launch.sh`, results in `data/turbo-left/card-2026-11-17-short/`): an empty kernel costs 2.8 us per asynchronous
launch and 8 us with a sync, so the "no gain above 50 us of latency" outcome did not apply. JACOBI1D (20,000 launches of 0.1 ms in the simulator) was never
sent, for four reasons, each now a member of `Costs::short_launches()` (what the runtime uses; `VGPU_HOST_GPU_SHORT=0` is the rule before): the card time
of a kernel never run there was a fixed 100 us, and a launch under 200 us never went, so the prior stayed (now at most `sim_s / 20`, floor 40 us); the
start share of a kernel launched thousands of times was capped at 500 launches (5,000); the first card launch's time includes the module load (65 us for a
20 us kernel) and was kept for good (the prior is kept instead, later times at their least); and one launch that was 60 us late ended the card for its kernel
(three in a row now). JACOBI1D 2.3 s to 1.5 s, sc_gpu 0.96 s to 0.85 s; LargeKernelParameter saves 0.2 ms a launch against the 0.25 ms its start needs, and stays in the simulator.


## 24. The first big launch of a young process (#656, #659; Round r11)

A launch the size rules had no estimate for was sampled (section 8) only if its static count reached a floor or the process was old enough to
afford a sample; a first launch that counts 13 to 600 times under what it takes (mergeSort 80 ms counted, 1.4 s taken; `initRNG` 9.5 ms counted, 6 s taken) was
neither. Since r11 a launch of `kFirstBigThreads` (65,536) threads or more with a loop is excused both while the process has spent nothing on samples, so a
program whose big launches are all short pays one sample (3 to 14 ms) and no more. `VGPU_HOST_GPU_FIRST_BIG=<threads>` sets it, 0 is the rule before. Measured on an A10G:
mergeSort 2.30 s to 0.58 s, MC_EstimatePiInlineP 4.09 s to 0.29 s, MC_SingleAsianOptionP 4.09 s to 0.77 s; histogram, segmentationTreeThrust, reduction, interval,
polybench 3MM and GEMM and others take a sample that sends nothing and are within the rig's noise (`data/r11-first-launch/`). Not caught: warpAggregatedAtomicsCG, lavaMD.

