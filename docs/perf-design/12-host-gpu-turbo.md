# Host-GPU mode, turbo: kernels whose result is the card's own

Status: experimental, opt-in (`VGPU_HOST_GPU=turbo`). Measured on one A10G in *Round sp59*
([rounds/11-rounds-sp36-sp41.md](rounds/11-rounds-sp36-sp41.md#round-sp59-turbo-host-gpu-mode-on-an-a10g)): over 59 programs of PolyBench/GPU, Rodinia
and HeCBench the geometric mean speed-up is 1.23x against 1.12x for exact mode (summed wall time 1.75x against 1.20x), every program's output is the
simulator's, and no kernel was refused for an instruction of no class. Sections 5 and 8 say what the card confirmed and what it changed.
Tracking issue: pantheonsim #587. Read [09-host-gpu-spike.md](09-host-gpu-spike.md) first: this mode is the same machinery with a wider
door.

## 1. Why

`VGPU_HOST_GPU=exact` sends a launch to the machine's real GPU only when the card computes the simulator's bits. Round sp42 measured what
that leaves behind: of the kernels the mode refused for their code, most held an approximate function (`MUFU`), and the rest a tensor-core
instruction, packed half arithmetic or an atomic. Round sp44 measured how far the card is from the simulator on the approximate functions:
4% to 25% of all float patterns differ, by one or two units in the last place.

Turbo mode sends those kernels to the card as well. What the program then gets is the card's value, which is what the program gets on
real hardware, and which is close to the simulator's but not its bits. That is a different promise from exact mode's, so it is a different
mode, and exact mode is byte for byte what it was: the same set, defaults, decisions and summary text (a unit test holds a copy of the set).

## 2. The classes

Every instruction of a kernel is put in one class, once per kernel (`turbo_class` in `include/vgpu/host_gpu.hpp`), and a kernel is
admitted by the classes it holds. It is still a whitelist: a mnemonic nobody named is `never`.

| class | instructions | default | switch |
|---|---|---|---|
| exact | today's exact set | admitted | (cannot be switched off) |
| mufu | every `MUFU` form, `RCP64H` and `RSQ64H` included | admitted | `VGPU_HOST_GPU_TURBO_OFF=mufu` |
| half | `HADD2`, `HMUL2`, `HFMA2`, `HSETP2`, `HSET2`, `HMNMX2` | admitted | `=half` |
| float-red | `RED` / `REDG` on a float type (no value returned) | admitted | `=red` |
| call | `CALL` and `RET`, when every call is relative, to a label inside the kernel's own code section | admitted | `=call` |
| extra | `IDP`, `VABSDIFF`, `VABSDIFF4`, `VIADDMNMX`, `VIMNMX3`, `FMNMX3`, `F2IP`, `FSWZADD`, `REDUX`, `LDGSTS`, `LDGDEPBAR`, `LDSM`, `MOVM`, `FFMA2`, `FMUL2`, `FADD2`, integer `REDG` | admitted | `=extra` |
| tensor | `HMMA`, `IMMA`, `DMMA` | admitted | `=tensor` |
| atomic-int | `ATOM`, `ATOMG`, `ATOMS` on integers with `ADD`, `MIN`, `MAX`, `AND`, `OR`, `XOR`, `INC`, `DEC`; `RED` with `INC`, `DEC`, `POPC` | admitted | `=atomics` |
| atomic-float | `ATOM`, `ATOMG`, `ATOMS` on a float type (a value is returned) | refused | `VGPU_HOST_GPU_ATOMICS=1` |
| atomic-swap | any `CAS`, `CAST.SPIN`, `EXCH` | refused | `VGPU_HOST_GPU_ATOMICS=1` |
| never | `TEX`, `TLD`, `TLD4`, `TXD`, `SULD`, `SUST`; `QMMA`, `BMMA`, the sm_90 and sm_100 tensor and tensor-memory instructions; `JMP`, `NANOSLEEP`, `BPT`; a call that leaves the section; `ATOMS.ARRIVE`; an instruction with a field the decoder cannot name; anything else | refused | none |

Why each admitted class is safe to run on the card, as distinct from numerically equal: the card executes the module's own cubin, so any
instruction is "supported" there. What the mode must guarantee is that the card has everything the kernel reads. The shared checks do
that (section 3), and they are about memory and launch shape, not about opcodes. The classes above touch registers, shared memory and
global memory through the same addresses as `LDG` and `STG`. The `never` class holds what does not: textures and surfaces live in the
simulator's own tables, and the rest is unnamed because nobody has looked, not because it is known to be unsafe. The `extra` class is the
judgement call of this list: no measurement backs it yet, which is why it has a switch of its own.

A kernel of the exact set (the division idiom with `VGPU_HOST_GPU_DIV=1` included) is admitted whatever is switched off.

A refused kernel is named once on stderr, with the first instruction that refused it, so that a benchmark shows what is still missing:

```
host_gpu_refused: kernel=<name> class=never|off|atomics insn=<the instruction as nvdisasm prints it>
```

### The order-dependent classes

An integer atomic that returns the old value leaves the same memory whatever the order (the operations commute), but each thread's
returned value depends on the order the threads arrived in. A program that uses the value as an index (stream compaction, a work
queue) writes the same set of elements in another order on the card than in the simulator, and both are right. A float reduction adds in
arrival order, so its sum differs in the last places from run to run. Turbo mode admits both by default; `VGPU_HOST_GPU_TURBO_OFF=atomics,red`
keeps them in the simulator when a reproducible result matters more than speed.

A float atomic whose value is used, a compare-and-swap and an exchange are behind `VGPU_HOST_GPU_ATOMICS=1`. Section 2 of the spike found
the card disagreeing with itself on programs built on them (HeCBench `bh` tree building, `aop`): there is no single card answer, a verify
run can only report that the two sides differ, and a program's output can change from run to run. `tools/probes/hostgpu_atomic_repeat.cu`
measures which kinds a card repeats.

## 3. What is shared with exact mode, unchanged

Everything except the class scan: the launch-shape checks (cooperative and cluster launches, texture objects, graph conditionals, warp
size), arguments that point into mapped host memory, device memory outside the allocation table, modules with variables, external calls or
device-side launches, the size rule and its start cost, the first-launch rule, the sample of a launch's first blocks (section 8), the bans, the
64 KiB residency protocol, the list of diagnostics that switch the mode off, and `VGPU_HOST_GPU_MAX_MIB`.

Three places were checked for an assumption that silently depended on the exact set:

* **Calls.** The eligibility scan reads one code section. A kernel of the exact set could not call anything (the division idiom's slow
  path is checked instruction by instruction). In turbo mode a `CALL` is admitted only when it is relative and its label is an address
  inside the section the scan read; a call through a register, to an absolute address, or into another section makes the kernel `never`,
  because it would run code nobody classed. Modules with external symbols were already refused. Recursion is the card's to run, on the
  card's own stack, as it is when the program runs natively.
* **Kernels that do not return.** `cuCtxSynchronize` has no deadline, a running kernel cannot be cancelled, and while it runs the card's
  memory cannot be read back, so there is no fallback. In exact mode the card computes the simulator's bits, so a kernel loops on the card
  exactly when it would loop in the simulator. In turbo mode that argument is gone: a loop whose exit depends on a value the card
  computes differently, or a spin lock, is the card's to run. A correct program terminates on a real GPU, so the risk is the risk of running
  the program natively; what is new is that the simulator's own protection (`VGPU_MAX_STEPS`) is one of the diagnostics that switch the mode
  off. The guard is `VGPU_HOST_GPU_TIMEOUT` (seconds, default 600 in turbo mode, none in exact mode): a launch that has been on the card
  longer makes the process say so and end with status 124. Spin locks and compare-and-swap loops are in the opt-in class as well.
* **The ski-rental account and the bans** count time, not opcodes: nothing to change. The simulator's time for a kernel that turbo
  mode admits is learnt the same way (two simulated launches, or one long one).

Two bugs of exact mode's scan were found on the way:

* A kernel refused for its module (variables, external calls) was reported and counted as "not in the exact set" on every launch after its
  first, because the cached verdict kept no reason. The reason is cached now; such launches are counted as unsupported launch shapes every
  time. Fixed for both modes.
* `exact_mods_allowed` looks for the modifiers `F32`, `F64`, `F16` on a `RED`, but the decoder prints a float reduction's type as one
  modifier, `F32.FTZ.RN`. So exact mode admits float reductions on sm_75 to sm_89 (on sm_90 and later the mnemonic is `REDG`, which exact
  mode does not name). Fixed in a commit of its own, so that it can be judged apart from turbo mode: exact mode now refuses them, which
  is the one change to exact mode's set, and it is a change towards what the set was written to be. Turbo mode admits them as its
  `float-red` class. Whether a card run had ever offloaded such a kernel in exact mode was not checked; sp42's verify runs found no
  exact-set kernel that differed, which says the programs sampled there hold none, or that their sums happened to agree.

## 4. Verify by tolerance

`VGPU_HOST_GPU_VERIFY=1` (or `all`) runs each launch on the card and then in the simulator, as in exact mode. The simulator's result is the
one the program gets, and the card copy is given up before the simulator runs, so the next launch on the card starts from the simulator's
memory again: a difference does not carry into the next launch, and verify measures each kernel alone. (Until Round sp59 the card copy was
kept through the simulator's launch, in both modes: the card then ran on its own earlier results, and the simulator's atomics lost updates
in the gate. The price of the fix is that a verify run pushes all live memory before every launch.) Without verify the program
gets the card's values and differences can compound over launches; that is the mode, and only an end-to-end comparison of the program's
output (the benchmark harness, section 6) says what it amounts to.

Exact mode compares bytes. Turbo mode compares the same live ranges word by word (`include/vgpu/host_gpu_tol.hpp`), after skipping
64 KiB chunks that are equal. For each 4-byte word, aligned by address, whose bits differ:

1. as binary32: pass when `|card - sim| <= atol + rtol * |sim|`;
2. else as binary64, the aligned 8 bytes that hold the word, when they lie inside the range and the memory around the word does not
   look like binary32 data: the same rule;
3. else as two binary16 halves: pass when both halves pass.

A NaN matches any NaN under readings 1 and 2, and only the same bits under reading 3 (the low 16 bits of a float look like a half NaN
one time in 32). An infinity matches only the same infinity, and a word that is a binary32 infinity or NaN on one side is not tried as
halves. A word that passes no reading is outside tolerance. The defaults are
`VGPU_HOST_GPU_ATOL=1e-8` and `VGPU_HOST_GPU_RTOL=1e-5`; a value that is not a number zero or more turns the mode off with a message.
`VGPU_HOST_GPU_TOL_TYPE=f32|f64|f16` replaces the guess by one reading for every word (see the limits).

Each launch with a word outside tolerance is reported (the first three per kernel, all with `VGPU_HOST_GPU_LOG=1`): words compared,
different, outside, the largest absolute and relative difference, and the first address outside with both values in hex. At exit:

```
host_gpu_verify: in_set_identical=.. in_set_different=.. other_identical=.. other_different=..     (bytes, as in exact mode; "in set" = admitted)
host_gpu_turbo: mode=turbo launches_card=.. launches_sim=.. card_s=.. sim_s=.. words_compared=.. words_different=.. words_outside=.. words_subnormal=.. kernels_outside=.. max_abs=.. max_rel=..
host_gpu_refusals: refused_never=.. refused_atomics=.. refused_off=.. refused_exact=.. refused_safety=.. too_small=.. failed=..
host_gpu_outside: kernel=.. words_compared=.. words_different=.. words_outside=.. max_abs=.. max_rel=.. first=0x<address>:card=0x<word>:sim=0x<word>     (one per kernel with a word outside)
```

`host_gpu_turbo:` and `host_gpu_refusals:` are printed by turbo mode whenever the summary is (not under `VGPU_QUIET` alone), verify or
not: `launches_card`, `launches_sim`, `card_s` and `sim_s` are the coverage numbers of a plain run. Exact mode prints the two lines only
with `VGPU_HOST_GPU_LOG`, so that its summary is unchanged; there `refused_exact` counts the launches whose kernel turbo mode would take.
`max_abs` and `max_rel` are taken under the reading that passed (binary32 for a word outside tolerance, when both sides are finite), and
`max_rel` only over differences above `atol`: below it the absolute term alone passes the word, and the ratio of two near-zero readings
(the low bits of a double, two small integers) says nothing.

### What changed from the first plan, and why

The plan was "else if the 8 bytes pass as binary64". Taken alone that rule makes verify blind to half of every float array: for a word at
an 8-byte boundary the next word, whatever it holds, reads as the high half of a double, and any change of the low half is at most 2^-20
of that double, which is under any rtol above 1e-6. So reading 2 is tried only where the simulator's bytes do not look like binary32
(`looks_like_f32`: in the 256 bytes around the word, three in four of the nonzero words at 8-byte boundaries have the exponent of a
magnitude between 2^-40 and 2^40; with fewer than four such words the word itself decides), and never for a word in the high half that is
a binary32 infinity or NaN on one side. A unit test changes each element of a float array by half its value and requires each to be
outside.

### Limits

Memory has no types, so the compare guesses, and these are the guesses that can be wrong:

* **Small integers pass.** Two integers below 2^23 read as two binary32 denormals, whose difference is far under `atol`. An index that is
  off by one is inside tolerance. Such words are counted (`words_subnormal`); a kernel with many is one to read by hand, or to run with
  `VGPU_HOST_GPU_ATOL=0`, which makes every denormal difference count.
* **Large integers and pointers fail**, as they should, but so does valid data that is merely in another order (section 2).
* **Where the data is not recognisably binary32** (sparse arrays, magnitudes outside 2^-40 to 2^40, structures) reading 2 can pass a wrong
  low word of an 8-byte pair.
* **Half precision is not judged by the guess.** Two halves in a word read, under reading 1, as one float whose exponent is the top
  bits of the high half. Halves below about 0.25 make a float below `atol`, so every difference between them passes; a difference in the
  low half is a change of the float's low mantissa bits and passes up to a quarter of its range. Memory that holds halves has to be
  compared with `VGPU_HOST_GPU_TOL_TYPE=f16` (every word as two halves, nothing else) and an rtol that takes one unit in the last place
  of a half, `VGPU_HOST_GPU_RTOL=2e-3`; float and integer memory of the same launch then shows as outside, so it is a way to look at one
  kernel, which is how the tensor probe uses it. `f32` and `f64` force the other readings the same way.
* **The guess has small leaks of its own** that the forced readings do not: a float whose low 16 bits change between `0x0000` and
  `0x8000` (0.4% of its value) passes as two halves, a zero of either sign and an equal one.
* **Within tolerance is not the same answer.** A difference of one unit in the last place can send a comparison the other way: a
  branch, a clamp, an iteration count, an index derived from a float. Verify reports what the memory looks like after one kernel, not
  what the program would have done next with the card's values.
* **Programs on compare-and-swap do not repeat on the card.** With `VGPU_HOST_GPU_ATOMICS=1` a difference from the simulator is not
  evidence of anything; compare the card with itself first.
* **Registers and shared memory are not compared**, only device memory in the allocation table, as in exact mode.

## 5. What the card said (Round sp59, one A10G, nvcc 13.0)

1. `nvidia/tests/e2e/run_host_gpu.sh`, `run_host_gpu_stub.sh` and `tools/card-ci/run-card-ci.sh` pass: the expf kernel runs on the card in turbo mode,
   verify finds no word outside the default tolerance (2.1 M of 132 M words differ, by at most 1.2e-7 relative), exact is unchanged.
2. `tools/probes/hostgpu_turbo_card_probes.sh sfu <out>`: `tanh` differs on 5.6% of all float patterns and is outside the default tolerance on 86,770 of
   them (largest difference 8e-6). `sin` and `cos` are another matter: the card's are fixed-point, about 3.6e-7 absolute error, so a sine near zero is far
   outside a relative tolerance (37.6% of the inputs of ordinary size are outside the default for `sin`, 0.3% for `cos`), and for arguments of many turns
   the simulator's own value is not the card's at all. **A kernel that takes `sinf` or `cosf` needs `VGPU_HOST_GPU_ATOL=1e-6` to verify clean**, and its
   results near zero differ from the simulator's by up to 3.6e-7 in a plain run.
3. `... atomics <out>`: integer atomics leave the same memory in 7 of 7 repeats and return other values every time; float sums differ by up to 23 ulp
   from run to run; compare-and-swap and exchange never repeat. The defaults stand.
4. `... tensor <out> <build>`: `HMMA`, `IMMA`, `DMMA` (and `BMMA`, not yet named) gave the simulator's bits in 394 launches, on inputs whose products
   are exact; an accumulation that rounds was not tested.
5. `tools/bench/hostgpu/run.py` (section 6) over the three corpora: the round's tables. Turbo adds five programs to exact's (one of them 78x); the
   `never` class refused nothing; `extra` was met in one kernel, which verified clean. What limits both modes is the first-launch rule, not the classes.

## 6. The benchmark harness

`tools/bench/hostgpu/run.py` runs the prebuilt programs of `tools/bench/real` (PolyBench/GPU, Rodinia, the reduced HeCBench list; the same
layout, arguments and scratch directories) as sim, exact and turbo, `--runs` times each, interleaved, and optionally once as turbo with
`VGPU_HOST_GPU_VERIFY=all`. It writes JSON and a markdown table: wall time, speed-up over sim, launches on the card of all launches, seconds
on the card and in the simulator, launches kept in the simulator by reason, verify's counts, and the program's output against the sim
run's: identical, numeric (the same words, every number with a fraction or exponent within `atol + rtol * |sim|`; integers and hashes
must be equal), or different, with the largest numeric difference. A run past `--timeout` is killed (its own process group) and recorded.
The form `exactdiv` is exact with `VGPU_HOST_GPU_DIV=1`; a program slower than `--long-secs` in the simulator gets `--long-runs` runs of each form.
`--dry-run` is a self-test that needs no card; it is the ctest `hostgpu_bench_selftest`.

```
python3 tools/bench/hostgpu/run.py --build build --corp /path/to/corp --gpu nvidia/a10g --runs 3 \
    --forms sim,exact,turbo,verify --timeout 600 --json hostgpu.json --md hostgpu.md
```

## 7. Tests

| test | where it runs | what it pins |
|---|---|---|
| `test_host_gpu_logic` | every CI job | the mode names; exact's set against a frozen copy and every decoder mnemonic; each class; joined modifiers; calls; the switches and the off list; the sampling rule (how many blocks, what a process can afford, the estimate, what repays the card, when not to sample) |
| `test_host_gpu_tol` | every CI job | the rule, the three readings, NaN, infinity, denormals, signed zero, range ends, the float-array hole, parsing of the tolerances |
| `e2e_host_gpu_stub` | CI jobs with nvcc (a stand-in driver, no card) | turbo reaches the card path for expf, rcp and division beside expf; exact does not; the lines for scripts; the atomics opt-in and the off list; unreadable values; verify against a card that computes nothing; the timeout; a contended reduction under verify |
| `hostgpu_bench_selftest` | every CI job | the harness: parsing, output classes, the table, a hang that is killed |
| `e2e_host_gpu`, `tools/card-ci/run-card-ci.sh` | a machine with a card only (passed on an A10G, Round sp59) | section 5, item 1 |

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
   page tables); a copy that has not answered is ended and the launch is the simulator's.
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
it on the card from the memory as it is, and each is what it was before this round. In exact mode the card computes the simulator's bits for the
launch, as for any launch it takes; in turbo mode the card's own, as section 5 found.

What the copy can do outside its memory, and why it does not:

* *Output.* The process's buffered output is flushed before the fork, and the copy ends with `_exit`: no buffer is written twice, no exit handler
  or destructor of the program runs. A kernel that prints is not admitted to the card in either mode (its module has an external call), so it is
  never sampled.
* *The real driver.* A process whose driver start has begun is not sampled: the copy never holds a driver context. (After the start, the rules of
  sp43 and the static first-launch rule find the start paid and decide as before.)
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
122 s (what is left is its CPU reference), HeCBench black-scholes 25.1 s to 6.1 s, binomial 7.45 s to 1.36 s and shmembench 6.17 s to 0.30 s in
turbo. Geometric mean over 61 programs, against the simulator: 1.12x to 1.18x in exact mode, 1.14x to 1.27x in turbo. Outputs are the
simulator's. No program with a sampled launch was more than 1% slower; the estimate was 1.5 to 1.6 times under the simulator's real time, and
the card's kernel time 270 to 770 times under the estimate, not 8.

**What it did not reach, and progress inside a block (Round sp61).** A grid of fewer than 256 blocks has too few blocks to time: a copy would
have to run one of them to its end, and on such a grid a block is a large share of the launch. PolyBench COVAR and CORR spend their time in a
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
`VGPU_HOST_GPU_SAMPLE=whole`, the rule of sp60b): COVAR 64.7 s to 41.8 s in exact mode and 64.4 s to 41.9 s in turbo, CORR 64.8 s to 42.0 s and 64.8 s to
42.1 s (1.54x in both modes; what is left is the programs' CPU reference). Each long kernel's one launch, 24.5 s in the simulator, took 1.67 s on the card
path, driver start and copies included; the sample cost 10 ms (COVAR) and 14 ms (CORR). In exact mode both kernels are admitted, and the outputs are the
simulator's in every form. Everything sp60b kept is kept: the fork, the copy's no-cache rule, the age and cost
limits, no sample once the driver's start has begun, the wait for copies at exit; a grid the whole-block sample takes is sampled as before.

**What was tried first and removed** (Round sp60b has the numbers): moving a launch while the simulator runs it (a watcher thread, a copy of the
whole live memory before the launch, the simulator stopped between two blocks) worked against the stand-in driver, and the copy cost too much
for launches that turn out short; sampling the first blocks in the process itself, with every page they touch saved through the card gate and
put back, was exact (every byte compared) and made a sampled block 14 to 26 times slower than a block of the launch, which is both the cost and
the error of the estimate.

## 9. What is next

From Round sp59, less what section 8 does: a start-cost account that weighs the gain for programs of many short launches (they lose 0.1 to 0.2 s
or gain nothing); the card's fixed-point `MUFU.SIN` / `MUFU.COS` in the simulator; `BMMA` in the tensor class and a tensor probe whose sums round;
other cards and toolkits.
