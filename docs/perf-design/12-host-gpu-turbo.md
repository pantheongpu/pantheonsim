# Host-GPU mode, turbo: kernels whose result is the card's own

Status: experimental, opt-in (`VGPU_HOST_GPU=turbo`). **Turbo is not promised to be bit-accurate, and since 2026-11-17 it sends to the host GPU every kernel the card can
physically run, including the kernels of a simulated profile of another architecture** (a simulated H100 on an RTX 3060: the card runs what it can, from its own
image of the program): section 15 says what changed and why; the earlier text of this document describes the rules that `VGPU_HOST_GPU_FIT=0` keeps. Turbo prints a notice
when it comes on that its results may differ from the simulator's, and they can, by a few bits. `VGPU_HOST_GPU=exact` is unchanged and still bit-exact. Measured on one A10G in *Round sp59*
([rounds/11-rounds-sp36-sp41.md](rounds/11-rounds-sp36-sp41.md#round-sp59-turbo-host-gpu-mode-on-an-a10g)): over 59 programs of PolyBench/GPU, Rodinia
and HeCBench the geometric mean speed-up is 1.23x against 1.12x for exact mode (summed wall time 1.75x against 1.20x), every program's output is the
simulator's, and no kernel was refused for an instruction of no class. Sections 5, 8 and 9 say what the card confirmed and what it changed.
Round sp66 widened what both modes send (section 10; on an A10G the 45 programs it changes go from 1.20x to 1.60x in turbo mode and 1.08x to 1.12x in exact mode). Tracking issue: pantheonsim #587. Rounds sp75 and sp76 (sections 12 and 13) were run on an A10G on 2026-11-15: a kernel's first launch, a block that outlasts the wait and a launch sampled with the driver up; HeCBench aidw 181.57 s to 44.41 s, the three tensor GEMMs of the CUDA Samples 16 to 209 s to 1.3 to 3.8 s, outputs identical ([data/sp75/card.md](data/sp75/card.md), [data/sp76/card.md](data/sp76/card.md)). Read [09-host-gpu-spike.md](09-host-gpu-spike.md) first: this mode is the same machinery with a wider
door.

## 1. Why

`VGPU_HOST_GPU=exact` sends a launch to the machine's real GPU only when the card computes the simulator's bits. Round sp42 measured what
that leaves behind: of the kernels the mode refused for their code, most held an approximate function (`MUFU`), and the rest a tensor-core
instruction, packed half arithmetic or an atomic. Round sp44 measured how far the card is from the simulator on the approximate functions:
4% to 25% of all float patterns differ, by one or two units in the last place.

Turbo mode sends those kernels to the card as well. What the program then gets is the card's value, which is what the program gets on
real hardware, and which is close to the simulator's but not its bits (for the approximate functions this describes the simulator before the model below; what still differs is tensor, half, atomic and reduction arithmetic and the order of atomics). That is a different promise from exact mode's, so it is a different
mode, and exact mode is byte for byte what it was: the same set, defaults, decisions and summary text (a unit test holds a copy of the set).

Since 2026-11-15 the simulator models the approximate-math unit itself ([numerics/math-functions.md](../numerics/math-functions.md)): each of
the ten `MUFU` sub-operations gives the card's bits on all 2^32 inputs, measured on an RTX 3060, an A10G and an H100 PCIe (Blackwell inferred), unless `VGPU_MUFU_EXACT=0`.
The 4% to 25% above is therefore the distance to the simulator of before the model (and of `VGPU_MUFU_EXACT=0`). Nothing in either mode
changed with it: exact mode still refuses a kernel with a `MUFU` outside the division and square root idioms, and turbo mode's `mufu` class
is what it was. Widening the exact set to `MUFU` kernels is open: it needs a card run with `VGPU_HOST_GPU_VERIFY=all` on the cards the model
was measured on, and a rule for the cards where it is only inferred.

## 2. The classes

Every instruction of a kernel is put in one class, once per kernel (`turbo_class` in `include/vgpu/host_gpu.hpp`), and a kernel is
admitted by the classes it holds. Since 2026-11-17 (section 15) a mnemonic nobody named is `unlisted` and turbo admits it: the card runs the cubin's own words whatever the
simulator's decoder calls them. `never` is only what the card cannot run for the simulator. What a kernel needs of its module besides code (variables,
external functions) is decided apart, by the same rules in both modes, in section 3.

| class | instructions | default | switch |
|---|---|---|---|
| exact | today's exact set | admitted | (cannot be switched off) |
| mufu | every `MUFU` form, `RCP64H` and `RSQ64H` included | admitted | `VGPU_HOST_GPU_TURBO_OFF=mufu` |
| half | `HADD2`, `HMUL2`, `HFMA2`, `HSETP2`, `HSET2`, `HMNMX2` | admitted | `=half` |
| float-red | `RED` / `REDG` on a float type (no value returned) | admitted | `=red` |
| call | `CALL`, `RET` and `LEPC` (the return address of an absolute call; Round sp66), when every call is relative, to a label inside the kernel's own code section, or (Round sp66) absolute, to the first instruction of another code section of the module, whose code is then classed with the kernel's | admitted | `=call` |
| extra | `IDP`, `VABSDIFF`, `VABSDIFF4`, `VIADDMNMX`, `VIMNMX3`, `FMNMX3`, `F2IP`, `FSWZADD`, `REDUX`, `LDGSTS`, `LDGDEPBAR`, `LDSM`, `MOVM`, `FFMA2`, `FMUL2`, `FADD2`, integer `REDG`; `B2R`, `NANOSLEEP` (Round sp66) | admitted | `=extra` |
| tensor | `HMMA`, `IMMA`, `DMMA`, `BMMA` (A10G measured 2026-11-16, 36 of 36 launches byte-identical under `VGPU_HOST_GPU_VERIFY=all`; integer popcount arithmetic, so other cards that have it are inferred, not measured) | admitted | `=tensor` |
| atomic-int | `ATOM`, `ATOMG`, `ATOMS` on integers with `ADD`, `MIN`, `MAX`, `AND`, `OR`, `XOR`, `INC`, `DEC`; `RED` with `INC`, `DEC`, `POPC` | admitted | `=atomics` |
| atomic-float | `ATOM`, `ATOMG`, `ATOMS` on a float type (a value is returned) | admitted (refused until Round sp66) | `VGPU_HOST_GPU_ATOMICS=0` |
| atomic-swap | any `CAS`, `CAST.SPIN`, `EXCH` | admitted (refused until Round sp66) | `VGPU_HOST_GPU_ATOMICS=0` |
| never | `TEX`, `TLD`, `TLD4`, `TXD`, `TXQ`, `TMML`, `SULD`, `SUST`, `SUATOM`, `SURED` (their handles are indices into the simulator's tables); `BPT` (a breakpoint or trap ends the card's context); since the review of PR #693 also the sm_90 and later forms that touch state the simulator keeps in its own format or that no card run has shown safe: every tensor-memory-access form (`UTMALDG`, `UTMASTG`, `UTMAREDG`, `UTMAPF`, `UTMACMDFLUSH`, `UBLKCP`, `UBLKRED`: the shim's `CUtensorMap` is the simulator's own layout, not the hardware descriptor), the warpgroup, cluster and tensor-memory forms (`HGMMA` and kin, `SYNCS`, `UCGABAR`, `USETMAXREG`, `ELECT`, `STSM`, `LDTM`, `UTCMMA` ...) and `JMP` (it can leave the section the scan read); any other call that leaves the section (through a register, into the middle of a section, outside the module's code; the absolute call to `vprintf` excepted, section 10) | refused | none |
| unlisted | whatever no class names: `QMMA` (sm_89 FP8), `ATOMS.ARRIVE`, an unnamed mnemonic, an instruction with a field the decoder cannot name (admitted since 2026-11-17; refused until then, as `never`). Exact mode refuses all of it | admitted | `=unlisted` |

Why each admitted class is safe to run on the card, as distinct from numerically equal: the card executes the module's own cubin, so any
instruction is "supported" there. What the mode must guarantee is that the card has everything the kernel reads. The shared checks do
that (section 3), and they are about memory and launch shape, not about opcodes. The classes above touch registers, shared memory and
global memory through the same addresses as `LDG` and `STG`. The `never` class holds what does not: textures and surfaces live in the
simulator's own tables, and a trap ends the card's context. What is unnamed (`unlisted`) is admitted since 2026-11-17 because nobody has looked, not because it is
known to be unsafe, and a result that is the card's own is what turbo promises: no measurement backs the class, which is why it has a switch of its own, as `extra` has. The `extra` class is the
judgement call of this list: no measurement backs it yet, which is why it has a switch of its own.

A kernel of the exact set (the division idiom with `VGPU_HOST_GPU_DIV=1` and the square root idiom with `VGPU_HOST_GPU_SQRT=1` included, Round sp68) is admitted whatever is switched off; turbo mode itself takes every `MUFU` anyway, so the square root idiom changes nothing in it.

A refused kernel is named once on stderr, with the first instruction that refused it, so that a benchmark shows what is still missing:

```
host_gpu_refused: kernel=<name> class=never|off|atomics insn=<the instruction as nvdisasm prints it>   (`unlisted` is admitted: it is named only with VGPU_HOST_GPU_TURBO_OFF=unlisted, as class=off)
```

### The order-dependent classes

An integer atomic that returns the old value leaves the same memory whatever the order (the operations commute), but each thread's
returned value depends on the order the threads arrived in. A program that uses the value as an index (stream compaction, a work
queue) writes the same set of elements in another order on the card than in the simulator, and both are right. A float reduction adds in
arrival order, so its sum differs in the last places from run to run. Turbo mode admits both by default; `VGPU_HOST_GPU_TURBO_OFF=atomics,red`
keeps them in the simulator when a reproducible result matters more than speed.

A float atomic whose value is used, a compare-and-swap and an exchange were behind `VGPU_HOST_GPU_ATOMICS=1` until Round sp66, and are admitted by
default since; `VGPU_HOST_GPU_ATOMICS=0` keeps them in the simulator. Section 2 of the spike found the card disagreeing with itself on programs built
on them (HeCBench `bh` tree building, `aop`): there is no single card answer, a verify run can only report that the two sides differ, and a program's
output can change from run to run. `tools/probes/hostgpu_atomic_repeat.cu` measures which kinds a card repeats. Why the default changed (section 10):
that is what the program gets on the card when it runs natively, which is turbo mode's promise; the returned values of integer atomics, admitted
from the start, do not repeat either; the one risk that is new, a compare-and-swap loop that spins longer on the card than in the simulator, is bounded
by the watchdog (`VGPU_HOST_GPU_TIMEOUT`); and over the corpus every such kernel ran on the card (Round sp59's opt-in runs and its verify runs, which
send every launch: `aop`, `atomicPerf`, `bh`) and finished, with the programs' output the simulator's.

## 3. What is shared with exact mode, unchanged

Everything except the class scan: the launch-shape checks (cooperative and cluster launches, warp size), arguments that point into mapped host memory, device memory outside the allocation table, what a kernel needs of its module (section 10:
its variables, which the card is given at the simulator's addresses, and the external functions it calls, of which turbo mode alone admits
`vprintf`), the size rule and its start cost, the first-launch rule, the sample of a launch's first blocks (section 8), the bans, the
64 KiB residency protocol, the list of diagnostics that switch the mode off, and `VGPU_HOST_GPU_MAX_MIB`.

Three places were checked for an assumption that silently depended on the exact set:

* **Calls.** The eligibility scan reads one code section. A kernel of the exact set could not call anything (the division idiom's slow
  path is checked instruction by instruction). In turbo mode a `CALL` is admitted only when it is relative and its label is an address
  inside the section the scan read; a call through a register, to an absolute address, or into another section makes the kernel `never`,
  because it would run code nobody classed. Since Round sp66 an absolute call to the *first instruction of another code section of the
  module* is followed instead (`call_to_entry`): that is a device function the compiler kept out of line, in a section of its own (every
  `-rdc` build; the CUDA Samples' cooperative-groups and tensor-core kernels), the driver links the call to its own copy of that same code,
  and the scan classes the callee's section with the kernel's, its own calls followed the same way, each section once. The census of the
  round found these calls behind 157 s of simulator time on the Samples, more than every safety refusal together. Modules with external symbols were already refused. Recursion is the card's to run, on the
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
4. `... tensor <out> <build>`: `HMMA`, `IMMA`, `DMMA` (and `BMMA`, admitted 2026-11-16: `e2e_bmma_gemm`, 36 launches, ran on an A10G under `VGPU_HOST_GPU_VERIFY=all` with 0 different words, `data/turbo-left/card-2026-11-16-bmma/`; the exact set does not hold it) gave the simulator's bits in 394 launches, on inputs whose products
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
The form `exactdiv` is exact with `VGPU_HOST_GPU_DIV=1`; `exactr0` and `turbor0` are exact and turbo with `VGPU_HOST_GPU_START_REPEATS=0`
(section 11), `exactl0` and `turbol0` with `VGPU_HOST_GPU_START_LAUNCHES=0`, `exactb0` and `turbob0` with `VGPU_HOST_GPU_SAMPLE_BLOCK=0` (section 12); a program slower than `--long-secs` in the simulator gets `--long-runs` runs of each form.
`--dry-run` is a self-test that needs no card; it is the ctest `hostgpu_bench_selftest`.

```
python3 tools/bench/hostgpu/run.py --build build --corp /path/to/corp --gpu nvidia/a10g --runs 3 \
    --forms sim,exact,turbo,verify --timeout 600 --json hostgpu.json --md hostgpu.md
```

## 7. Tests

| test | where it runs | what it pins |
|---|---|---|
| `test_host_gpu_logic` | every CI job | the mode names; exact's set against a frozen copy and every decoder mnemonic; each class; joined modifiers; calls; the switches and the off list; the sampling rule (how many blocks, what a process can afford, the estimate, what repays the card, when not to sample); the rule for a launch while the card holds the memory (both sides by the formula, the margin and the floor found by bisection, the Costs it reads, when it does not act); the bans' reference while the card holds memory; the zero-grid check; the kinds of external functions and what each mode admits, the texture instructions, the `printf` switch and the atomics default, the card image (each relocation kind applied, the tables shortened, what it refuses) and the code relocation writer (Round sp66); the bounded blocks (Round sp75: the limit of a sampled grid and of a small one, the cap, what a stopped block's time sends, the least time a sample sends, which small grids are sampled and their floor, the reasons a launch is not sampled) |
| `test_host_gpu_tol` | every CI job | the rule, the three readings, NaN, infinity, denormals, signed zero, range ends, the float-array hole, parsing of the tolerances |
| `e2e_host_gpu_stub` | CI jobs with nvcc (a stand-in driver, no card) | turbo reaches the card path for expf, rcp and division beside expf; exact does not; the lines for scripts; the atomics default, `VGPU_HOST_GPU_ATOMICS=0` and the off list; unreadable values; verify against a card that computes nothing; the timeout; a contended reduction under verify; kernels with module variables in both modes, the image the card gets (no relocation of a variable left), the constant bank looked up and filled before the launch, `printf` in turbo only, a live texture object, a kernel that calls `malloc` (Round sp66, `host_gpu_vars.cu`, steps 30 to 36); a short launch that follows a long one to a card that keeps memory and computes nothing while the card holds the memory, and stays with `VGPU_HOST_GPU_STREAK=0` or when the program read the memory back (steps 24 to 26); a zero grid while the card holds the memory gets the simulator's error (step 27); a verify run pushes again only the chunks that differed (Round sp67, step 28); the square root idiom only with `VGPU_HOST_GPU_SQRT=1` (Round sp68, step 29); 400 launches of one kernel, each a hundredth of the start cost, start the driver after the 200th, and not with `VGPU_HOST_GPU_START_REPEATS=0` nor when there are 150 (Round sp69, step 37); a launch whose every block outlasts the wait is sent on a block stopped at the limit, the card given the memory as it was, and with `VGPU_HOST_GPU_SAMPLE_BLOCK=0` its copy does not answer; a grid of 96 blocks with no followed loop is sampled by bounded blocks, and not with the switch off (Round sp75, steps 38 to 40) |
| `hostgpu_bench_selftest` | every CI job | the harness: parsing, output classes, the table, a hang that is killed |
| `e2e_host_gpu`, `tools/card-ci/run-card-ci.sh` | a machine with a card only (passed on an A10G, Round sp59) | section 5, item 1; module variables, `printf` (the order of the lines, stdout a file) and a live texture object on the card, the output the mode-off output to the byte (Round sp66) |

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
it on the card from the memory as it is, and each is what it was before this round. In exact mode the card computes the simulator's bits for the
launch, as for any launch it takes; in turbo mode the card's own, as section 5 found.

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
122 s (what is left is its CPU reference), HeCBench black-scholes 25.1 s to 6.1 s, binomial 7.45 s to 1.36 s and shmembench 6.17 s to 0.30 s in
turbo. Geometric mean over 61 programs, against the simulator: 1.12x to 1.18x in exact mode, 1.14x to 1.27x in turbo. Outputs are the
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
launch shape, memory), and the card path is the one every resident launch takes. In exact mode the card computes the simulator's bits for an admitted
kernel, so outputs cannot change; in turbo mode the card's values are the mode's promise, as for any launch it sends. It never starts the driver (a
program whose launches are all short never pays the start, as before) and never acts when the card holds nothing (the previous launch that touched the
memory ran in the simulator, or the host has read everything back).

What it can cost: a launch it sends whose kernel takes longer on the card than its simulator time plus the pull (the bans end that kernel after one such
launch, as for any kernel); and a launch it sends whose pull happens anyway, because the launch after it is one the card cannot take (a refused kernel
then pulls everything): the launch's card path against its simulator time, a few microseconds.

**What the card said** (one A10G, [Round sp65](rounds/11-rounds-sp36-sp41.md#round-sp65-host-gpu-mode-a-launch-while-the-card-holds-the-memory); medians of 3
runs against `VGPU_HOST_GPU_STREAK=0`): PolyBench LU 11.6 s to 1.5 s in turbo mode and in exact mode with `VGPU_HOST_GPU_DIV=1` (22 GB moved each way
to 16 MiB), Rodinia gaussian 0.99 s to 0.53 s, GRAMSCHM 271 s to 225 s in turbo (one run; its CPU reference is the rest), HeCBench axpby 1.11x and 1.14x.
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
section) keeps the module in the simulator with the reason. This is sound for exact mode as much as for turbo mode: the kernel reads the bytes it
would read in the simulator, at the addresses it would read them.

**External functions, by the kernel.** The external functions a kernel can reach are its call graph's and its own code's call relocations
(`sass::reachable`), and only those decide, by kind (`extern_kind`, `include/vgpu/host_gpu.hpp`): a kernel that calls `malloc` or `free` stays in
the simulator (the card's heap is the card's own, outside the window: a pointer from it kept in memory means nothing to the simulator), so does one
that calls `assert` (a failed assert on the card loses the context, and with it data only the card holds), one that launches kernels (the child
grids, their code and their addresses would be the card's, and the scan has not seen them), and one that calls anything else not named. A kernel
of the same module that calls none of them is judged by its own code. Turbo mode admits a kernel that calls `vprintf`: the absolute call to it is
the one call that may leave the kernel's section (the driver links it at load). The card writes the arguments to its printf buffer and the driver
prints them when the host synchronises with the kernel, which the card path does at once (`cuCtxSynchronize`); the mode flushes the process's
stdout just before such a launch, so the card's lines come after everything the program printed before the launch and before anything it prints
after, which is the simulator's order. The text is the card's formatting of the card's values, as on real hardware; a kernel that prints more than
the card's printf buffer (1 MiB unless the program sets `cudaLimitPrintfFifoSize`, which the simulator does not pass on) loses lines on the card,
as it does natively. **What the card said (Round sp66): nvcc 13.0's sm_86 code calls `vprintf` through a register** (`MOV` of the address's two halves,
which the loader patches, then `CALL.ABS.NOINC R2`), a call the scan does not follow, so on that toolkit a kernel that prints is still refused
(class `never`); the admission covers the immediate form (`CALL.ABS.NOINC` with the address in the instruction, relocation 58 or 75), and no
kernel of the corpus reached the card through it, so the output order above is argued, not measured. Following the register form needs the scan to
tie the register to the `MOV` pair; not done. Exact mode does not admit `printf` (its promise is about memory), nor does a verify run (each launch would print twice, once on each side). `VGPU_HOST_GPU_TURBO_OFF=printf` keeps such kernels out.

**Texture objects.** A kernel reads a texture or surface only through a texture or surface instruction (`TEX`, `TLD`, `TLD4`, `TXD`, `TXQ`, `TMML`,
`SULD`, `SUST`, `SUATOM`, `SURED`: `texture_mnemonic`), which neither the exact set nor any turbo class admits; the handle itself is a number. So a
launch is no longer refused because the device has a texture object: a kernel that fetches is refused by its code (class `never`), and every other
kernel is judged as if no texture existed. Sound for both modes.

**Graphs.** Every kernel launched by `cudaGraphLaunch` is given the graph's conditional handles, and until Round sp66 that refused every such
launch. A kernel reaches the handles only by calling `cudaGraphSetConditional` (or by launching a graph from the device), and a kernel that
calls either is kept in the simulator by what it calls; so a graph's other kernels are judged as any kernel is. Sound for both modes.

**What was not lifted, and why.** The device heap and device-side launches (above); memory mapped with `cuMemMap` or shared with another
process (the window mirrors the allocation table: an alias of one physical page at two addresses, or another process's memory, cannot be made the
same on the card without mirroring the mappings themselves); arguments that point into mapped or managed host memory (the card could reach that
memory through `cuMemHostRegister` at the same address where the device says so, `CU_DEVICE_ATTRIBUTE_CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM`, but
the mode would have to register every such range before a launch and the residency protocol does not cover memory the card reads over the bus; in
the census 3.4 s are behind it, nearly all of them HeCBench `zerocopy`, a program written to time zero-copy access); cooperative and cluster launches and warp size (launch shapes the card path does not build). `VGPU_HOST_GPU_WIDER=0` restores the safety rules and the atomics opt-in of before the round, for A/B runs (the benchmark's forms `exactn` and `turbon`); the class scan's additions (calls into another section, `B2R`, `NANOSLEEP`, `LEPC`) are not switched by it.

## 11. The driver start of a program of many short launches (Round sp69; both modes)

The driver start (0.15 to 0.25 s) is bought by a launch that repays it alone, or, once the program has simulated as long as the start costs
(the account of sp43), by a launch that saves a share of it by itself: the start over 50 launches, 5 ms (the integration of sp64 to sp68, for
four short CUDA Samples that paid a start they did not repay). That share kept the start from Rodinia hotspot, whose 1,000 launches of about
1 ms save four times the start between them: 2.27x in turbo mode in Round sp66, 1.00x with the share.

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

On an A10G hotspot's turbo run goes from 1.06 s (1.00x) to 0.57 s (1.85x; 2.05x under the account alone, which buys the start 48 launches
sooner); the four Samples send nothing and stay at the simulator's time; Rodinia sc_gpu (launches of 0.6 ms, under the floor) is not helped;
no program of the integration's keep list is more than 3 % slower, and every output is the one it was.

`VGPU_HOST_GPU_START_REPEATS=0` is the rule before. Numbers, the per-kernel table and the tests:
[Round sp69](rounds/11-rounds-sp36-sp41.md#round-sp69-host-gpu-the-driver-start-of-a-program-of-many-short-launches).

## 12. A block that outlasts the wait (Round sp75; both modes)

The summary run of 2026-11-13 left eight programs whose long launches the card never got (pantheonsim #656, #657): the CUDA Samples' three
tensor-core GEMMs (16 to 208 s, one launch each), HeCBench aidw (365 s in turbo mode, two of its four launches simulated), and four short
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
sampled (the benchmark's forms `exactb0` and `turbob0`). `VGPU_HOST_GPU_SAMPLE=whole` keeps the bound and samples no small grid. The
`host_gpu_sample:` line of a bounded sample ends with `small=`, `unfinished=` and `block_limit_s=`, and with `VGPU_HOST_GPU_LOG` the mode now
says why each launch stayed: the `host_gpu_kept` reasons name the rule (`size rule: no estimate, not sampled: ...`, `size rule: the sampled
estimate does not repay the card path and the driver start`, `size rule: the estimate does not beat the card path`, ...), and a
`host_gpu_nosample:` line gives the numbers for a kernel's first two launches, and `host_gpu_overhead: kept_launches=.. decide_s=.. scan_s=..
after_s=.. kernels=..` what the mode's own work cost for the launches that stayed (deciding them, of which scanning each kernel's code once,
and learning from them afterwards).

**On a card** (2026-11-15, one A10G, the summary run's rig and build: [data/sp75/card.md](data/sp75/card.md)). In turbo mode
tf32TensorCoreGemm goes from 209 s to 3.0 s, bf16TensorCoreGemm from 157 s to 3.8 s, cudaTensorCoreGemm from 16.1 s to 1.3 s and
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
| the fork itself harms the process that holds the driver | nothing in the mode can show this without a card. A forked child that never calls the driver is what every program makes that calls `system()` or `popen()` after its first CUDA call; the copy shares the driver's open files, closes its own descriptors when it ends (which releases nothing while the process holds them) and maps nothing new. **Measured on an A10G, 2026-11-15** ([data/sp76/card.md](data/sp76/card.md)): 20 of 20 runs as expected in exact and in turbo mode (`sampled=2 sent=2 timeouts=0 after_driver=1 copies_left=0`), no new `NVRM` or `Xid` line, no process left; the fork costs 0.13 ms before the driver's start and 5 ms with the driver loaded |

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

`VGPU_HOST_GPU_SAMPLE_STARTED=0` is the rule before: no launch is sampled once the driver's start has begun (the benchmark's forms `exactd0`
and `turbod0`). The `host_gpu_sampled:` line's `after_driver=` counts the samples taken with the driver up, the decision line of such a
sample reads `sampled with the driver up: ... against a card path of ... ms without a start`, and with `VGPU_HOST_GPU_LOG` the pull before a
sample is said (`... MiB only the card held pulled before a sample`).

**On a card** (2026-11-15, one A10G, driver 595.112.07, CUDA 13.0, every stage of `data/sp76/card-run.sh`; [data/sp76/card.md](data/sp76/card.md); `d0` is `VGPU_HOST_GPU_SAMPLE_STARTED=0`). Measured:

* *The fork.* The card test passes with step 13 (the first fork of a process that holds the real driver). Twenty of twenty runs as expected in exact and in turbo mode, no new `NVRM` or `Xid` line, no process left. The fork costs 0.13 ms before the start and 5 ms with the driver loaded (what the stand-in could not show: it was 0.5 ms there).
* *Programs changed* (turbo against `turbod0`, median of 10 runs; aidw one run, atomicAggregate 3): aidw 181.57 s to 44.41 s (4 of 4 launches on the card, 3 before); mergeSort 3.18 to 2.12 s (34 of 49 launches on the card to 42); histogram 3.27 to 2.30 s (58 of 68 to 62); segmentationTreeThrust 6.52 to 5.25 s (241 of 514 to 243); atomicAggregate 1.48 to 0.89 s (4 of 6 to 5). Outputs identical (aidw's was not compared).
* *Programs kept* (18 programs, 3 runs): every one within 3 % of its `d0` form but particlefilter_float (1.07 s against 0.88 s), which takes no sample with the driver up and is the program's own race (below).
* *The rest of the corpus* (113 programs, one run): no program that takes a sample with the driver up is slower. Seven programs over 0.3 s read 3 to 11 % slower with the same decisions in both forms; five are within 2 % in the `keep` stage. samples/interval (1.09) and samples/reduction (1.07) were **not rerun**, so the single-run reading stands as unconfirmed: a rerun is `data/sp76/rerun-interval-reduction.sh`.

What the card did not show and nothing here measured: other cards, drivers and toolkits; a program with hundreds of MiB live on the rig (the page faults after the fork). If the fork does harm on another card, `sample_started` in `src/runtime/host_gpu.cpp` is the default to turn off.

**Two things the round measured and did not change** (numbers in the round's text):

* *Rodinia particlefilter_float*, which Round sp75's card run left unsettled (turbo mode 5 to 8 % slower than with
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

From Round sp59, less what section 8 does: a start-cost account that weighs the gain for programs of many short launches (they lose 0.1 to 0.2 s
or gain nothing; section 11 is a first step: it counts the launches a kernel has had, not yet what the card saved on them); the card's fixed-point `MUFU.SIN` / `MUFU.COS` in the simulator (done 2026-11-15 for every `MUFU` sub-operation, section 1); a tensor probe whose sums round (`BMMA` is in the tensor class since 2026-11-16);
other cards and toolkits.

From Round sp75: a first launch while the driver is up (a trial in the simulator that the card's copy undoes: aidw's second kernel, and every
kernel a program first launches after its start was bought); a static count that weighs a loop's body by its turns where the bound is an
immediate (the floor of section 8 for `initRNG`); the bounded blocks as the fallback of a sample inside the blocks that finds no warp in a
followed loop.

From Round sp76 (which took the first of those another way: the launch is sampled with the driver up, section 13; its card run is done, above): a rerun of
samples/interval and samples/reduction on a card (`data/sp76/rerun-interval-reduction.sh`); a static count that weighs a loop's body by its
turns, which needs the flow scan to follow a loop that is left through a call to the kernel's `EXIT` and the census of every kernel it would
newly have sampled (design with numbers: [data/turbo-left/README.md](data/turbo-left/README.md)); the open turbo issues #660 and #661 have card scripts there (`card-short-launch.sh`, `card-bmma-verify.sh`); what the page faults after a fork cost a program with
hundreds of MiB live on the rig (the floor of warpAggregatedAtomicsCG's two kernels); whether the size of a process's environment tilts
particlefilter's race (the card run's `pf` stage ran, but `data/sp76/card.md` records only that the program's own race moves its time: its group counts per form are not kept).

## 15. Turbo fit: every kernel the card can run (2026-11-17)

**Decision (owner, 2026-11-17).** `VGPU_HOST_GPU=turbo` is not promised to be bit-accurate. It sends to the host GPU every kernel the card can physically run,
including the kernels of a simulated profile of another architecture (a simulated H100, sm_90, on an RTX 3060, sm_86: whatever the card can run, however it can).
Its results can differ from the simulator's by a few bits, and the mode says so once when it comes on (`host_gpu: VGPU_HOST_GPU=turbo runs the kernels it can on the host
GPU ... results may differ from the simulator's`; not under `VGPU_QUIET`). `VGPU_HOST_GPU=exact` is unchanged: its set, its gates (the card must be the simulated
model: same architecture, same SM count), its decisions and its output.

What changed in turbo mode, and why:

1. **The card is given its own image of each module.** Until now the card loaded the very cubin the simulator ran, chosen for the *simulated* profile, so a
   profile of another architecture than the card's switched the whole mode off at start-up. Now the shim keeps every image of a module's fatbin (cubins and PTX; only in turbo,
   `nvidia/src/fatbin_sass.cpp card_source_from_fatbin`, attached to each kernel as `ptx::EntryFn::card`) and the runtime chooses, once the card's architecture is
   known, the image the driver itself would: a cubin for exactly the card's sm, else the newest cubin of its major with a lower or equal minor, else the newest PTX whose
   `.target` the card can run (the driver compiles it), else none and the module stays in the simulator (`include/vgpu/host_gpu_source.hpp pick_card_image`; the rules are those of
   `sass::runs_on` and `nvidia/src/fatbin.cpp pick_cubin/pick_ptx`, and a unit test holds the first equal to the simulator's over every pair of architectures). The
   start-up gates on architecture and SM count are gone in turbo (kept in exact); a module with no image the card runs is refused per module, with its reason
   (`no image of the module runs on the card (sm_86): it holds SASS sm_90, PTX sm_90`), counted in `refused_image`.
2. **The scan reads the card's image.** A card cubin that is not the simulator's is read with `sass::analyze` (the loader's placement and relocation of code with no device
   memory taken), so the instruction classes, the calls and the externs are those of the code the card runs. Its module variables must be laid out as the simulator's are
   (symbol, section, offset, size, section size all agree, else the module is refused, never guessed), and the relocation patch that points them at the simulator's copy is made on the card's
   image; every kernel's parameters must have the simulator's sizes. PTX is given to the card only when its text declares no variable, texture or surface, has no texture or surface
   instruction, `trap` or `brkpt`, makes no call through a pointer and calls no external function but `vprintf` (`scan_ptx_for_card`, a conservative text scan).
3. **Instruction classes.** A mnemonic no class names is class `unlisted` and is admitted (exact refuses it, its whitelist is untouched); `never` is what needs the
   simulator's tables (texture and surface instructions), `BPT`, a call the scan cannot follow (an unlisted `CALL`, one with a modifier the decoder cannot name, is followed like any other),
   and the sm_90+ forms listed in section 2 (tensor-memory access reads the simulator's own tensor map; the rest has no card evidence: no sm_90 card was available, so none of them was measured).
   A module the card compiles from PTX gets the classes of its text (atomics, approximate math, half, tensor, calls) and the same switches, `VGPU_HOST_GPU_TURBO_OFF` and `VGPU_HOST_GPU_ATOMICS=0`, as a cubin does; its class is that of all its kernels together (one atomic keeps the whole module out). PTX has no `unlisted` class, and its sm_90+ forms (`cp.async.bulk`, `wgmma`, `setmaxnreg`, `elect`, cluster and `mapa` forms ...) are class never. `VGPU_HOST_GPU_TURBO_OFF=unlisted` keeps them out.
4. **A fit check** before a launch goes to the card: block dimensions and threads against the card's limits and the kernel's own (`cuFuncGetAttribute`: the most threads it
   can run, registers, static shared memory), registers per block, static plus dynamic shared memory against the opt-in limit, grid dimensions, and the simulator's memory
   window against the card's total memory (`fits_card`, `window_fits`). A launch that does not fit stays in the simulator and the mode goes on (`refused_fit`). The local
   memory a thread's frame needs cannot be checked beforehand (the driver allocates it at launch). A launch the card refuses after the check ends the mode as before: the simulator runs
   it, and when the card holds data the simulator lacks the process ends with an internal error naming the kernel; a kernel that does not return is ended by the watchdog (status 124).
5. **`VGPU_HOST_GPU_FIT`** (default `1` in turbo): every kernel that passes the safety checks and fits the card goes to the card, whatever its size: no size rule, no sample of its
   first blocks, no first-launch rule, no start cost, no bans, no streak rule, and not the 2 GiB `VGPU_HOST_GPU_MAX_MIB` pipe cap (the card's capacity is what is checked). The card is
   started at the first launch that passes the cheap checks, since the image depends on it. **Why it is the default:** "every kernel the card can run" cannot hold with the size rule
   (a one-warp launch would stay in the simulator), and the owner asked for it. The cost is real for small programs (a driver start in every process, and a push or
   pull of memory whenever the simulator and the card alternate): measured on an A10G, 2DCONV 0.93 s to 2.0 s and 0.01 to 0.06 s programs 0.24 to 0.5 s, while the long ones are as fast or faster (below). On a WSL2 host turbo with FIT=1 turns itself off at once with a notice (the driver start never returns there). `VGPU_HOST_GPU_FIT=0` keeps every earlier rule (and the lazy driver start; the verdicts are redone
   when the card turns out to be of another architecture than the profile).
6. **Kept as they were** (safety, not bit-exactness): cooperative and cluster launches (the card path is a plain `cuLaunchKernel`), host-mapped arguments, VMM, IPC and managed memory, module
   variables the patch cannot apply, externs other than `vprintf`, texture objects, no code section, the diagnostics that need the simulator, and the watchdog.

Built later, narrowly, in section 16.3 (this paragraph described it as not built): retargeting `.target sm_90` PTX to the card's architecture (rewrite the target, let the driver compile what the card's ISA accepts; blockers are `wgmma`, clusters, TMA, `elect`,
`setmaxnreg`... and a launch shape the card lacks). `scan_ptx_for_card` and the image choice are where it would hook in. Multi-card is out of scope.

Limits: a simulated profile with an sm_90-only build (`-arch=sm_90`, no older cubin or PTX) has no image an sm_86 card runs; its kernels stay in the simulator. Libraries loaded with `cuLibraryLoadData` give the card the image the
simulator was given (the fatbin's other images are not kept there). Results of kernels that read the simulated profile's properties (SM count, shared memory size) follow the simulated profile, as the program sees it.

Tests: `tests/unit/test_host_gpu_logic.cpp` (class table, `pick_card_image`, `fits_card` boundaries, `scan_ptx_for_card`, `window_fits`, `parse_fit`) and steps 45 to 52 of
`nvidia/tests/e2e/run_host_gpu_stub.sh` (stand-in driver, no card; steps 10 to 44 run with `VGPU_HOST_GPU_FIT=0`, which pins the rules they were written for). Mutation evidence is in
[data/turbo-fit/README.md](data/turbo-fit/README.md).

### Measurements (AWS, 2026-11-17; every number measured, on the card named)

Setup: 2 x A10G (sm_86, 80 SMs) and 1 x L4 (sm_89, 58 SMs), 14 programs built for sm_86 (PolyBench, Rodinia, HeCBench), one run each, `tools/bench/hostgpu/run.py --forms sim,turbo,verify`. Rows: [data/turbo-fit/results.md](data/turbo-fit/results.md).

| case | geometric mean over the simulator | notes |
|---|---:|---|
| A10G, profile a10g, origin/main turbo | 2.09x | size rule; small programs stay in the simulator |
| A10G, profile a10g, this branch, FIT=1 | 1.79x | long programs as fast or faster; small ones pay the driver start (+0.2 to +1 s) |
| A10G, profile a10g, this branch, FIT=0 | 2.11x | the earlier rules |
| A10G, profile h100 (sm_90 on sm_86), origin/main | 1.00x | no card path: 0 of 7,708 launches reached the card (the simulator runs PTX: "not a SASS module"; results.md section E) |
| A10G, profile h100, this branch, FIT=1 | 2.61x | FDTD-2D 198 s to 4.3 s, LU 63 s to 1.1 s |
| L4, profile h100, this branch | 2.45x | the card runs its sm_86 cubin (lower minor of the same major) |
| L4, profile a10g (80 SMs simulated, 58 real), this branch | 1.69x | exact mode would be off here |

Differences from the simulator: 0 words and 0 bits in every verified launch of every program (measured); the programs hold almost no approximate math, so the size of the differences of an SFU-, half- or atomic-heavy program on these cards is unknown here.
Rodinia srad_v2 faults on the card (an out-of-bounds read of the benchmark) and falls back to the simulator with identical output (measured). Not measured: sm_90-only builds, PTX-only builds on a card, `cuLibrary` modules.

### Known limits of sending every launch (FIT=1)

Sending everything the card can run also sends what the simulator would have run differently, and a launch that goes wrong on the card is not recoverable once the card holds data the simulator lacks:
* A launch the card faults on (an out-of-bounds read the simulator tolerates, Rodinia srad_v2; measured) falls back to the simulator only while the card holds nothing the simulator lacks (srad_v2 was the first card launch). After earlier card launches the process ends with an internal error naming the kernel (the message says which switch avoids it).
* A kernel that waits for blocks the card cannot hold at once (a persistent or spin-barrier kernel sized for the simulated profile's SM count, an H100 profile on a 28-SM card) or for another stream (a producer/consumer pair the simulator overlaps and the card path serialises: one launch at a time) can hang; the watchdog ends the process with status 124 after `VGPU_HOST_GPU_TIMEOUT` (600 s). No static guard is built (it would need the kernel's occupancy on the card and a recognised polling pattern; the simulator's polling analysis is not available to this scan). Inferred, not measured: no such program was among the 14 measured.
* The fit check cannot see the local memory a kernel's frames need (the driver allocates it at launch).
* Avoid with `VGPU_HOST_GPU_FIT=0` (the size rules keep most of these in the simulator) or `VGPU_HOST_GPU_TURBO_OFF=<class>`.
* Follow-ups, not done: the module tables here are keyed by pointers with no unload hook (a reused address could meet a stale entry; older than this change); a forked child inherits a started driver whose context it cannot use (its launches fall back, counted failed); the WSL guard matches "microsoft" or "wsl" in /proc/version and prints in every process; the dxg branch has a unit test but no stub step.

## 16. Turbo next: what the first measurements said, a driver start that overlaps set-up, PTX retargeting (2026-11-17)

Tags: [M] measured on one A10G (sm_86, 80 SMs, driver 595, CUDA 13.0), one run each unless a number says three; [I] inferred; [U] unknown. Rows: [data/turbo-fit/next-results.md](data/turbo-fit/next-results.md). The before build is turbo-fit fae3230b.

### 16.1 Programs the first round did not measure [M]

On the a10g profile with `VGPU_HOST_GPU_FIT=1`, 18 HeCBench programs with math, atomics or sorting and 15 CUDA Samples (atomics, reductions, scans, four tensor-core GEMMs), each as sim, turbo and `VGPU_HOST_GPU_VERIFY=all`:
* Speed-ups are large where the simulator is slow: bf16TensorCoreGemm 183 s to 3.8 s, tf32TensorCoreGemm 238 s to 2.9 s, cudaTensorCoreGemm 16 s to 1.2 s, atomicAggregate 4.5 s to 0.25 s, radixsort2 3.2 s to 0.30 s, mergeSort 7.0 s to 0.49 s, histogram 11.6 s to 1.3 s; the short ones are slower than the simulator by the driver start (binaryPartitionCG 0.02 s to 0.29 s, scalarProd 0.06 s to 0.28 s).
  Three HeCBench programs the simulator could not finish in 240 s (aidw, atomicReduction, binomial) took 44.3, 2.1 and 1.3 s in turbo (native 44.4, 1.9, 1.3 s).
* **Differences from the simulator are real here, unlike the first round's programs.** Words that differ under verify: HeCBench aop 32,595 words (largest absolute 2.27e33, relative 1.15e-5, none outside the default tolerance), bh 23,488 words of which 12,583 are outside tolerance (largest absolute 3.4e-41, a denormal; 1 kernel), attention 1 word (abs 2.2e-3, rel 5.4e-7), CUDA Samples warpAggregatedAtomicsCG 16,000,741 words and simpleAtomicIntrinsics 2 (none outside tolerance; the order of atomics), immaTensorCoreGemm differs in its verify output text (abs 0.01). Every other program: 0 words. Bits differing are not in this table (the benchmark script reads words only): [U].
* HeCBench atomicCost: the simulator run was killed (rc -9) and the turbo run took 72 s with a different output text (38 words against none): not understood [U].
* Three repeats of the 14 earlier programs (same build): the spread between runs is under 10 % except sub-second programs (a few hundredths of a second); native A10G times equal turbo's for the short ones (0.24 s is the process's own CUDA start-up).
* sm_90-only builds on the h100 profile before this round (PolyBench GEMM, ATAX, 2DCONV, FDTD-2D, HeCBench atomicAggregate, silu; each as PTX-only, SASS-only and `-arch=sm_90`): 0 launches on the card in all 18 runs ("no image of the module runs on the card"); turbo equals the simulator's time plus 0.2 s.

### 16.2 Driver start on a helper thread [M for the effect, I for the safety argument]

The start was already a helper thread (`begin_start`: load libcuda, cuInit, the primary context, the address-window reservation) that the launching thread waited for, with a 10 s deadline. What changed: with `VGPU_HOST_GPU_FIT=1` it is begun when the simulated device is created
(`Runtime::prestart`, from `Device::Device`), so it overlaps the program's own set-up; the first launch waits for it as before. Why it is safe: the start thread touches only the driver's function table and the context, which nothing reads before `await_start`; the context is bound on the launching thread by `cuCtxSetCurrent` before every card call (as before); every use is under the launch lock; a failed start (cuInit fails, no device, the window cannot be reserved, no answer in 10 s) turns the mode off once, with the message, as before and the launch that waits for it falls to the simulator; with `VGPU_HOST_GPU_FIT=0`, in exact mode and with `VGPU_HOST_GPU_PRESTART=0` the start stays lazy (stub step 56).
Measured, three runs each, same programs: PolyBench 2DCONV 0.89 to 0.75 s, ATAX 0.42 to 0.32 s, Rodinia backprop 0.46 to 0.34 s, LU 1.10 to 1.00 s, silu 0.28 to 0.26 s; no change for bfs (0.24 s), srad_v2 (0.44 s), hotspot, GEMM, FDTD-2D, black-scholes; HeCBench atan2 0.25 to 0.29 s (three runs each, slower). The gain is the program's own set-up time that the start now hides, at most the start's 0.1 to 0.15 s; programs that finish in the simulator in a few milliseconds cannot gain (0.24 s is also what the same program costs natively), and one small program lost 0.04 s [M], cause unknown [U]. A verify pass of the new build over the 14 programs: 0 words different. Cheap predicate to skip the start for tiny programs: none found (the program's length is unknown when it starts).

### 16.3 Retargeting sm_90 PTX for sm_80, sm_86 and sm_87 cards [M]

A module whose only usable image is PTX for plain `.target sm_90` (no cubin the card runs, no older PTX) used to stay in the simulator. Now (`pick_retarget_image`, `retarget_ptx_text`, `scan_ptx_for_card(..., retarget)`): on an sm_80, sm_86 or sm_87 card the `.target` line is rewritten to the card's (the `.version` is never lowered) and the driver compiles the PTX, unless the text scan finds a form ptxas 13.0 rejects for those architectures (the experiment's micro-tests: `mbarrier.try_wait`, `expect_tx`/`complete_tx`, count-form `mbarrier.arrive`, vector and 128-bit atomics, `fence.proxy.async`, `%aggr_smem_size`, `min`/`max` with `.relu`, packed 16-bit integer arithmetic, fp8/fp4 converts and mma, the cluster directives, and the forms refused in section 15), or declares any module variable (rule 1: the card's module has its own copy). sm_90a PTX, other hosts (sm_75, sm_89, sm_90 and later) and `code=sm_90`-only builds (no PTX) are not retargeted. The summary counts `retargeted=` (modules the driver compiled) and `retarget_refused=` (refused by the scan).
Measured on the A10G with the h100 profile, builds for sm_90 only (table in next-results.md section 1c):
* PTX-only builds and `-arch=sm_90` builds (which also hold PTX) now run on the card: FDTD-2D 119 to 4.2 s (1,500 launches on the card), atomicAggregate 14.0 to 0.25 s, silu 0.81 to 0.26 s, ATAX 1.13 to 0.32 s, 2DCONV 0.99 to 0.77 s, GEMM 0.75 to 0.52 s; `VGPU_HOST_GPU_VERIFY=all` found 0 words different in all of them.
* The driver's compile is paid on the first launch of a module: GEMM built as PTX only spent 2.5 s of card time on it (3.0 s in all, against 1.5 s in the simulator), a cost the driver's cache avoids on later runs of the same PTX [I: the cache, not measured].
* Builds with `code=sm_90` only (no PTX): still no card path (0 launches on the card), as the scan cannot help them.
* A form the text scan does not know (`atom.add.noftz.bf16x2`, sm_90): the driver refuses the compile with error 218 (invalid PTX); the message says so, the module stays in the simulator, the result equals the simulator's, turbo goes on (next-results.md section 3b). Loud, not silent.
* Not measured: sm_80 or sm_87 cards, modules with referenced-only variables (rule 2: not built), kernels that assume sm_90's occupancy or shared memory beyond the card's (the fit check refuses what exceeds the card; persistent kernels sized for 132 SMs are the known hang risk of section 15), the JIT time of large modules (the experiment's ptxas took 16 s on a 25 MB PTX [M, ptxas not the driver]; the driver's start deadline of 10 s does not cover the module load [I]).

### 16.4 Launch limits of the simulated profile [M, stub; the real card could not tell]

Turbo's fit check knows the card's limits, not the simulated profile's, and the simulator's launch validation ran only on launches the simulator itself ran. A launch the profile refuses (launch bounds, block or grid dimensions, registers, shared memory, a cluster) could therefore reach the card. Fixed: turbo mode now calls the simulator's check (`Device::check_launch`) before asking the card, so the launch is refused with the plain simulator's error; a zero grid or block is still the mode's own to refuse with its log line. Found on a stand-in card that a `__launch_bounds__(128)` kernel launched with 256 threads was sent to the card before the fix (the real card would have refused it by the kernel's thread limit, which the fit check reads, so the real-card effect is smaller; not run) [M for the stand-in]. For shared memory the runtime API already refuses a launch the profile refuses (T4 profile, 80 KiB of dynamic memory, A10G card: same error in plain and turbo, before and after the fix) [M, rig]; an H100 profile launch of 120 KiB (over the A10G's 99 KiB) is the simulator's to run and gives the plain simulator's result before and after [M, rig]. Registers, threads and grid limits are the same on every NVIDIA profile and card pair available, so a mismatch there cannot be made on real hardware [I].

### 16.5 Known limits, new and old

Retargeted modules inherit section 15's limits and add: the 10 s driver start deadline does not bound the module compile; one failing module compile is counted once per kernel launch that meets it (`failed` and `refused_image`); a PTX module's classes are those of the whole module.

