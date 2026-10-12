# Host parallelism: what was measured, changed, and left alone (r7-sched)

Written 2026-10-12 from the code at `443e79c0` plus this branch. Every line
reference is to that tree (after this branch where it says so). Items marked
NOT MEASURED were not; nothing here is an estimate presented as a result.

## How a grid reaches host threads

* PTX engine: `launch_grid` (`src/exec/interpreter.cpp`, `nthreads` at ~13386).
  One thread when the launch is cooperative, uses a non-default scheduler
  (`ordered`), has one block, or its kernel can allocate on the device
  (`allocates_while_running`, ~13268). Otherwise `min(worker_count, SMs/cluster)`,
  `worker_count` being `VGPU_THREADS` or `host_cpus()` (~13284), capped by the
  block count. Blocks are handed out in chunks of `units / (8 * nthreads)` from
  one `fetch_add` (~13428).
* SASS engine: `Runner::run` (`src/sass/exec.cpp`, ~875-945). The same caps,
  written a second time. Blocks are handed out one at a time from
  `next_unit_`; that counter is also what `clusterlaunchcontrol.try_cancel`
  takes the next unit from (`exec_ops.inc`, `exec_clc`), so the hand-out order
  is observable by a kernel and was not changed.
* `VGPU_THREADS=1` runs the blocks inline on the calling thread, in order
  (`launch.hpp` ~line 294).

## H1. Thread creation per launch: CONFIRMED, FIXED

Before: both engines created `nthreads` `std::thread`s per launch and joined
them (`interpreter.cpp` ~13410, `sass/exec.cpp` ~941 on the old tree).

Measured (`tiny.cu`: `out[i] += ...`, grid G, 32 threads, 10000 launches in a
loop, rtx3060 profile, host load average 12-14 on 20 CPUs throughout, so
absolute numbers are inflated; A and B alternated 5 times, median of the 5):

| engine | grid | VGPU_THREADS | before (us/launch) | after (us/launch) |
| --- | --- | --- | --- | --- |
| SASS | 1 | default (runs inline) | 5.4 | 5.6 |
| SASS | 8 | default (8 threads) | 267 | 43 |
| SASS | 64 | default (20 threads) | 657 | 218 |
| PTX (`VGPU_SASS=0`) | 8 | default | 213 | 31.5 |

(Spread, min..max over the 5 runs: SASS grid 8 191..356 before, 35..57 after.)
With one thread both versions run inline and match, which is also why grid-1
launches did not change.

What was changed (`include/vgpu/exec/worker_pool.hpp`, `src/exec/worker_pool.cpp`):
a pool of parked threads made on first use. The semantics that were kept, each
with a test (`tests/unit/test_worker_pool.cpp`):

* `run_parallel(n, body)` runs `body(0..n-1)` on `n` distinct threads that are
  alive together (a body may wait for another; the old code allowed that).
* The caller never runs a body, only waits (the caller holds the API lock and
  owns `thread_local` call state).
* A body starts with the caller's floating-point environment (`fegetenv`) and
  signal mask. The old threads inherited both from the creating thread. This
  matters: `Interpreter`'s constructor reads `fegetround()` on the worker
  (`interpreter.cpp` ~2240), and a body that dies between `fesetround` and its
  restore would otherwise leave a directed mode on a thread that now lives on.
  CPU affinity and scheduling attributes are those of the thread that created
  the worker; they change no result.
* Nested calls and calls from several host threads at once take idle workers
  and make more, so none waits for a worker another holds.
* Fork: the pool records its process id; a child that finds a pool of another
  pid abandons it and builds its own. No lock is taken on that path.
* The only engine `thread_local` is `t_bc` (`interpreter.cpp` ~1233), cleared
  at the start of every texture fetch, so persisting across launches is safe.
  `grep thread_local src include` found no other state the executors read.

Idle workers poll for 40 us after a job when launches have been arriving less
than 40 us apart, and for 2 us otherwise, then sleep. Without the polling a
sleeping worker costs a futex wake: with plain condition variables only
(an earlier version of the pool, 2000 launches) the grid-8 SASS case was
108-124 us per launch, against 25-29 us with the polling, same machine, same
quarter hour. A microbenchmark of `run_parallel` alone (empty bodies, 5000
calls) measured 131 us per call for 8 workers and 199 us for 20 when they had
to be woken, and 2-40 us (noisy) when hot.

Not claimed: CPU-seconds. `/usr/bin/time` under this WSL kernel under-reports
short-lived threads (the old per-launch threads), so user+sys of before and
after are not comparable here. Wall time is what is reported.

## H2. Block hand-out and per-block reuse: NO CHANGE (measured, nothing to gain)

* Chunked hand-out / work stealing: PTX already chunks; SASS takes one block per
  `fetch_add` (~5 per microsecond at most, a block is >= 4 us even for a
  one-warp kernel). No mutex or atomic appears in a `perf` profile of
  1024-block launches at 20 threads (no mutex, lock or futex function among the
  sampled symbols; user-space samples only, the kernel's are not visible in
  this VM), so the per-block `stats_mu_` lock and the
  counter are not contended. Not changed.
* Sorted scheduling: block cost is unknown before it runs. Not applicable.
* NUMA / core pinning: this host (a WSL VM) has no NUMA topology to measure on.
  NOT MEASURED, not changed.
* Per-worker reuse of Block/Warp allocations: the SASS worker already keeps one
  `Block` for all its blocks (`sass/exec.cpp` `worker` lambda), reusing the
  vectors; `init_block` resets by `w = Warp{}` (a 3 KB struct plus a vector
  move). `perf record` (cpu-clock, 2 kHz, VGPU_THREADS=1) of 1024 one-warp
  blocks: `init_block` 2.5% of samples, `malloc`+`cfree` 2.4% (some of it from
  `count_access`, `r7-sass`'s code), and for 256-thread blocks with a 16 KB
  shared tile (`big.cu`) `init_block` is below the top 15 symbols (all >=1.5%).
  Reusing the register-file vector across blocks could save part of the malloc
  share, i.e. at most ~1-2%, below this machine's noise (load 12+), so it was
  not attempted. The PTX engine builds a `BlockState` per block and
  `setup_block` is 8.9% of a 1024-block grid of one-warp blocks (`p2.data`);
  that file belongs to `r7-ptx`, the default engine is SASS, and the gain would
  apply only to tiny blocks. Not changed. `step_block` allocates a
  `std::vector<size_t> runnable` on every call (~2642); noted for `r7-ptx`.

## H3. Skipping idle work: NO CHANGE

Exited warps are skipped by one mask test in the warp loops
(`sass/exec.cpp` ~1074, `if (!(w.alive & ~w.exited)) continue`) and
`run_warp_until_yield` returns at once for a `Done` warp. A tail block whose
threads all fail the bounds check still runs its prologue, and those
instructions are counted (`instructions`, `thread_instructions`); skipping
them would change reported counters, which this round forbids. Dropped.

## H4. Warps of one block on several host threads: ANALYSIS ONLY, NOT IMPLEMENTED

Potential. The pantheon stress workloads at the sizes the runner uses
(`--grid_size 4`, docs/pantheon-workloads.md): `VGPU_TRACE=1` on the two stress
binaries available locally, 3 s each:

    mma_virus   481 launches: all grid 4x1x1, block 256x1x1  (8 warps)
    fp64_virus  513 launches: all grid 4x1x1, block 256x1x1

So a launch has 4 blocks to give 20 host CPUs (`min(blocks, cpus)` threads),
each block has 8 warps. If the 8 warps of a block could run concurrently the
usable parallelism would be 32 against 4 now: at most 5x on a 20-CPU host
(20/4), less in practice. That bound is arithmetic, not a measurement; NOT
MEASURED beyond the launch shapes above.

Why it is not done. What makes today's runs reproducible is that the order in
which the warps of a block interleave is a fixed function of the program:
round-robin, up to 4096 instructions per turn in SASS (`sass/exec.cpp` ~1078),
the `Scheduler` in PTX. Running them concurrently changes, for kernels that
are race-free in the CUDA sense, nothing a correct program can see; but this
simulator also promises (ARCHITECTURE.md, GOVERNANCE.md:30) reproducible runs
and exact counters, and these observable things depend on the interleave:

1. Order of integer atomics whose return value is used (slot allocation,
   stream compaction): the output permutation changes run to run.
2. Order of floating-point atomics and reductions: the bits change.
3. `%clock`/`%clock64`: one counter per block bumped per warp instruction
   (`interpreter.cpp` ~2598, 2817, 3166).
4. The race detector's per-block shadow memory and epoch (`SharedShadow`,
   ~218; `VGPU_RACE`), which would race with itself.
5. Block-level state that every warp mutates: named barriers, mbarriers, the
   wgmma snapshots, Tensor Memory, `bar.red` accumulators (`BlockState`, ~2531;
   SASS `Block`, `sass/exec.cpp` ~120).
6. A kernel that does have a shared-memory race between two `__syncthreads`:
   its result today is one fixed answer; concurrent it is one of several. That
   is the very class of program VGPU_THREADS=1 exists to debug.
7. `printf` order within a block.

Items 1, 2, 3, 6 are unavoidable for any kernel that uses those features, and
whether a given kernel does cannot be decided from the instruction stream in
general (a store index is data). So determinism cannot be proved, and an opt-in
switch would give an `--fast` mode whose results differ from the default's on
exactly the programs the simulator is used to debug. Per the brief, nothing was
implemented. If someone wants it later, the safe subset is "kernels whose only
cross-warp communication is shared memory ordered by `bar.sync`, with no
atomics, no `%clock`, no race detection", checked by a conservative scan of the
instruction stream, plus an equality test against the sequential run on a
corpus; that is a project of its own.

## H5. Concurrent kernels from different streams: see `streams.md`

## H6. Cooperative launches on a single worker: ANALYSIS ONLY, NOT IMPLEMENTED

A cooperative launch makes every block resident and runs them round-robin on one
worker (`docs/cooperative.md`; PTX `run_grid_cooperative`, `interpreter.cpp`
~2362, `kSlice = 256`; SASS `sass/exec.cpp` ~903, 256 instructions per warp
turn, ~921). One worker is also the only schedule in which a spinning warp
(the grid barrier is a sense-reversing counter in device memory) is always
preempted by a bounded slice, so the blocks it waits for always run.

A multi-worker version is possible in principle: each worker owns a fixed subset
of the resident blocks and round-robins its own; the barrier's spin then waits
on another worker's progress rather than on its own round. It cannot hang as
long as every block is owned by some running worker. What it gives up is the
property docs/cooperative.md names as the reason for one worker: the order is
fixed and the same every run. After a `grid.sync()` the phases are ordered by
the barrier, but within a phase the interleave of blocks differs from run to
run, so integer-atomic slot order and floating-point atomic order (items 1, 2
of H4) become run-dependent where today they are not. Whether any workload would gain is
not known: a grep of the pantheon stress suite's sources (`pantheon/kernels`,
47 directories) finds no `cooperative_groups`, `cudaLaunchCooperativeKernel` or
`grid.sync`, so there is no measured benefit to trade for the loss of
reproducibility. Not implemented; the single-worker behaviour is
covered by `a_cooperative_launch_still_runs_on_one_thread` (test_sched) and
`cooperative_grid` / `sched_pool` (e2e).
