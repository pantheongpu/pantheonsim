# Study 5: multi-process sharding and a sampling mode

**Recommendation.**

* Multi-process sharding of a grid: **DO NOT.** It gains nothing on one host (blocks already run on every core), and
  across hosts it needs a coherent shared device memory the project does not have.
* Sampling (run 1 in N blocks and extrapolate): **DO NOT** as a feature of the simulator. It does not fit the purpose
  of a functional simulator or of the workloads that motivate it. The reduced-intensity knobs the workloads already
  have are the supported way to shrink a stress run.

## 1. Sharding

**What exists.** The block loop runs on `host_cpus()` threads by default (`Runner::run`, exec.cpp:879-884; the PTX
engine similarly, interpreter.cpp:13285; launch.hpp:288-294 documents `VGPU_THREADS=1` as the strictly ordered
option). Measured scaling on this machine (int_virus, 64 blocks, base engine, wall seconds per launch, load ~14
from other agents): 1 thread 0.400, 2: 0.196, 4: 0.104, 8: 0.079, 16: 0.052 (7.7x on 16). So the
grid is already sharded; a process-level shard would run the same blocks on the same cores.

**What would be needed for it to mean anything** (more cores than one host has, i.e. several hosts):

* All device memory shared and coherent between the shards. `MemoryManager` keeps allocations as chunk tables in
  one process (memory.cpp; the spill file in memory_backing.cpp is created and owned by one pid and is inherited by
  forks, not shared). The one cross-process path is exported IPC handles, which move that allocation into a
  `memfd` mapped `MAP_SHARED` (memory.cpp:660-690, 1156); ordinary `cudaMalloc` memory is not shared.
* Cross-block atomics and ordering. Atomics that can cross blocks take a striped `std::mutex` per process
  (`atomic_lock`, exec.cpp:193), and plain aligned accesses are relaxed host atomics on the chunk bytes (ARCHITECTURE.md D2).
  Between processes the mutexes would have to become process-shared or lock-free on the shared mapping, and every
  kernel that relies on cross-block flags (CUB's decoupled look-back, mentioned in D2; cooperative launches, which
  require every block resident at once, exec.cpp:847) needs the blocks it waits for to make progress
  in another process.
* Launch-level state: device-side `malloc` (single-threaded by design, exec.cpp:887-892), dynamic-parallelism children
  run after the parent grid, clusters (blocks resident together), `%smid` assignment by host thread (exec.cpp:`sreg`
  0x43), counters (merged by summation, `LaunchStats::add`, so shard-able), CUPTI records, faults.
* Determinism: the contract (ARCHITECTURE.md, "Determinism contract") holds for race-free kernels only at
  `VGPU_THREADS` > 1 already; sharding does not make it worse for them and cannot restore it for racy ones.

For a single host the conclusion is arithmetic: threads are already the cheaper form of the same parallelism (no
second address space, no serialisation of memory). For several hosts there is no distributed-memory design to build
on, and building one is a different product. The speed-up that remains is per-core, which is studies 1-2.

## 2. Sampling

Run every Nth block (or a prefix of the grid) and scale the counters, extrapolating the rest.

* It changes results by definition: output buffers contain only the sampled blocks' writes; any workload that
  verifies its own output (pantheon's `--verify` pass, the golden-kernel comparison that int_virus runs, tests that
  checksum results) would report failures that are not failures, and frameworks would compute with garbage.
  The same-bits rule of this round ("without changing a single result bit") cannot apply to it.
* Extrapolating *counters* is also not sound: they are exact counts from addresses and lane masks, deliberately
  not a statistical estimate (TODO.md:1380 "Performance counters here are not hardware counters"); a grid whose blocks
  differ (edge tiles, data-dependent branches, atomics contention) has a count that depends on which blocks
  were sampled.
* It would not give a performance number either: there is no timing model (project decision of 2026-09-27;
  TODO.md "Not possible by design"), so "throughput" extrapolated from a sample is a count of simulated
  operations per host second, which is a property of the simulator, not the GPU.
* Purpose: the pantheon workloads are saturation tests (TODO.md:1565-1567: a GPU retires ~10^13 ops/s, the CPU
  interpreter ~10^8, so they run at reduced intensity through their own knobs; `scripts/run-pantheon-workloads.sh`
  sets `--kernel_loops 2 --grid_size 4`). Those knobs shrink the problem *and keep the run self-consistent*; a sampled
  grid does neither.
* If a labelled mode were ever wanted (for a soak where only "does it launch and not fault" matters), it belongs in
  the workload's CLI, not the engine, where the workload can also skip its own verification and say so. I would not
  build it in the simulator.

## 3. What I did not do

No prototype: nothing here is a speed question that a measurement could settle; the measurements used are the
thread-scaling run above and code reading. NOT MEASURED: process-level scaling (no implementation exists).
