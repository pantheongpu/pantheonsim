# Study 3: memoising deterministic kernels

**Idea.** Key a launch on (module, kernel, grid, arguments, input memory); if seen before, replay the recorded writes
instead of executing.

**Recommendation.** **DO NOT make it a default. DO LATER, as an explicit opt-in (`VGPU_MEMO=1`, name illustrative),
and only after tiers 1-2 exist,** restricted to launches that pass a static eligibility check. The potential is
real and large on some of the project's own workloads (below), but the correctness conditions are strict, the
payoff is exactly where "the simulator ran the kernel" is the point of the test, and every default-on version
would make a kernel's side effects depend on history.

## 1. What a launch does besides computing (what a replay must reproduce)

From `Device::launch` / `run_kernel` (runtime.cpp:1040-1085) and the SASS runner:

* writes to device memory (global, and shared/local that die with the block): replayable as a write set;
* `LaunchStats` returned to the runtime and reported (`report_counters`, VGPU_COUNTERS, `vgpu ncu` records through
  `VGPU_COUNTERS_FILE`): replayable only if stored with the entry; CUPTI callbacks and activity records and
  telemetry (`note_kernel`, which publishes utilisation as wall time spent in kernels) see a launch that took no time;
* device-side `printf` (exec/device_printf.hpp): output order and content must be replayed;
* device `malloc`/`free` (heap state on the `MemoryManager`), dynamic-parallelism child launches
  (`exec::devrt`), cooperative-launch workspace, graph launches (`device_launches`): not replayable;
* fault injection: `fault_->check_lost`, `maybe_hang` run before the launch (runtime.cpp:1033-1037) and the
  `AccessFault` hook corrupts loads while armed (include/vgpu/memory.hpp, `AccessFault::load_pending`): with any fault armed the launch
  *must* run;
* atomics and cross-block ordering: with the default one-thread-per-core grid (exec.cpp:879) a float atomic or racy
  kernel does not have one answer; a memo would pick one and freeze it;
* clocks: the SASS clock registers return the warp's instruction count (exec.cpp:632-637), which is
  deterministic, so unlike hardware they do not disqualify a kernel; `%smid` depends on the host thread
  (exec.cpp:`sreg` 0x43) and does when the grid is larger than the SM count;
* managed/UVM memory and host-mapped ranges (`host_maps_`): the page's other writer is the host;
* texture/surface objects, constant banks, module `__device__` variables: reads outside the argument pointers;
* the race detector, sanitizer-style modes, `VGPU_KERNEL_DIGEST`, `VGPU_SASS_TRACE`, the debugger: the launch must run.

## 2. The input set is the hard part

A kernel may read any live allocation (pointer tables, a pointer stored in memory, module globals, constants
banks, textures). The argument words are only the roots. Sound options:

* hash **all** live allocations (cost: the touched bytes of device memory per launch; in the pantheon runs below,
  megabytes; for an LLM-sized device memory, gigabytes: more than running the kernel);
* track reads (shadow bitmaps: slows every launch, and the engine's loads are the hot path);
* version counters per allocation, bumped on every write: cheap, but a re-run of an idempotent kernel bumps the
  counter although the bytes are the same, which is exactly the case that would hit, so hits would need content
  hashes of the written ranges after all;
* restrict to launches whose reads are provably inside the argument pointers' allocations: needs static analysis
  of the SASS for any load whose address does not derive from a parameter; the decoder gives the instructions,
  but pointer provenance through shared memory and registers is a data-flow analysis the project does not have.

## 3. Measured potential on the project's own workloads

`VGPU_KERNEL_DIGEST` (runtime.cpp:158-215) logs, after every launch, a hash of each allocation the arguments point
into. I ran eight unmodified pantheon workloads on the simulator (a10 profile, `VGPU_THREADS=1`, grid 4, 2 loops,
3 s each, binaries from `pantheon/build-vgpu-86`, `data/memo-fixed-point-pantheon.txt`) and counted launches whose
post-state hash equals that of the previous launch of the same kernel. Because the next launch starts from the previous launch's post-state, an equal
pair means the kernel reproduced its own pre-state: a memo keyed on content would have hit (these workloads pass identical arguments each launch,
which the digest does not record: an assumption).

| workload | launches | fixed-point launches |
| --- | --- | --- |
| compute_virus | 14,143 | 14,142 (100.0%) |
| memory_retention_bake | 7,401 | 7,399 (100.0%) |
| cache_latency | 2,030 | 2,028 (99.9%) |
| memory_bank_thrash | 1,497 | 1,496 (99.9%) |
| fp64_virus | 653 | 652 (99.8%) |
| omni_virus | 1,140 | 1,136 (99.6%) |
| int_virus | 6,260 | 1 (0.0%): `sink[tid] += ...` accumulates, every launch differs |
| pulse_virus | 60 | 1 (1.7%): same |

So for six of eight a memo would have skipped essentially every launch after the first, and for two it would
never hit. Limits of this measurement: only the argument-reachable allocations were hashed; 3 s of a tiny grid;
the digest shows the effect on state, not that the *reads* were covered.

## 4. Why not by default

* These are **stress** workloads. On hardware, repeating the launch is the experiment. In the simulator the same
  launch is also what exercises the pantheon tool's logic around it (its telemetry loop, verification pass,
  injected faults via `--inject_error` and `vgpu fault`). Skipping the execution would turn "the kernel ran N
  times" into "the first run's result was reused N-1 times", which a workload that reports a throughput or
  counts launches will happily print as a number. The workloads' reported TOPS are already meaningless
  (no timing model, the project's decision of 2026-09-27); with a memo they would also be unbounded.
* A silent wrong answer is the unforgivable bug class (ARCHITECTURE.md D5). Any eligibility bug becomes a stale
  result with no diagnostic, and the memo would be keyed on history, so a failure appears only on the second
  launch of a long test.
* It cannot help the accumulating workloads (int_virus, pulse_virus), which are the most expensive ones
  (72 s and 161 s at full size, TODO.md:1457-1459). Tiers 1-2 help all of them.
* It interacts with every observability feature in section 1.

## 5. If built

Opt-in only; eligibility refused (launch runs normally) when: any fault armed or injected; atomics/`RED`/`ATOM*`,
`SHFL`-style cross-lane ops are fine but cross-block ordering ops are not; printf, malloc/free, device launches,
`BAR`-dependent global communication across blocks, cooperative launches; textures/surfaces/managed/host-mapped
memory in reach; race or trace modes; more than one host thread **unless** the kernel has no atomics and no
overlapping writes (needs the same analysis). Key: relocated code hash, grid/block/cluster, shared bytes, argument
bytes, content hash of all live allocations (bounded by a size limit; over the limit, ineligible). Entry: the
write set (ranges and bytes, the diff against pre-state), the `LaunchStats`, any printf bytes. A replay must apply
the counters and the CUPTI/telemetry notifications exactly as a run does. Verification mode: run every Nth hit for real and
compare, loudly, to catch eligibility bugs.

NOT MEASURED: the cost of hashing all allocations on these runs; hit rates at full size; any workload outside
the eight above (PyTorch, llama.cpp: framework kernels mostly consume fresh data and would rarely hit).
