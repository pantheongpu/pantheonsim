# Scope: an opt-in host-GPU mode (`host_gpu`)

Status: scope, 2026-10-23. Nothing is built. It follows [07-host-gpu.md](07-host-gpu.md) (the measurements) and the user's decision to scope an opt-in mode
with a rounding disclaimer. All costs are estimates by reading the code, not measurements. Off by default, always.

## 1. What it is

A launch whose kernel is eligible runs on the real card of the machine; every other launch, and every other API call, stays in the simulator. The user turns it on:

    # pantheonsim.yaml                 # or VGPU_HOST_GPU=exact|approx|off
    host_gpu: exact                    # off (default) | exact | approx

* **`exact`**: only kernels whose every SASS instruction has a fully defined result (see 3) are offloaded. The result is bit-identical to the simulator's by
  construction, so no rounding disclaimer is needed. This is the form with an equivalence argument.
* **`approx`**: also offloads kernels with SFU/fast-math, tensor-core MMA, float atomics and the like. Results are the card's, may differ in the last bits
  and, through accumulation, further. The disclaimer is printed once at start-up and written into the report. This is the user's proposal.

Not a card swap: the simulated profile must equal the host card (see 4). "Simulate an H100 on an A10G" stays out of scope (07, verdict 2: 1 of 16 workloads runs).

## 2. Architecture (the cheapest design that keeps pointers valid)

1. **Same addresses on the card.** At context creation reserve the simulator's device address ranges on the card with `cuMemAddressReserve` (the driver honoured
   84% to 100% of reservations in `probe_va`). If a reservation is refused the mode turns itself off for the process with one notice. This keeps pointers
   stored inside device memory valid, which is what a mirror at other addresses cannot.
2. **Chunk residency.** The simulator's sparse VRAM already tracks 64 KiB chunks. Add a state per chunk: `host`, `card`, `both-clean`. The simulator touching a
   `card` chunk copies it back first (the check sits next to the existing materialise-on-first-write test in `src/core/memory.cpp`; its cost on the hot path
   must be measured and is a gate, 6). An offloaded launch first pushes every `host` chunk of the allocations it may reach (all live allocations, since
   pointers inside memory cannot be followed) and marks written chunks `card`. Page-granular copies; unchanged chunks are not re-sent.
3. **Launch.** Load the cubin with `cuModuleLoadData` once per module (70 us measured), launch with the original grid, block, shared size and arguments, then
   `cuStreamSynchronize` on a private stream: the simulator's streams are synchronous, so the offloaded launch is synchronous too.
4. **Everything else is unchanged.** Allocation, copies, events, graphs, libraries (cuBLAS stays the simulator's; offloading libraries is out of scope, 07 verdict 5).

## 3. Eligibility (a per-kernel static check, cached per function)

Refuse (run in the simulator) when any of:

* the profile is not the host card (4), or the module has no SASS for the card's architecture and PTX JIT would be needed (the card would compile it differently);
* any diagnostic that needs the simulator is on: `counters`/profile runs, `race`, `strict`, `scheduler` != default, `fault*`, CUPTI subscribers or `VGPU_TRACE`,
  `VGPU_SASS_TRACE`, `VGPU_MAX_STEPS`, `count` > 1, a debugger. The mode is rejected at start-up with a message when combined with these (not silently);
* the kernel uses clusters, cooperative launch, dynamic parallelism, device `printf`/`assert`/`malloc`, texture/surface objects, or unified-memory
  addresses the card does not map;
* (`exact` only) the SASS contains MUFU, any `.approx`/fast-math form, tensor-core MMA, float atomics/reductions (order), shuffles with undefined lanes, or
  anything outside a whitelist of integer ops, IEEE `FADD/FMUL/FFMA/FSETP`, converts with defined rounding, loads, stores and integer atomics on distinct
  addresses. Whitelist built from `src/sass/tier_ops.inc`'s forms, which are already held to the interpreter's bits.
  **Not provable statically** and so a stated limit: data races and reads of uninitialised memory (the card's answer is one valid outcome; the simulator's is
  another; 07 found `srad_v2` reading out of bounds and 11 of 164 programs differing card vs card).

## 4. Identity rule

`host_gpu` requires: profile name equals the host card's model, or an explicit `host_gpu_device: <index>` plus an equal compute capability and SM count.
Shared memory, SM count, driver and runtime versions the program reads are then the simulator's profile values, which equal the card's except driver/runtime
strings (13.0 vs 13.2, 07 section 3.2): reported as the simulator's, as today.

## 5. What the user loses and gets, in the disclaimer's words

Lost for offloaded launches: counters, fault injection, OOB/use-after-free/race diagnostics, deterministic scheduling, CUPTI records (07 section 5.4).
`approx` only: results can differ from the simulator's and from another machine's (SFU rounding, tensor-core accumulation order). A per-kernel line in the
summary says how many launches and how many warp-instruction-equivalents ran on the card, so a result is never silently mixed.

Gain (07, E1): 8x to 105x on the device phase of short kernel-bound runs, 122x to 26,000x at full launch size. No gain, or a loss, below roughly 100 us of
work per launch (card launch 7.7 us vs 2.5 us simulated, plus copies): the mode must keep a threshold (`host_gpu_min_threads`, default from measurement) and use
the simulator below it.

## 6. Plan and gates (each phase ends with a decision; nothing starts without the user's go)

| phase | content | estimate | gate to continue |
| --- | --- | --- | --- |
| 0 | `vgpu check ./prog`: run native and simulated, compare (07 verdict 4). Independent of the rest; gives the data on how often `exact` would have held | 1 to 2 weeks | none; useful alone |
| 1 | spike, no product merge: VMM reservation + chunk residency + one-kernel offload for the 16 harness workloads, `exact` check as a hash compare against the simulator | 4 to 6 weeks | hot-path cost of the residency test < 1% on `bench.sh`; identical hashes on all eligible workloads; speed-up on `int_virus`-class > 10x including copies |
| 2 | eligibility scanner (3), config and refusals (3, 4), summary line, docs, tests wired to a GPU runner | 4 to 6 weeks | the same-bits test passes on a card runner; CI cost agreed (a GPU runner does not exist today) |
| 3 | `approx`, disclaimer, async streams only if measured as needed | 2 to 4 weeks | decided after Phase 2 data |
| later | HIP/AMD (same design, unmeasured: no AMD GPU with ROCm was available), multi-GPU racks | not estimated | |

Full API coverage (the 170 entry points, graphs, managed memory, IPC; 07 section 5.3) is not needed for this design: only launches and the memory they touch go
to the card; the rest stays simulated. This is why the estimate is smaller than 07's 3 to 6 months for a full passthrough, and it is the main risk to that
estimate: the residency model must be right for every API that touches device memory (memcpy, memset, graphs, VMM, IPC, managed). Each such path needs a
"copy back first" hook, and one missed path is silent corruption. Tests: a differential run of every e2e program with `host_gpu` forced and random
launch/read interleavings, and mutation checks on each hook.

## 7. Open questions for the user

1. Is `exact`-only (no disclaimer needed) enough to start, with `approx` later?
2. A GPU runner for CI (an AWS g5 instance on demand, or a self-hosted card) is required from Phase 2; who pays and where?
3. Phase 0 first (cheap, useful alone), or straight to the Phase 1 spike?
