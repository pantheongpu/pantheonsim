# Study 10: host-GPU mode, Phase 1 audit (sp34): what exists, what every memory path does, what is refused

Status: 2026-11-07, branch `sp34-hostgpu` (from origin/main 7d8d98a4). Off by default; with `VGPU_HOST_GPU` unset the simulator is bit-identical to main
(nothing on any access path changed, see 4). It follows [08](08-host-gpu-opt-in-scope.md) (scope, Phase 1 gates) and [09](09-host-gpu-spike.md) (the spike and the
persistent card copy). Phase 1 of 08 section 6 was already built by the spike branches that are merged on main (#440 persistent copy, #459 copy-buffer reuse /
traffic ban / first-launch rule / lazy driver start / division idiom / card CI); this round audits it against 08 and closes what the audit found.

## 1. Status audit: 08 Phase 1 against main

| 08 asks for | on main | verdict |
|---|---|---|
| VMM reservation of the simulator's device ranges, same addresses on the card | `Impl::ensure_mapped`: one 2 MiB-granule reservation per window, backed up to the highest live address (09 section 1: honoured on a native driver) | built; **cannot run on WSL2** (section 6) |
| chunk residency Host/Card/Both, "copy back first" hooks | `Residency` (include/vgpu/host_gpu.hpp), `CardGate` in `MemoryManager` (resolve / find_ordinary_miss / run_allocation / free / share), `host_gpu_track.hpp` | built; hooks inventoried below, **one gap found and closed** (VMM / IPC / kernel heap memory) |
| one-kernel offload, synchronous private stream | `Runtime::try_launch`, module loaded once per module, `cuLaunchKernel` + `cuCtxSynchronize` | built |
| eligibility scanner from `tier_ops.inc` forms | `exact_opcode_allowed`, `division_idiom_only`, per-kernel cache | built |
| size rule 5.1 + `host_gpu_log` | `should_offload`, `SimTimeTable`, ban rules, first-launch rule; `VGPU_HOST_GPU_LOG=1` | built; `host_gpu_log` now a config key |
| `host_gpu: off|exact` config, `VGPU_HOST_GPU` | env only | **config keys `host_gpu`, `host_gpu_log` added** |
| rejected loudly with diagnostics that need the simulator (08 sections 3 and 4) | a list of 10 names, one of them (`VGPU_FAULT`) not a variable the simulator reads, the message hidden by `VGPU_QUIET`, the scheduler's default name counted as a diagnostic | **list rebuilt from what the simulator reads; said under `VGPU_QUIET`; a typo in the mode is said** |
| identity rule (08 section 4) | `want_sm` / `want_mp` compared with the card at start | built |
| hot-path cost of the residency test < 1% | nothing on `hot_location`; one relaxed load in `resolve` / `free` / `free_all` / `share` | measured in section 5 |
| identical hashes on all eligible workloads | `VGPU_HOST_GPU_VERIFY=1|all`; 21,804 + 23,248 launches identical over 64 programs (09 section 7) | done on A10G; **not repeatable on the local 3060s** (section 6) |
| no workload slower with the mode on | LU 0.82x vs write-through was the exception; #459's traffic ban targets it | verified on A10G by #459, not re-measured here (no card) |

Reused, not redone: everything in the table marked built (merged branches `host-gpu-proto`, `host-gpu-persistent`, `fix-hostgpu-handback`, #459 `sp14-hostgpu`;
`vgpu check` of Phase 0 is a separate tool and was not touched).

## 2. Inventory: every way into device bytes, and what the card copy does about it

The card holds a copy of the table of live allocations (`MemoryManager::live_ranges()`), at the simulator's addresses. A stale card copy is wrong only when the
simulator reads or writes an allocation's bytes without asking the gate, or when device bytes exist that are not in the table. Grep of `src/` and `nvidia/`:
all device bytes are reached through `MemoryManager` (no other file keeps a chunk pointer), and the launch hook is the one `Device::launch`
(`src/runtime/runtime.cpp`), which every launch passes: `cudaLaunchKernel`, `cuLaunchKernel`, cooperative and graph kernel nodes (`vgpu_driver::run_kernel` is the
simulator's own run, reached from `Device::launch`'s callers only), library kernels, and `vgpu demo`.

| path (entry point) | callers | mechanism | test (`tests/unit/test_card_gate.cpp`) |
|---|---|---|---|
| `write`, `read`, `fill` | cudaMemcpy* (all directions, async forms, 2D/3D), cudaMemset*, cuMemcpy*, graph memcpy/memset nodes, tests | `resolve()` asks `before_access(addr, len, write)` while any allocation is card-valid | `host_copies_ask_the_gate_with_their_range_and_direction`, table case rows `write`, `read`, `fill` |
| `copy_to` (device to device, peer copies) | cudaMemcpyPeer, D2D copies | declines an allocation with a card copy; the caller falls back to `read` + `write` | table rows `copy_to (source)`, `copy_to (destination)` |
| `load_scalar`, `store_scalar` | simulator kernels | out of the lookup memory while card-valid, so `resolve()` | `a_remembered_allocation_is_forgotten_...`, rows `load_scalar`, `store_scalar` |
| `load_run`, `store_run`, `load_batch`, `load_broadcast`, `store_batch`, `store_strided`, `batch_window`, `atomic_word_ptrs`, `rmw_batch`, `rmw_scalar`, `atomic_rmw`, `rmw_if_changed`, `load_fields`, `load_texels`, `load_quad`, `store_quad`, `load_quad_run`, `store_quad_run` | simulator kernels (SASS, PTX, tensor and texture units) | start with `find_ordinary` / `run_allocation`, which decline a card-valid allocation; the caller takes the scalar path | `run_accesses_decline_...`, one table row each |
| `free`, `free_all` | cudaFree, context reset | `before_free` always (also with no card copy: the gate tracks every live allocation) | `a_free_tells_the_gate_*` (2), `a_zero_count_...` |
| `share`, `adopt`, `abandon` (IPC) | cudaIpcGetMemHandle / OpenMemHandle | `share` pulls the card copy first and drops it; `adopt` memory is never in the table | `exporting_an_allocation_brings_the_card_copy_back_first`, `card_blocker_*` |
| `map`, `unmap`, `set_access`, `create_handle`... (VMM) | cuMemMap and friends | bytes live outside the allocation table: **launches refused** while any mapping exists | `card_blocker_names_device_memory_outside_the_allocation_table` |
| `heap_alloc`, `heap_free` | kernel `malloc` / `free` | heap blocks are outside the table: **launches refused** while the heap holds a block | same test; end to end `run_host_gpu_stub.sh` step 6 |
| `map_host`, `host_address`, `host_load_batch` (pinned, managed, registered, mapped IPC) | cudaHostAlloc, cudaMallocManaged, cudaHostRegister | outside the device windows; an argument that points into it refuses the launch (existing). A pointer *stored in device memory* that points into it is not followed | limit, see 3 |
| texture and surface objects, clusters, cooperative, graph conditionals, non-32 warp | launch shapes | refused per launch (`kernel_eligible`) | `e2e_host_gpu`, `e2e_host_gpu_stub` |
| launches of other kernels between offloaded ones | `Device::launch` | `before_simulated` gives the card copy back before the simulator runs | `e2e_host_gpu` (card), verify mode |

New in this round: the VMM, IPC and heap rows (`MemoryManager::card_blocker()`), which closes the one path of 08's list ("memcpy, memset, graphs, VMM, IPC, managed")
that had no hook and no refusal: before, a launch could run on the card while a kernel argument or a pointer stored in memory named a VMM-mapped address. The card
has that address unmapped or backed with zeros inside the span `ensure_mapped` fills, so the answer was a fault or silent zeros.

## 3. Known limits (said, not hidden)

* A pointer stored in device memory that points into host-mapped memory (pinned, managed) is not followed: the card faults on it (loud: in persistent mode that is an
  internal error while data is card-resident, never wrong bytes). Arguments that point into host-mapped memory are refused up front.
* Another host thread running simulator launches while an offloaded launch ends is not protected (09 section 7.2).
* A data race, an uninitialised read or an out-of-bounds read the card tolerates cannot be seen statically (08 section 3).
* A debugger, CUPTI subscribers and the telemetry kernel records have no switch the mode can test; the launches they would observe are not observed.
* `host_gpu_min_work` / `host_gpu_margin` of 08 section 5.1 are not knobs: the rule's constants are named in `include/vgpu/host_gpu.hpp` (09 section 6 "no
  environment overrides"); unchanged.

## 4. What changed in code

* `MemoryManager::card_blocker()` (include/vgpu/memory.hpp, src/core/memory.cpp), used by `Runtime::kernel_eligible` (not cached: a mapping can appear between two
  launches). Nothing on an access path changed.
* `kIncompatibleVars`, `diagnostic_switched_on`, `mode_text_is_known` (include/vgpu/host_gpu.hpp): the start-up refusal is pure logic now and unit tested.
  The message is printed even under `VGPU_QUIET`; `VGPU_HOST_GPU=<not off, exact or turbo>` prints "not a mode; mode off".
* Config keys `host_gpu` (`off`, `exact`) and `host_gpu_log` (src/cli/config.cpp, docs/config.md, README.md, pantheonsim.example.yaml).
* Tests: the table test over every `MemoryManager` entry, `card_blocker_*`, `mode_text_known_and_diagnostic_switches`, step 6 of `run_host_gpu_stub.sh`.


## 5. Tests and mutation evidence (server2, Release, g++ 13; compile-checked with g++-12 as CI does)

`test_card_gate` 10/10 (was 7 tests before this round: +the table test, +`card_blocker_*`), `test_host_gpu_logic` 43/43 (+`mode_text_known_and_diagnostic_switches`),
`test_config` 17/17 (the key table and the starter file take the new keys), `e2e_host_gpu_stub` PASS with step 6 (run on server2 with the stand-in driver; the
program was compiled with CUDA 13.0 nvcc on the WSL box because server2 has no toolkit: `host_gpu_lazy.cu` + `host_gpu_lazy_heap.cu`).
In the table test, 7 entries went ahead and asked the gate with the right direction, 18 declined (the batch, run and atomic forms: their caller then takes the scalar path).

Mutants (each applied to a copy, rebuilt, all three tests run, source restored and touched; cases failing):

| mutant | killed by |
|---|---|
| M1 `resolve()` does not ask the gate | `a_remembered_allocation_is_forgotten...`, `host_copies_ask_the_gate...`, table test |
| M2 `find_ordinary_miss` no longer declines a card-valid allocation | `a_remembered_allocation...`, `run_accesses_decline...`, table test |
| M3 `run_allocation` no longer declines | `run_accesses_decline...`, table test |
| M4 / M5 / M6 `card_blocker` ignores VMM mappings / IPC regions / the kernel heap | `card_blocker_names_device_memory...` (each) |
| M7 `free` does not tell the gate | both `a_free_tells_the_gate_*` |
| M8 `share` does not pull first | `exporting_an_allocation_brings_the_card_copy_back_first` |
| M9 the scheduler's default name counts as a diagnostic | `mode_text_known_and_diagnostic_switches` |
| M10 `VGPU_FAIL_ALLOC` not in the list; M11 `approx` is a known mode; M12 `0` switches a diagnostic on | same |
| M13 heap check removed from `card_blocker`, whole simulator rebuilt | `run_host_gpu_stub.sh` step 6 ("a launch went to the card while the kernel heap held a block") |

Not covered by a mutant: the config keys (a removed key is not caught by any test: the keys
write plain environment variables that `kIncompatibleVars` and `mode_from_text` read, and both ends are tested); `host_gpu_log` behaviour (a debug print).
Not CI-runnable and not changed: `e2e_host_gpu` (needs a card; hosted CI prints its SKIP, the card workflow `card-host-gpu.yml` runs it).

## 6. The local RTX 3060s (WSL2): the mode cannot run there, measured

Probe (`m1_probe_va`, forked child with a watchdog, CUDA 13.0, driver on WSL2, one process, under the card lock):

| `cuMemAddressReserve` hint + map + kernel + read back, 64 MiB | result |
|---|---|
| 0x20'0000'0000 (128 GiB), 0x40'0000'0000, 0x60'0000'0000, 0x7f'0000'0000, 0x80'0000'0000 (512 GiB) | exact, 0.2 s |
| 0x100'0000'0000 (1 TiB) | no answer in 10 s, killed |
| **0x2000'0000'0000, the simulator's device 0 window** | **no answer in 10 s, killed** |

The card's address space under WSL2 ends between 512 GiB and 1 TiB; the simulator's windows start at 32 TiB. So the same-address design (08 section 2, step 1)
has no way to start on these cards, and the hang-and-deadline path (09 section 1, "not exercised against a real hang") was run for real: `lazy_prog big` with
`VGPU_HOST_GPU=exact`, simulated profile `nvidia/rtx3060`, the sim built on server2 and copied: after 10 s "the driver did not answer within 10 s (an address outside the card's
range hangs on some systems); mode off", the program finished in 12.4 s with the same four hashes as the mode-off run, exit 0, nothing left running, both cards
healthy afterwards (`nvidia-smi`). That closes the one untested item of 09 section 5.2 and also shows the mode degrades safely.

Consequence for step 4 of the brief: **there are no mode-on numbers on the 3060** (speed-up, offloaded and refused launches, hash equality, residency cost on a card).
The A10G numbers of 09 (21,804 + 23,248 in-set launches bit-identical over 64 programs; FDTD-2D 33x, aligned-types 253x, babelstream 35x, base64e 31x over the simulator;
LU 0.82x vs write-through before #459) are the latest card evidence and were not repeated.
The only way to get this card to run the mode is to give the simulator a window below 512 GiB when the mode is on (the base, 1 TiB stride and 16 windows are
constants used in 13 places: `include/vgpu/memory.hpp`, `src/core/memory.cpp`, `nvidia/src/runtime_api.cpp`); that changes every device pointer and the
per-device VA budget (16 windows of 16 GiB would fit; a device with more than 16 GiB of address use would not) and needs a decision, not a side effect of this branch.
The alternative is a native-Linux card (an A10G on AWS, not approved).

## 7. Gate verdicts

| gate (08 section 6, Phase 1) | verdict |
|---|---|
| hot-path cost of the residency test < 1% | **met by construction and by microbenchmark.** Nothing is on `hot_location`, `find_ordinary`'s hit path or any per-access path of an allocation that is not card-resident (09 section 7.2). `bench_gate` (server2, one pinned core, best of 7, 20 M accesses): `load_scalar` 3.20 ns with no gate, 3.41 / 3.22 ns with a gate set and nothing / another allocation card-resident (run-to-run noise is about 0.2 ns: the two a-allocation rows of the same run differ by 0.2 ns while the code is the same), `store_scalar` 3.11 / 3.29 / 3.11 ns, `load_batch` 1.25 ns per lane in all three. An access to an allocation that IS card-resident takes `resolve()` and the gate: 29 ns, which a simulator launch never pays because `before_simulated` gives the card copy back first. With the mode off the gate is not set at all. No workload-level A/B was run: the code on the access paths did not change in this branch, so a harness A/B of branch against main would only measure noise |
| no workload slower with the mode on | **not re-measured** (no card; section 6). #459's traffic ban and first-launch rule are the fix for LU, measured on the A10G |
| identical hashes on all eligible workloads | **not re-measured** (section 6). The verify mode and the A10G corpus run stand |
| off by default and bit-identical when off | yes: the new code runs only after `VGPU_HOST_GPU=exact`; with it unset the only change is `card_blocker()` which is never called, and the config keys write nothing unless set. The one visible change with the mode OFF: none. With a typo in `VGPU_HOST_GPU` a line is printed (before: silent off) |
