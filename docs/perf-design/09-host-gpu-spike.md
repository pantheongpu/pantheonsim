# Study 9: the host-GPU spike (round sp8, measurements and a prototype)

Status: measured 2026-10-28 on an AWS g5.xlarge (A10G 23 GB, driver 595.112, 4 vCPU, Ubuntu 24.04, host CUDA 13.0 toolkit, simulator built Release with
gcc 13, not the CI image), branch `sp8-hostgpu`. It follows [07-host-gpu.md](07-host-gpu.md) (the ceiling: card 8x to 26,000x faster on kernels) and
[08-host-gpu-opt-in-scope.md](08-host-gpu-opt-in-scope.md) (the scope and the user's decisions: opt-in, `exact` first, a per-launch size/cost rule, functional
results only). Raw outputs and tools: [data/round13](data/round13) (`verify_all.sh`, `bench_exact.sh`, `refusals.sh`, `optrace.sh` and the M1 to M3 tools). Nothing here is on by default: the mode is `VGPU_HOST_GPU=exact`, absent otherwise.

## 0. Result in twelve lines

1. **The card accepts the simulator's own addresses.** On a native Linux driver the whole simulator range, 16 windows of 1 TiB from `0x2000'0000'0000`, reserves
   exactly (`cuMemAddressReserve`), `cuMemCreate`/`cuMemMap` back it in 2 MiB granules, a kernel reads and writes through the same pointer values and the host reads
   them back. One rule matters: **the driver honours a fixed hint only for reservations of 32 MiB or more**; 2 to 16 MiB reservations are placed by the driver
   (which is why 07 saw 84 of 100): reserve a whole window once (M1).
2. **Replaying real launches on the card agrees with the simulator to the bit for 236 of 265 sampled launches (89%)**: 74 of 74 PolyBench, 51 of 51 Rodinia, 111 of
   140 HeCBench. All 29 that differ (22 deterministic, 7 card-nondeterministic) belong to kernels holding an SFU instruction (`MUFU.EX2/RCP/SQRT/LG2/RSQ`) or a
   compare-and-swap atomic loop; there is no unexplained difference (M2).
3. **The card path costs about 7 us per launch plus the data**: 12 GB/s in and 10 GB/s out for contiguous copies from 1 MiB, but only 4.4/4.0 GB/s when the
   simulator's 64 KiB chunks are copied one call at a time (M3). With the measured simulator times of the 265 launches (198.6 s in all), a rule "offload when the
   simulator needs more than twice the card path" would have saved 192 s (97%) with every launch moving all live memory both ways, and lost 0.3 s when applied to all.
4. **A prototype exists and works** (M4): `VGPU_HOST_GPU=exact` runs eligible SASS launches on the real card, write-through (the simulator's memory stays the
   truth, so no hook is added to the memory manager). 23,248 launches of kernels in its exact set were bit-identical to the simulator's, none different (section 5.1). Whole processes: SYRK 10.2 s to 1.3 s, SYR2K 19.5 s to 2.3 s,
   FDTD-2D 159 s to 68 s; and a bad estimate in the first version sent 253 small launches of PolyBench 3DCONV to the card and made it **38x slower**, which is fixed (two samples, the smallest, and a ban after one bad launch).
5. **What stays in the simulator** is large: any kernel with an SFU instruction (every `expf`, `sqrtf`, `__fdividef`; and every IEEE division, as it starts with
   `MUFU.RCP`), float or compare-and-swap atomics, tensor cores, textures, module variables, device calls, clusters, cooperative launches, mapped host memory, and
   everything small. The honest summary of the cost of exactness is in section 6.

## 1. M1: the card's virtual address space (standalone probe, no simulator)

Tool: [m1_probe_va.cu](data/round13/m1_probe_va.cu), raw: [raw/raw_m1_sweep.txt](data/round13/raw/raw_m1_sweep.txt), [raw/raw_m1_more.txt](data/round13/raw/raw_m1_more.txt). The simulator's
windows are in `include/vgpu/memory.hpp`: device d owns `0x2000'0000'0000 + d x 1 TiB` (16 windows, up to `0x3000'0000'0000`), functions at `0x6ffb'0000'0000`,
kernels at `0x6ffc'0000'0000`, and the `kShared/kLocal/kParam` generic windows at `0x6ffe..0x6fff'0000'0000` (`include/vgpu/sass/exec.hpp`; the SASS engine's fake
windows, which the card never sees: they are the engine's own, and the card has its own shared and local windows).

| question | answer on the A10G, driver 595.112, native Linux |
|---|---|
| granularity (`cuMemGetAllocationGranularity`) | 2 MiB minimum and recommended; the simulator's `kVmmGranularity` is the same |
| usable VA range | up to 2^47: hints at `0x4000'0000'0000` honoured; `0x8000'0000'0000` and above: `CUDA_ERROR_OUT_OF_MEMORY` at once, no hang |
| the simulator's windows | **all 16 x 1 TiB windows reserve exactly, in one process at the same time**; 64 MiB at the base, +1 MiB, `0x2fff'0000'0000`, `0x6ffb'0000'0000` (function window) and `0x7ffe'0000'0000` all honoured and usable (map, kernel write, host read back verified) |
| places the driver keeps for itself | hints at `0x2'0000'0000` and `0x100'0000'0000` (1 TiB) were moved: the driver's own mappings; neither is in a simulator window |
| smallest honoured reservation | **32 MiB.** A reservation of 2, 4, 6, 8 or 16 MiB at an aligned, free hint is placed by the driver (`0x7395...`, near host addresses); 32 and 64 MiB are exact; a 32 MiB reservation at a hint that is not 32 MiB aligned is rounded up to the next boundary (`0x2000'0020'0000` became `0x2000'0200'0000`). This explains 07's "84 of 100 honoured" for 2 MiB reservations |
| `cuMemAddressReserve` size | must be a multiple of 2 MiB (64 KiB: invalid value); a hint not 2 MiB aligned is ignored |
| `cuMemCreate`/`cuMemMap` | sizes multiples of 2 MiB; a partial map of a larger reservation works (8 MiB reserved, 2 MiB mapped at +0, the same physical granule mapped again at +4 MiB works); overlapping map fails; `cuMemSetAccess` on an unmapped hole fails; unmap of half a granule fails |
| big maps | 4 GiB and 20,000 MiB (of 22.3 GiB) mapped at the window base, kernel and readback verified, 0.48 and 0.51 s including process start |
| concurrent processes | 4 processes at once, each reserving and mapping 256 MiB at the same address 20 times, running a kernel and reading it back: all exact and verified (VA is per process; the physical memory is the card's). 5.9 s wall for 4 processes against 4.3 s for 1: the driver serialises some of it |

**The hang (WSL2).** On the local RTX 3060 under WSL2 a hint outside the card's range does not fail: `cuMemAddressReserve` never returns (memory note: windows at
`0x1000'0000'0000` and above hang, `0x7ffe'0000'0000` returns an error, and the card's VA space ends between 512 GiB and 1 TiB). This was not re-run here (nothing runs on the WSL
machine); the native card gave an error or the address every time, so the A10G rig cannot show the hang. Design, both parts implemented:
* the probe tool runs every risky call in a forked child with a deadline (the child has no CUDA state, so `fork` is safe there) and kills it on timeout;
* the simulator is multi-threaded and has initialised nothing of CUDA before the first offloaded launch, but `fork` after threads exist is unsafe for a child that loads a driver,
  so the mode starts the real driver, reads the card's identity and reserves the window **on a detached thread with a 10 s deadline** (`src/runtime/host_gpu.cpp`, `Impl::start`); on a timeout
  it says so once and the mode is off (the thread stays stuck in the driver; the runtime object is leaked, not destroyed). Not exercised against a real hang.
* A WSL2 card could only be served by moving the windows below the card's range (under about 512 GiB), which changes the simulator's addresses for everyone: not proposed.

## 2. M2: replaying real launches on the card and comparing to the bit

Capture (simulator side, throwaway, env-gated by `VGPU_HOSTGPU_CAPTURE=<dir>` in `Device::launch`): for the first 2 or 3 launches of each kernel, the whole device memory before
the launch (every live allocation: base, size, bytes), after it, the launch (name, grid, block, dynamic shared, argument bytes) and the simulator's time. Replay
([m2_replay.cpp](data/round13/m2_replay.cpp), pure driver API, driven by [m2_run.sh](data/round13/m2_run.sh)): the program's own fatbin is read from its `.nv_fatbin` section
and loaded with `cuModuleLoadData`, the allocations are put at their simulator addresses (one reservation over the span, 2 MiB granules backed), the kernel is launched with the
captured arguments, the memory is read back and compared with the simulator's per 64 KiB chunk. Every launch is replayed twice on the card, so a difference is classed
"card nondeterministic" when the card disagrees with itself. Programs: the corpora's own builds and arguments (round 12's lists), `nvidia/a10g` profile, `VGPU_THREADS=4`.

| corpus | programs | launches replayed | identical | differ (deterministic on the card) | differ, card nondeterministic |
|---|---:|---:|---:|---:|---:|
| PolyBench/GPU | 20 | 74 | **74** | 0 | 0 |
| Rodinia (12 given) | 10 with a capture | 51 | **51** | 0 | 0 |
| HeCBench (33 of the reduced list) | 27 with a capture | 140 | 111 | 22 | 7 |
| **total** | | **265** | **236 (89%)** | **22** | **7** |

Raw: [raw/m2_polybench.txt](data/round13/raw/m2_polybench.txt), [raw/m2_rodinia.txt](data/round13/raw/m2_rodinia.txt), [raw/m2_hecbench.txt](data/round13/raw/m2_hecbench.txt) (one line per launch: grid, block,
memory, changed bytes, chunks differing, 32-bit words differing, the card's kernel time, push and pull times, the simulator's time, verdict, first differing address and values).

**Why the 29 differ.** The instruction mix of each differing kernel (the SASS engine's own trace, `VGPU_SASS_TRACE` on warp 0 of block 0, [optrace.sh](data/round13/optrace.sh)):

| program (kernels) | launches | what the kernel holds | what the data show |
|---|---:|---|---|
| silu (forward, backward, 3 variants) | 10 | `MUFU.EX2`, `MUFU.RCP` (expf) | 1 to 6 ulp, deterministic on the card |
| bilateral | 3 | `expf` | 2 to 3 ulp |
| s3d (`ratt`, `rdsmh`, `ratt10`) | 3 | `MUFU.EX2`, DFMA, `MUFU.RCP64H` | 1 ulp (the SFU divergence of the earlier notes) |
| perplexity (`sigmas_kernel`) | 1 | `MUFU.EX2`, `MUFU.LG2`, `MUFU.RCP` | 125 ulp |
| binomial | 2 | `MUFU.EX2` | 171 ulp |
| aobench | 1 | `MUFU.RCP`, `MUFU.SQRT`, FTZ | the image differs by one level in a pixel |
| attention (kernel1, kernel2) | 2 | `MUFU.EX2`, `MUFU.RCP` | 1 ulp; kernel1 also changes between two card runs |
| aop (`prepare_svd`, `partial_beta`, `final_beta`, `partial_sums`) | 6 | `ATOMS.CAST.SPIN.64` (a double `atomicAdd` as a compare-and-swap loop), `MUFU.RCP64H` | the order of the adds; 5 of 6 change between two card runs |
| bh (`TreeBuildingKernel`) | 1 | `ATOMG.E.CAS` | tree layout depends on the winner of each CAS: the card disagrees with itself |

Every kernel that differed holds an SFU instruction or a CAS; **no differing kernel was free of both**, and the 236 identical launches include kernels with FFMA, double precision (`DFMA`,
`DADD`), shared memory and barriers, shuffles, `FRND`/`F2I`/`I2FP` conversions, integer `RED` and a float `RED.E.ADD.F32` (asmooth: the adds happened not to collide, which is
luck, not a guarantee: the exact set excludes it). **Also identical although they contain `MUFU.RCP`**: the IEEE divisions of `black-scholes`, `atan2`, `adam` and PolyBench `CORR`
(`MUFU.RCP` plus Newton refinement plus `FCHK`): the refinement makes the result independent of the approximation's low bits. The exact set is nevertheless conservative and
refuses them; allowing the `MUFU.RCP` + `FCHK` division idiom needs an exhaustive comparison of the simulator's `MUFU.RCP` with the card's first (not done).

**Not replayed, and why** (the denominator of the table counts only replayed launches): `srad_v2` (the simulator stops at the out-of-bounds read the card hides, so there is no
"after"), `black-scholes` (live memory over the 768 MiB capture cap), `aidw` and `atomicCost` (a launch ran past the 90 s capture timeout in the simulator), `zerocopy` (the
replay tool crashed on kernels that use mapped host memory, which the exact set also refuses), `nn` (its data list names files that were not copied), and the programs whose
kernels never launched in the sampled prefix.

**What this proves and does not.** The sample is the first 2 to 3 launches of each kernel, not whole programs, and the inputs are those programs' own (random or generated
data, all kinds of values). It shows that for kernels with no SFU and no order-dependent atomic the SASS engine and the real A10G agreed on every sampled launch (the 125 PolyBench and Rodinia launches here, and the 23,248 launches
in the exact set of section 5.1), which is the evidence the exact set is built on. It is not a proof: a data race, a read of uninitialised
memory, an out-of-bounds read the card tolerates (srad_v2) or an instruction whose divergence needs a rare input would not show in a sample.

## 3. M3: the cost model

### 3.1 What moving memory costs

[m3_xfer.cpp](data/round13/m3_xfer.cpp), raw: [raw/m3_xfer.txt](data/round13/raw/m3_xfer.txt). The simulator's memory is 64 KiB chunks, each its own host allocation; best of 7 (3 above 256 MiB).

| fixed costs | |
|---|---:|
| empty kernel, launch and `cuCtxSynchronize` | 7.0 us (2.4 us to submit alone) |
| 4-byte copy in / out | 5.9 / 7.9 us |
| one 64 KiB chunk, pageable, in / out | 16.2 / 21.6 us |
| `cuMemHostRegister` + unregister of one 64 KiB chunk | 200 us (3,500 ms for 1 GiB: pinning in place is not viable) |
| CUDA context creation and the real driver's start (from 07) | about 150 ms once per process |

| MiB moved | contiguous pageable in / out | contiguous pinned in / out | 64 KiB chunks, one call each, in / out | gather to pinned staging then one copy, in / out |
|---:|---:|---:|---:|---:|
| 1 | 8.0 / 5.6 GB/s | 12.3 / 12.2 | 4.9 / 3.8 | 8.4 / 7.3 |
| 16 | 12.5 / 9.6 | 13.4 / 13.1 | 4.5 / 4.0 | 5.8 / 5.7 |
| 256 | 13.1 / 10.1 | 13.5 / 13.2 | 4.4 / 4.0 | 5.8 / 5.8 |
| 1,024 | 13.1 / 10.1 | 13.5 / 13.2 | 4.5 / 4.0 | 5.8 / 5.8 |

So a **1 GiB** push costs 82 ms contiguous, 230 ms chunk by chunk; the card reads or writes its own memory at 100s of GB/s, so the link is the cost. The prototype copies
contiguous ranges in 32 MiB batches (the simulator's `read` already gathers chunks into one buffer), i.e. the "contiguous pageable" column plus the simulator's own copy.

### 3.2 When the card path wins

[m3_model.py](data/round13/m3_model.py) ([raw/m3_model.txt](data/round13/raw/m3_model.txt)) applies these costs to the 265 replayed launches, using each launch's measured simulator time, the card kernel
time measured in the replay, and three data scenarios: *cold* (every live byte pushed, the changed bytes pulled: a lone launch between simulator phases; this is what the prototype
does), *dirty* (only the changed bytes move each way) and *chain* (nothing moves: a run of launches on card-resident data). Rule: offload when `sim > 2 x (7 us + kernel + copies)`.

| scenario (chunk-at-a-time copies) | launches where the card path wins | simulator time saved | would be lost by offloading everything |
|---|---:|---:|---:|
| cold | 205 of 265 | 192.4 s of 198.6 s (96.9%) | 0.3 s |
| dirty | 255 of 265 | 194.1 s (97.7%) | 0.0 s |
| chain | 261 of 265 | 195.1 s (98.2%) | 0.0 s |

(Contiguous pinned copies: cold 223 launches, 97.8%.) By the simulator's time for the launch: under 1 ms, 34 of 71 win cold and 67 of 71 in a chain; 1 to 10 ms, 43 of 62 cold; every
launch above 10 ms wins in every scenario (63 + 58 + 11 launches). **The size rule, as a break-even** (card kernel 0.1 ms, margin 2):

| data moved each way | card path | offload only if the simulator needs more than |
|---:|---:|---:|
| 64 KiB | 0.12 to 0.14 ms | 0.2 to 0.3 ms |
| 1 MiB | 0.26 to 0.61 ms | 0.5 to 1.2 ms |
| 16 MiB | 2.6 to 8.1 ms | 5 to 16 ms |
| 256 MiB | 40 to 128 ms | 81 to 256 ms |
| 1 GiB | 162 to 513 ms | 323 to 1,025 ms |

In warp instructions (the simulator does tens to 150 million per CPU-second) that is: a launch is worth offloading above roughly **0.2 ms of simulation when its data fit in a
few chunks, and about 0.3 s of simulation (30 M warp instructions on one thread) per GiB that has to move**. Grid size is a weak proxy on its own: median simulator time per launch
was 0.34 ms for launches under 1,024 threads, 1.1 ms for 1,024 to 16,384, 13.6 ms for 16 K to 262 K and 106 ms above that; on the card the same launches take 27 to 160 us.

**Where it never wins, or cannot be tried** (from the measurements and the code, not each one run):
* small launches: below about 0.2 ms of simulation nothing the card does can beat 7 us plus a copy; 71 of the 265 replayed launches took under 1 ms in the simulator, most of the Rodinia
  `needle`, `lud` diagonal, `gaussian` and `bfs` launches (a 0.03 to 0.3 ms simulation against a card path of 0.04 to 0.1 ms plus the copy: break-even at best);
* programs made of many small kernels with host phases between them: the whole-process time
  is dominated by the host reference computation, which both sides pay (PolyBench `GEMM`: simulator 0.88 s GPU phase + 0.35 s CPU reference, with the mode 0.15 s + 0.40 s);
* the card's own fixed cost: context creation about 150 ms per process, JIT of PTX-only modules about 150 ms (the prototype takes SASS modules only);
* streams (the simulator's are synchronous, the prototype's launch is synchronous too: nothing overlaps), events (the simulator's are host clock), graphs (replayed as launches
  through the same path: not tested), printf and assert (device `vprintf` is an extern: refused), device-side malloc and dynamic parallelism (refused), texture objects (refused),
  unified/managed memory and pinned mapped host memory (refused when an argument points into it; a pointer stored inside memory cannot be seen and would fault on the card, which
  then ends the mode and reruns the launch in the simulator), cooperative launches and clusters (refused), module variables (`__device__`, `__constant__`: refused, the card's copy would
  not see `cudaMemcpyToSymbol` writes).

## 4. M4: the design for a card copy that outlives a launch

The prototype (section 5) is write-through: **every offloaded launch copies all live memory to the card and the changed 64 KiB chunks back.** That is correct without touching the
memory manager, and it is the "cold" row of 3.2; it is wrong only in cost for a run of launches on the same large buffers. The state machine below removes that cost. It is implemented as pure logic and unit
tested (`Residency` in `include/vgpu/host_gpu.hpp`, random differential test in `tests/unit/test_host_gpu_logic.cpp`), **not wired in**.

**Per-chunk states**, one byte per 64 KiB chunk of an allocation: `Host` (the simulator's bytes are the truth), `Card` (the card's), `Both` (equal). Transitions:

| event | rule |
|---|---|
| allocation | all chunks `Host` (zeros are the simulator's sparse table) |
| simulator write, whole chunk | chunk becomes `Host`, nothing copied |
| simulator write, part of a chunk | a `Card` chunk is pulled first; chunk becomes `Host` |
| simulator read | `Card` chunks pulled first (become `Both`) |
| launch on the card | `Host` chunks pushed (become `Both`); after the kernel every chunk of every allocation the kernel may reach is `Card` (it may write any: pointers in memory cannot be followed); a card-side dirty record would make this exact (not available: the card has no dirty bits; a hash or a write-set from a prior simulator run of the kernel could) |
| free | the card's mapping of that range stays (granules are never unmapped in the prototype); reuse starts `Host` |

**Where it hooks the existing `MemoryManager`** (all of the chunk table is in `src/core/memory.cpp` and `include/vgpu/memory.hpp`: 27 `chunks[...]` reads and 40 `is_uniform`/`uniform_byte` uses):

* *Your sentinel idea, as it fits the code.* The chunk table already has a tagged pointer: a pointer with the low bit set is a **uniform chunk** (the byte in the upper bits,
  `is_uniform`, `uniform_chunk`), and every hot path (`hot_location`, `load_batch_in`, `scalar_location`, `rmw_locate`, `load_fields`, `read_chunks`, `store_run`, ...) already
  tests `!chunk || is_uniform(chunk)` and takes a slow path for those. A `Card` chunk could be a second tag (`uniform_byte` value 0 with bit 9 set, for instance) that every one of those slow paths must
  recognise first and pull from. The cost on the hot path is **zero instructions** (the test exists); the cost is **correctness surface**: each of the ~40 uniform-aware sites needs the new branch
  or it reads garbage from a card-resident chunk, and `materialize`, `fill`, `write`, `store_run`, `copy` and `free_all` need to know to pull first (or discard). One missed site is silent corruption.
* *The alternative I would try first, which touches almost nothing on the hot path:* gate at the **allocation**, not the chunk. `resolve()` (the one place that finds an allocation, with
  the address and length of the access) and `find_ordinary_miss()` (where the per-thread lookup cache is filled) are the only ways to an `Allocation`. Give an `Allocation` a `Residency*` that is null
  unless a launch has left chunks on the card; `find_ordinary_miss` does **not** put such an allocation in a thread's lookup cache (so `hot_location` never finds it), and `resolve` calls
  `sim_read`/`sim_write` for `[addr, addr+len)` and pulls what it names before returning. When the last `Card` chunk of an allocation is pulled the pointer is dropped and the allocation caches
  again. Launching on the card bumps the table epoch (`BigReaderLock`) once, so every thread's cache entry for it is dead. Cost: nothing on the hot path for an allocation that never
  went to the card; a launch-time epoch bump; slow-path lookups (the existing table walk) for an allocation with chunks on the card. The risk is the other entries into chunks that skip `resolve`:
  `batch_window`, `run_allocation`, `scalar_location` and `rmw_locate` all begin with `find_ordinary` and so are covered by the cache rule; `read`, `write`, `fill` and the copy calls go through `resolve`.
* **Launch hook** (`Device::launch`, `src/runtime/runtime.cpp`, written): before `run_kernel`, `hostgpu::Runtime::for_device()` (null when the mode is off: one load) and `try_launch()`; a `false` returns
  to the unchanged simulator path. After a simulator run, `note_simulated` teaches the size rule the kernel's time.
* **Calls that must synchronise state** (each needs a "pull `Card` chunks first" or "mark `Host`" hook; the write-through prototype needs none of them): `cudaMemcpy*` all directions and the async forms
  (`MemoryManager::read/write`), `cudaMemset*` and `fill`, `cudaFree`/`free_all` (drop residency), `cudaMallocManaged` and registered host memory (never `Card`: refused as arguments),
  `cuMemMap`/`unmap`/`set_access` and VMM handles (`map`, `unmap`, `retain_handle*`: a remap changes which bytes an address means: drop and re-push), IPC (`share`, `adopt`: file-backed; refuse),
  peer copies, `cudaMemcpyToSymbol` and module symbols (refused), texture and surface objects (refused), `cuMemcpy2D/3D` (go through `read/write`), graph memcpy and memset nodes (same calls),
  host-initiated kernels from libraries (cuBLAS and friends call `Device::launch` or the same memory API in the simulator, so they are covered by the hook but not offloaded: library kernels are
  PTX or simulator-internal and the cuBLAS bits differ, 07 section 6.2).
* **Launch types excluded in v1** (each a `kernel_eligible` check in `src/runtime/host_gpu.cpp`): non-SASS modules (PTX-only and library-internal kernels); any instruction outside the exact set (section 2 and
  `exact_mnemonic_allowed`: SFU, tensor cores, textures, `ATOM*`/`RED` float/CAS/EXCH, `CALL`/`RET`, packed half arithmetic); modules with `__device__`/`__constant__` variables, externs
  (`vprintf`, `malloc`, `__assertfail`), or device-side launches; cooperative, cluster and non-32 warp-size launches; texture tables, graph conditionals; arguments that point into mapped host
  memory; device 1 and up; any diagnostic that needs the simulator (`VGPU_COUNTERS`, `VGPU_SASS_TRACE`, `VGPU_KERNEL_DIGEST`, faults, race, strict, a non-default scheduler, `VGPU_MAX_STEPS`: the mode refuses to start with a message
  rather than silently dropping them); a card that is not the simulated device (compute capability and SM count must equal the profile's: **a simulated H100 on an A10G is not offered**);
  launches under the size rule; and any launch after a failure on the card (a faulting kernel poisons the context: the mode ends with a message, the simulator runs that launch).

## 5. The prototype (commit-by-commit in the branch; build: `src/runtime/host_gpu.cpp`, logic: `include/vgpu/host_gpu.hpp`)

`VGPU_HOST_GPU=exact` (env only; the YAML key `host_gpu:` of 08 is not wired). Steps of one launch, all in `hostgpu::Runtime::try_launch`:
(1) `kernel_eligible` (cached per kernel; scans the kernel's `.text` instructions against the exact set; checks module and launch shape);
(2) the size rule: history from the simulator's earlier runs of the kernel (per block, `SimTimeTable`) or, for a kernel the simulator has not run, a lower bound
`blocks x warps x static instructions / 100 M per s`;
(3) `ensure_mapped` backs the simulator's window up to the highest live address with 2 MiB granules (zero-filled the first time; kept until exit; refuses when the card has less than 256 MiB more than needed);
(4) all live allocations are pushed in 32 MiB batches to their own addresses, the kernel runs from the module's cubin (`sass::Module::image`, kept only when `VGPU_HOST_GPU` is set) with the
captured launch, synchronously; (5) the card's bytes are read back and **only 64 KiB chunks that differ are written into the simulator**.
A failure before the kernel runs leaves the simulator's memory untouched and the simulator runs the launch. The real driver is loaded next to the shim with `dlopen(RTLD_DEEPBIND)` (the two
`libcuda`s coexist in one process: checked, the shim reports the simulated device and the real driver the card, section 5.1); entry points are looked up under their exported `_v2` names (the plain
names are the old 32-bit-size calls: my first version wrote through them and mapped nothing).

`VGPU_HOST_GPU_VERIFY=1|all` (and `VGPU_HOST_GPU_LOG=1`): run the launch on the card, keep the card's bytes, run it in the simulator, compare the memory (the program gets the simulator's result);
`all` does it for kernels outside the exact set too (the evidence for the set); the exit summary counts identical and different launches for both groups.

### 5.1 Measurements of the prototype

**Same bits, whole programs, every launch** ([raw/verify_all.txt](data/round13/raw/verify_all.txt)): `VGPU_HOST_GPU_VERIFY=all` over the same three corpora (64 program runs, 60 s each at most, the
size rule bypassed): every launch the card can run ran on both and the memory was compared after each.

| group | launches | identical | different |
|---|---:|---:|---:|
| kernels **in the exact set** | 23,248 (in 35 programs) | **23,248** | **0** |
| kernels outside it (SFU, CAS, float atomics, calls) | 3,833 | 3,553 | 280 |

The 280 are the kernels of section 2 (silu, aop, binomial, s3d, bilateral, aobench, perplexity, black-scholes, bh); the 3,553 identical ones outside the set are why the set is called conservative
(for example IEEE divisions). No kernel in the set ever differed. (The sample is what those programs launch in 60 s; JACOBI1D alone contributes 20,000 tiny launches.)

**Speed**, process wall time and the programs' own "GPU Time" (the device phase), mode off against `exact`, one run each on the 4-vCPU rig (the simulator at `VGPU_THREADS=4`; the CPU reference
the PolyBench programs run themselves, 0.3 to 0.4 s, is in the process time), output identical in every row ([raw/bench_exact_default.txt](data/round13/raw/bench_exact_default.txt), [raw/bench_exact_static.txt](data/round13/raw/bench_exact_static.txt)):

| program | process off | on, default rule | on, `VGPU_HOST_GPU_STATIC=1` | device phase off / on (static) | launches on the card |
|---|---:|---:|---:|---:|---:|
| GEMM | 1.27 s | 1.56 s | **0.60 s** | 0.90 / 0.15 s | 1 |
| SYRK | 10.17 s | 10.39 s | **1.29 s** | 9.1 / 0.17 s | 1 |
| SYR2K | 19.51 s | 19.78 s | **2.32 s** | 17.4 / 0.17 s | 1 |
| 3MM | 3.75 s | 3.81 s | **1.31 s** | 2.5 / 0.16 s | 3 |
| 2DCONV | 1.18 s | 1.40 s | 0.97 s | 0.57 / 0.27 s | 1 |
| GEMVER | 1.14 s | 1.27 s | 0.93 s | 0.77 / 0.48 s | 1 of 3 |
| MVT | 2.23 s | 2.59 s | 2.63 s | 1.96 / 2.10 s | 0 (kernels too small by the rule) |
| 3DCONV (254 launches of 3 ms, 128 MiB live) | 1.00 s | 1.24 s | 1.24 s | 0.68 / 0.82 s | 0 (the size rule keeps them in the simulator) |
| FDTD-2D (1,500 launches of ~100 ms, 48 MiB live) | 158.9 s | **68.5 s** | 67.9 s | 154.8 / 64.0 s | 1,494 |

Reading the table, honestly:
* **The default rule is history-only** (a kernel's first two launches run in the simulator, which times them, and the estimate is the smallest per-block time seen): it gives **2.3x on FDTD-2D**, nothing on
  programs that launch each kernel once (GEMM, SYRK, 3MM: every kernel is launched once, so no history ever exists), and costs **0.15 to 0.35 s per process** when nothing is offloaded
  (the real driver's start, context and the 1 TiB reservation happen at the first launch; the 0.10 to 0.15 s difference in the device phase of the "stays in the simulator" rows). The mode is not free
  for small programs; it is a mode for runs that are long.
* The **static switch** estimates a never-run kernel from its code and grid: it turns GEMM, SYRK, SYR2K and 3MM into 0.15 to 0.17 s device phases (**8x to 8.4x on whole processes** for SYRK and SYR2K, whose
  simulation took 9 and 17 s) but **is not sound**: the first version of the rule used it by default and sent PolyBench 3DCONV's 253 small launches to the card, **38x slower than the simulator** (26.2 s against 1.0 s: 128 MiB
  pushed and compared per launch for a 3 ms simulation, because a stencil grid that covers the array has threads that leave early, which no static count sees). That is what the two fixes in the rule are for: **the estimate needs two
  simulated samples and uses the smallest** (the first launch pays decoding and set-up), and **a launch that took longer on the card path than the simulator was expected to need ends the offloading of that kernel**
  (one bad launch is the most it can cost). Both are unit tested. With them the static switch is safe on 3DCONV (0 on the card, 1.24 s) and keeps its wins.
* **FDTD-2D shows the price of write-through**: 1,500 launches that the simulator needs 104 ms each for take 43 ms each on the card path, and ~42 ms of that is the copy of 48 MiB in and back. The kernel itself is 0.12 to 0.18 ms.
  With a card copy that persists (section 4) those launches would cost the 7 us launch plus their kernel, about 0.2 ms: **500x on that phase instead of 2.4x**. That is the case for the residency work.
* The prototype's per-launch cost of moving everything is why programs with large live memory and cheap kernels (3DCONV) are refused. The size rule decided correctly there (the table's 3DCONV row).

**Tests** (all in the branch; none run on the WSL machine): `test_host_gpu_logic` (12 cases: the mode text, the exact set against the measured kernels of section 2, the residency protocol against a ground-truth model over 200 random
histories, the size rule and its margin, the two-sample estimate, the ban) passes, and also under `-fsanitize=address,undefined` (`halt_on_error`, leaks on) with no report; it needs no card and no CUDA, and is wired in CMake like the other unit tests.
`e2e_host_gpu` (`nvidia/tests/e2e/run_host_gpu.sh`, `host_gpu_exact.cu` + `host_gpu_exact_mod.cu`) skips with `SKIP:` when there is no `nvidia-smi`, no card, no profile for it or the mode does not start, and on the rig passes: output equal with the mode on and off,
5 launches on the card and the expf kernel, the one-warp launch and the module-variable kernel refused as intended, and `VGPU_HOST_GPU_VERIFY=1` finds every kernel of the set identical. **Mutation checks** (rig copy, each reverted): (a) `MUFU` added to the exact set fails
`exact_set_refuses_what_differed_in_the_spike`, `exact_set_matches_the_measured_kernels` and `e2e_host_gpu` ("a kernel of the exact set gave other bits on the card": `use_expf`, 64 chunks differ in every run, which also shows the verification catches real divergence);
(b) the write-back of the card's result into the simulator removed fails `e2e_host_gpu` (the program's checksums change); (c) `Residency::sim_read` not pulling fails `residency_keeps_both_sides_coherent` and `residency_moves_only_what_it_must`.
`ctest -R lint_`: 23 of 24 pass on the rig; `lint_curl_retries` fails there only because the rig copy of the tree has no git repository. Not run: TSan (no shared state in the logic under test; the runtime takes one mutex per offloaded launch),
the CI image build, the hosted CI.

### 5.2 Limits of this prototype (known, not hidden)

* Write-through copies every live byte each way per launch: correct, and the cost is the M3 "cold" row. Chains of launches need section 4's residency.
* **Whole-module refusal**: one `__device__`/`__constant__` variable in a module (a translation unit) keeps all its kernels in the simulator. A relocation scan of each kernel's code would refine it.
* The first two launches of every kernel, and every launch of a kernel launched once, run in the simulator (unless the unsound static switch is on).
* One offloaded launch at a time (a mutex); synchronous; one device (the simulator's device 0, the card's device 0); no `approx` mode; env-only (the `host_gpu:` YAML key of 08 is not wired).
* A kernel that reads outside its allocations (srad_v2) is reported by the simulator and silently tolerated by the card: for such a program the mode would hide the report, **if the kernel were in the set and the read stayed inside the window the card has mapped**.
  The size rule and the exact set do not protect against this; it is the "undefined behaviour the card tolerates" case of 08.
* Counters, the performance model's inputs, CUPTI records, fault injection, race and OOB checks, deterministic scheduling and printf order do not exist for a launch on the card (08, section 5); the exit summary says how many launches that applies to.
* Not tested: a card that is not an A10G (the identity check compares compute capability and SM count only); graphs and streams around offloaded launches; programs with several host threads launching at once; the start-up hang on WSL2.


## 6. What it adds up to, and what I would do next

**Is it worth it?** For the user's stated use, a long simulation of the card that is in the machine, **yes, as an opt-in**, and the evidence is stronger than I expected in two places and weaker in one:

* *Stronger*: the card takes the simulator's addresses with one reservation per window; 23,248 launches of kernels in a conservatively chosen exact set were bit-identical to the simulator's on the A10G, with no difference; and every kernel that differed
  was explained by an SFU instruction or a CAS loop. The risk of silent wrong answers in `exact` mode comes from what a sample cannot show (races, reads of uninitialised or out-of-window memory, a rare input for an instruction the set trusts), not from known arithmetic.
* *Stronger*: the speed-ups where the rule can see them are large (8x on SYRK and SYR2K processes, 2.3x on FDTD-2D with the crudest data movement, 150 ms device phases for simulations of 2.5 to 17 s).
* *Weaker*: the exact set excludes a great deal of real code (every `expf`, `sqrtf`, division, float atomic: of the 24 HeCBench programs of the reduced list that launched a kernel in the 60 s sampled, 18 had at least one kernel outside it), and the size rule cannot make a good decision for a kernel it has never seen run. A real workload
  of PyTorch kernels or a cuBLAS-heavy program (library kernels are the simulator's own, never offloaded) would see little of this until the set grows. The set can grow with evidence: the verify mode is the tool that decides a mnemonic's place (`MUFU.RCP` + `FCHK` division and the `RED`/`ATOM` integer forms are the first candidates).

**Recommended next step, in order** (each a decision for you, none started):
1. Persist the card copy (section 4): the allocation-level gate at `resolve()`/`find_ordinary_miss()` rather than the per-chunk sentinel; it removes the copy that costs FDTD-2D 98% of its time on the card path. Gate for merging: no change in the hot-path benchmarks (it adds nothing to `hot_location`), a differential test of random launch and access interleavings against
   the write-through mode, and a mutation check per synchronising call listed in section 4.
2. Make the first launch decidable: run the first launch of a kernel on the card when the card's own measured kernel time shows the simulator would need far longer than the copy (the card kernel time is known after one launch; the simulator is at least ~8x slower per 07), with the ban rule as the net. Cheaper than a sound static estimate, which does not exist.
3. A card CI runner for `e2e_host_gpu` (an A10G on demand, a few minutes per run), as 08 asked: until then the test only runs where a card is attached, and hosted CI runs the logic unit test and the SKIP.
4. Grow the exact set with `VGPU_HOST_GPU_VERIFY=all` runs over more programs (the HeCBench/Samples lists, PyTorch kernels), one mnemonic family at a time, keeping the set a whitelist.

**Not worth doing**: a different simulated card than the host's (07, still true: the identity check refuses it); `approx` before the exact path has users; offloading libraries (their bits differ, 07 section 6.2); an in-simulator watchdog by `fork` (use the deadline thread); and the YAML key before the env switch has had a release of use.

**Costs and cautions to keep in view**: 0.15 to 0.35 s of fixed cost per process when the mode is on and nothing is offloaded; no counters or diagnostics for card launches; the card is a shared machine resource (another process on the same GPU slows the launches and could exhaust its memory: the mode backs only the memory in use and refuses with
less than 256 MiB spare); a faulting launch ends the mode for the process; AMD is not covered (no card to measure on).

## 7. Step A and the persistent card copy (2026-10-28/29, a second A10G rig)

The question: does a card copy that outlives a launch (section 4) pay? Rule set beforehand: stop if the gain over write-through is under about 1.5x on programs that matter (more than a few seconds of simulation that
write-through already offloads) or only FDTD-2D-like cases benefit; otherwise build it.

### 7.1 Profile, before any code

Method: every corpus program (64 runs, 100 s at most each) ran with `VGPU_HOST_GPU_VERIFY=1` and `VGPU_HOST_GPU_TRACE`, which records every allocation, free, host copy in, out or fill of device memory (`MemoryManager` hook) and every launch
(kernel, grid, the simulator's own time, whether it is in the exact set, its time on the card, the allocations its arguments name): [raw/traces](data/round13/raw/traces) (gzip). [residency_model.py](data/round13/residency_model.py) replays them under four
policies with the rule of the runtime (two simulator launches per kernel first, margin 2, the ban) and the measured costs: *off*, *write-through* (0.85 ns per live byte, measured on FDTD-2D), *persist* (sound: a launch the simulator runs first pulls everything
and leaves everything Host; an offloaded launch pushes the Host chunks and leaves every live chunk Card) and *argsonly* (unsound upper bound: only the allocations the arguments name). Output: [raw/residency_model.txt](data/round13/raw/residency_model.txt),
[raw/residency_summary.txt](data/round13/raw/residency_summary.txt). The simulator times in it are the simulator's alone (a first version of the trace also timed the card path in verify runs and made 3DCONV look 40x slower than it is; found and fixed).

* 64 programs, 58 with launches, **34,236 launches**; 25,722 are in the exact set; 21,087 follow another in-set launch with no host copy between (the "chained" case); write-through offloads 4,499 of them, a persistent copy 4,869.
* Device-phase time over all programs: off 629 s, write-through 439 s, **persistent 358 s** (1.23x over write-through in total: most of the corpus's simulation time is in programs neither mode offloads: CORR and COVAR 36 s each (SFU), SYRK and SYR2K (one launch each)).
* Programs with more than 3 s of simulation that write-through already offloads: **6**.

| program | off | write-through | persistent | gain |
|---|---:|---:|---:|---:|
| FDTD-2D (552 launches seen) | 57.8 s | 24.2 s | 0.8 s | 32x |
| aligned-types (HeCBench) | 64.0 s | 8.9 s | 1.5 s | 5.9x |
| babelstream | 31.3 s | 26.0 s | 5.6 s | 4.6x |
| base64e | 58.0 s | 11.7 s | 1.7 s | 6.9x |
| GRAMSCHM | 52.4 s | 39.4 s | 25.4 s | 1.55x |
| LU | 55.7 s | 20.0 s | 14.2 s | 1.41x |

Eight programs gain 1.5x or more over write-through (those five, axpby 9.4x, JACOBI2D 3.4x, needle 1.5x): **3,571 of the 34,236 launches (10%) are in them, 1,773 launches (5%) are ones persistence newly lets the card take or makes cheaper**, and their device time goes from 111 s
(write-through) to 35 s. In the other 50 programs persistence changes nothing (single-launch programs, SFU kernels, kernels too small). Four of the six large programs gain 4.6x to 32x, so it is not only FDTD-2D; two (the LU and GRAMSCHM shapes, where
in-set and out-of-set kernels alternate and every simulator launch gives the card copy up) gain 1.4x to 1.55x in the model.

**Cost with data already on the card**, measured ([raw/resident_launch_cost.txt](data/round13/raw/resident_launch_cost.txt), `M2_LOOP`): a launch plus synchronisation 7.9 to 8.2 us for tiny kernels; the real kernels of FDTD-2D 111 us (step 1, 2) and 143 us (step 3) against 98 to 116 ms
in the simulator: **~900x**; 2DCONV 279 us against 575 ms (2,060x); GEMM 365 us against 902 ms; JACOBI2D 23 us against 20 to 27 ms (~1,000x); bfs and gaussian kernels 8 us against 0.15 to 0.7 ms (20 to 88x). So the "~500x" for FDTD-2D holds for the launch itself (queued launches without a sync each
2.6 us for tiny kernels, 105 us for FDTD-2D); on the whole program it is lower (section 7.3).

Decision: the rule is not triggered (four large programs gain 4.6x to 32x, and the corpus's chained launches are 21,087 of 34,236), so **it was built**.

### 7.2 What was built

* **State**: `Residency` (include/vgpu/host_gpu.hpp), one byte per 64 KiB chunk: Host, Card, Both, as in section 4, per tracked allocation (`Impl::res` in src/runtime/host_gpu.cpp).
* **The gate** (`vgpu::CardGate`, include/vgpu/memory.hpp; src/core/memory.cpp). `Allocation::card_valid` counts chunks whose card copy is valid (Card or Both). While it is above zero the allocation is **out of every thread's lookup memory**
  (`find_ordinary_miss` and `run_allocation` decline it; `card_void_lookups` voids what threads remember, with `note_free` under a quiet write lock), so each access to it takes `resolve()`, which asks the gate first (`before_access(addr, len, write)`; a kernel-style scalar store is told apart by its op text, because it passes `writing=false`).
  `free`, `free_all` and `share` tell the gate before they take their locks. **Nothing was added to `hot_location`, `find_ordinary`'s hit path or any per-access path of an allocation that is not card-resident**; the cost elsewhere is one relaxed load of a counter in `resolve`, `free`, `free_all`, `share`.
* **Gate actions**: a read pulls the Card chunks it covers (they become Both); a write pulls a Card chunk it covers only partly, then makes the chunks Host. A launch the simulator runs gives everything back first (`before_simulated`; this is a speed measure: the gate alone is also correct, because every simulator access to a card-valid allocation takes the slow path and marks it, but a kernel's accesses would then each pay a lock).
* **Launch** (`try_launch`): push the Host chunks (merged into runs of up to 32 MiB), run, mark every live chunk Card; nothing comes back. The size rule charges the push only (a later pull is paid when the host reads); the ban still ends a kernel whose launch took longer than the simulator was expected to.
  `VGPU_HOST_GPU_WRITE_THROUGH=1` keeps the previous mode for comparison.
* **Failure**: a kernel that fails on the card while the card holds data the simulator does not raises an internal error (the data cannot be recovered); before, such a failure fell back to the simulator. A faulting in-set kernel is therefore fatal once data is resident.
* **Not supported** (state, single device, synchronous launches, no `approx`, as before) plus: another host thread running simulator launches at the same moment as an offloaded launch's end is not protected (the gate and the lookup voiding assume the launching thread is the only one touching device memory then).

Verification and tests:
* `VGPU_HOST_GPU_VERIFY=1` now exercises the persistent bookkeeping: the card runs the launch on its resident copy, the simulator runs it too on its own memory (which stays the truth), and after each launch the card's whole memory is compared with the simulator's. A missed sync point leaves the card with stale input and shows as a difference.
  **Corpus: 64 programs, 21,804 launches of kernels in the exact set compared, 21,804 identical, 0 different** ([raw/verify_resident.txt](data/round13/raw/verify_resident.txt), 26 programs launched an in-set kernel in the 100 s each ran).
* Unit tests (no card): `test_host_gpu_logic` (14: the state machine against a ground-truth model over random histories including simulator launches, `chunk_span`, `chunk_runs`, the rule) passes plain, under ASan+UBSan (leaks on, halt) and under TSan; `test_card_gate` (7: the memory manager's side with a fake gate:
  a remembered allocation is voided, run accesses decline, copies ask with range and direction, free and reset tell the gate, export pulls first, zero cost with no resident allocation, an allocation is ordinary again at zero) passes plain, ASan+UBSan and TSan (built from memory.cpp, memory_backing.cpp, error.cpp).
* `e2e_host_gpu` extended: between launches on the card the program does a partial write into a resident chunk, a memset over whole chunks, a device-to-device copy, a free and a reallocation, a partial read, and a simulator launch; the printed checksums must equal the mode-off and write-through runs, the persistent run must move under half of write-through's bytes (109 + 95 MiB against 312 + 312), and verify must find 31 in-set launches identical.
* **Mutation checks** (each applied to the rig copy, rebuilt, reverted): write sync dropped (`e2e_host_gpu` fails: "a kernel of the exact set gave other bits"); read sync dropped (fails: checksums differ); free not told to the gate, either in the memory manager or in the gate (fails); lookup exclusion in `find_ordinary_miss` removed (`test_card_gate` 4 of 6); exclusion in `run_allocation` removed (5 of 6);
  `card_void_lookups` emptied (4 of 6); scalar store not treated as a write (5 of 6); export not pulling (6 of 7, after adding that test); the earlier `MUFU`-allowed and write-back-removed mutations still fail. **One is not caught, by design**: dropping `before_simulated`'s reset leaves output and bits unchanged, because the gate alone keeps it correct; it only costs speed.
  (Two of the guards, the lookup memory and the run accesses, are only reachable by simulator kernels and library code, which the end-to-end program does not run between launches on resident data; that is why `test_card_gate` exists.)

### 7.3 Wall times, same programs, A10G rig (4 vCPU), output compared with timing lines dropped

[raw/bench_resident.txt](data/round13/raw/bench_resident.txt); the "out DIFF" rows differ only in lines that print a measured rate or overlap percentage (checked by hand for base64e and babelstream: the program's own PASS lines and every result line are identical).

| program | off | write-through | persistent | persistent vs write-through | vs off | moved (in / out, MiB) wt -> persistent |
|---|---:|---:|---:|---:|---:|---|
| FDTD-2D | 161.9 s | 68.5 s | **4.9 s** | 14x | 33x | 71,715 / 71,715 -> 48 / 16 |
| aligned-types | 2,682.6 s | 1,091.8 s | **10.6 s** | 103x | 253x | 1,047,322 / 1,047,322 -> 1,240 / 1,192 |
| babelstream | 561.1 s | 380.6 s | **15.9 s** | 24x | 35x | 571,010 / 571,010 -> 1,152 / 0.3 |
| base64e | 376.5 s | 53.0 s | **12.0 s** | 4.4x | 31x | 49,344 / 49,344 -> 6,619 / 8,745 |
| JACOBI2D | 1.01 s | 0.50 s | 0.40 s | 1.25x | 2.5x | 275 / 275 -> 7.6 / 7.6 |
| rod/sc | 0.44 s | 0.51 s | 0.37 s | 1.4x | 1.2x | 141 / 141 -> 101 / 51 |
| axpby | 0.47 s | 0.58 s | 0.41 s | 1.4x | 1.15x | 272 / 272 -> 12 / 47 |
| LU | 56.9 s | **16.6 s** | 20.3 s | **0.82x (slower)** | 2.8x | 23,008 / 23,008 -> 22,496 / 22,496 |
| GRAMSCHM | 299.9 s | 314.6 s | 279.6 s | 1.13x | 1.07x | 98,208 / 98,208 -> 98,208 / 98,176 |
| 3DCONV, ADI, GEMM, SYRK, gaussian, needle | 0.02 to 10 s | same or +0.2 s | same | 1.0x | 0.9x to 1.0x | 0 (the rule keeps them in the simulator) |

* The profile's prediction held for the big winners (FDTD-2D 32x predicted, 14x measured on the whole program because the first two launches of each kernel and the host phase remain; aligned-types, babelstream, base64e within 1.5x of the model).
* **LU and GRAMSCHM do not benefit.** Their launches alternate between kernels in the exact set and kernels that are not, so every simulator launch gives the card copy up and the next offloaded launch pushes it again: 22 GB moved either way. Persistent is 18% **slower** than write-through on LU (cause not isolated: the traffic is the same, so the gate, the chunk bookkeeping and the extra pulls are the suspects).
  An upper bound from the model for a finer reach (only the allocations a launch's arguments name, unsound as pointers inside memory are not followed): the same numbers as `persist`, so even that would not help these two: the alternating kernels share their buffers.
* Fixed cost of the mode when nothing is offloaded stays 0.15 to 0.35 s per process (3DCONV 1.04 to 1.28 s, GEMM 1.32 to 1.54 s).

### 7.4 What this adds up to

**Worth it**, for runs of repeated launches on the same buffers; the gain is 14x to 103x over write-through, 31x to 253x over the simulator on four programs of 6 to 45 minutes of simulation (FDTD-2D, aligned-types, babelstream, base64e), no effect on 50 of the 64 corpus programs, a loss on LU (-18% against write-through)
and nothing on GRAMSCHM. Share of the corpus: 10% of launches are in programs that gain 1.5x or more, 5% of launches change. The profile's weak side stands: single-launch programs (GEMM, SYRK, SYR2K, 3MM) and SFU kernels are untouched by persistence; they need the first-launch decision (section 6, step 2) and a grown exact set.
Suggested defaults: persistent on (it is), with a ban that also looks at the traffic: a kernel whose offloaded launches move more than a fixed share of the live memory each time (LU's) should go back to the simulator after two launches.
