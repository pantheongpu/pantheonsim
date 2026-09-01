# VirtualGPU architecture

## The layer stack

The non-negotiable structural rule: **no per-SKU emulators.** One shared
engine, parameterized by data.

```
vendor frontend            libvgpucuda (CUDA driver API)      [future: HIP/ROCm]
        |
architecture semantics     PTX subset -> AST                  [future: sm_xx feature gates,
        |                                                      AMD GCN/CDNA IR]
common CPU engine          SIMT warp interpreter + virtual memory + scheduler
        |
device profile             profiles/nvidia/*.yaml  (data, not code)
```

An H100 and an H200 differ only in their profile data. Hopper and Blackwell
will differ in feature gates over the same interpreter. NVIDIA and AMD differ
in frontend + ISA, but share the execution engine and memory model (wavefront
width 64 is a parameter, not a rewrite — enforced today by an explicit
`warp_size == 32` check that names the gap).

## Fidelity levels

- **L1 Discovery** — devices exist, enumerate, report properties. (Working)
- **L2 Runtime behavior** — contexts, memory, copies, errors; streams/events
  next. (Core working)
- **L3 Compute** — PTX kernels execute functionally on CPU. (Working subset)
- **L4 Ecosystem** — cuBLAS/cuDNN/NCCL/PyTorch. (Not attempted; design only)

## Repository map

```
include/vgpu/…, src/
  core/      profiles, registry, restricted-YAML, virtual memory, errors
  ptx/       lexer, parser, AST                       (vendor frontend, NVIDIA)
  exec/      SIMT interpreter, scheduler              (common CPU engine)
  runtime/   Device/Runtime facade (primary-context model)
  cuda/      libvgpucuda.so — C ABI driver-API shim   (vendor frontend, NVIDIA)
  cli/       vgpu binary
include/vgpu_cuda.h   clean-room driver-API subset header (public ABI)
profiles/  nvidia/{a10,a100,h100,h200,b200}.yaml  amd/ (placeholder)
tests/     unit/ framework/ c_harness/ kernels/
```

C++20, CMake, **zero external dependencies** (hand-rolled restricted-YAML +
micro test framework) so a bare CI container can build it.

### Why C++ (vs Rust)

The deliverable is a C-ABI `libcuda`-compatible `.so`, future JIT work is
LLVM (a C++ library), and the audience (CUDA shops, CI images) always has a
C++ toolchain. Rust's safety was the counter-argument; we compensate with
bounds-checked accessors everywhere and dense tests. Decided 2026-09-01;
revisit only with a concrete failure mode in hand.

## Key decisions

### D1. Warp-vectorized interpretation (not thread-per-lane)

A warp is the execution unit: 32 lanes advance in lockstep; each virtual
register holds 32 lane values; predicates and activity are 32-bit masks. No OS
threads per GPU thread — a block is a vector of warp states advanced by a
scheduler loop on one CPU thread.

Divergence: a branch that splits the active mask pushes the not-taken
`(pc, mask)` onto a per-warp stack and follows the taken side; when a path
retires (`ret`), the next parked path pops. Paths reconverge at `ret`.
Consequence: `bar.sync` under divergence is rejected with a precise error
(IPDOM reconvergence is the planned fix, not a redesign — the stack becomes
reconvergence-point-keyed).

The interpreter is deliberately naive (hash-map register files, per-lane
loops). Correctness, determinism, and diagnosability outrank speed; the
upgrade path is `PTX -> internal IR -> LLVM JIT` behind the same
`exec::launch()` boundary, so the runtime above never changes.

### D2. Scheduling is an abstraction, and always will be

`exec::Scheduler::pick(runnable_warps)` chooses which warp advances at each
yield point (barrier or retirement). Today: deterministic round-robin, so
every run is bit-for-bit reproducible. The interface exists *now* so that
`--scheduler random` (seeded) and `--scheduler adversarial` (searching legal
orders to break missing-synchronization assumptions) drop in without touching
the interpreter. This is the foundation of the "TSan for GPUs" ambition;
nothing in the engine may assume a fixed warp order beyond what a scheduler
provides.

Blocks currently run sequentially in fixed order (a legal schedule). Inter-
block parallelism on host threads can come later; it must preserve the
scheduler abstraction per block.

### D3. Virtual memory is sparse, monotonic, and paranoid

- Device VAs live at `0x7fff'0000'0000+`, far from host pointers; a host
  pointer passed as a device pointer is diagnosed by range.
- Backing is 64 KiB chunks materialized on first write: profiles can claim
  any VRAM size regardless of host RAM. Untouched device memory reads as
  **zero, deterministically** (real GPUs: undefined — documented divergence).
- VAs are never reused, so use-after-free stays detectable process-lifetime.
- Every access is bounds-checked; overruns into alignment padding are
  classified as overruns, not unknown pointers. Diagnostics carry allocation
  base/size, kernel, PTX line, lane, and profile — they are a product
  feature, not debug leftovers.
- Documented deliberate divergences from real hardware, all in the direction
  of catching bugs: deterministic zero-fill, trapping integer division by
  zero, trapping misaligned access.

Host is assumed little-endian 64-bit (static_assert-able; scalar loads use
memcpy semantics).

### D4. Profiles are data; unverified data is labeled

Profiles hold only functionally visible properties (limits, capabilities,
sizes) — no clocks, no bandwidths. Every profile carries `verified: false`
until a future hardware-characterization suite (physical GPUs as oracles)
confirms each value; the CLI prints that caveat. Placeholder values come from
public vendor documentation only. The restricted-YAML subset is documented in
`include/vgpu/yamlish.hpp`.

### D5. Errors: stable codes + rich text, and no silent gaps

One exception type (`vgpu::Error`) with a stable `Err` code (tests and the C
shim switch on it) and a human message built where the context lives. Layers
*rewrap* — parse errors gain kernel names, launches gain lane/profile — the
code survives rewrapping. The C ABI maps codes onto documented `CUresult`
values (kernel-side faults → `ILLEGAL_ADDRESS`; emulator gaps →
`NOT_SUPPORTED`) and prints the full message to stderr, because a bare enum
would waste the diagnosis.

Unsupported ≠ broken: anything outside the implemented subset (PTX
instruction, API, attribute) fails loudly, naming exactly what was missing.
Silent wrong answers are the one unforgivable bug class in an emulator.

### D6. Clean-room ABI discipline

`vgpu_cuda.h` is written from NVIDIA's public driver-API documentation;
numeric values (error codes, attribute ids) follow the documented ABI so real
headers/binaries agree, but no proprietary header text or code is copied.
The same policy will apply to NVML, HIP, and ROCm-SMI surfaces later.

## Differential testing (designed-for, not built)

The characterization suite will run identical micro-tests on physical GPUs
and their virtual profiles, diff observable results, and turn every
divergence into a regression test + a profile fix:

```
generated CUDA/PTX micro-test -> real H100 ---+
                              -> vgpu H100 ---+--> diff -> conformance DB
```

Architectural accommodations already in place: deterministic execution
(diffs are meaningful), profile-driven behavior (fixes are data edits),
stable error codes (comparable outcomes), stats from every launch.

## Determinism contract

Same binary + same inputs + same profile + same scheduler ⇒ identical device
memory afterward, identical errors, byte-for-byte. Anything that would break
this (host-thread parallelism, address randomization) must be opt-in.
