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
- **L4 Ecosystem** — cuBLAS/cuBLASLt/cuDNN/cuFFT/cuRAND/cuSPARSE/cuSOLVER/NCCL/NVRTC/NPP/nvJPEG
  are implemented and verified against hardware (docs/libraries.md);
  PyTorch itself is not.

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
  of catching bugs: deterministic zero-fill, trapping misaligned access.

  Integer division by zero used to be one of them and no longer is. The
  reasoning was that a div-by-zero in a kernel is almost always a bug; real
  code falsified it. ggml's flash-attention passes zero for a stride the
  configuration does not use, takes a remainder from it, and discards the
  answer -- so trapping made those kernels unrunnable over arithmetic that was
  never going to matter. It now follows the hardware, deterministically: an
  all-ones quotient and a remainder of the dividend. `VGPU_TRAP_DIV_BY_ZERO=1`
  restores the trap for a debugging run.

  The general lesson is the one the register analysis taught earlier: a
  divergence that catches bugs is only worth having while it does not also
  reject correct programs. Compilers emit arithmetic whose result is dead, and
  a simulator that judges every instruction as if its result mattered will
  refuse code that hardware runs. Reading an undefined register is the same
  story -- it now reads as zero, and the diagnosis moved to the point where
  such a value would become a result: an address, a branch condition, or a
  store.

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

### D6. Two shims: documented runtime API over undocumented dark API

To host unmodified nvcc binaries we provide `libvgpucudart` — a drop-in
`libcudart.so.13` implementing the **documented** CUDA Runtime API plus the
documented nvcc host-registration ABI (`__cudaRegisterFatBinary`,
`__cudaRegisterFunction`, `__cudaPushCallConfiguration`, `__cudaGetKernel`,
`cudaLaunchKernel`). A chevron launch lowers onto these; our runtime pulls the
embedded PTX out of the fatbin and runs it on the SIMT engine. This is the
supported path and requires the app to link *shared* cudart.

The alternative — letting NVIDIA's *static* cudart run and satisfying it from
our `libcuda` — means implementing `cuGetExportTable`, a set of **undocumented,
version-specific, integrity-checked** vtables ("the dark API"). We implemented
enough to watch cudart bootstrap, but chose the documented runtime API as the
primary interface: it is stable, legible, and clean-room. Static-cudart hosting
stays behind an env flag as future work. The rule this encodes: prefer a
documented interface we can maintain over an undocumented one we must chase.

Two ABI subtleties that bit us and are now guarded:
- `cudaDeviceProp` is version-specific. The shim's copy MUST match the toolkit
  that compiled the app, or field writes land at the wrong offsets and smash
  the caller's stack. The build derives the ABI header from the `nvcc` on PATH
  (not a stale system copy) and `static_assert`s the struct size.
- CUDA 12.4+/13 lowers chevrons through `__cudaGetKernel` + `__cudaLaunchKernel`
  (a `cudaKernel_t` handle), not only the classic `cudaLaunchKernel(func, …)`.
  Both entry points are provided.

### D7. Address spaces are windows, not aliases

PTX `.shared` and `.local` addresses are **offsets within their space**, not
generic addresses: `mov.u32 %r, sharedvar` yields a 32-bit offset, and `cvta`
converts to/from the generic space. VirtualGPU models this with two reserved
VA windows (`kSharedVaBase`, `kLocalVaBase`); space-tagged accesses add the
window base, `cvta.<space>` adds it, `cvta.to.<space>` subtracts it, and
generic accesses route by address range. `.global`/`.const` already alias the
flat device VA range, so those conversions are identity.

Getting this wrong is silent: an early version treated every space as an alias,
so a 32-bit `st.shared [%r]` truncated a 64-bit pointer and wrote through
address 0. Shared memory is per-block and **zero-initialized** (hardware leaves
it undefined) — another deliberate determinism divergence.

### D8. Tensor-core fragments: a chosen, documented layout

PTX explicitly leaves the mapping of matrix elements to `wmma` fragment
registers **unspecified**. VirtualGPU therefore defines its own self-consistent
layout (m16n16k16: A/B halves indexed by lane and register, C/D f32 as a flat
row-major 256-element spread over 32 lanes x 8 registers) and computes
`D = A x B + C` from it.

Consequence, stated plainly: kernels that treat fragments as opaque
(`wmma.load` -> `wmma.mma` -> `wmma.store`) or that fill them uniformly get
hardware-matching results. A kernel that depends on NVIDIA's exact undocumented
element distribution — which is legal to observe but not to rely on — may
differ. That is the correct trade for a functional simulator, and the
characterization suite is where any real divergence would be caught.

### D9. Clean-room ABI discipline

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
