# TODO / status

Updated: 2026-09-01 (rev 4). See ARCHITECTURE.md for the design behind these.

## Implemented (tested)

- M0 build system: CMake + zero-dep C++20; `scripts/build.sh`, `scripts/test.sh`
- M1 profiles: a10/a100/h100/h200/b200 (placeholders, `verified: false`);
  `vgpu list-gpus`, `vgpu info --gpu <id> [--json]`
- M2 runtime core: devices, primary-context model, sparse virtual VRAM,
  H2D/D2H/D2D, OOB/UAF/double-free/interior-free/misalignment diagnostics
- M4 PTX parser: ld/st(param/global), mov, cvta, add/sub/mul/min/max/div/rem,
  and/or/xor/shl/shr, mad.lo, fma, mul.wide, setp, selp, predication (@/@!),
  bra, bar.sync 0, ret/exit; sregs tid/ntid/ctaid/nctaid/laneid;
  0f/0d float literals; precise unsupported-PTX errors
- M5 SIMT interpreter: 32-lane warps, mask divergence + reconvergence stack,
  functional cross-warp bar.sync, deterministic round-robin scheduler,
  step-budget infinite-loop guard, per-lane fault context
- M6 vectorAdd end-to-end: `vgpu demo vectoradd` (exact verification)
- M3 driver API: libvgpucuda.so — cuInit, version, device get/count/name/
  totalmem/attributes/CC, ctx create/destroy/set/get/sync, primary ctx
  retain/release, mem alloc/free/HtoD/DtoH/DtoD/getinfo, moduleLoadData(Ex)/
  unload/getFunction, cuLaunchKernel(kernelParams), error name/string;
  VGPU_GPU / VGPU_DEVICE_COUNT / VGPU_QUIET
- M7 (real): unmodified nvcc-compiled CUDA apps run on VirtualGPU via
  libvgpucudart (CUDA Runtime API + nvcc host-registration ABI +
  cuLibrary/cuKernel + fatbin PTX extraction incl. zstd). Verified with the
  external C11 driver-API harness AND an nvcc-compiled vectorAdd e2e test.
- Pantheon workloads: the pantheongpu stress/diagnostics kernels run
  unmodified (idle, memory_read/write, galpat, march_test, memory_hammer,
  atomic/int/compute virus). memory_read differential-matches a physical RTX
  3060 including fault-injection + device printf. See docs/pantheon-workloads.md.
- PTX additions: cvt (int<->float, rounding modes), neg, abs, prmt.b32, not,
  shf funnel shifts, atomics (add/min/max/and/or/xor/exch/cas), vector
  ld/st.v2/v4, predicate logic (and/or/xor/not.pred), .local memory frames,
  module .global variables, aggregate by-value params, device printf (vprintf),
  transcendentals (ex2/lg2/sin/cos/sqrt/rsqrt/rcp/tanh), bfe/bfi/brev/popc/clz,
  mad.wide, mov pack/unpack, NaN-aware setp forms, inline-asm register locals.
- Shared memory: static + dynamic (extern) .shared, per-block zeroed frames,
  ld/st/atom.shared, correct space-relative addressing (cvta to/from generic).
- Warp shuffles (shfl.sync up/down/bfly/idx, + predicate output) and
  vote/ballot.
- Tensor cores: wmma.mma m16n16k16 f32.f32 (row/col layouts), wmma.store.d.
- f16 (software IEEE binary16) and packed f16x2 arithmetic.
- CUDA Graphs: real stream capture -> record -> replay.
- Multi-GPU: peer access queries and cudaMemcpyPeer(Async) across virtual
  devices (all_reduce and p2p_thrasher take their real peer-DMA paths).
- Configurable virtual VRAM (VGPU_VRAM_MB); VGPU_TRACE coverage-growth logging.

## Hardware characterization

`tools/characterize.cu` + `tools/characterize-telemetry.sh` read a physical
device and emit a profile with `verified: true`.
`tools/characterize-cloud.sh <instance-type> <region>` does the whole thing on
a rented GPU -- launch, characterize, capture conformance references, and
terminate (termination is registered before launch and then confirmed, because
an instance left running bills by the hour).
`tools/compare-profile.py` diffs a measured profile against the one in the
tree, so corrections are visible rather than silently applied.

Verified against real hardware, four devices across three architectures:

| profile | device | how |
| --- | --- | --- |
| `nvidia/rtx3060` | RTX 3060 (sm_86) | local |
| `nvidia/a10` | A10 (sm_86) | Lambda `gpu_1x_a10` |
| `nvidia/a100-sxm4-40gb` | A100 SXM4 40GB (sm_80) | Lambda `gpu_1x_a100_sxm4` |
| `nvidia/h100` | H100 SXM5 80GB (sm_90) | Lambda `gpu_1x_h100_sxm5` |

All four match the physical device on **512 conformance values each** -- the
same binary run on hardware and on VirtualGPU, diffed.

What characterization corrected in the documentation-derived placeholders:
- A10 `vram_bytes` 25769803776 -> 23696375808 (datasheet "24 GB"; the device
  reports 22.07 GiB) and `temperature_max_c` 85 -> 98.
- H100 `vram_bytes` 85899345920 -> 85028896768.
Clocks, power caps, SM counts, shared-memory limits and PCI ids were already
right. Capacity values being wrong is exactly what this process is for.

Still documentation-derived: A100 80GB, H200, B200, and all three AMD
profiles. B200 had no Lambda capacity; AMD parts are not offered there.

## Register and occupancy modeling

PTX declares *virtual* registers, so counting declarations says nothing about
what a thread occupies. `src/ptx/regalloc.cpp` solves liveness over the
control-flow graph to a fixed point and takes the peak, counts a 64-bit value
as a register pair as the hardware does, keeps predicates in their own file,
and rounds to the allocation granularity.

That count is functional, not decorative:
- a block needing more registers than the device allows fails with
  "too many resources requested for launch", as on hardware;
- `__launch_bounds__` (`.maxntid` / `.reqntid`) is parsed and enforced;
- `cudaFuncGetAttributes` reports real `numRegs` and `localSizeBytes`, and
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` does the standard occupancy
  calculation instead of returning a placeholder.

Checked against a physical RTX 3060: a simple kernel reports **8 registers and
6 blocks/SM on both**. Measured against `ptxas -v` across the pantheon kernels,
this analysis lands a little *under* the real allocation -- 8 vs 14, 16 vs 20,
24 vs 26, 16 vs 24 -- because ptxas keeps values live longer than the data flow
requires in order to hide latency. So treat it as a lower bound on what a
thread needs: the launch refusal only fires when a kernel is genuinely
impossible, and the occupancy figure is optimistic by the same margin.

It used to err the other way, and far harder. Approximating a live range as
first-definition-to-last-use and then extending everything that touched a loop
across the whole loop body made every value in a grid-stride kernel look
simultaneously live: 328 registers for a kernel ptxas compiles into 14, and 10
of the 46 pantheon workloads refused to launch. The lesson is that a
conservative estimate is only safe while it stays under the hardware limit --
past that it stops being caution and starts being a false negative.

## Ecosystem tools that work today

- **pynvml** and anything built on it (nvitop, gpustat, monitoring agents,
  framework memory queries) — verified against a 4-GPU virtual rack.
- **lspci**, via generated PCI configuration space.
- **nvidia-smi / rocm-smi / rocm_agent_enumerator** — supplied by VirtualGPU
  (see docs/telemetry.md for why the stock nvidia-smi binary cannot be used).

**Vendor libraries.** cuBLAS, cuBLASLt, cuDNN, cuFFT, cuRAND, cuSPARSE,
cuSOLVER, NCCL, NVRTC, NPP and nvJPEG are implemented under their real sonames, each verified
against NVIDIA's own library on a physical GPU: cuDNN, cuFFT and cuSPARSE are
bit-identical on every value the conformance suite reports, cuSOLVER on
everything but one f32 eigenvalue, and NCCL on all 24 values at two ranks
across two physical GPUs. The math runs on the host rather than through the
interpreter, because a vendor library is not user code — see docs/libraries.md
for the boundary, the per-library scope, and what each one deliberately refuses.

NVRTC works by invoking the toolkit's own nvcc, which runs on the host and
needs no GPU -- so runtime-compiled kernels (CuPy, Numba, Triton, inductor)
reach the interpreter through the driver API like any other PTX.

Multi-GPU is verified against real hardware in two places: the local two-GPU
box for the differential suite, and a rented two-H100 SXM5 instance for the
same suite plus eight-rank NCCL over eight processes. That instance runs CUDA
12.8, so it also covers the older toolkit's LZ4 fatbins and soname majors.

Not yet: `nvidia-smi topo -m`, DCGM. PyTorch also ships thousands of its own kernels,
which would run on the interpreter, so `import torch` finding a usable GPU is
still a separate question from library coverage.

## Known out of scope (not CUDA)

- `rt_virus` needs **OptiX** (NVIDIA's ray-tracing library, loaded from
  libnvoptix.so.1) and `media_enc_virus` needs **NVENC**
  (libnvidia-encode.so.1). Both are separate NVIDIA subsystems, not CUDA;
  emulating them is a distinct project. They fail with the vendor library's
  own error rather than a VirtualGPU error.

## Partially implemented

- Divergence: min-PC reconvergence -- paths at the same pc merge and the lowest
  pc runs next, so bar.sync after a divergent region works. Not full IPDOM:
  irreducible control flow is not handled.
- cuCtxSetCurrent(NULL) pops rather than clearing a per-thread binding; the
  current-context stack is process-global, not thread-local.
- M7 proper: needs the CUDA *runtime* API shim + fatbin PTX extraction to run
  an unmodified nvcc-built binary (embedded-PTX driver-API apps work today).

## Not implemented (fails loudly, never silently)

- PTX: textures/surfaces, cp.async, wgmma, grid sync, inline-asm-only
  instructions
- Runtime: async copies, unified/managed memory, virtual memory mgmt API
  (cuMemAddressReserve…), host-pinned memory
- Frontends: cubin/SASS loading, cuGetProcAddress dispatch, AMD everything
  (HIP, ROCm-SMI, CDNA ISA)
- Tooling: `vgpu run` (LD_LIBRARY_PATH/LD_PRELOAD wrapper), `vgpu test
  --matrix`, trace record/replay, schedulers random/adversarial, race
  detection, OOM injection, characterization/differential-fuzz harness,
  conformance DB + compat scores

## Performance

Measure with `tools/bench.sh` (vectorAdd) and a register-heavy kernel.
Profile with gprof; guessing has been wrong every time so far.

Done (1.5x on memory-bound, 2.0x on ALU-bound):
- Register names are interned to dense ids at parse time; the interpreter
  indexes a flat register file instead of hashing a name per operand.
- Per-instruction scratch is no longer zero-initialized. `Lanes r{}` was a
  256-byte memset on *every* instruction -- about 6 GB of pointless memset in
  the ALU benchmark, and the single largest win found.
- Hot ALU paths choose the operation and width once per warp instead of
  per lane, and iterate only active lanes (`for_active`).
- Link-time optimization for Release builds.

Measured now: vectorAdd 2M elements ~145 ms; ~1.05G lane-ops of dense FMA
~0.85 s. Profiling says the remaining time is in the per-lane loops
themselves, which is where it should be for an interpreter.

Narrower lane storage was implemented and measured: registers are now split
into a 32-bit and a 64-bit file by declared width, with native 32-bit paths
for the hot integer, float and fma cases. Measured A/B against the previous
single 64-bit file:

| benchmark | wide | narrow |
| --- | --- | --- |
| vectorAdd 2M (memory-bound) | ~147 ms | ~148 ms |
| dense f32 FMA | ~0.86 s | ~0.83 s |
| 120 live registers | ~0.90 s | ~0.88 s |

So about 3% on register-heavy kernels and nothing on memory-bound ones --
far less than hoped. The reason is that the register file was already small
enough to sit in L1 (30 registers x 256 B is under 8 KB), so halving it does
not remove a bottleneck that was not there. It is kept because it is correct,
it halves per-warp register memory (which will matter as resident warp counts
grow), and a 32-bit register file with 64-bit values in pairs is what the
hardware actually does.

Next, in order of expected payoff:
1. **PTX -> internal IR -> LLVM JIT** (ARCHITECTURE.md D1). Profiling now puts
   the time in per-lane interpretation itself, which is exactly what a JIT
   removes. This is the real answer; further interpreter micro-optimization
   has hit diminishing returns.
2. Block-level parallelism across host threads, behind the scheduler
   abstraction so determinism is preserved.

A GPU still retires ~10^13 ops/s, so saturation-style stress tests are run at
reduced intensity via their own CLI knobs; see
scripts/run-pantheon-workloads.sh.

## Next milestones (order)

1. **Interpreter speed**: intern register names to dense indices at parse
   time (see Performance above) — the single biggest win available without
   the JIT.
2. **Scheduler: random mode** (seeded) + first differential scheduling tests,
   then the adversarial mode that makes VirtualGPU a race detector.
3. **Characterization harness v0**: run the same micro-tests on a physical
   GPU (bench/ rents them) and on virtual profiles, diff, and start flipping
   `verified` bits in the profiles.
4. **Static cudart hosting**: satisfy NVIDIA's undocumented driver export
   tables (cuGetExportTable dark API) so binaries built with the *default*
   (static) cudart also run without a `-cudart shared` rebuild. Partial
   groundwork exists in the driver shim; deferred as brittle/version-specific.
5. **More PTX as workloads demand it**: bf16, cp.async, mma.sync, textures.
