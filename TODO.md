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
device and emit a profile with `verified: true`. `profiles/nvidia/rtx3060.yaml`
was produced this way and is the only profile whose values come from hardware
rather than documentation; `lspci` resolves its generated PCI id to the correct
device name, and both conformance tests match the physical card exactly.

Run them on any GPU to add support for it:

```bash
nvcc -std=c++14 tools/characterize.cu -o vgpu-characterize -lcuda
{ ./vgpu-characterize 0 | sed '/^telemetry:/,$d'; tools/characterize-telemetry.sh 0; } \
  > profiles/nvidia/<model>.yaml
tests/conformance/run_conformance.sh    # then confirm the semantics match
```

The remaining profiles (A10/A100/H100/H200/B200, MI300X/MI325X/MI350X) are
still documentation-derived placeholders. Characterizing them needs access to
those parts.

## Register and occupancy modeling

PTX declares *virtual* registers, so counting declarations says nothing about
what a thread occupies. `src/ptx/regalloc.cpp` runs liveness over the
instruction stream (extending ranges across loop back edges) and a linear scan
for the peak, counts a 64-bit value as a register pair as the hardware does,
keeps predicates in their own file, and rounds to the allocation granularity.

That count is functional, not decorative:
- a block needing more registers than the device allows fails with
  "too many resources requested for launch", as on hardware;
- `__launch_bounds__` (`.maxntid` / `.reqntid`) is parsed and enforced;
- `cudaFuncGetAttributes` reports real `numRegs` and `localSizeBytes`, and
  `cudaOccupancyMaxActiveBlocksPerMultiprocessor` does the standard occupancy
  calculation instead of returning a placeholder.

Checked against a physical RTX 3060: a simple kernel reports **8 registers and
6 blocks/SM on both**. A register-heavy kernel reports 48 where hardware says
24 -- the estimate is conservative, because ptxas rematerializes and schedules
in ways this analysis does not model. Erring toward "needs more" is the safe
direction for something that refuses launches.

## Ecosystem tools that work today

- **pynvml** and anything built on it (nvitop, gpustat, monitoring agents,
  framework memory queries) — verified against a 4-GPU virtual rack.
- **lspci**, via generated PCI configuration space.
- **nvidia-smi / rocm-smi / rocm_agent_enumerator** — supplied by VirtualGPU
  (see docs/telemetry.md for why the stock nvidia-smi binary cannot be used).

Not yet: cuBLAS/cuDNN/NCCL (so no PyTorch), `nvidia-smi topo -m`, DCGM.

## Known out of scope (not CUDA)

- `rt_virus` needs **OptiX** (NVIDIA's ray-tracing library, loaded from
  libnvoptix.so.1) and `media_enc_virus` needs **NVENC**
  (libnvidia-encode.so.1). Both are separate NVIDIA subsystems, not CUDA;
  emulating them is a distinct project. They fail with the vendor library's
  own error rather than a VirtualGPU error.

## Partially implemented

- Divergence: reconverges at ret only; bar.sync inside divergence = clear
  error. Upgrade: IPDOM reconvergence points.
- cuCtxSetCurrent(NULL) pops rather than clearing a per-thread binding; the
  current-context stack is process-global, not thread-local.
- M7 proper: needs the CUDA *runtime* API shim + fatbin PTX extraction to run
  an unmodified nvcc-built binary (embedded-PTX driver-API apps work today).

## Not implemented (fails loudly, never silently)

- PTX: shared memory (__shared__), warp shuffles/vote/ballot, mul.hi, f16/bf16
  math, half-precision cvt, textures/surfaces, cp.async, tensor-core ops
  (wmma/mma/wgmma), grid sync, inline-asm-only instructions
- Runtime: streams (M8), events (M8), async copies, unified/managed memory,
  virtual memory mgmt API (cuMemAddressReserve…), host-pinned memory
- Frontends: CUDA runtime API (cudart), NVML, cubin/SASS loading,
  cuGetProcAddress dispatch, AMD everything (HIP, ROCm-SMI, CDNA ISA)
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
