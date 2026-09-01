# TODO / status

Updated: 2026-09-01. See ARCHITECTURE.md for the design behind these.

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
- M7 (stand-in): external C11 harness binary against the ABI alone — passes
  discovery/memory/launch/error-path checks on all five profiles

## Partially implemented

- Divergence: reconverges at ret only; bar.sync inside divergence = clear
  error. Upgrade: IPDOM reconvergence points.
- cuCtxSetCurrent(NULL) pops rather than clearing a per-thread binding; the
  current-context stack is process-global, not thread-local.
- M7 proper: needs the CUDA *runtime* API shim + fatbin PTX extraction to run
  an unmodified nvcc-built binary (embedded-PTX driver-API apps work today).

## Not implemented (fails loudly, never silently)

- PTX: shared/local memory, atomics, shuffles/vote/ballot, cvt, mul.hi,
  vector ld/st (v2/v4), f16/bf16, sat/approx/rounding variants, textures,
  cp.async, tensor-core ops (wmma/mma), grid sync
- Runtime: streams (M8), events (M8), async copies, unified/managed memory,
  virtual memory mgmt API (cuMemAddressReserve…), host-pinned memory
- Frontends: CUDA runtime API (cudart), NVML, cubin/SASS loading,
  cuGetProcAddress dispatch, AMD everything (HIP, ROCm-SMI, CDNA ISA)
- Tooling: `vgpu run` (LD_LIBRARY_PATH/LD_PRELOAD wrapper), `vgpu test
  --matrix`, trace record/replay, schedulers random/adversarial, race
  detection, OOM injection, characterization/differential-fuzz harness,
  conformance DB + compat scores

## Next milestones (order)

1. **M8** streams/events (default-stream semantics already hold: everything
   is synchronous; add handles + cross-stream sync semantics)
2. **Shared memory + atomics + shuffles** in PTX/interpreter — unlocks real
   reduction/scan kernels and the first interesting divergence bugs
3. **cudart shim**: `__cudaRegisterFatBinary`-family interception + fatbin
   parsing to pull embedded PTX out of unmodified nvcc binaries; then
   `vgpu run --gpu … ./app` wrapper (true M7)
4. **Scheduler: random mode** (seeded) + first differential scheduling tests
5. **Characterization harness v0**: same micro-tests on a physical GPU
   (bench/ rents them) vs virtual profiles; start flipping `verified` bits
