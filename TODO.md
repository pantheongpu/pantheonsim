# TODO / status

Updated: 2026-09-01 (rev 3). See ARCHITECTURE.md for the design behind these.

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
- Configurable virtual VRAM (VGPU_VRAM_MB); VGPU_TRACE coverage-growth logging.

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

The interpreter retires roughly 2x10^8 instructions/s (measured:
`vgpu demo vectoradd -n 2000000` in ~210 ms). A GPU retires ~10^13 ops/s, so
saturation-style stress tests must be run at reduced intensity via their own
CLI knobs (`--kernel_loops`, `--grid_size`); see
scripts/run-pantheon-workloads.sh. Closing even part of that gap needs the
IR/JIT path in ARCHITECTURE.md D1. The cheapest next win is interning
register names to dense indices at parse time: the interpreter currently
does a string-hash lookup per operand, which profiling points to as the
dominant cost (returning operands by reference instead of by value was
already tried and gained only ~5%, so the copies were already elided).

## Next milestones (order)

1. **M8** streams/events (default-stream semantics already hold: everything
   is synchronous; add handles + cross-stream sync semantics)
2. **Shared memory + atomics + shuffles** in PTX/interpreter — unlocks real
   reduction/scan kernels and the first interesting divergence bugs
3. **Static cudart hosting**: satisfy NVIDIA's undocumented driver export
   tables (cuGetExportTable dark API) so binaries built with the *default*
   (static) cudart also run without a `-cudart shared` rebuild. Partial
   groundwork exists in the driver shim; deferred as brittle/version-specific.
4. **Scheduler: random mode** (seeded) + first differential scheduling tests
5. **Characterization harness v0**: same micro-tests on a physical GPU
   (bench/ rents them) vs virtual profiles; start flipping `verified` bits
