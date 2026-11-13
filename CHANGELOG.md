# Changelog

PantheonSim is versioned by the `project()` line in `CMakeLists.txt`; `vgpu --version` prints it.
Each release is tagged `vX.Y.Z` and described on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases). It simulates GPUs functionally, so a
release says what runs and what it produces, never how fast a real GPU would run it.

## Unreleased

- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: while the card holds memory the simulator does not (the launch before ran
  there), a launch of a kernel with an estimate is weighed with what each side really moves: keeping it in the simulator costs its time plus
  pulling back every chunk only the card holds, sending it costs the launch, the kernel's time on the card and the push of what only the
  simulator holds. Before, a short launch between two card launches stayed in the simulator for its own time alone, pulled the whole live memory
  back, and the next card launch pushed it all again. Measured on one A10G (Round sp65): PolyBench LU 11.6 s to 1.5 s in turbo mode and with
  `VGPU_HOST_GPU_DIV=1`, Rodinia gaussian 0.99 s to 0.53 s, GRAMSCHM's traffic from 98 GB to 40 GB each way; outputs identical.
  `VGPU_HOST_GPU_STREAK=0` keeps the old rule; the summary's `host_gpu_streak:` line counts the launches sent this way
  ([design](docs/perf-design/12-host-gpu-turbo.md), section 9). A launch with a grid or block dimension of 0 now stays in the simulator, which
  reports its own configuration error, instead of failing on the card (it ended the mode with an internal error when the card held memory).
- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: a launch on fewer than 256 blocks, which the sample of whole blocks could not
  judge, is now sampled *inside* its blocks when its kernel has a counted loop: the forked copy runs each block until its warps stand in the
  loop, reads how far each lane's induction register is from the bound its compare reads, and times one turn of the loop on one warp; the
  launch's simulator time is the turns left at that cost. Measured on one A10G (Round sp61): PolyBench COVAR 64.7 s to 41.8 s and CORR 64.8 s to
  42.0 s in both modes (their long kernel, 24.5 s in the simulator, takes 1.67 s on the card path), outputs identical. Grids of 256 blocks or more
  are sampled as before; `VGPU_HOST_GPU_SAMPLE=whole` keeps the rule of Round sp60b ([design](docs/perf-design/12-host-gpu-turbo.md), section 8).
- Host-GPU mode (experimental, opt-in), both `exact` and `turbo`: a launch of a kernel with a loop that the mode knows nothing of yet is now *sampled*
  before the simulator is left to run it. The process is forked, the copy runs a few of the launch's blocks (up to 32, a few milliseconds), reports
  their times and ends; if they say that the launch repays the card path and the start of the real driver, the card runs the launch, otherwise
  the simulator does. Before, such a launch went to the card only on a static count of the kernel's instructions, which is low by the trip
  count of its loops, so programs that launch each kernel once kept their one long launch in the simulator. `VGPU_HOST_GPU_SAMPLE=0` switches
  the sampling off; `=log` reports what it would do with no card. Measured on one A10G (Round sp60b): PolyBench SYRK 5.16 s to 1.30 s, SYR2K
  9.81 s to 2.37 s, 2MM 172 s to 122 s in both modes, HeCBench black-scholes 25.1 s to 6.1 s in turbo; geometric mean over 61 programs from 1.12x
  to 1.18x (`exact`) and from 1.14x to 1.27x (`turbo`) against the simulator, outputs identical. Kernels launched on a few large blocks (PolyBench
  COVAR, CORR) are not reached ([design](docs/perf-design/12-host-gpu-turbo.md), section 8; pantheonsim #587).
- Host-GPU mode (experimental, opt-in): `VGPU_HOST_GPU=turbo` also runs on the machine's real GPU the kernels whose result is the
  card's own and not the simulator's bits: the approximate functions (`MUFU`), packed half arithmetic, float reductions, calls inside the
  kernel, tensor-core instructions and integer atomics that return a value; float atomics with a returned value, compare-and-swap and
  exchange only with `VGPU_HOST_GPU_ATOMICS=1`; textures and unnamed instructions never. `VGPU_HOST_GPU_TURBO_OFF` switches classes
  off. With `VGPU_HOST_GPU_VERIFY` the card's memory is compared with the simulator's by tolerance (`VGPU_HOST_GPU_ATOL`,
  `VGPU_HOST_GPU_RTOL`, `VGPU_HOST_GPU_TOL_TYPE`) instead of byte for byte, and the exit summary has lines for scripts (`host_gpu_turbo:`, `host_gpu_refusals:`,
  `host_gpu_outside:`). A launch that does not come back from the card ends the process after `VGPU_HOST_GPU_TIMEOUT` seconds (600 in
  turbo mode). `exact` mode is unchanged, except for two bugs: a kernel refused for its module is no longer reported as "not in the
  exact set" from its second launch on, and a float reduction (`RED.ADD.F32`, which adds in arrival order) is now refused on sm_75 to
  sm_89 as it was meant to be (the check did not recognise the decoder's spelling of the type). Measured on one A10G (Round sp59): geometric mean 1.23x over 59 programs
  of PolyBench/GPU, Rodinia and HeCBench against 1.12x for `exact`, outputs identical. `VGPU_HOST_GPU_VERIFY` now gives the card copy up before the
  simulator runs the launch, in both modes: the simulator's atomics could lose updates under verify, and the card ran on its own earlier results.
  [Design and limits](docs/perf-design/12-host-gpu-turbo.md); the benchmark harness is
  `tools/bench/hostgpu/run.py`, the card probes `tools/probes/hostgpu_turbo_card_probes.sh`. `docs/cli.md` now lists every
  `VGPU_HOST_GPU_*` variable (pantheonsim #587).
- Tier 2 (`VGPU_JIT=1`, still off by default), the cold run ([round sp63](docs/perf-design/rounds/11-rounds-sp36-sp41.md)): a section the
  share rule keeps waits for its cache lookup on the plain run loop too, so short programs no longer pay for the counting loop before
  the lookup (ADI, asmooth, sortKV, atan2, adam lost 1 to 9 % with `VGPU_JIT=1`; now within 0 to 2 %). The regions chosen for a section
  are recorded before its compile, so a compile that the end of the program cancels (silently now) is started at the next run's lookup
  instead of at the threshold (`VGPU_JIT_HISTORY=0` turns that off): asyncAllocation, shorter than its compile, is warm from its third run
  instead of never at one thread (at sixteen threads such compiles still never finish). The cache directory is named by the library's GNU build id, other builds' directories untouched for 14 days are
  removed, and the cache is capped (`VGPU_JIT_CACHE_MAX_MB`, default 256, least recently loaded objects first). The notice about a missing
  compiler or an unusable cache is given at most once a day per cache directory, and `VGPU_JIT=auto` turns the JIT on without those notices.
  Results and counters are unchanged.
- Setup action: a release carries the simulator prebuilt for linux-x86_64, one tarball and `.sha256` per runner and toolkit the
  action sets up (ubuntu-24.04 with CUDA 12.0, 12.6, 13.0 or none, ubuntu-22.04 with CUDA 12.6 or none), built and attached by
  `.github/workflows/release-binaries.yml` when a `v*` tag is pushed. The action, pinned at that release, downloads the tarball,
  checks its sha256 and unpacks it instead of building the simulator (8 to 24 minutes on a cold cache), and falls back to the cache
  and the source build when there is no tarball for it or the download or checksum fails. New input `prebuilt` (`false` always
  builds), outputs `prebuilt` and `prebuilt-asset`; `docs/github-action.md` gives the measured setup times (pantheonsim #599).
- SASS engine on hosts with AVX-512 (F/BW/DQ/VL and VNNI): binary64 add, multiply and fused multiply-add (`DADD`, `DMUL`, `DFMA`) run
  sixteen lanes to a step instead of four. Programs that compute in `double` run 3 to 39 % faster on a Zen 4 host (eight of nine
  measured gain 5 % or more: lavaMD 1.18x, RSBench 1.39x, LULESH 1.07x); single-precision programs and hosts without AVX-512 are
  unchanged, the results are the same bits, and `VGPU_SIMD=avx2` keeps the 256-bit kernel. New unit test `test_lanes512` (skipped on
  a host without AVX-512).
- Tier 2 (`VGPU_JIT=1`, still off by default): a code section the cache has nothing for no longer runs in the counting loop all the
  way to its compile threshold. It waits on the plain run loop, where only the warp instructions of each call are added up, and is
  counted region by region again for the last `VGPU_JIT_PROFILE` (4,000,000) instructions before the threshold. Programs that never
  reach the threshold paid 2 to 16 % for the counting with `VGPU_JIT=1`; results and counters are unchanged
  ([round sp57](docs/perf-design/rounds/11-rounds-sp36-sp41.md), which also says why the JIT stays opt-in).
- CI and setup action: when the apt mirror host of a GitHub-hosted runner image (azure.archive.ubuntu.com) does not resolve, the
  toolchain install points apt at the first of archive.ubuntu.com and four mirrors outside ubuntu.com whose name does resolve,
  instead of failing three times with exit 100 (`scripts/apt-mirror-fallback.sh`, run before every hosted-runner `apt-get update`;
  it changes nothing when the mirror resolves, and says so when no fallback resolves either).
- AMD: a scalar load whose destination is VCC (`s_load_dwordx2 vcc, s[94:95], 0x0`, which LLVM writes when it runs short of scalar
  registers) fills VCC. It wrote the loaded pair to s0 and s1, which on gfx90a hold the private segment buffer, so the kernel's next
  stack access stopped with "buffer_load_dword through a buffer that adds the lane's id" (Kokkos kernels such as OPALX's
  MultipoleT on the simulated MI250X, pantheonsim #579). Every SMEM destination that is not an SGPR (vcc_lo, vcc_hi, the pair,
  the trap temporaries) is written where it lives.
- AMD: a scratch access sums its scalar offset and its instruction offset in 32 bits, as the ISA gives the address
  (`flat_scratch + swizzle(sgpr_offset + inst.offset, threadID)`): `scratch_store_dwordx4 off, v[0:3], s0 offset:32` with s0 = -32
  is the work-item's byte 0, where it reached for 2^32 and stopped with "past the 1136 bytes a work-item reserved" (pantheonsim #578).
- AMD: a packed float instruction (`v_pk_add_f32`, `v_pk_mul_f32`, `v_pk_fma_f32`) that reads the constant 0 reads zero in both halves;
  `v_pk_add_f32 v[2:3], v[2:3], 0 neg_lo:[1,1] neg_hi:[1,1]`, a float negate, stopped with "takes the second part of a constant" and left
  mwa_hyperbeam's single-precision `fee_kernel` output at zero (pantheonsim #580). A nonzero constant's second half stays refused.

- CI: a heavy job (SASS, ollama, ollama-amd, ASan, TSan) takes a reservation before it samples the memory, not after. `ci/jobs.sh --check`
  now takes a share (the profile's minimum) in a ledger under `/tmp/pantheonsim-ci-reserve`, which the pool's containers bind-mount, waits
  while the shares leave no room (capacity 5 GiB: one heavy job per host, `CI_RESERVE_CAPACITY_GIB` raises it), lets the machine settle
  5 s and only then samples. A new last step, `ci/jobs.sh --release` (`if: always()`), gives the share back; one a killed runner never
  released expires after 4 hours. Before, two jobs on the two runners of one host both passed on the same free memory and the host's
  out-of-memory killer took them down (pantheonsim #575, #576). The ollama downloads retry once when the archive arrives partial, and
  the clean-up step no longer fails on a step-summary file the runner has already removed. The test is `lint_ci_jobs` (the ledger) and
  `lint_ci_reservation` (every job releases and mounts it).
- Setup action: the build cache is saved by an `actions/cache/save` step right after the build (on a miss only) instead of by
  `actions/cache`'s post step, which runs only when the whole job succeeds; a caller whose tests failed no longer throws the build away
  and rebuilds it for about 15 minutes next run (pantheonsim #582).
- CUDA runtime and driver: a kernel's `maxThreadsPerBlock` (`cudaFuncGetAttributes`, `CU_FUNC_ATTRIBUTE_MAX_THREADS_PER_BLOCK`)
  counts whole warps per sub-partition, as the launch check does. A kernel of 104 registers a thread reported 608 threads and a
  launch of 608 was refused ("cannot place a single block ... limited by registers"); it reports 512, as an RTX 5080 does for
  the same kernel.
- PTX: `ld.weak` and `st.weak` are accepted (the explicit spelling of the default memory semantics, PTX ISA 6.0 and later). They
  stopped GiMMiK 4.0's matrix-multiply kernels, and so PyFR's CUDA backend, at module load (pantheonsim #554).
- NVRTC (the built-in compiler, used when no toolkit is visible): `--ftz=`, `--prec-div=` and `--prec-sqrt=` are accepted beside
  `--fmad=`, as the real NVRTC takes them; PyFR passes `--ftz=true` to every compile (pantheonsim #555).
- CUDA driver: `cuModuleLoadDataEx` fills `CU_JIT_ERROR_LOG_BUFFER` (and the size slot) when a load fails, and malformed PTX is
  `CUDA_ERROR_INVALID_PTX` rather than `CUDA_ERROR_NOT_SUPPORTED`; a `.version` past 9.3 is `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`, and
  text that does not begin with `.version` is `CUDA_ERROR_INVALID_IMAGE`. `cuLinkAddData` checks its PTX as it is added and fails
  the same three ways, with the log in the link state's error buffer, instead of `cuLinkComplete` succeeding on garbage. All
  measured on an RTX 3060 (gocudrv's `TestRealModuleJITLog` and `TestRealLinkerBadPTX`, pantheonsim #558).
- PTX: a `.reg` range may declare up to 1,048,576 registers (it was 65,536). ptxas allocates physical registers and counts only what
  a kernel uses, so nvcc's unrolled BN254 field arithmetic declares 125,511 and loads on a GPU; the per-block limit on registers a
  kernel uses is unchanged (pantheonsim #559).
- PTX: lanes of one group that sit at two copies of a `vote.sync` naming the whole group (nvcc tail-duplicates a ballot into both arms
  of an `if`) now vote together, as they do on a card, where each lane gets its group's ballot. The simulator ran the first copy
  for its lanes alone, and the `bar.sync` after it stopped with "cannot be reached by every lane of the warp", as in fastMD's
  neighbour-list build (pantheonsim #574). Measured on an RTX 3060 (sm_86, CUDA 13.0).

- SASS: a loop that spins with `__nanosleep` as its back-off (lanes of one warp racing for a slot with `atomicCAS`, the losers
  sleeping until the winner publishes) runs to the end. ptxas puts no YIELD in such a loop, so the lane that won stood at the
  BSYNC closing it and the spinners ran until the exec limit; a NANOSLEEP now lets the other lanes and warps go first, as a YIELD
  does, and the spin breaker applies (eliphatfs/cuacd's mesh_audit edge table, pantheonsim #553).
- `VGPU_HOST_STATS=1` prints one line of host-kernel counters when a simulated program exits (CPU user and system time, page faults,
  context switches, threads, huge-page use, heap-chunk `madvise` and `mmap` counts, worker-pool threads and calls, kernel, CPU count and
  THP mode); `VGPU_HOST_STATS=30` also prints one every 30 seconds, flushed, and `VGPU_HOST_STATS_FILE` redirects them. For finding why a
  run is slower on one kernel than another (pantheonsim #551).
- CI: the workloads summary and the "Pantheon workloads failing on main" issue name the machines whose job wrote no results (a hosted
  runner lost mid-build leaves no report, and the count alone did not say which) (pantheonsim #545).
- PTX and SASS `min`/`max`: the SASS tier could not decode `HMNMX2`, the instruction behind `min`/`max` on `f16`, `f16x2`, `bf16` and
  `bf16x2` (`__hmin2`, `__hmax2`, `min.NaN.f16`, ...), so a kernel using one stopped with an illegal-memory-access error; it now runs
  them, with `.ftz`, `.NaN` and `.xorsign.abs` as the PTX engine and an RTX 3060 give. `FMNMX.XORSIGN` (`min.xorsign.abs.f32`) ignored
  its sign rule on SASS and returned an unsigned magnitude. The PTX engine accepts the three-input `min`/`max{.ftz}{.NaN}{.abs}.f32` of
  sm_100 that ptxas emits (it was an unsupported-PTX error). `max.s32.relu`, `__viaddmax_s32_relu` and the other `.relu` forms of
  pantheonsim #547 already ran since #451; new e2e test `e2e_minmax_forms` sweeps every sibling form (the relu and DPX ones are in
  `e2e_dpx_intrinsics`).
- SASS engine, packed half arithmetic: with `HMNMX2` decoded, modules such as one holding `min.f16` no longer fall back to the PTX engine
  wholesale, which exposed `HADD2`, `HMUL2` and `HFMA2` forms the SASS engine got wrong against an RTX 3060 (51 of 380 forms of
  `e2e_ptx_half_forms`, and 4 of `e2e_ptx_forms`): `.FTZ` on an f16 result (subnormal inputs and result to a signed zero), `.SAT` (a -0 result is +0), `.RELU`
  (a negative or -0 result is +0), `F2FP` with a directed rounding mode (`cvt.rz`, `.rm`, `.rp` to f16 and bf16) and its one-conversion
  `MERGE_C` form (`cvt.rn.relu.bf16.f32`, which was read as an f16 pair) and `.SATFINITE` (`cvt.rn.satfinite.f16x2.f32`, whose
  flag was read as part of the type). They now give the card's bits, on the interpreter and
  the tier (the tier leaves those forms to the interpreter). `e2e_ptx_half_forms` runs on the SASS interpreter, the tier and the PTX engine
  and fails if the SASS did not run.
- cuFFT: an in-place real transform (`cufftExecR2C`, `D2Z`, `C2R`, `Z2D` on one buffer, no advanced layout) reads and writes the
  padded real array cuFFT documents (each row of the fastest axis is `2 * (n / 2 + 1)` reals, a batch is the complex distance
  apart). It read the array as dense, so 2D and 3D results were wrong and so was a 1D batch (pantheonsim #548).
- cuBLAS and cuBLASLt: `CUBLAS_TF32_TENSOR_OP_MATH`, `CUBLAS_COMPUTE_32F_FAST_TF32` and the FP16/BF16 fast compute types round the
  FP32 inputs of the host-computed GEMMs (A and B, not C) as a tensor core does: TF32 to nearest with ties away from zero
  (measured on an RTX 3060), FP16 and BF16 to nearest even, for `Sgemm`, `SgemmEx`, `Cgemm`, the batched forms, `GemmEx` and
  `cublasLtMatmul`, from the size the card starts using tensor cores; `NVIDIA_TF32_OVERRIDE=0` is honoured. Before, the result was the plain
  FP32 one in every mode (pantheonsim #549; the rules and what is modelled are in `nvidia/docs/cublas.md`).
- PTX engine, `wmma`: a fragment holds its matrix elements where a card puts them, for every shape and type (f16, bf16, tf32, s8/u8,
  s4/u4, b1, f64 operands; f16, f32, s32 and f64 accumulators). It used a layout of its own, so code that reads or writes
  fragment elements one by one (`acc.x[i] *= (a.x[i] > 0)`, how tiny-cuda-nn masks a ReLU's gradient, and the epilogues of CUTLASS and
  hand-written kernels) met the wrong element: tiny-cuda-nn's `FullyFusedMLP` double-backward test gave mean relative errors of 0.6
  to 0.8 for three of its five rows on the default SASS engine too, because a module holding one SASS instruction the engine did not decode (the packed half
  maximum, `HMNMX2`, until pantheonsim #547) ran on the PTX engine wholesale. The three code paths that held old layouts (f16 and bf16 `m16n16k16` with an f32 accumulator, tf32, and the generic
  one) are one now, and the tile route takes tf32 and the f16/bf16 `m16n16k16` forms with f16 accumulators too. The layout is the
  Turing-and-later one, measured on an RTX 3060 (a Volta card, which was not measured, may differ)
  (pantheonsim #550, new test `e2e_wmma_fragment_layout`).
- PTX engine: a `wmma.load` whose registers nothing reads is dead code and does nothing, as on a card where ptxas removes it. tiny-cuda-nn's
  fused backward pass loads a fragment of its last layer from a null pointer and uses none of it, which stopped the program with an
  invalid-pointer error at address 0 on the simulator only (pantheonsim #550, in `test_exec3`).
- CUDA runtime and driver: a zero-byte `cudaMalloc`, `cudaMallocManaged`, `cudaMallocHost`, `cudaHostAlloc` or `cudaMallocAsync` succeeds
  and gives a NULL pointer, as on a GPU, where it gave a distinct non-NULL one (Gpufit tests its unused buffers against NULL); the
  driver's `cuMemHostAlloc`/`cuMemAllocHost` of 0 bytes do the same, and `cuMemAlloc(0)` is `CUDA_ERROR_INVALID_VALUE` with the
  result cleared (pantheonsim #515).
- CUDA runtime and driver: `cudaMemGetInfo` and `cuMemGetInfo` accept a NULL for the free or the total pointer, or both, and fill
  the other, as on a GPU. They used to answer invalid value, which stopped cuda-api-wrappers' device_management example
  (pantheonsim #519).
- CUDA driver: calls that work with no current context on a GPU now do. A stream-ordered call that names a stream of its own
  (`cuMemAllocAsync`, `cuMemAllocFromPoolAsync`, `cuMemFreeAsync`, the `Async` copies and fills including `cuMemcpy3DAsync`,
  `cuStreamSynchronize`, `cuStreamQuery`) takes its context from the stream; `cuMemFree` and `cuMemFreeHost` use the context that
  owns the pointer; `cuMemAddressReserve` and `cuMemAddressFree` need none. The legacy, per-thread and NULL streams and the
  synchronous copies still answer `CUDA_ERROR_INVALID_CONTEXT`. This is how cuda-api-wrappers drives the API (retain the primary
  context, push it for one call, pop it), and five of its examples stopped on it (pantheonsim #518).
- CUDA driver: `cuPointerGetAttribute(CU_POINTER_ATTRIBUTE_CONTEXT)` (and `cuPointerGetAttributes`) returns the context a
  `cuMemAlloc`, `cuMemAllocPitch`, `cuMemAllocManaged` or `cuMemHostAlloc` allocation was made in, not the one current on the
  calling thread (pantheonsim #521).
- CUDA driver: `cuModuleGetFunction` returns the same `CUfunction` for the same module and name every time, and it is the handle
  `cuModuleEnumerateFunctions` lists, so programs that compare handles (cuda-api-wrappers' module_management example) find them
  equal (pantheonsim #520).
- CUDA runtime: `cudaLaunchHostFunc` (and `cudaLaunchHostFunc_v2`) enqueues the function and returns; it runs on a worker thread for
  the stream, after the stream's earlier host work and before its later work. An event recorded behind it, and `cudaStreamQuery`, report
  `cudaErrorNotReady` until it has finished, as does `cudaEventElapsedTime`; `cudaStreamSynchronize`, `cudaEventSynchronize` and
  `cudaDeviceSynchronize` wait for it; `cudaStreamWaitEvent` on a pending event holds the other stream back; the legacy default stream
  orders against blocking streams. A host function that waits for something the caller does after enqueueing it no longer hangs the
  call (pantheonsim #487). Copies and fills still run to completion inside their calls; kernel launches are the next entry;
  the driver API's `cuLaunchHostFunc` is the entry after that. Programs that never queue a host function see no change. `cudaErrorNotReady` from a query is no longer left as the last error (`cudaGetLastError`), as on a card.
- CUDA driver: `cuLaunchHostFunc` (and `cuLaunchHostFunc_v2`) enqueues the function and returns, as the runtime's does and as a GPU does
  (pantheonsim #487, the driver half). It runs on the stream's worker thread, in order with the stream's other host functions, and what
  the stream is given afterwards waits for it: `cuLaunchKernel` and its variants, every `cuMemcpy*` and `cuMemset*` form (the
  synchronous ones on the legacy stream wait for blocking streams' host functions only), stream memory operations and
  `cuStreamAddCallback`. `cuStreamQuery` is `CUDA_ERROR_NOT_READY` while one is pending, `cuStreamSynchronize` and `cuCtxSynchronize` wait
  for it, and a driver event recorded behind it is `CUDA_ERROR_NOT_READY` from `cuEventQuery` and `cuEventElapsedTime` until it has finished
  (`cuEventSynchronize` waits, `cuEventElapsedTime` measures to the moment it finished), `cuStreamWaitEvent` holds the other stream back.
  A host function that waits for something the caller does after queueing it no longer hangs the call. Programs that never queue a host
  function see no change. Not covered: stream-ordered allocation and free, prefetch and attach, external semaphores and `cuMemMapArrayAsync`
  do not wait for earlier host functions, and `cuStreamAddCallback` still runs its callback in the call (after the earlier host
  functions) instead of later.
- CUDA runtime: a kernel launched with `cudaLaunchKernel` (`<<<>>>`) on a stream made with `cudaStreamNonBlocking` returns before the
  kernel has run, so `cudaEventQuery` and `cudaStreamQuery` on work behind it report `cudaErrorNotReady`, and so does
  `cudaEventElapsedTime`, until it has finished (pantheonsim #523; the issue's repro now prints 600, 600, then 0 after the
  synchronize, as an RTX 3060 does). The kernel runs on the stream's worker thread under the API lock, so every call that takes the
  lock (copies, frees, allocations, the driver API) waits for it as it waited inside the launch, and a copy or launch on the same
  stream, `cudaStreamSynchronize`, `cudaEventSynchronize`, `cudaDeviceSynchronize` and `cudaStreamWaitEvent` order after it. Results,
  counters, `VGPU_KERNEL_DIGEST`, error text and sticky faults are the same as before; a launch the device refuses for its
  configuration is still refused in the launch. A kernel that ends in a failure that is not a sticky fault (ran past
  `VGPU_MAX_STEPS`) reports it at the launching thread's next call instead of at the launch. It is not used on the default streams,
  with a profiler or `VGPU_TRACE`, `VGPU_RACE`, `VGPU_MAX_STEPS` or `VGPU_STRICT`, for cooperative, cluster and graph launches, while
  a managed allocation exists, or while a CUDA library shim (cuBLAS, cuDNN, cuFFT, ...) is loaded, whose math runs on the calling
  thread with no lock; those launches run inside the call as before. `VGPU_ASYNC_LAUNCH=all` adds streams made with `cudaStreamCreate`;
  `VGPU_ASYNC_LAUNCH=0` turns it off. Design and limits: docs/perf-design/streams.md.
- PTX: `mov.u32%r1, %laneid;` (no space between the type and the operand, which nvcc emits for inline asm written `"mov.u32%0, ..."`) is
  accepted, as ptxas does; a `%` only ever opens a register name, so it now ends the opcode word (pantheonsim #522).
- PTX: `elect.sync _|%p, mask;` and `elect.sync %r|_, mask;` accept the sink `_` for either result, as ptxas does (cubecl's
  `plane_elect`) (pantheonsim #503).
- PTX: `setp` and `set` on `.bf16` (and `.bf16x2`) compare the values, not the raw bits: the sign counts, `-0 == +0`, and NaN is
  unordered, so `setp.nan.bf16`, `setp.neu.bf16` and the ordered forms give the answers they give on `.f16` (pantheonsim #502).
- PTX: `shfl.sync`, `vote.sync`, `match.sync`, `redux.sync` and `bar.warp.sync` wait for every lane that any running lane's member mask
  names, not just the first lane's. With two half-warp tiles, the tile that did not contain the first lane went on while a member of
  its own tile was still in a long loop and read 0 for it (HierarchicalKV's `tlp_v2_upsert*`). Only the lanes whose members are away
  wait; the rest of the path runs on (pantheonsim #488).
- libcudart exports the nine `cudaGL*` interop functions the CUDA 13.0 runtime does (`cudaGLRegisterBufferObject`, `cudaGLMapBufferObject`,
  `cudaGLSetGLDevice`, `cudaGLGetDevices`, ...). A program that only names one of them, such as COLMAP's SiftGPU, failed to start when linked with
  `-z now` ("undefined symbol: cudaGLRegisterBufferObject"). With no OpenGL context on a simulated GPU they answer what an RTX 3060 answers with none:
  `cudaErrorInvalidGraphicsContext` for register, unregister and `cudaGLGetDevices`, `cudaErrorInvalidValue` for map, unmap and the map flags.
- `VGPU_MEMO=1` (opt-in, off by default and then unchanged): launch-level result caching. A launch whose kernel, grid, arguments,
  environment and device-memory history have been seen before is not run; its stored write set, counters and kernel digest are applied
  (`VGPU_MEMO_DIR` keeps entries between runs, so a second run of a deterministic program hits from its first launch;
  `VGPU_MEMO_VERIFY` runs every hit for real as well and compares). The device's memory is named by a hash chain over what the
  program did to it, not by its content, and anything the cache cannot describe (virtual memory management, IPC, pinned or managed
  memory a kernel can reach, device malloc and printf, cooperative and cluster launches, textures, atomics on several threads, a
  profiler, the race detector, an armed fault, ...) restarts the chain, so the cost on a miss is a budgeted before-and-after hash of
  the allocations the launch can write and the answer is never a stale one. Knobs: [docs/cli.md](docs/cli.md); rules, tests, mutants
  and measurements: [docs/perf-design/rounds/10-rounds-sp19-sp27.md](docs/perf-design/rounds/10-rounds-sp19-sp27.md#round-sp35-launch-level-result-caching-built-tested-measured-branch-sp35-memo-base-originmain-9f80d40e).
- The container image builds again. `scripts/package-build.sh` left `shim/libcudss.so.1` pointing into `cudss1/`, a directory the image does not
  have, so the build stopped at "links that lead nowhere" from the cuDSS 1.0 change onward. A lint test (`lint_package_build`) now builds a small
  tree with the same layout.
- SASS: `ULDC` with a uniform base register (`ULDC.64 UR6, c[0x3][UR8+-0x72e0]`) takes its offset as a signed 16-bit number and wraps in the
  64 KiB bank, as `LDC` does. ptxas uses it to reach constant data past byte 32768, and the simulator read zero there on the Ampere and Ada
  profiles (an inverse NTT in FIDESlib came out all zeros) (pantheonsim #499).
- `vgpu shell --rootfs`: opt-in private root. The session gets its own root on a tmpfs (entered with `pivot_root` inside the mount
  namespace the shell already makes): the host's `/usr`, `/bin`, `/sbin`, `/lib*` and the `/etc` entries a loader, resolver, TLS
  and shell need come through read-only; `/root`, `/home`, `/opt`, `/tmp`, `/var/tmp` and `/work` start empty and are discarded on
  exit; `/dev` is minimal; `/proc`, `/sys` and the simulated GPU surfaces are as before; `/usr/local/cuda` and `/opt/rocm` appear
  only where the host has a toolkit. `-v HOST:DEST[:ro]` (repeatable) adds host paths the way `docker run -v` does and `--home DIR`
  keeps `/root`. Where user namespaces are refused it exits `1` rather than fall back to the host's files. Default behaviour is
  unchanged. See [docs/machine-simulator.md](docs/machine-simulator.md#a-private-root---rootfs).
- PTX: `abs.s16` takes the sign from bit 15 of the register. It used to read the whole 32-bit register as s32, so a negative
  value written with zero upper bits (`mov.b16 a, 0xff81`, `cvt.u16.u32`) came back unchanged instead of as `0x007f`.
- PTX accepts unary `~` on an integer constant (`and.b64 %rd3, %rd3, ~0x2;`), alone, nested, with `-` or in
  parentheses, at the width of the instruction; the lexer used to stop with `unexpected character '~'`
  (pantheonsim #483).
- PTX: a negated predicate source (`and.pred %p2, !%p0, %p1;`, also `or`, `xor`, `not.pred` and the predicate of `selp`) is accepted. drjit-core
  emits it for boolean operations, and the module used to fail to load with `expected operand, got '!'`.
- PTX: `ldu.global` (the uniform load, scalar and `.v2`/`.v4`) runs as the `ld.global` it stands for.
- PTX: the array-element address `var[imm]` (`st.local.b32 arr[0], %f0;`, also `[arr[1]]`) is accepted for `.local`, `.shared`,
  `.global` and `.const` arrays. The index counts elements of the declared type; one outside the array is refused.
- PTX: the array-element address with a register index, `st.local.b32 arr[%r15], %f16;` and `ld.local.b32 %f100, arr[%r99];` (a 32- or
  64-bit register; `.local`, `.shared`, `.global` and `.const` arrays), is accepted: the address is the variable's plus the register times
  the element size, per lane, unchecked as on the card. Dr.Jit emits it for every opaque-index array access, and the module used to fail to
  load with `bad integer literal '%r0'` (pantheonsim #530).
- The runtime exports the OpenGL interop functions of `cuda_gl_interop.h` (`cudaGraphicsGLRegisterBuffer`, `cudaGraphicsGLRegisterImage`,
  `cudaGLGetDevices`, `cudaGLSetGLDevice` and the legacy `cudaGL*BufferObject` family), so a program linked with `-z now` that merely
  references one (COLMAP's SiftGPU) starts instead of dying with `undefined symbol: cudaGLRegisterBufferObject`. They answer as an RTX 3060
  with no GL context does: `cudaErrorInvalidGraphicsContext`, `cudaErrorInvalidValue` for a buffer that was never registered
  (pantheonsim #528). `tests/lint/check_header_exports.py` now checks `cuda_gl_interop.h` too.
- PTX: a `.func` can take its parameters and return its value in `.reg`s (`.func (.reg .b32 r) f(.reg .u64 data)`), and a call
  (direct, or through a `.callprototype` with `.reg`s) passes registers or numbers for them.
- libcuda exports the OpenGL, EGL and VDPAU interop functions (`cuGraphicsGLRegisterBuffer`,
  `cuGraphicsGLRegisterImage`, `cuGLGetDevices`, `cuGLCtxCreate`, the legacy `cuGL*BufferObject` family,
  `cuGraphicsEGLRegisterImage`, `cuEGLStream*`, `cuVDPAU*`, `cuGraphicsVDPAURegister*`) and `cuGetProcAddress`
  finds them. They answer `CUDA_ERROR_NOT_SUPPORTED`, since a simulated GPU has no display stack; before, a loader
  such as Dr.Jit that looks the names up when it loads libcuda turned its CUDA backend off.
- `cuLaunchKernel`, `cuLaunchKernelEx` and graph kernel nodes take the `extra` argument form
  (`CU_LAUNCH_PARAM_BUFFER_POINTER`, `CU_LAUNCH_PARAM_BUFFER_SIZE`, `CU_LAUNCH_PARAM_END`) that cudarc and Dr.Jit
  launch with. The buffer holds the kernel's parameters at the offsets `cuFuncGetParamInfo` reports. A list with no
  buffer or size, a size short of the parameters, a repeated or unknown key, no `CU_LAUNCH_PARAM_END`, or both
  `kernelParams` and `extra` is `CUDA_ERROR_INVALID_VALUE`; it used to be `CUDA_ERROR_NOT_SUPPORTED` for every list.
- cuDSS 1.0 runs. The simulator's cuDSS is now presented as `libcudss.so.1` (1.0: its version macros,
  `cudaDataType_t` signatures, `cudssSetJitCacheDir` and the pinned-memory handler functions, the four new
  config parameters, Bunch-Kaufman accepted) with `libcudss.so.0` (0.8) kept beside it, unchanged, so programs
  built against either major run. A program built against NVIDIA's 1.0, such as SCS's `run_tests_cudss`, no
  longer stops with "needs libcudss.so.1, but this build of the simulator has no libcudss.so" (#426).
- A kernel run from PTX no longer fails to launch when its register estimate is up to twice the block's register
  file. The estimate is the peak of live virtual registers and ptxas needs fewer, so ggml's `rms_norm_f32`
  (88 estimated, 64 allowed at 1,024 threads) stopped Ollama running Gemma 3 on the RTX PRO 6000 profile, the one
  profile whose Ollama build has no cubin. The excess is charged as spill; a launch several times over still fails.
- The GitHub Action has a `source-dir` output, the simulator's source tree. A project that has the pantheon
  workloads can run `tests/workloads/run_pantheon_workloads.sh` from its own CI with one `uses:` and no second
  checkout (`docs/pantheon-workloads.md`).
- The pantheon workload runner reports a workload that printed `Skipping: <reason>` as skipped, with the reason,
  not as passed.

## [0.2.0] - 2026-10-24

### Getting started
- `make` installs what is missing (it asks first) and builds; `./run` starts a simulated GPU machine from a
  settings file. The README is those two commands; the rest is in `docs/getting-started.md`, `docs/config.md`
  and `docs/cli.md`. The simulator uses half your CPUs unless you say otherwise.
- `scripts/ci-setup.sh` does for GitLab CI, CircleCI, Forgejo, Jenkins and Buildkite what the GitHub Action does
  (`docs/ci-other-platforms.md`).
- Every document was checked against the code, and the profile counts, command lists, ROCm versions and
  limitations were corrected. 48 profiles (38 NVIDIA, 10 AMD); 14 NVIDIA and 3 AMD are read from physical cards.
- `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, issue templates (bug, something unsupported, a wrong GPU profile) and a
  pull request template say how to send a change and what it needs.

### Faster
- The SASS engine's pre-decoded tier is on by default; the AMD engine has a tier of its own, and matrix and
  packed-math fast paths. Atomics, kernel launch and the first launch of a fatbin cost less. A second tier that
  compiles hot straight-line code (`VGPU_JIT=1`) is available and off by default. Every fast path produces
  the same bits as the path it replaces, and the old path stays available through the documented `VGPU_*`
  switches so the two can be compared (`docs/cli.md`).
- Blocks that wait on one another (grid-wide spins, decoupled look-back) now run together, taking turns when
  they cannot all be resident. This applies to SASS, PTX and AMD code.

### NVIDIA
- New profile: RTX PRO 4500 (read from a card).
- CUPTI traces were compared with the real thing on 19 rented GPUs; `torch.profiler` traces are checked against a
  card in CI. `VGPU_PLATFORM=linux` switches CUPTI to the answers a Linux driver gives (the default is `wsl`,
  `nvidia/docs/cupti.md`).
- SASS: a cloned device function runs its own copy; a yielding warp rotates through its lane groups.
- LuisaCompute's CUDA backend runs (scalar `suld`/`sust`, `txq`, the device runtime library in `cuLinkAddData`);
  cuTensorNet and cuda-samples fixes.

### AMD
- `rocprofv3` traces of small HIP programs match what it recorded on a real RX 6800 and RX 6700 XT.
- gfx942 and gfx950 were swept against AMD's instruction guides; gfx90a images, kernel-to-host calls (hostcall),
  PC sampling, the ROCm Validation Suite and vLLM graphs run (`amd/README.md`).

### Known limits
- Functional only: there is no timing model, so a run on a physical GPU before shipping still counts.
- Programs must link the shared CUDA runtime (`nvcc -cudart shared`); a statically linked runtime needs NVIDIA's
  undocumented driver export tables (this is why CuPy does not work).
- Nsight Systems and Nsight Compute as NVIDIA ships them, and OptiX, are not supported. See `docs/status.md`.

## [0.1.0] - 2026-10-15

First tagged release: simulated NVIDIA and AMD GPUs, so CUDA and HIP programs run unmodified on a machine with
no GPU (development, teaching and CI). GitHub Actions through `pantheongpu/setup-pantheonsim`, a container on
`ghcr.io/pantheongpu/pantheonsim`, or a build from source.
