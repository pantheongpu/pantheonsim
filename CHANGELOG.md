# Changelog

PantheonSim is versioned by the `project()` line in `CMakeLists.txt`; `vgpu --version` prints it.
Each release is tagged `vX.Y.Z` and described on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases). It simulates GPUs functionally, so a
release says what runs and what it produces, never how fast a real GPU would run it.

## Unreleased

- Under `vgpu run --preload`, where the driver is loaded privately (#597), CUPTI and NVSHMEM now find it: they looked driver
  functions up in the global scope only, which a private library is not in. CUPTI's callbacks named a placeholder for the current
  context instead of the one the program's driver calls find current, and `nvshmemx_cumodule_init` and `nvshmemx_culibrary_init`
  failed, so a module loaded with the driver API never got its `nvshmemi_device_state_d`. They now ask the runtime's lookup, which
  tries the private driver first (`nvidia/src/driver_symbol.hpp`). So does the runtime's own check for driver-API managed memory
  before deferring an asynchronous copy, which still looked in the global scope only. `e2e_preload_torch_like` runs a third program
  (`preload_torch_peers.cpp`), and `lint_driver_lookups` keeps any shim other than the driver from looking a driver function up in
  the global scope alone.
- `cuProfilerStart`, `cuProfilerStop`, `cudaProfilerStart` and `cudaProfilerStop` answer as an RTX 3060 does (CUDA 13.0); they
  used to succeed always. The driver's return `CUDA_ERROR_NOT_INITIALIZED` before `cuInit`, `CUDA_ERROR_INVALID_CONTEXT` with no
  current context, `CUDA_ERROR_CONTEXT_IS_DESTROYED` when the current context is dead (destroyed, or a primary context released
  to nothing or reset), and a dead context's fault. The runtime's are runtime calls like any other: they start the runtime and make
  the primary context current on the calling thread, and answer a dead context's fault or `cudaErrorNoDevice`. And after
  `cuDevicePrimaryCtxReset`, the runtime's next call makes the primary context active again, as on the card
  (`cuDevicePrimaryCtxGetState` kept reporting it inactive). New test `test_profiler_control`.
- Runtime: every call that fails makes its error the thread's last error, as on an RTX 3060. Only the calls that went through the
  runtime's guard did: `cudaEventElapsedTime` on an event never recorded, `cudaStreamWaitEvent` with an unknown flag,
  `cudaStreamBeginCapture`/`cudaStreamEndCapture` refusals, `cudaMemcpy2D` with a pitch too small, `cudaMemcpy3D` and `cudaMemset3D`
  with bad arguments, a captured event's `cudaErrorCapturedEvent` and others left `cudaGetLastError` at success. A query's not-ready
  still sets nothing, and a call the simulator makes inside another does not count (found by differential testing against an RTX 3060;
  `e2e_card_edges_last_error`).
- Runtime: a copy kind that is none of the five is `cudaErrorInvalidMemcpyDirection` (21) from `cudaMemcpy`, `cudaMemcpyAsync`, the 2D
  copies and `cudaMemcpy3D`, as on an RTX 3060, not `cudaErrorInvalidValue` (`e2e_card_edges_copy_kind`).
- Runtime: 2D pitches are checked as an RTX 3060 checks them. `cudaMemset2D` of more than one row with a pitch narrower than the row is
  `cudaErrorInvalidValue` and writes nothing; it used to fill the overlapping rows and succeed. `cudaMemcpy2D` of a single row accepts a
  pitch of 0 (it was `cudaErrorInvalidPitchValue`), and a set pitch narrower than the row is `cudaErrorInvalidValue` there
  (`e2e_card_edges_pitch`).
- Runtime: `cudaEventCreateWithFlags`, `cudaHostAlloc` and `cudaMallocManaged` refuse flags the card refuses (an unknown bit;
  Interprocess without DisableTiming; managed memory with 0 or both attach flags) with `cudaErrorInvalidValue` and create nothing.
  They took any value (`e2e_card_edges_alloc_flags`).
- Runtime: once a stream capture is invalidated, launches, `cudaMemcpyAsync`, `cudaMemsetAsync`, `cudaEventRecord` and
  `cudaStreamWaitEvent` on its streams fail with `cudaErrorStreamCaptureInvalidated` until `cudaStreamEndCapture`, as on an RTX 3060;
  they were recorded and succeeded. `cudaEventSynchronize` and `cudaEventQuery` of a captured event also invalidate a Relaxed-mode
  capture, and answer `cudaErrorInvalidValue` once it is invalidated; `cudaStreamEndCapture` from a stream that joined the capture
  invalidates it. `e2e_capture_streams` no longer continues a capture after those two calls (`e2e_card_edges_capture`).
- Runtime: `cudaPointerGetAttributes` of an address in the device range that no live allocation covers (a freed pointer, one past
  the end) answers success and `cudaMemoryTypeUnregistered` with device -2, as an RTX 3060 does for any address CUDA does not know. It
  was `cudaErrorInvalidValue`, which also became the last error (`e2e_runtime_conformance`).
- cuBLAS: `cublasSetPointerMode` refuses a mode other than HOST and DEVICE, and `cublasSetMathMode` a value above 31, keeping the old
  setting; both were stored, and an unknown pointer mode made later calls read host scalars as device pointers.
  `cublasGetStatusName`/`cublasGetStatusString` give the card's names and sentences (ARCH_MISMATCH, MAPPING_ERROR, EXECUTION_FAILED
  and LICENSE_ERROR were all "CUBLAS_STATUS_INTERNAL_ERROR", and the string was the name). An empty `cublasSgemm` may pass a null C,
  and `cublasSetMatrix`/`cublasGetMatrix` with a short leading dimension no longer leave a runtime last error
  (`e2e_card_edges_cublas`).
- cuSOLVER: `getrs` refuses an unknown operation, negative sizes and short leading dimensions; it solved with them, and `n = -1`
  aborted the program with `std::length_error`. `potrf`, `potrs`, `syevd` and `gesvd` refuse unknown fill modes and jobs, negative
  sizes and short leading dimensions with LAPACK's info (minus the argument's position), `getrf` with `lda < m` succeeds with info -4,
  and `gesvd` with `m < n` is `CUSOLVER_STATUS_INVALID_VALUE` with info 0 (it was NOT_SUPPORTED). `gesvd` with `jobu = 'O'` or
  `jobvt = 'O'` works, writing the vectors over A; it was refused. `cusolverDnSetAdvOptions` refuses unknown functions and algorithms
  (`e2e_card_edges_cusolver`).
- cuSPARSE: `cusparseCreateCsr/Csc/Coo` refuse an unknown index base and more entries than the matrix has places,
  `cusparseCreateDnMat` an unknown order and a leading dimension below a column (COL) or row (ROW), and `cusparseSpMV` an unknown
  operation, a null alpha or beta, a non-CSR algorithm for a CSR matrix, and a compute type the card does not offer for the data
  (`CUSPARSE_STATUS_NOT_SUPPORTED`, also from `cusparseSpMV_bufferSize`). The `cusparseSetMat*` setters refuse values their enums do
  not have, destroying a null handle or descriptor succeeds, and `cusparseGetErrorName`/`String` give the card's texts
  (`e2e_card_edges_cusparse`).
- cuBLAS: `?gemv` takes what an RTX 3060's cuBLAS 13.0 takes. A negative `m` or `n` is an empty product and a zero
  increment is a stride of zero (`incx = 0` reads `x[0]` for every element; with `incy = 0`, `y[0]` keeps row 0's result):
  the simulator refused them with `CUBLAS_STATUS_INVALID_VALUE`, leaving `y` as it was where the card computes it. An
  operation other than N is a transpose (`CUBLAS_OP_CONJG` is `conj(A)` for `cgemv`/`zgemv` and `A` for the real types; it
  was taken as `A^H`), and `cgemv`/`zgemv` refuse an `lda` below `m`, which they read past. The other level-2 routines
  and the GEMMs (`?gemm`, `cublasHgemm`, `?gemm3m`) print reference BLAS's line for the first bad argument on stdout, as
  the card does (" ** On entry to SGER   parameter number 5 had an illegal value"); `?symv`/`?hemv` refuse silently, and
  `?gemv` prints nothing (found by differential testing; `e2e_card_edges_blas2`).
- Runtime: `cudaMemcpy3D` and `cudaMemcpy3DAsync` between pitched pointers check pitches and `ysize` as an RTX 3060 does. One
  row of one slice accepts a pitch of 0 (it was `cudaErrorInvalidPitchValue`) and refuses a set pitch narrower than the
  row with `cudaErrorInvalidValue`, the 3D form of the one-row `cudaMemcpy2D` fix; more than one slice needs a `ysize` of
  at least the rows (0 included: the simulator took 0 for the rows and copied); a set `ysize` that does not hold the rows
  from `pos.y` is `cudaErrorInvalidValue` (it copied overlapping slices). An extent with a side of 0 copies nothing; the
  simulator copied one row or one slice for a height or depth of 0 (`e2e_card_edges_pitch`).
- cuSOLVER: `potrf` (`cusolverDn?potrf`, `cusolverDnXpotrf`) of a matrix that is not positive definite leaves A as an
  RTX 3060 does: the columns before the failing one factored, NaN on the failing diagonal, and then, from the lower
  triangle, the rest factored with that column taken as 0; from the upper, NaN in every row from the failing one. The
  simulator left A untouched (measured up to n = 32; the card factors larger matrices in blocks, which changes the part
  after the failing column). `getrf` without pivoting multiplies the column below a zero pivot by 0 and goes on, as
  the card does; it skipped the column, leaving it as it was (`e2e_card_edges_cusolver`, `test_host_dense`).
- cuSPARSE: `cusparseSpMV_bufferSize` asks for the workspace the card asks for (11 bytes and up for a CSR matrix with
  entries) and `cusparseSpMV` refuses a null buffer then; it asked for 0 and took null. A null handle, matrix or vector
  is `CUSPARSE_STATUS_INVALID_VALUE` from both (it was NOT_INITIALIZED, or success for the size).
  `cusparseCsr2cscEx2`/`_bufferSize` take `CUSPARSE_CSR2CSC_ALG1` only and refuse a null handle, the size is never 0,
  and a null buffer is refused (`e2e_card_edges_cusparse`).
- Driver: `cuCtxSetCurrent` takes a destroyed context, as an RTX 3060 does (it was `CUDA_ERROR_INVALID_CONTEXT`), and every
  call that needs a context answers `CUDA_ERROR_CONTEXT_IS_DESTROYED` while the current one is dead: destroyed, a primary
  context released to nothing, or reset. `cudaDeviceReset` now resets the driver's primary context as
  `cuDevicePrimaryCtxReset` does, so the driver's calls are `CUDA_ERROR_CONTEXT_IS_DESTROYED` until the runtime's next
  call that uses the device; they succeeded. That call, after `cuCtxSetCurrent(NULL)` or destroying the program's own
  context, makes the primary context current again, as on the card (the thread was left with none). The runtime's
  queries (`cudaGetDeviceCount`, `cudaGetDevice`, `cudaGetDeviceProperties`, `cudaDeviceGetAttribute`,
  `cudaPointerGetAttributes`) no longer make the primary context current or active, as on the card
  (`e2e_card_edges_context`).
- AMD: the SDWA, DPP and VOP3P modifiers and the denormal and IEEE modes around them, audited against the guides and an RX 6800
  (follow-up to pantheonsim #613). The output modifier (`mul:2`, `mul:4`, `div:2`) did nothing on gfx942 and gfx950 only with IEEE mode
  or output denormals on; it now does nothing then on gfx90a, gfx1030 and gfx1100 too (CDNA2 to CDNA4 guides, 6.5, the RDNA3 guide,
  7.2.3.1, and the card, which
  a code object's defaults always hit), and applied, a result it makes denormal is flushed keeping its sign while a zero becomes +0;
  the tier leaves an output modifier to the interpreter there. `v_div_scale_f32` and `v_div_scale_f64` wrote their result unclamped
  and unscaled: the card clamps it (a NaN to 0) and scales it like any VOP3 float result (every target, gfx1250's too). neg_lo and neg_hi on the packed integer
  instructions (`v_pk_add_u16` and the rest) were ignored by the interpreter and the tier; they flip the half's top bit, as the card
  does. gfx90a and gfx1030 take the guides' rule for `v_min`/`v_max` (f16, f32, f64, packed): -0 is below +0, and with IEEE mode a
  signalling NaN gives a NaN, without it the other operand (the CDNA2 guide's pseudocode, and the card); they took the host's fmin and
  fmax; there a denormal operand is a zero where MODE flushes input
  denormals (the result is an operand, which the host's DAZ left as it was). A half's denormals are flushed where `float_denorm_mode_16_64`
  says (the card does; they were always kept). gfx1030's SDWA comparisons (`v_cmp_*_sdwa`) were refused as not implemented. gfx11's
  VOP3 DPP with op_sel read the first source from its own lane, not the shuffled one. On gfx1030 `v_dot2_f32_f16` flushes a denormal
  addend whatever MODE says and keeps denormal halves, as the card does. New test `test_amd_gcn_mods`
  (`amd/tests/data/asm_mods.s`: 50 SDWA and DPP, 36 VOP3P and VOP3 and 24 gfx11 VOP3 DPP instructions over 1024 edge inputs, in three modes,
  gfx90a, gfx942, gfx1030 and gfx1100, tier off and on; the gfx1030 kernels run on the card gave exactly what the test expects).
  test_amd_gcn_same_bits: 96 lines re-recorded, the ones these change (its header lists them). Not fixed: gfx1030's VOP2 16-bit
  results keep the destination's high half on the card (`v_min_f16_e32`), where the executor zeroes it as gfx9 does.
  `amd/tests/hipcc/rdna3.cpp` expected `v_add_f32_e64 ... div:2` to halve in a default-mode (IEEE on) hipcc kernel; it now expects the
  modifier ignored (`rdna3.gfx1100` and `rdna3.w64.gfx1100` rebuilt: host code only, the device code is the same).
- AMD: the clamp bit of the integer instructions saturates their result on every target, as the CDNA3 and CDNA4 guides say (6.5, and
  "Supports saturation" on each add and subtract) and as an RX 6800 does. The VOP3b adds and subtracts with a carry out (`v_add_co_u32`,
  `v_sub_co_u32`, `v_subrev_co_u32`, `v_addc_co_u32`, `v_subb_co_u32`, `v_subbrev_co_u32`, RDNA's `_co_ci_` forms) and `v_mad_u64_u32` /
  `v_mad_i64_i32` ignored it (gfx9's decoder did not read the bit, and neither the interpreter nor the tier applied it), so
  `v_add_co_u32_e64 ... clamp`, LLVM's unsigned saturating add on gfx8, wrapped; clamped, a sum past the type's range is its maximum, a
  difference below zero is zero, and the carry out stays the unclamped one's. `v_mad_u32_u24`, `v_mad_i32_i24`, `v_mul_u32_u24`,
  `v_mul_i32_i24`, `v_sad_u8`, `v_sad_hi_u8`, `v_sad_u16`, `v_msad_u8`, `v_mul_lo_u16` and `v_mad_legacy_u16` ignored it too, and
  `v_subrev_u32`, `v_subrev_u16`, `v_mad_u16`, `v_mad_i16`, `v_mad_u32_u16` and `v_mad_i32_i16` applied it on gfx942 and gfx950 only.
  `v_mad_i64_i32`'s scalar bit is bit 64 of the 65-bit result (the guides' pseudocode, and the card's): the sign of the exact result, not
  whether it overflowed int64, which it was. New test `test_amd_gcn_int_clamp` (gfx90a, gfx942 and gfx1030, tier off and on; its gfx1030
  kernel run on the card gave what the test expects) (follow-up to pantheonsim #613).
- SASS: a lane group the simulator lets another group of its warp past in the middle of a loop (the turn a group gives after many
  backward branches while other lanes wait, which is the simulator's own fairness and not a YIELD) gets its uniform registers and
  predicates back as it left them when it runs again. ptxas keeps a counted loop's index on one side of a divergent branch in a uniform
  register that the other side reuses, because the hardware runs one side to its BSYNC or EXIT unless it meets a YIELD; the other side's
  write landed in the loop. pravirkr/loki's sm_90 EP scorer (`kernel_ep_shift_add_score_c2c`) walked UR4 through a row of shared memory
  while the lanes of invalid leaves loaded a pointer into UR4, and after the 64th trip stopped with "a shared address (0x8ded482c) naming
  block 141 of a cluster of 1"; the PTX path and the card were right. `SR_CgaCtaId` was not involved: it reads 0 in a launch without
  clusters. New unit test `test_sass_uniform_switch`, SASS and PTX (pantheonsim #642).
- Runtime: a copy, fill or kernel launch issued on a stream behind a host function that has not finished returns at once and runs
  after it, in stream order, as on an RTX 3060. It used to run inside the call after waiting for the host function, so a host function
  that waits for something the program does after the call (Celeritas' `AsyncSnapshotTest.device_pending`) hung the program.
  `cudaMemcpyAsync`, `cudaMemsetAsync`, the 2D and 3D copies and fills between pointers, the symbol copies, `cudaMemcpyPeerAsync`,
  `cudaFreeAsync` and kernel launches on any stream the runtime made are queued on the stream's lane when the memory they touch is
  device or page-locked memory; they see what the host function wrote, and events, stream queries and every synchronizing call wait for
  them. Copies of pageable memory, managed memory and arrays still wait in the call. `cudaFree`, `cudaFreeHost`, `cudaHostUnregister`
  and `cudaDeviceReset` wait for queued work first. The 2D, 3D and symbol copies on a stream also follow host functions on that stream
  now (on a non-blocking stream they ran before them), and `cudaMemcpyPeer` and the synchronous symbol copies follow host functions on
  the legacy default stream. New e2e test `e2e_async_host_copies`; [design](docs/perf-design/streams.md) (pantheonsim #634).
- `nvidia-smi nvlink` takes the options of the real subcommand, not only `-s` and `-e`: `-c`/`--capabilities`, `-cBridge`/`--checkBridge`,
  `-gt d|r`, `-i` and `-l` after `nvlink`, several displays at once, and `-h` or no option for nvidia-smi's own usage text. A GPU without NVLink
  prints its name and exit 0 for every display, as the real tool does; on the SXM profiles `-c` prints each link's capabilities as
  `nvmlDeviceGetNvLinkCapability` reports them, `-gt` zero traffic, and `-cBridge` no link lines (the links run through the baseboard, not a
  bridge). The remote-end and setting displays (`-p`, `-R`, `-ec`, `-gLowPwrInfo`, `-gBwMode`, `-gLWidth`, `-info`) are refused on a GPU with
  links, the changes (`-re`, `-sBwMode`, ...) everywhere, and an unknown option prints `Option "X" is not recognized.` and exits 2 as nvidia-smi
  does. jpmorganchase/QOKit runs `nvidia-smi nvlink -cBridge` with `check=True` when its tests are collected (pantheonsim #640).
- AMD: a load into accumulation registers that the wave had not used yet, made for the whole wave at once (`global_load_dwordx2 a[38:39], v2,
  s[6:7] offset:200`, every lane at one address, as a Kokkos kernel loads a pointer out of a structure), fills every register it names. The
  register file grows to the highest accumulation register named, and growing it for the second register moved the first one's row, which
  then went to the freed copy: a38 kept what it held before, and the next load through the pointer stopped with "invalid-pointer ...
  address is not inside any device allocation" at a small address (llnl/exadis on the simulated MI300X, pantheonsim #621). The file now grows
  to the last register before any row is taken (the whole-wave loads of two, three and four dwords and the whole-wave store of three or
  four). It surfaced with host-mapped memory taking the whole-wave path (#507): exadis's data is managed memory, which went lane by lane before.
- cuRAND: `curandGeneratePoisson`, `curandCreatePoissonDistribution`, `curandDestroyDistribution`,
  `curandGeneratePoissonMethod`, `curandGenerateSeeds` and `curandGetProperty` are exported. A program that called
  one stopped with "undefined symbol" (RubixML/numpower's `NumPower::poisson`; pantheonsim #629). The methods are
  the ones an RTX 3060's cuRAND uses for each generator and lambda: Marsaglia-Tsang-Wang's square histogram, built
  with the card's aliases, so `curand_discrete` in a kernel and a Sobol' generator's Poisson integers are the card's;
  the inversion and the rounded normal above the card's cut-offs ([libraries](nvidia/docs/libraries.md); `e2e_rand_poisson`).
- AMD: loading a library's code objects takes 40 % less time. The 62 Tensile code objects of a small rocBLAS program (`amd/tests/rocblas/blas`,
  sgemm, dgemm and the rest, ROCm 7.1 rocBLAS on a simulated MI300X) loaded in 0.44 s instead of 0.75 s and the program ran in 1.61 s
  instead of 1.96 s (median of 8 alternating runs on an ubuntu-24.04 runner; best 1.53 s against 1.84 s). The time was page faults
  and copies, not arithmetic: a kept bundle entry is inflated straight into its place instead of through a buffer that is then
  copied, the zstd buffer is kept from one load to the next, the code object is read where it lies instead of from a copy, `.text`
  and a linked image are written once instead of zeroed first, and the buffers of tens of megabytes are asked for in 2 MB pages
  (`madvise(MADV_HUGEPAGE)`); the process took 190,000 page faults instead of 350,000. The bytes read are the same:
  `test_amd_codeobject_same_bits` gives the recorded digests, and the 2,838 code objects ROCm 7.1 ships (rocBLAS's and hipBLASLt's
  libraries) give the same digests before and after. `VGPU_LOAD_PLAIN=1` reads them as before (`test_amd_codeobject_same_bits_plain`);
  `test_amd_bundle` adds bundles larger than the buffer, read both ways. AMD's `rocblas-test` gains nothing from it: its start-up
  (57 s a process, reading `rocblas_gtest.data`, against 0.1 to 0.4 s of loading) is AMD's client, not the simulator (pantheonsim #565).
- NVRTC without the toolkit's libnvrtc (the shim's nvcc path, or `VGPU_NVRTC=nvcc`) now searches includes as NVRTC does: no host
  directory, the toolkit headers in `__CUDACC_RTC__` mode, the headers given to `nvrtcCreateProgram` and the `-I`/`--include-path`
  directories. A program that brings its own `<cstdint>` (ThrustRTC 0.3.20) or defines `int64_t` itself (PyTorch's jiterator) compiles,
  where nvcc's host headers made it fail with "invalid redeclaration of type name int64_t"; `#include <stdint.h>` and the other host headers
  fail to open, as with NVRTC. Where the toolkit's headers share `/usr/include` with glibc's (Debian's and Ubuntu's
  `nvidia-cuda-toolkit`), the compile sees links to the CUDA headers alone. What NVRTC predefines (`size_t`, `ptrdiff_t`, `clock_t`, `NULL`, `std::move`, `std::forward`,
  `std::initializer_list`, `printf`, `assert`) is there without an include, measured on CUDA 13.0's NVRTC; the PTX matches NVRTC's for the
  programs compared. ThrustRTC's fill and reduce run under `vgpu run` on `nvidia/a100` (ThrustRTC 0.3.20's own occupancy calculator does not
  know compute capability 9.x, so its kernels fail on an H100, real or simulated). The nvcc path also no longer leaves an empty
  `$TMPDIR/vgpu-nvrtc-<pid>` directory behind per process. New test `e2e_nvrtc_paths_nvcc` (pantheonsim #619).
- AMD: `hipModuleGetFunction` no longer prints "the code object has no kernel named ..." when another loaded module has the
  kernel. rocBLAS and hipBLASLt (Tensile) find a GEMM kernel by asking each loaded code object in turn and clear the
  `hipErrorNotFound` of the misses, so every new GEMM kernel printed the message up to once per earlier module, which read as a
  failed GEMM in runs whose GEMMs were right (double and complex-double GEMMs through rocBLAS 7.2 on `amd/mi300x` checked against a
  host reference). The error code and `hipGetLastError` are unchanged, and a kernel that no loaded module has is still reported (#636).
- Driver: `cuGetProcAddress` / `cuGetProcAddress_v2`, and with them `cudaGetDriverEntryPoint` and `...ByVersion`, resolve a
  function's base name by the CUDA version asked for, as an RTX 3060 does: `cuGraphExecUpdate` at 12000 or later is
  `cuGraphExecUpdate_v2` (it was not found, so cuda-core 1.3.0 refused the driver and every numba-cuda launch failed),
  `cuCtxCreate` is `_v3` at 12000 and `_v4` at 13000, `cuStreamGetCaptureInfo` is `_v3` at 13000 (it was `_v2`), and so on for every
  name. The table comes from NVIDIA's typedef headers (`PFN_<name>_v<version>`; `scripts/gen_driver_proc_versions.py`,
  `nvidia/src/driver_proc_versions.inc`), with the per-thread-stream forms for `CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM`. As on
  the card, a name asked for at a version older than the function is `CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT`, and a name that
  is not found returns `CUDA_SUCCESS` with a null function and the status (it returned `CUDA_ERROR_NOT_FOUND`). The card maps
  `cuGraphInstantiate` to `cuGraphInstantiate_v2` at every version from 11.0, not to `cuGraphInstantiateWithFlags`, and so does the
  shim. New unit test `test_driver_proc_address` compares all 3,976 answers the card's driver gave (497 names at 11040, 12000,
  12040 and 13000, both stream modes; `tools/probes/driver_proc_address_probe.cpp`) with the shim's; `lint_driver_proc_versions`
  checks the table against the toolkit at hand (pantheonsim #626).
- Driver: `cuProfilerInitialize` is exported (deprecated, still in the CUDA 13 driver; cudarc 0.19 binds it when it loads libcuda and
  panicked without it). It answers `CUDA_ERROR_NOT_INITIALIZED` before `cuInit` and `CUDA_ERROR_NOT_SUPPORTED` after, whatever the
  arguments and with or without a context, as an RTX 3060 does. Of the other names the CUDA 13.0 driver stub exports, the shim
  now also exports the per-thread forms of the CUDA 11 stream memory-operation and capture calls (`cuStreamWaitValue32_ptsz`,
  `cuStreamBatchMemOp_ptsz`, `cuStreamBeginCapture_ptsz`, `cuStreamGetCaptureInfo_ptsz` and the rest, which the card hands a CUDA
  11 caller that asks for the per-thread default stream) and the unsuffixed `cuMemGetAddressRange`; only the debugger's `cudbg*`
  entry points remain unexported (pantheonsim #627).
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
- Kernels no longer die with SIGFPE when the host program enabled floating-point traps (`feenableexcept`, gfortran
  `-ffpe-trap=invalid,zero,overflow`, which SPECFEM3D's configure puts in `FLAGS_CHECK`). The engines did the device's arithmetic
  under the host's traps, on the launching thread and on the worker threads that copy its environment, so the first inf or NaN a
  kernel made killed the process; an RTX 3060 runs the same program with the traps on. A launch (PTX, SASS and AMD) now runs with
  the traps masked, and the host gets its whole floating-point environment back afterwards, traps, rounding mode and sticky flags,
  none raised by the kernel (`include/vgpu/exec/host_fp.hpp`). New unit tests `test_launch_fp_traps` and `test_amd_gcn_fp_traps`
  (pantheonsim #605).
- Runtime: driver functions taken from `cudaGetDriverEntryPoint` / `cudaGetDriverEntryPointByVersion` by a program that does not
  link libcuda (gputools does this) see what the runtime has done, as on an RTX 3060: the driver is initialised (they returned
  `CUDA_ERROR_NOT_INITIALIZED`), a thread the runtime has used has its primary context current (they returned
  `CUDA_ERROR_INVALID_CONTEXT` from `cuModuleLoadData`), and a thread it has not used has none. Before any runtime call,
  `cudaGetDriverEntryPoint` initialises the driver and makes no context current, as on the card. New e2e test
  `e2e_driver_entry_late` (pantheonsim #604).
- PTX: `ld.param::entry`, `ld.param::func` and `st.param::func`, the explicit `.param` state spaces of PTX ISA 8.3, are accepted as
  `ld.param` / `st.param`; jax 0.12's kernels failed to load ("unrecognized ld/st modifier '.param::entry'"). `st.param::entry`
  stays refused, as ptxas refuses it. Unit test `ld_param_entry_and_func_state_spaces` in `test_exec3` (pantheonsim #589).
- Runtime: `cudaDestroyTextureObject` and `cudaDestroySurfaceObject` of a handle that is no object (0, one never made, one already
  destroyed) return `cudaSuccess` and leave the last error alone, as an RTX 3060 does; they returned `cudaErrorInvalidValue`, which
  MaCh3's teardown left in the last error for CUDAProb3's next check to exit on. Unit test in `test_runtime_exports`
  (pantheonsim #618).
- cuBLAS: a dot product (`?dot`, `?dotu`, `?dotc`, `DotEx`, `DotcEx`, their `_64` forms) in the default host pointer mode
  with its result pointer in device memory stores the result there, as an RTX 3060 does; it died with SIGSEGV inside the
  library, which stopped every device XC integration of GauXC (its `gdot` makes this call). The other reductions (`?nrm2`,
  `?asum`, `i?amax`, `i?amin` and their Ex and `_64` forms), and a dot product with n <= 0, crash on the card with a device
  result in host mode; they now return `CUBLAS_STATUS_INVALID_VALUE`, said once on stderr, instead of crashing. Measured per
  routine on the card ([cuBLAS](nvidia/docs/cublas.md)). New unit test `test_cublas_host_mode_device_result` (pantheonsim #638).
- NPP, CUDA 12 build: `nppGetStreamContext`, `nppGetStream` and `nppSetStream` (NPP 12's own stream, which CUDA 13's NPP removed)
  are exported from the shim, filling the context as NPP 12.3 does on an RTX 3060. New e2e test `e2e_npp_stream_context` (skips
  on CUDA 13) (pantheonsim #617).
- NPP: `nppiFilterGaussBorder_8u_C1R_Ctx` and its plain CUDA 12 form, with every mask NPP gives it (3x3 to 15x15), bit for bit
  as NPP 13.0 and NPP 12.3 on an RTX 3060 on random, constant and impulse images. Measured: a separable single-precision filter,
  vertical then horizontal, each an fma chain over the taps in the order 1, 0, 2, ..., n-1, truncated to 8 bits (so a constant
  image can come out a count low, 3 to 2 with 3x3, which is what an earlier attempt could not explain); the taps are the card's
  (the 32f form's impulse response is exactly their products), with sigma 0.6 + 0.4 * (n/2) rather than the header's formula.
  Replicate is the only border, the source image's edge replicates (pSrc is the pixel at `oSrcOffset`), and the argument checks
  and their order are the card's: null pointer, size, offset out of range, border mode, mask size, then an empty ROI succeeds and
  a destination step below the ROI width is `NPP_STEP_ERROR`. The card's output is pinned in the `npp_imgproc` golden
  (`e2e_library_goldens`) (pantheonsim #617).
- HIP: `hipGraphInstantiateWithParams` takes `hipGraphInstantiateFlagUpload` and `hipGraphInstantiateFlagDeviceLaunch`, as ROCm's
  HIP does there (all four flags); it refused anything but AutoFreeOnLaunch and UseNodePriority with `hipErrorInvalidValue`, so
  Colvars, which instantiates its graphs with Upload and an `uploadStream`, stopped on its first frame. With Upload the graph is
  uploaded on `uploadStream` as `hipGraphUpload` does, which queues no work here (the executable graph is already the one a launch
  runs); `result_out` and `errNode_out` are set as ROCm sets them. DeviceLaunch only shows in `hipGraphExecGetFlags`, as in ROCm.
  `hipGraphInstantiateWithFlags` still refuses both, as ROCm does. New e2e test `amd_graph_instantiate_params` (pantheonsim #628).
- HIP: a failed `hipModuleLoad` is the thread's last error, as in ROCm's HIP, where every API returns through `HIP_RETURN`: a missing
  file (`hipErrorFileNotFound`) and an empty one (`hipErrorInvalidImage`) were returned but not recorded, so `hipGetLastError` then said
  `hipSuccess`. OpenMM's HIP platform reads `hipGetLastError` after a miss in its kernel cache and stopped with "Detected HIP error: no
  error" at the first kernel of every `TestHip*` program. `hipModuleGetFunction` and `hipModuleGetGlobal` now check their arguments in
  ROCm's order and with its codes: a null module is `hipErrorInvalidResourceHandle` (`hipModuleGetFunction` said `hipErrorInvalidValue`)
  and an empty kernel name is `hipErrorInvalidValue`. New e2e test `amd_module_last_error`, which also holds the other module and link
  entry points to recording their errors (pantheonsim #635).

- Peer access: after `cudaDeviceEnablePeerAccess` (or `cuCtxEnablePeerAccess`), a kernel on one simulated device reads, writes and
  atomically updates the peer device's allocations in place, as on a card (unified addressing). It stopped with `[invalid-pointer]
  ... (belongs to a different device)` and `cudaErrorIllegalAddress`, so a program that asks `cudaDeviceCanAccessPeer` and takes the
  direct route when the answer is yes failed (accl-kaust/GPU-IBS on two simulated H100s, pantheonsim #598). Without peer access, or
  after `cudaDeviceDisablePeerAccess`, the access still faults, as it does on two RTX 3060s (measured). Only an access that missed every
  allocation of the device looks at the peer; launch caching (`VGPU_MEMO=1`) and host-GPU mode stand aside while a peer link exists.
  New test `e2e_peer_kernel` (SASS and PTX paths).
- `vgpu run --preload`: PyTorch starts again, and its profiler lists what it did before. Both broke when the driver became a private load
  (pantheonsim #597, below). The runtime looked the driver up in the global scope only, so it never initialised it nor handed it its
  primary context, and PyTorch 2.10's `torch.cuda` init, which asks `cuDevicePrimaryCtxGetState` with no `cuInit` of its own, failed with
  "CUDA driver error: initialization error" on every check. The runtime now keeps the privately loaded driver's handle and asks it
  first, and `cudaGetDriverEntryPoint` no longer reloads that driver with `RTLD_GLOBAL` (#604's late-load path), which put it in the
  global scope after all. And each shim carries its own copy of the profiling engine: the unversioned driver in the global scope was
  what made every shim use one, so without it cuBLAS's internal copies were reported as the program's `cudaMemcpy`. The engine's
  functions now sit in a version node every shim shares (`VGPU_PROFILING_1`). New test `e2e_preload_torch_like`: programs that use
  the shim as PyTorch does, with no PyTorch.
- `vgpu run --preload`: `import ThrustRTC` (PyPI 0.3.20) no longer crashes with a segmentation fault (pantheonsim #597). ThrustRTC
  keeps the driver's and NVRTC's functions in exported pointers named like them (`cuInit`, `nvrtcVersion`, ...), and with the shim's
  `libcuda.so.1` preloaded its references to its own `cuInit` bound to the shim's function, whose code its `dlsym` result then
  overwrote (NVIDIA's own `libcuda.so.1` preloaded does the same). The driver's libraries and NVRTC are now loaded privately
  (`RTLD_LOCAL`) by the preloaded runtime as it starts (`VGPU_PRELOAD_PRIVATE`), so every later request for those sonames still gets
  the simulator's, and the global scope no longer holds their functions. New test `e2e_preload_fnptr`. ThrustRTC's kernels still do
  not compile through the simulator's NVRTC (its own `cstdint` meets the host's `/usr/include`).
- `e2e_sass_jit` (steps shard): a fault run whose compile did not happen (it outlasted its time limit on a loaded runner, or the
  compiler was killed) no longer fails the comparison with the tier. The JIT's notice on stderr ("pantheonsim: JIT: ... stay on the
  tier") was the only difference: the fault itself was reported the same, byte for byte, in 72 runs of the tier, the JIT and the JIT
  without exceptions on an idle and on a loaded machine (pantheonsim #611). The fault runs give a compile 840 s and take a run that
  could not compile once more; a failure now names the notice or shows the diff, and a compiler killed on its first compile checks the
  retry.
- `test_lockstep_look`: `a_block_that_stands_still_is_still_found_out_and_slept_on` no longer depends on how many turns a thread
  takes in 100 ms of wall time, which failed under load and under ThreadSanitizer (pantheonsim #612). `lockstep_worker` takes the
  nap it sleeps between passes as a callable (the engines' sleeps, unchanged); the test counts the naps: the first after the third
  look, then 5, 10, 20 ... 500 microseconds, and none for a block that computes.
- AMD: `v_div_fixup_f32`/`_f64` and `v_div_fmas_f32`/`_f64` with the VOP3 `clamp` bit or an output modifier (`mul:2`, `mul:4`,
  `div:2`) wrote the result unchanged, in the interpreter and in the tier. clang folds `fmin(fmax(a / b, 0), 1)` into the clamp of the
  division's last instruction (`v_div_fixup_f64 ... clamp`) at -O2 and up, so such code got the unclamped quotient (llnl/exadis on gfx942).
  They now clamp and scale as every other VOP3 float result does; `v_div_fixup_f16` already did. New test `test_amd_gcn_div_clamp`
  (pantheonsim #613).
- AMD: an atomic add (or any read-modify-write) of a dispatch that started while another dispatch ran alone could be lost. The lone
  dispatch decided at the start of an atomic instruction that it needed no lock, and the other's atomic landed between its read and its
  write. A dispatch now marks such an instruction before it looks at the count of running dispatches, and a dispatch that starts waits
  for that instruction to end; results of programs that run one dispatch at a time are unchanged. New test
  `a_dispatch_that_starts_while_another_runs_alone_does_not_lose_atomic_adds` in `test_amd_gcn_atomics` (pantheonsim #610).
- AMD: a kernel built with `.amdhsa_uses_dynamic_stack 1` (recursion, indirect or virtual calls) gets
  max(`hipLimitStackSize`, its static `private_segment_fixed_size`) bytes of scratch per work-item, as ROCm's HIP gives it
  (CLR `VirtualGPU::submitKernelInternal`), instead of only its static size; an HSA dispatch packet's `private_segment_size` is
  honoured where it is larger, `HSA_EXECUTABLE_SYMBOL_INFO_KERNEL_DYNAMIC_CALLSTACK` reports the descriptor's bit, and
  `hipDeviceSetLimit(hipLimitStackSize)` rounds up to 16 bytes and refuses more than a wave's scratch offset addresses (131004 bytes on
  gfx9), as CLR does. New tests `test_amd_gcn_dyn_stack` and `test_amd_hip_dyn_stack` (pantheonsim #585).
- Tests: `e2e_group_ballot_barrier`, a cell-list neighbour build in the shape of fastMD's `nl_build_kernel` (groups of four lanes
  at different copies of a group-masked `__ballot_sync`, then a full-mask `__shfl_down_sync` and `__syncthreads`), on a simulated
  H100 as PTX and as SASS, every count and listed neighbour checked against the host; it passes on an RTX 3060. fastMD's
  `VerletList.GrowsOnOverflowAndStaysComplete` (and the rest of `test_verlet`) and `Simulation.LayerStressSumRules` pass on main
  on both paths (pantheonsim #574).
- cuBLAS: the band and packed matrix-vector routines (`tbmv`, `tpmv`, `gbmv`, `sbmv`/`hbmv`, `spmv`/`hpmv`, every type) read the
  band where it is stored and loop over it, instead of building the full n x n matrix on every call. SCS's diagonal scaling
  (`cublasDtbmv` with k = 0 on a vector of 1500) mapped 34 MiB, took 8,000 page faults and summed 2.25 million products on each
  call; with transparent huge pages at `madvise`, as GitHub's runners ship them, that was most of the run. SCS's `max_ent` solve
  on a 7.0 kernel with `madvise`: 574 s before, 5.5 s after, the same iterations; the answers are the full loops' bits (an
  infinity or NaN in x still runs the full loops, which make NaN of the zeros it meets). `tbsv`/`tpsv` still use the full matrix.
  Making a device also fixes glibc's heap thresholds where its dynamic rule ends up (32 MiB mmap, 64 MiB trim), so a
  host-computed library call that loads and frees a few MiB of operands (cuSPARSE SpMV) no longer gives the heap back to the
  kernel and faults it in again each time (`VGPU_HOST_HEAP=0`, or a threshold set in the environment, keeps glibc's own).
  The setup action has a `huge-pages` input (default `true`): on a GitHub-hosted runner whose transparent huge pages are
  `madvise` it sets `always` for the job when sudo works, says what it did, and never fails the step
  (`scripts/thp-always.sh`). New tests `test_cublas_band_heap`, `test_host_heap` and `lint_thp_always` (pantheonsim #551).
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
