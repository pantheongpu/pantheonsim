# Changelog

PantheonSim is versioned by the `project()` line in `CMakeLists.txt`; `vgpu --version` prints it.
Each release is tagged `vX.Y.Z` and described on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases). It simulates GPUs functionally, so a
release says what runs and what it produces, never how fast a real GPU would run it.

## Unreleased

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
  call (pantheonsim #487). Kernels, copies and fills still run to completion inside their calls, so an event recorded behind a long
  kernel is still complete at once (#523 stays open); `cuLaunchHostFunc` in the driver API still runs inline. Programs that never queue
  a host function see no change. `cudaErrorNotReady` from a query is no longer left as the last error (`cudaGetLastError`), as on a card.
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
- License: from the next release PantheonSim is under the Business Source License 1.1, which keeps it free to use, change and run in
  your own CI and allows one restriction: you may not offer it to other people as a hosted or managed service. Each version becomes
  Apache-2.0 four years after release. Versions up to 0.2.0 stay Apache-2.0 (`LICENSE-APACHE-2.0`). `docs/licensing.md` has the
  details, and contributions now need the Contributor Agreement.
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
