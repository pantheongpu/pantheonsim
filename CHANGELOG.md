# Changelog

PantheonSim is versioned by the `project()` line in `CMakeLists.txt`; `vgpu --version` prints it.
Each release is tagged `vX.Y.Z` and described on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases). It simulates GPUs functionally, so a
release says what runs and what it produces, never how fast a real GPU would run it.

## Unreleased

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
