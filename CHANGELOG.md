# Changelog

PantheonSim is versioned by the `project()` line in `CMakeLists.txt`; `vgpu --version` prints it.
Each release is tagged `vX.Y.Z` and described on the
[releases page](https://github.com/pantheongpu/pantheonsim/releases). It simulates GPUs functionally, so a
release says what runs and what it produces, never how fast a real GPU would run it.

## [0.2.0] - unreleased

### Getting started
- `make` installs what is missing (it asks first) and builds; `./run` starts a simulated GPU machine from a
  settings file. The README is those two commands; the rest is in `docs/getting-started.md`, `docs/config.md`
  and `docs/cli.md`. The simulator uses half your CPUs unless you say otherwise.
- `scripts/ci-setup.sh` does for GitLab CI, CircleCI, Forgejo, Jenkins and Buildkite what the GitHub Action does
  (`docs/ci-other-platforms.md`).
- Every document was checked against the code, and the profile counts, command lists, ROCm versions and
  limitations were corrected. 48 profiles (38 NVIDIA, 10 AMD); 14 NVIDIA and 3 AMD are read from physical cards.

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
