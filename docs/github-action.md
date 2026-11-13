# Running CUDA and HIP tests in GitHub Actions

GitHub's hosted runners have no GPU. This action gives a job simulated NVIDIA
or AMD GPUs, so CUDA and HIP code can be compiled and its tests run on every
push, with a run on physical GPUs kept for a nightly or release job.

Use it as `pantheongpu/setup-pantheonsim@v0`
([setup-pantheonsim](https://github.com/pantheongpu/setup-pantheonsim)). Each
release of that action pins a simulator commit that was tested with it, and
`@v0` follows the latest `v0.x.y` release. This repository's own `action.yml`
(`pantheongpu/pantheonsim@<ref>`) is what that action runs; pointing a workflow
at `@main` builds whatever `main` is that day, so use it only to try an
unreleased fix.

```yaml
jobs:
  cuda-tests:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: pantheongpu/setup-pantheonsim@v0
        with:
          gpu: nvidia/h100
          count: 2
      - run: |
          nvcc -arch=compute_90 my_test.cu -o my_test
          vgpu run ./my_test
```

For AMD, name an Instinct profile. The action installs ROCm's own `hipcc`:

```yaml
      - uses: pantheongpu/setup-pantheonsim@v0
        with:
          gpu: amd/mi300x
          count: 2
      - run: |
          hipcc --offload-arch=gfx942 my_test.hip -o my_test
          vgpu run ./my_test
```

To compile with a newer CUDA than Ubuntu's, give its version:

```yaml
      - uses: pantheongpu/setup-pantheonsim@v0
        with:
          gpu: nvidia/h100
          cuda-toolkit: '12.6'
```

## What it does

1. Installs CMake and a compiler for the GPU:
   - NVIDIA: Ubuntu's CUDA toolkit (CUDA 12.0 on `ubuntu-24.04`), or the
     `nvcc` of the version you name from NVIDIA's repository.
   - AMD: ROCm's `hipcc`, with the clang and device libraries it compiles
     with, from AMD's repository. Only the compiler: the simulator is the
     runtime, so no driver or ROCm runtime is installed.
2. Gets the simulator at the commit the release pins: the prebuilt tarball
   from that release when it has one for this runner and toolkit (its sha256
   checked), otherwise a build from source, cached against a hash of its
   source and the toolkit version so later runs skip it. See
   [Setup time](#setup-time).
3. Puts the simulator's tools first on `PATH` and sets `VGPU_GPU` and
   `VGPU_DEVICE_COUNT` for the rest of the job.

After it, `nvidia-smi` (or `rocm-smi` and `rocm_agent_enumerator`) reports
the GPUs you asked for, `nvcc` is the real
compiler with `-cudart shared` added (so unmodified build systems link the
runtime the simulator stands in for), and `vgpu run ./program` runs a program on
the simulated GPUs. A program `hipcc` built also runs directly, because the
action points ROCm's `libamdhip64` at the simulator. `vgpu test --matrix ./program` runs it on every measured
profile and compares the output; it exits 3 when a profile's result differs
and 4 when the program did not run at all (`vgpu test --help`).

## Inputs

| Input | Default | Meaning |
| --- | --- | --- |
| `gpu` | `nvidia/t4` | The profile to simulate (`vgpu list-gpus`), e.g. `nvidia/h100` or `amd/mi300x` |
| `count` | `1` | How many GPUs |
| `cuda-toolkit` | `apt` for NVIDIA, `none` for AMD | `apt` installs Ubuntu's toolkit; a version like `12.6` or `13.0` installs that `nvcc` from NVIDIA; `none` uses one the job already installed |
| `rocm` | `7.1` for AMD, `none` for NVIDIA | The ROCm version whose `hipcc` to install -- `6.4`, `7.0`, `7.1` and `7.2` are the releases whose programs are checked on the simulator; `none` uses one the job already installed |
| `library-path` | `true` | Puts the simulator's libraries on `LD_LIBRARY_PATH`, so programs run directly (ctest, scripts) as well as under `vgpu run`. Set `false` when the build links the toolkit's own vendor libraries (see below) |
| `prebuilt` | `true` | Uses the release's prebuilt simulator when there is one for this runner and toolkit and its sha256 matches; `false` always builds from source |
| `huge-pages` | `true` | On a GitHub-hosted runner whose transparent huge pages are `madvise` (as they ship), sets them to `always` for the job when sudo works, so the simulated program's own memory gets 2 MiB pages too; the step says what it did and never fails on it. Self-hosted runners are left alone; `false` leaves the setting as it is |

## Outputs

| Output | Meaning |
| --- | --- |
| `build-dir` | Where the simulator was built |
| `shim-dir` | Its CUDA libraries, for `LD_LIBRARY_PATH` when not using `vgpu run` |
| `prebuilt` | `true` when the simulator came from the release's prebuilt tarball, `false` when it was restored from the cache or built |
| `prebuilt-asset` | The release asset name it looked for (empty off x86-64) |

## Setup time

Building the simulator from source is most of the action's time on a cold
cache. Measured for the whole action step on GitHub's hosted runners:

| Job | Built from source (cold cache) | Restored from the cache (warm) | From a release's prebuilt tarball |
| --- | --- | --- | --- |
| `ubuntu-24.04`, NVIDIA, Ubuntu's CUDA 12.0 | 13 to 22 min | 1.3 to 1.7 min | 4.9 min (4.85 of it the apt install) |
| `ubuntu-24.04`, NVIDIA, CUDA 12.6 from NVIDIA | 15 to 20 min | 1.4 to 1.8 min | 1.8 min |
| `ubuntu-24.04`, NVIDIA, CUDA 13.0 from NVIDIA | 20 min | | 1.5 min |
| `ubuntu-24.04`, AMD, ROCm 7.1 | 8 to 13 min | 0.6 to 0.7 min | 0.5 min |
| `ubuntu-22.04`, NVIDIA, CUDA 12.6 from NVIDIA | 18 to 24 min | 3.0 min | 3.0 min |
| `ubuntu-22.04`, AMD, ROCm 7.1 | 9 to 15 min | 0.8 min | 1.1 min |

(Cold and warm: the pantheonsim [Action workflow](../.github/workflows/action.yml),
9 runs on 2026-11-11 and 12, the
[setup-pantheonsim tests](https://github.com/pantheongpu/setup-pantheonsim/actions/workflows/test.yml),
6 runs from 2026-10-27 to 2026-11-12, and the source builds of one
[release-binaries](../.github/workflows/release-binaries.yml) run; one warm run
took 5.7 min. Prebuilt: that run's check jobs, one each, which served the
tarballs from the runner itself, so the 18 to 35 MB download from GitHub's
release host is not in them; fetching, checking and unpacking took at most 1 s.
Everything else in a warm or prebuilt step is installing the toolkit.)

The cache is per repository and keyed on the simulator's source, so the first
run in a repository, and the first after each simulator change, is the cold
one, and `@main` is cold after every commit to the simulator.

A release tag carries the simulator prebuilt for these runners and toolkits
(`ubuntu-24.04` with CUDA 12.0, 12.6 or 13.0 or for AMD, `ubuntu-22.04` with
CUDA 12.6 or for AMD), attached by the
[release-binaries workflow](../.github/workflows/release-binaries.yml) as
`pantheonsim-<version>-linux-x86_64-<os>-cuda<version>-<source hash>.tar.gz`
with a `.sha256` beside it. With the action pinned at such a release (or at its
commit), it downloads the tarball, checks the sha256 and unpacks it instead
of building the simulator. Anything else -- `@main`, a
toolkit or runner with no tarball, a container job (the tarball is built for
the hosted runner's directory), a failed download or a checksum that does not
match -- falls back to the cache and the source build, with a notice saying
why. `prebuilt: false` skips the tarball.

## Things to know

- Build kernels for an architecture the profile supports: `-arch=compute_75`
  runs on a T4 and every newer card, `compute_90` needs Hopper. On AMD, build
  for the card's target: `gfx942` for the MI300X and MI325X, `gfx950` for the
  MI350X.
- The simulator's CUDA libraries are built against the job's toolkit, so their
  version always matches the `nvcc` that compiled the program.
- Test suites run unchanged. With `library-path: true` (the default), a
  program run directly, by `ctest` or a script, reaches the simulated GPUs
  just as it does under `vgpu run`. Programs are linked with the shared CUDA
  runtime even where a build asks for the static one (`-cudart static`,
  `-lcudart_static`, CMake's default), because the static runtime cannot talk
  to a simulated driver.
- A build that links the toolkit's own vendor libraries (`-lcublas`,
  `-lcusparse`, `-lcufft`, ...) needs `library-path: false`, with its tests run
  under `vgpu run`. With the default, the simulator's libraries are on
  `LD_LIBRARY_PATH` while the job builds, and the linker looks there for the
  real libraries' own dependencies. NVIDIA's `libcublas.so`, for one, needs
  internal symbols of its `libcublasLt.so` that the simulator's does not have,
  so the link fails with undefined references such as
  `cublasLtLegacyGemmSSS`. `vgpu run` puts the simulator's libraries in place
  for the test run alone:

  ```yaml
  - uses: pantheongpu/setup-pantheonsim@v0
    with:
      gpu: nvidia/t4
      library-path: false
  - run: make                  # links the toolkit's libcublas
  - run: vgpu run ./run_tests  # runs against the simulator's
  ```

  Linking against the simulator's own libraries instead (`-L` the `shim-dir`
  output) also works.
- A HIP project that calls CMake's `find_package(hip)` can need more of ROCm
  than the action installs. The action installs only the compiler: it extracts
  `hsa-rocr-dev` (the HSA runtime's CMake config) but not `hsa-rocr` (the
  `libhsa-runtime64.so` that config names), and not `comgr`. Projects using the
  action have had to install `comgr` and `hsa-rocr` themselves, from AMD's
  repository (the action has already added it), in a step after the action.
- CMake projects get the shared runtime through a toolchain file the action
  names in `CMAKE_TOOLCHAIN_FILE`, since CMake links with the host compiler
  where the `nvcc` wrapper cannot reach. A project that sets its own
  toolchain file needs `set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)` in it.
- Runners: `ubuntu-24.04` and `ubuntu-22.04`, and container jobs (which run as
  root without `sudo`). On 22.04, Ubuntu's own CUDA toolkit is 11.5, older than
  the simulator supports, so an NVIDIA job there gets CUDA 12.6 from NVIDIA.
- It checks behaviour, not performance: timings mean nothing here.

Not on GitHub? See [ci-other-platforms.md](ci-other-platforms.md).
