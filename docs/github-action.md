# Running CUDA and HIP tests in GitHub Actions

GitHub's hosted runners have no GPU. This action gives a job simulated NVIDIA
or AMD GPUs, so CUDA and HIP code can be compiled and its tests run on every
push, with a run on physical GPUs kept for a nightly or release job.

```yaml
jobs:
  cuda-tests:
    runs-on: ubuntu-24.04
    steps:
      - uses: actions/checkout@v4
      - uses: pantheongpu/pantheonsim@main
        with:
          gpu: nvidia/h100
          count: 2
      - run: |
          nvcc -arch=compute_90 my_test.cu -o my_test
          vgpu run ./my_test
```

For AMD, name an Instinct profile. The action installs ROCm's own `hipcc`:

```yaml
      - uses: pantheongpu/pantheonsim@main
        with:
          gpu: amd/mi300x
          count: 2
      - run: |
          hipcc --offload-arch=gfx942 my_test.hip -o my_test
          vgpu run ./my_test
```

To compile with a newer CUDA than Ubuntu's, give its version:

```yaml
      - uses: pantheongpu/pantheonsim@main
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
2. Builds the simulator at the ref the job named, and caches the build against
   a hash of its source and the toolkit version, so later runs skip it.
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
| `rocm` | `7.1` for AMD, `none` for NVIDIA | The ROCm version whose `hipcc` to install; `none` uses one the job already installed |

## Outputs

| Output | Meaning |
| --- | --- |
| `build-dir` | Where the simulator was built |
| `shim-dir` | Its CUDA libraries, for `LD_LIBRARY_PATH` when not using `vgpu run` |

## Things to know

- Build kernels for an architecture the profile supports: `-arch=compute_75`
  runs on a T4 and every newer card, `compute_90` needs Hopper. On AMD, build
  for the card's target: `gfx942` for the MI300X and MI325X, `gfx950` for the
  MI350X.
- The simulator's CUDA libraries are built against the job's toolkit, so their
  version always matches the `nvcc` that compiled the program.
- It checks behaviour, not performance: timings mean nothing here.
