# Running CUDA tests in GitHub Actions

GitHub's hosted runners have no GPU. This action gives a job simulated NVIDIA
GPUs, so CUDA code can be compiled and its tests run on every push, with a run
on physical GPUs kept for a nightly or release job.

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

## What it does

1. Installs CMake and, unless `cuda-toolkit: none`, Ubuntu's CUDA toolkit
   (CUDA 12.0 on `ubuntu-24.04`).
2. Builds the simulator at the ref the job named, and caches the build against
   a hash of its source and the toolkit version, so later runs skip it.
3. Puts the simulator's tools first on `PATH` and sets `VGPU_GPU` and
   `VGPU_DEVICE_COUNT` for the rest of the job.

After it, `nvidia-smi` reports the GPUs you asked for, `nvcc` is the real
compiler with `-cudart shared` added (so unmodified build systems link the
runtime the simulator stands in for), and `vgpu run ./program` runs a program on
the simulated GPUs. `vgpu test --matrix ./program` runs it on every measured
profile and compares the output.

## Inputs

| Input | Default | Meaning |
| --- | --- | --- |
| `gpu` | `nvidia/t4` | The profile to simulate (`vgpu list-gpus`) |
| `count` | `1` | How many GPUs |
| `cuda-toolkit` | `apt` | `apt` installs Ubuntu's toolkit; `none` uses one the job already installed |

## Outputs

| Output | Meaning |
| --- | --- |
| `build-dir` | Where the simulator was built |
| `shim-dir` | Its CUDA libraries, for `LD_LIBRARY_PATH` when not using `vgpu run` |

## Things to know

- Build kernels for an architecture the profile supports: `-arch=compute_75`
  runs on a T4 and every newer card, `compute_90` needs Hopper.
- The simulator's CUDA libraries are built against the job's toolkit, so their
  version always matches the `nvcc` that compiled the program.
- It checks behaviour, not performance: timings mean nothing here.
