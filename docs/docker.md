# PantheonSim in a container

Pull the simulator instead of building it. The image runs on any Linux host with
Docker: no GPU, no NVIDIA or AMD driver, no `--gpus` flag.

```bash
docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus
docker run --rm -e VGPU_GPU=nvidia/h100 ghcr.io/pantheongpu/pantheonsim:latest nvidia-smi
docker run --rm -v "$PWD:/work" -w /work ghcr.io/pantheongpu/pantheonsim:latest vgpu run ./my_program
```

## Tags

| Tag | CUDA libraries built against |
| --- | --- |
| `<version>-cuda13.0`, `<version>`, `latest` | CUDA 13.0 (`libcudart.so.13`) |
| `<version>-cuda12.8` | CUDA 12.8 (`libcudart.so.12`) |

Pick the tag whose CUDA major matches the toolkit your program was built with: a
program built with CUDA 12 needs `libcudart.so.12`, and the CUDA 13 image does not
carry it. Every image has the AMD side too (HIP, ROCm 6.4 and 7.x sonames).

## What is in it, and what is not

It holds the simulator (`vgpu`), the simulated libraries under their real sonames
in `/opt/pantheonsim/shim`, and the `nvidia-smi`, `rocm-smi` and `amd-smi`
drop-ins. `VGPU_GPU` picks the GPU (default `nvidia/t4`; `vgpu list-gpus` lists
them), `VGPU_DEVICE_COUNT` how many.

It has **no compiler**: neither `nvcc` nor `hipcc`. A program has to arrive built,
or be built in an image that has one. The simulator's shim is not on
`LD_LIBRARY_PATH`; `vgpu run` puts it there for the program it starts.

## Compiling inside a container

Copy the simulator into an image that already has your compiler.
[`docker/example/Dockerfile`](../docker/example/Dockerfile) does it for CUDA on
NVIDIA's own `devel` image and builds a program with it:

```bash
docker build -f docker/example/Dockerfile \
  --build-arg CUDA=13.0.0 --build-arg SIM=ghcr.io/pantheongpu/pantheonsim:latest \
  -t my-cuda-sim docker/example
docker run --rm my-cuda-sim
```

The same two lines work on any base image: `COPY --from=<pantheonsim image>
/opt/pantheonsim /opt/pantheonsim`, then `ENV PATH=/opt/pantheonsim/bin:$PATH
VGPU_SHIM_DIR=/opt/pantheonsim/shim`. For HIP, use AMD's `rocm/dev-ubuntu-24.04`
the same way.

The compiler is not in PantheonSim's image because NVIDIA's CUDA Toolkit license
lists the components others may redistribute (its Attachment A), and `nvcc` is not
among them. Pulling NVIDIA's image yourself keeps that between you and NVIDIA.

## Building the image

```bash
docker build --target cuda --build-arg CUDA_VERSION=13.0 -t pantheonsim-ci:cuda13.0 .github/ci-image
docker build -f docker/Dockerfile --build-arg BUILDER=pantheonsim-ci:cuda13.0 -t pantheonsim .
```

The first command makes the toolkit stage (it stays in the build; the final image
copies out only what `scripts/package-build.sh` selects). The
[Docker images workflow](../.github/workflows/docker.yml) does this for each CUDA
version, runs each image, and pushes it when a version tag is pushed.
