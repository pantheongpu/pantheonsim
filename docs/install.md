# Install on a clean machine

Every command on this page is run, as written, by the [clean-install check](ci.md#clean-install-and-version-matrix)
(`tests/install/run_doc_steps.sh`) on a bare `ubuntu:24.04` container, so a missing package or a wrong
command here fails CI. The blocks are for Ubuntu 22.04 and 24.04 on x86-64, with no GPU and no GPU driver. On a
machine that already has `make`, `git` and a C++ compiler, start at the toolkits.

## Prerequisites

A bare machine has none of the tools to fetch and build the simulator. Use `sudo` as your user (leave it out
as root).

<!-- install-step: prereqs -->
```bash
sudo apt-get update
sudo apt-get install -y make git wget gnupg ca-certificates
```

## CUDA toolkit, for `nvcc`

Install the compilers before you build the simulator: it builds its CUDA and HIP libraries against the toolkit it finds, and a simulator built with no toolkit has no `libcudart.so` for a program to load. (Installed later, run `make clean` and `make build` again.) Skip a toolkit you do not need.

Install the toolkit packages only, not the driver (`cuda-toolkit-*`, not `cuda` or `cuda-drivers`). Pick the
version with `CUDA`; 12.4 to 13.2 are the ones checked every night (12.4 is on Ubuntu 22.04 only: NVIDIA's
24.04 repository starts at 12.5).

<!-- install-step: cuda -->
```bash
CUDA=${CUDA:-13.0}
. /etc/os-release
wget -q "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${VERSION_ID//./}/x86_64/cuda-keyring_1.1-1_all.deb"
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends "cuda-toolkit-${CUDA//./-}"
export PATH="/usr/local/cuda-$CUDA/bin:$PATH"
nvcc --version
```

## ROCm, for `hipcc`

Install the user-space compiler packages only, not `amdgpu-dkms` (there is no GPU to drive). Pick the version with
`ROCM`: 6.4, 7.0, 7.1 and 7.2 are the ones checked every night.

<!-- install-step: rocm -->
```bash
ROCM=${ROCM:-7.1}
. /etc/os-release
sudo apt-get install -y --no-install-recommends libnuma1 libxml2 "$(apt-cache policy libelf1t64 | grep -q 'Candidate: [0-9]' && echo libelf1t64 || echo libelf1)"
sudo mkdir -p /etc/apt/keyrings
wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key | gpg --dearmor | sudo tee /etc/apt/keyrings/rocm.gpg > /dev/null
echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/$ROCM $VERSION_CODENAME main" | sudo tee /etc/apt/sources.list.d/rocm.list
printf 'Package: *\nPin: origin repo.radeon.com\nPin-Priority: 1001\n' | sudo tee /etc/apt/preferences.d/rocm > /dev/null
sudo apt-get update
sudo apt-get install -y --no-install-recommends hipcc hip-dev rocm-llvm rocm-device-libs
export PATH="/opt/rocm/bin:$PATH"
hipcc --version
```

## Build

Run these in a checkout of the repository. `make deps` installs CMake and a C++20 compiler when they are missing
(`YES=1` answers its question), `make doctor` says whether you are ready, and `make build` takes a few minutes.

<!-- install-step: build -->
```bash
make deps YES=1
make doctor
make build
make demo
```

The demo ends with a line starting `PASS`.

## Hello world

A CUDA program, built with the toolkit's `nvcc` and run on a simulated H100:

<!-- install-file: hello.cu -->
```cuda
#include <cstdio>

__global__ void twice(int* out) { out[threadIdx.x] = 2 * threadIdx.x; }

int main() {
  int* d;
  int h[64];
  cudaMalloc(&d, sizeof h);
  twice<<<1, 64>>>(d);
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  for (int i = 0; i < 64; ++i)
    if (h[i] != 2 * i) { std::printf("wrong at %d\n", i); return 1; }
  std::printf("hello from the simulated GPU: ok\n");
  return 0;
}
```

Build it with `-cudart shared` and run it on a simulated H100:

<!-- install-step: hello-cuda -->
```bash
nvcc -cudart shared hello.cu -o hello_cuda
make run APP=./hello_cuda GPU=nvidia/h100
```

A HIP program, for an AMD GPU:

<!-- install-file: hello.hip -->
```cpp
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void twice(int* out) { out[threadIdx.x] = 2 * threadIdx.x; }

int main() {
  int* d;
  int h[64];
  if (hipMalloc(&d, sizeof h) != hipSuccess) return 2;
  twice<<<1, 64>>>(d);
  if (hipMemcpy(h, d, sizeof h, hipMemcpyDeviceToHost) != hipSuccess) return 2;
  for (int i = 0; i < 64; ++i)
    if (h[i] != 2 * i) { std::printf("wrong at %d\n", i); return 1; }
  std::printf("hello from the simulated GPU: ok\n");
  return 0;
}
```

Build it for the GPU you simulate (`gfx942` is the MI300X) and run it:

<!-- install-step: hello-hip -->
```bash
hipcc --offload-arch=gfx942 hello.hip -o hello_hip
make run APP=./hello_hip GPU=amd/mi300x
```

## Docker instead

The image carries the simulator and no compiler:

<!-- install-step: docker -->
```bash
docker run --rm ghcr.io/pantheongpu/pantheonsim:latest vgpu list-gpus
```

## In GitHub Actions

The setup action does the toolkit steps above for you: see [github-action.md](github-action.md).
