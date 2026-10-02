# The toolchains CI's hosted jobs used to apt-install on every run.
#
# On GitHub's hosted runners `apt-get install nvidia-cuda-toolkit` usually
# takes a minute or two, but the archive mirror sometimes slows to a crawl:
# on 2026-10-01 single installs took 70 minutes and 3 hours 16 minutes, and a
# job died on a runner shutdown two hours into one. These images are built
# once (.github/workflows/ci-image.yml), pushed to GHCR, and pulled by the
# jobs (`container:`), which then install nothing from Ubuntu's archive.
#
# One Dockerfile, one target per toolkit (each its own image, so no job ever
# has two nvccs on PATH and picks the wrong one's ptxas):
#
#   cuda12.0  Ubuntu 24.04's nvidia-cuda-toolkit (CUDA 12.0) and nvprof, with
#             GCC 12 for nvcc and clang/LLVM for the AMD code-object tests:
#             the build, sanitizer, TSan, workloads and lazy-modules jobs.
#   cuda      CUDA from NVIDIA's apt repository, CUDA_VERSION (13.0, 12.8...):
#             the compiler and the headers of every library the simulator
#             stands in for. The ollama and SASS jobs (13.0) and the CUDA 12.x
#             leg (12.8).
#
# Every image says which it is in PANTHEONSIM_CI_IMAGE, so a step can tell it
# is running in one.
#
# Anything a test probes for (python3, bc, lspci, objdump, readelf, taskset)
# is installed too: a test that skips here but ran on the bare runner would be
# lost coverage nobody sees.

FROM ubuntu:24.04 AS base
ENV DEBIAN_FRONTEND=noninteractive LANG=C.UTF-8
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      build-essential cmake ninja-build git ca-certificates wget curl zstd xz-utils unzip \
      python3 python3-venv bc time file pciutils procps util-linux sudo gnupg \
 && rm -rf /var/lib/apt/lists/*

FROM base AS cuda12.0
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends \
      g++-12 nvidia-cuda-toolkit nvidia-profiler clang llvm \
 && rm -rf /var/lib/apt/lists/* \
 && nvcc --version | tail -2
# nvcc 12.0 rejects GCC 13's headers; the workflows also set this, but a job
# that forgets cannot then fail on it.
ENV NVCC_PREPEND_FLAGS="-ccbin g++-12" PANTHEONSIM_CI_IMAGE=cuda12.0

FROM base AS cuda
# Dotted: 13.0, 12.8.
ARG CUDA_VERSION=13.0
# The compiler and the header of every library the simulator stands in for:
# each shim is compiled against its library's own header, so a missing one
# fails the build. cuda-crt is a package of its own only in the releases that
# split it out of cuda-nvcc. cuobjdump and nvdisasm are the SASS tests' oracle.
RUN v=$(echo "$CUDA_VERSION" | tr . -) \
 && wget -qO /tmp/keyring.deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb \
 && dpkg -i /tmp/keyring.deb && rm /tmp/keyring.deb \
 && apt-get update -qq \
 && pkgs="cuda-nvcc-$v cuda-cudart-dev-$v cuda-nvml-dev-$v cuda-cupti-dev-$v cuda-nvrtc-dev-$v \
          libcublas-dev-$v libcufft-dev-$v libcurand-dev-$v libcusparse-dev-$v libnvjpeg-dev-$v \
          libcusolver-dev-$v libnpp-dev-$v cuda-cuobjdump-$v cuda-nvdisasm-$v" \
 && if apt-cache show "cuda-crt-$v" >/dev/null 2>&1; then pkgs="$pkgs cuda-crt-$v"; fi \
 && apt-get install -y -qq --no-install-recommends $pkgs \
 && rm -rf /var/lib/apt/lists/* \
 && "/usr/local/cuda-$CUDA_VERSION/bin/nvcc" --version | tail -2
ENV PATH=/usr/local/cuda-$CUDA_VERSION/bin:$PATH CUDA_HOME=/usr/local/cuda-$CUDA_VERSION \
    NVCC_PREPEND_FLAGS="" PANTHEONSIM_CI_IMAGE=cuda$CUDA_VERSION
