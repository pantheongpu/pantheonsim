# PantheonSim in a container: the simulator, its NVIDIA and AMD libraries under
# their real sonames, and the nvidia-smi / rocm-smi drop-ins. No GPU and no
# driver on the host. No compiler either: it neither carries NVIDIA's nvcc nor
# AMD's hipcc, so a program has to arrive already built (or be built in an image
# that has a compiler -- see docs/docker.md for how to add the simulator to one).
#
#   docker build -f docker/Dockerfile --build-arg BUILDER=<toolchain image> -t pantheonsim .
#
# The build stage needs a CUDA toolkit: the simulator's CUDA libraries are built
# against its headers and carry its major version (libcudart.so.13 from CUDA
# 13, libcudart.so.12 from CUDA 12). .github/ci-image/Dockerfile builds such an
# image, and the toolkit stays in the build stage; the final image holds only
# what scripts/package-build.sh copies out.
ARG BUILDER=pantheonsim-ci:cuda13.0
ARG BASE=ubuntu:24.04

FROM ${BUILDER} AS build
WORKDIR /src
COPY . .
RUN cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
 && cmake --build build -j"$(nproc)" \
 && scripts/package-build.sh build /opt/pantheonsim

FROM ${BASE}
ENV DEBIAN_FRONTEND=noninteractive
# What the simulator links besides itself: the C++ runtime.
RUN apt-get update -qq \
 && apt-get install -y -qq --no-install-recommends libstdc++6 ca-certificates \
 && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/pantheonsim /opt/pantheonsim
# bin/ first: nvidia-smi, rocm-smi and vgpu are the simulator's. The shim is
# not put on LD_LIBRARY_PATH, so a toolkit layered under this image still links
# its own libraries; `vgpu run` puts it there for the program it starts.
ENV PATH=/opt/pantheonsim/bin:$PATH \
    VGPU_SHIM_DIR=/opt/pantheonsim/shim \
    VGPU_GPU=nvidia/t4
LABEL org.opencontainers.image.title="PantheonSim" \
      org.opencontainers.image.description="Simulated NVIDIA and AMD GPUs on a CPU: CUDA and HIP programs run unmodified" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/pantheongpu/pantheonsim"
CMD ["vgpu", "list-gpus"]
