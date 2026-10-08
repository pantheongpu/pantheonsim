# The simulator added to an image that has a compiler. The compiler is NVIDIA's,
# pulled from NVIDIA's own image by whoever builds this; nothing of it is in
# PantheonSim's.
#
#   docker build -f docker/example/Dockerfile \
#     --build-arg CUDA=13.0.0 --build-arg SIM=ghcr.io/pantheongpu/pantheonsim:latest -t my-cuda-sim .
#   docker run --rm my-cuda-sim
ARG CUDA=13.0.0
ARG SIM=ghcr.io/pantheongpu/pantheonsim:latest
FROM ${SIM} AS sim

FROM nvidia/cuda:${CUDA}-devel-ubuntu24.04
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends libstdc++6 && rm -rf /var/lib/apt/lists/*
COPY --from=sim /opt/pantheonsim /opt/pantheonsim
# bin/ first: its nvcc is the real one with `-cudart shared` added, which a
# program needs to run on the simulator.
ENV PATH=/opt/pantheonsim/bin:$PATH \
    VGPU_SHIM_DIR=/opt/pantheonsim/shim \
    VGPU_GPU=nvidia/h100
WORKDIR /work
COPY vector_add.cu .
RUN nvcc -o vector_add vector_add.cu
CMD ["vgpu", "run", "./vector_add"]
