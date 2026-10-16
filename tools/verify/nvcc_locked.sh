#!/usr/bin/env bash
# nvcc for building the corpora's programs: the machine-wide build lock, nice, a memory cap, the
# CUDA 13.0 toolkit (13.2 segfaults on the local cards) and a shared CUDA runtime, so the
# simulator's libcudart can be substituted when the program is run.
exec flock "${VGPU_BUILD_LOCK:-$HOME/.cache/pantheonsim-build.lock}" nice -n 10 \
  bash -c 'ulimit -v 12582912; exec "$@"' _ "${CUDA_HOME:-/usr/local/cuda-13.0}/bin/nvcc" -cudart shared "$@"
