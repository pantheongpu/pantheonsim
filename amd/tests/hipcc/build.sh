#!/usr/bin/env bash
# Rebuilds the hipcc-built executable the HIP launch test runs. It needs ROCm;
# the executable is checked in, so the test itself does not.
#
#   amd/tests/hipcc/build.sh [rocm-path]
set -euo pipefail
cd "$(dirname "$0")"
rocm=${1:-${ROCM_PATH:-/opt/rocm}}
export ROCM_PATH=$rocm HIP_PATH=$rocm HIP_CLANG_PATH=$rocm/lib/llvm/bin HIP_DEVICE_LIB_PATH=$rocm/amdgcn/bitcode
"$rocm/bin/hipcc" --version 2>/dev/null | grep "HIP version" || true
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 chevron.cpp -o chevron.gfx942
echo "wrote $(pwd)/chevron.gfx942"

# The device code alone, for the decoder and the executor to be checked
# against, and the listing of it from the same toolchain's llvm-objdump.
"$rocm/bin/hipcc" -O3 -std=c++17 --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
  -c ops.hip -o ops.gfx942.o
"$rocm/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx942 ops.gfx942.o |
  sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > ops.gfx942.dis
echo "wrote $(pwd)/ops.gfx942.o and its listing ($(wc -l < ops.gfx942.dis) instructions)"

# A GEMM through rocWMMA, whose loads and stores put each matrix element
# where the hardware's matrix instructions expect it. Needs rocwmma-dev.
"$rocm/bin/hipcc" -O3 -std=c++17 --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
  -c wmma.cpp -o wmma.gfx942.o
"$rocm/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx942 wmma.gfx942.o |
  sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > wmma.gfx942.dis
echo "wrote $(pwd)/wmma.gfx942.o and its listing ($(wc -l < wmma.gfx942.dis) instructions)"
