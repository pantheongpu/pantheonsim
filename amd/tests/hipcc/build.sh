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
