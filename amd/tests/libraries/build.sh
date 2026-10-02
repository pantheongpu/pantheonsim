#!/usr/bin/env bash
# Rebuilds the programs run_libraries.sh runs: hipFFT, hipRAND, hipSOLVER and
# hipSPARSE called by hipcc-built programs for gfx942. It needs ROCm with the
# four libraries' headers (hipfft-dev, hiprand-dev, hipsolver-dev and
# hipsparse-dev); the programs are checked in, so the test does not.
#
#   amd/tests/libraries/build.sh [rocm-path] [hipcc's rocm-path]
set -euo pipefail
cd "$(dirname "$0")"
rocm=${1:-${ROCM_PATH:-/opt/rocm}}
hip=${2:-$rocm}
export ROCM_PATH=$hip HIP_PATH=$hip HIP_CLANG_PATH=$hip/lib/llvm/bin HIP_DEVICE_LIB_PATH=$hip/amdgcn/bitcode
for p in fft:hipfft rand:hiprand solver:hipsolver sparse:hipsparse; do
  "$hip/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 -I"$rocm/include" "${p%%:*}.cpp" -o "${p%%:*}.gfx942" \
    -L"$rocm/lib" -l"${p#*:}"
  echo "wrote $(pwd)/${p%%:*}.gfx942"
done
