#!/usr/bin/env bash
# Rebuilds spmm, the hipSPARSELt program amd_hipsparselt runs. It needs ROCm's
# hipcc, hipSPARSELt's and hipSPARSE's headers (the hipsparselt-dev and
# hipsparse-dev packages), and the libhipsparselt.so PyTorch's ROCm wheel
# ships to link against; the program is checked in, so the test needs none of
# them.
#
#   amd/tests/hipsparselt/build.sh <rocm> <include dir with hipsparselt/> <torch/lib>
set -euo pipefail
cd "$(dirname "$0")"
rocm=${1:-${ROCM_PATH:-/opt/rocm}} include=${2:-$rocm/include} lib=$3
export ROCM_PATH=$rocm HIP_PATH=$rocm HIP_CLANG_PATH=$rocm/lib/llvm/bin HIP_DEVICE_LIB_PATH=$rocm/amdgcn/bitcode
"$rocm/bin/hipcc" -O2 -std=c++17 -I"$include" spmm.cpp -o spmm -L"$lib" -lhipsparselt
echo "wrote $(pwd)/spmm"
