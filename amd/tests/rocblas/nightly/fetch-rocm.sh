#!/usr/bin/env bash
# Unpacks ROCm 7.1 packages from AMD's apt repository into a directory, with
# no install and no root: fetch-rocm.sh DIR build|run
#
#   run    what rocblas-test loads: rocBLAS (and its kernels), hipBLASLt,
#          roctracer's roctx, rocprofiler-register, comgr, and ROCm SMI (the
#          client asks it for clocks the simulator's ROCm SMI does not
#          answer). The HIP and HSA runtimes are the simulator's.
#   build  those, and what builds AMD's rocBLAS test client: the compiler
#          (rocm-llvm), HIP's headers and hipcc, the device libraries,
#          rocBLAS's headers, rocm-cmake.
set -euo pipefail
dir=$1 what=${2:-run}
repo=https://repo.radeon.com/rocm/apt/7.1/pool/main
v=70100-20~24.04_amd64.deb
run=(r/rocblas/rocblas_5.1.0.$v h/hipblaslt/hipblaslt_1.1.0.$v r/roctracer/roctracer_4.1.70100.$v
     r/rocprofiler-register/rocprofiler-register_0.6.0.$v c/comgr/comgr_3.0.0.$v r/rocm-core/rocm-core_7.1.0.$v
     r/rocm-smi-lib/rocm-smi-lib_7.8.0.$v)
build=(r/rocblas-dev/rocblas-dev_5.1.0.$v r/rocm-llvm/rocm-llvm_20.0.0.25425.$v h/hip-dev/hip-dev_7.1.25424.$v
       h/hipcc/hipcc_1.1.1.$v h/hip-runtime-amd/hip-runtime-amd_7.1.25424.$v h/hsa-rocr/hsa-rocr_1.18.0.$v
       h/hsa-rocr-dev/hsa-rocr-dev_1.18.0.$v r/rocm-device-libs/rocm-device-libs_1.0.0.$v
       r/rocm-cmake/rocm-cmake_0.14.0.$v)
pkgs=("${run[@]}")
[[ $what == build ]] && pkgs+=("${build[@]}")
mkdir -p "$dir" "$dir.debs"
for p in "${pkgs[@]}"; do
  f="$dir.debs/$(basename "$p")"
  [[ -s $f ]] || curl -fsSL --retry 3 -o "$f" "$repo/$p"
  dpkg-deb -x "$f" "$dir"
done
rm -rf "$dir.debs"
ls -d "$dir"/opt/rocm-7.1.0
