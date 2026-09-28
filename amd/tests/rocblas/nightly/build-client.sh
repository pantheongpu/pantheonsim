#!/usr/bin/env bash
# Builds AMD's rocblas-test (the rocBLAS test client, MIT licensed) from its
# rocm-7.1.0 source against the rocBLAS release it tests:
#
#   build-client.sh ROCM_DIR OUT_DIR
#
# ROCM_DIR holds fetch-rocm.sh's "build" packages. OUT_DIR gets rocblas-test,
# its data file, and the one library it loads from outside the run packages
# (the compiler's OpenMP runtime).
set -euo pipefail
root=$(cd "$1" && pwd)/opt/rocm-7.1.0
out=$(mkdir -p "$2" && cd "$2" && pwd)
work=${TMPDIR:-/tmp}/rocblas-client
rm -rf "$work" && mkdir -p "$work"
git clone -q --depth 1 --branch rocm-7.1.0 --filter=blob:none --sparse https://github.com/ROCm/rocm-libraries "$work/src"
git -C "$work/src" sparse-checkout set projects/rocblas
# ROCm's CMake helpers, which the client's install rules use.
printf 'list(APPEND CMAKE_MODULE_PATH "%s/share/rocmcmakebuildtools/cmake")\ninclude(ROCMInstallTargets)\n' "$root" > "$work/init.cmake"
export ROCM_PATH=$root HIP_PATH=$root HIP_CLANG_PATH=$root/lib/llvm/bin HIP_DEVICE_LIB_PATH=$root/amdgcn/bitcode PATH=$root/bin:$PATH
cmake -S "$work/src/projects/rocblas/clients" -B "$work/build" -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCHMARKS=OFF -DBUILD_CLIENTS_SAMPLES=OFF -DLINK_BLIS=OFF \
  -DBUILD_FORTRAN_CLIENTS=OFF -DBLA_VENDOR=OpenBLAS \
  -DCMAKE_CXX_COMPILER="$root/lib/llvm/bin/amdclang++" -DCMAKE_C_COMPILER="$root/lib/llvm/bin/amdclang" \
  -DCMAKE_PREFIX_PATH="$root" \
  -DCMAKE_CXX_FLAGS="-I$root/include/rocblas/internal -I$root/include/rocblas -I/usr/include/x86_64-linux-gnu/openblas-pthread" \
  -DCMAKE_PROJECT_INCLUDE="$work/init.cmake" -DGPU_TARGETS=gfx942 -DAMDGPU_TARGETS=gfx942 -DROCM_PATH="$root"
cmake --build "$work/build" -j "${JOBS:-$(nproc)}"
cp "$work/build/staging/rocblas-test" "$out/"
cp "$work/build/staging/rocblas_gtest.data" "$out/"
cp -L "$root/llvm/lib/libomp.so" "$out/"
ls -la "$out"
