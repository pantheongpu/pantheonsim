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
# Device-side printf: the program, and its device code with the listing, since
# the path ROCm's device library takes through a hostcall is thousands of
# instructions of its own.
# The same program unoptimized, as CMake builds HIP with no build type: device
# library calls that are not inlined, scalars spilled into lanes and read back
# with every lane off, and the long form of nearly every vector instruction.
"$rocm/bin/hipcc" -O0 -std=c++17 --offload-arch=gfx942 chevron.cpp -o chevron.O0.gfx942
"$rocm/bin/hipcc" -O0 -std=c++17 --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
  -c chevron.cpp -o chevron.O0.gfx942.o
"$rocm/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx942 chevron.O0.gfx942.o |
  sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > chevron.O0.gfx942.dis
echo "wrote $(pwd)/chevron.O0.gfx942, chevron.O0.gfx942.o and its listing ($(wc -l < chevron.O0.gfx942.dis) instructions)"
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 printf.cpp -o printf.gfx942
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
  -c printf.cpp -o printf.gfx942.o
"$rocm/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx942 printf.gfx942.o |
  sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > printf.gfx942.dis
echo "wrote $(pwd)/printf.gfx942, printf.gfx942.o and its listing ($(wc -l < printf.gfx942.dis) instructions)"
# Occupancy, device attributes, and a kernel reaching another device's memory.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 runtime.cpp -o runtime.gfx942
echo "wrote $(pwd)/runtime.gfx942"
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 textures.cpp -o textures.gfx942
echo "wrote $(pwd)/textures.gfx942"
# A cooperative launch, whose work-groups wait for one another at a grid
# barrier: the program, and its device code with the listing.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 cooperative.cpp -o cooperative.gfx942
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 --offload-device-only --no-gpu-bundle-output \
  -c cooperative.cpp -o cooperative.gfx942.o
"$rocm/lib/llvm/bin/llvm-objdump" -d --mcpu=gfx942 cooperative.gfx942.o |
  sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > cooperative.gfx942.dis
echo "wrote $(pwd)/cooperative.gfx942, cooperative.gfx942.o and its listing ($(wc -l < cooperative.gfx942.dis) instructions)"

# Streams that run at once: kernels on two streams handing values to each
# other, events, stream waits, the null stream's ordering, host functions.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 streams.cpp -o streams.gfx942
echo "wrote $(pwd)/streams.gfx942"

# gfx942's 8-bit floats, converted by the device's instructions and checked
# against the header's own software conversion on the host.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 fp8.cpp -o fp8.gfx942
# The same for gfx950, whose 8-bit floats are the OCP formats.
"$rocm/bin/hipcc" -O2 -std=c++17 -DVGPU_FP8_OCP --offload-arch=gfx950 fp8.cpp -o fp8.gfx950
echo "wrote $(pwd)/fp8.gfx942 and fp8.gfx950"

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
