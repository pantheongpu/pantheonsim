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

# HIP's calls beyond the everyday ones -- device flags and UUIDs, contexts, the
# reserved stream handles, callbacks, waiting on memory, every launch form --
# checked against what ROCm's HIP answers.
"$rocm/bin/hipcc" -O2 -std=c++17 -Wno-deprecated-declarations --offload-arch=gfx942 api.cpp -o api.gfx942
echo "wrote $(pwd)/api.gfx942"

# HIP's copies, fills and allocations of every shape, and managed memory's
# advice.
"$rocm/bin/hipcc" -O2 -std=c++17 -Wno-deprecated-declarations --offload-arch=gfx942 memory.cpp -o memory.gfx942
echo "wrote $(pwd)/memory.gfx942"
# Graphs and stream capture: nodes of every kind, executable graphs and what
# changes them, capture across streams and its modes, graph memory.
"$rocm/bin/hipcc" -O2 -std=c++17 -Wno-deprecated-declarations --offload-arch=gfx942 graphs.cpp -o graphs.gfx942
echo "wrote $(pwd)/graphs.gfx942"
# The rest of ROCm's exports: __managed__ variables, libraries, fat
# binaries, the run-time linker and HCC's launch, which load a code object
# built on its own.
"$rocm/bin/hipcc" -O2 --genco --offload-arch=gfx942 exports_kernel.cpp -o exports_kernel.gfx942.co
"$rocm/bin/hipcc" -O2 -std=c++17 -Wno-deprecated-declarations --offload-arch=gfx942 exports.cpp -o exports.gfx942
echo "wrote $(pwd)/exports.gfx942 and exports_kernel.gfx942.co"
# Modules: a code object built for gfx9-4-generic only, which gfx942 runs as
# a member of that family, and the program that loads it.
"$rocm/bin/hipcc" -O2 --genco --offload-arch=gfx9-4-generic modules_kernel.cpp -o modules_kernel.generic.co
"$rocm/bin/hipcc" -O2 -std=c++17 -Wno-deprecated-declarations --offload-arch=gfx942 modules.cpp -o modules.gfx942
echo "wrote $(pwd)/modules.gfx942 and modules_kernel.generic.co"
# Atomics: a flat one landing in LDS, a kernel's and a host thread's on one
# pinned counter, and a float max by compare-and-swap.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 atomics.cpp -o atomics.gfx942
echo "wrote $(pwd)/atomics.gfx942"
# Events shared with another process, which it forks for.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 ipc.cpp -o ipc.gfx942
echo "wrote $(pwd)/ipc.gfx942"
# Every error code's name and text; rocm/errors.expected is the same program
# on ROCm's own libamdhip64.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 errors.cpp -o errors.gfx942
echo "wrote $(pwd)/errors.gfx942"
# How many devices a program is shown, under the visible-devices variables.
"$rocm/bin/hipcc" -O2 --offload-arch=gfx942 visible.cpp -o visible.gfx942
echo "wrote $(pwd)/visible.gfx942"

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
# gfx950's block-scaled matrix instructions, v_prng_b32 and the lane swaps.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx950 gfx950.cpp -o gfx950.gfx950
echo "wrote $(pwd)/gfx950.gfx950"
# The sparse matrix instructions: gfx942's, and gfx950's with K doubled.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 smfmac.cpp -o smfmac.gfx942
"$rocm/bin/hipcc" -O2 -std=c++17 -DVGPU_GFX950 --offload-arch=gfx950 smfmac.cpp -o smfmac.gfx950
echo "wrote $(pwd)/smfmac.gfx942 and smfmac.gfx950"
# MODE's round and denormal modes: as hipcc builds by default, and built to
# flush single-precision denormals.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 numerics.cpp -o numerics.gfx942
"$rocm/bin/hipcc" -O2 -std=c++17 -DVGPU_FLUSH -fgpu-flush-denormals-to-zero --offload-arch=gfx942 numerics.cpp \
  -o numerics.flush.gfx942
echo "wrote $(pwd)/numerics.gfx942 and numerics.flush.gfx942"
# RDNA3 (gfx1100): VOPD, WMMA, double literals; and wave64 DPP.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1100 rdna3.cpp -o rdna3.gfx1100
"$rocm/bin/hipcc" -O2 -std=c++17 -DVGPU_W64 -mwavefrontsize64 --offload-arch=gfx1100 rdna3.cpp -o rdna3.w64.gfx1100
echo "wrote $(pwd)/rdna3.gfx1100 and rdna3.w64.gfx1100"
# RDNA4 (gfx1201): gfx12's WMMA layout, the scalar float unit, split barriers.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1201 rdna4.cpp -o rdna4.gfx1201
echo "wrote $(pwd)/rdna4.gfx1201"
# RDNA2 (gfx1030): SDWA, M0-relative registers, permlane16 and DPP row_share
# and row_xmask, in wave32 and wave64.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx1030 rdna2.cpp -o rdna2.gfx1030
"$rocm/bin/hipcc" -O2 -std=c++17 -DVGPU_W64 -mwavefrontsize64 --offload-arch=gfx1030 rdna2.cpp -o rdna2.w64.gfx1030
echo "wrote $(pwd)/rdna2.gfx1030 and rdna2.w64.gfx1030"
# hipCUB and rocThrust, which are headers compiled into the program, for a
# wave64 target and a wave32 one in one program. Their headers come with
# ROCm's hipcub-dev and rocthrust-dev, and need that ROCm's own hipcc
# (PRIM_ROCM, where it is another ROCm than the one above). rocThrust uses
# libhipcxx's <cuda/std/version> where it finds one, so a machine with CUDA's
# libcu++ in /usr/include builds this with it moved out of the way. rocPRIM
# builds every kernel for each GPU's tuning, so the bundle is compressed and
# the program stripped: 2.5 MB rather than 40.
prim=${PRIM_ROCM:-$rocm}
"$prim/bin/hipcc" -O2 -std=c++17 -isystem "$prim/include" --offload-arch=gfx942 --offload-arch=gfx1100 \
  --offload-compress -s prim.cpp -o prim.all
echo "wrote $(pwd)/prim.all"
# Where a work-group runs (__smid), on every target, in one program.
"$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=gfx942 --offload-arch=gfx950 --offload-arch=gfx90a \
  --offload-arch=gfx1030 --offload-arch=gfx1100 --offload-arch=gfx1201 smid.cpp -o smid.all
echo "wrote $(pwd)/smid.all"
# Arrays, textures and surfaces, for each RDNA generation's image resources.
for arch in gfx1030 gfx1100 gfx1201; do
  "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=$arch images.cpp -o images.$arch
done
echo "wrote $(pwd)/images.gfx1030, images.gfx1100 and images.gfx1201"

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

# Memory-test patterns (walking bits, address in address, checkerboard, moving
# inversions, seeded random, block copies, strides) for each architecture the
# memory test runs on. The checked-in programs were built with the Ubuntu 24.04
# `hipcc` (ROCm 5.7.1, clang 17), which is why they ask for libamdhip64.so.5
# and why gfx950 and gfx1201 are absent: that compiler does not know them.
# (amd/tests/e2e/run_memtest_patterns.sh maps the shim under that name.) With
# ROCm 6 or 7 the same line builds those too.
for arch in gfx942 gfx90a gfx1030 gfx1100; do
  "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=$arch memtest.cpp -o memtest.$arch
done
echo "wrote $(pwd)/memtest.gfx942, .gfx90a, .gfx1030 and .gfx1100"

# Work-group shapes (ids of every dimension, LDS between the waves of a group),
# for the four architectures the e2e test runs. Built with the Ubuntu 24.04
# hipcc (ROCm 5.7.1), whose code objects are version 4 -- the case where
# gfx90a, gfx942 and gfx1100 still keep a work-item's ids packed in v0.
for arch in gfx942 gfx90a gfx1030 gfx1100; do
  "$rocm/bin/hipcc" -O2 -std=c++17 --offload-arch=$arch workgroup.cpp -o workgroup.$arch
done
echo "wrote $(pwd)/workgroup.gfx942, .gfx90a, .gfx1030 and .gfx1100"

# The same programs for gfx1250 (CDNA 5, MI455X; wave32), which the runner executes on a simulated MI455X. Built
# by ROCm 7.2's hipcc, which knows gfx1250; no card has run them.
for src in atomics memory graphs cooperative streams smid errors pointers runtime gfx1250 wmma1250 trload1250 async1250 permlane1250 barrier1250; do
  "$rocm/bin/hipcc" -O2 -std=c++17 -Wno-deprecated-declarations --offload-arch=gfx1250 $src.cpp -o $src.gfx1250
  echo "wrote $(pwd)/$src.gfx1250"
done
