#!/usr/bin/env bash
# The single-process form of NCCL: one thread issues every rank's call between
# ncclGroupStart and ncclGroupEnd, over several virtual devices. That pattern
# would deadlock any implementation that blocked inside the collective, so it is
# worth its own test at a rank count the differential run cannot reach.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
ranks="${1:-4}"
out="${TMPDIR:-/tmp}/vgpu-nccl-group.$$"

command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libnccl.so.2" ]] || { echo "SKIP: libvgpunccl not built"; exit 0; }

mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
nvcc -std=c++14 -arch=sm_86 -Wno-deprecated-gpu-targets -cudart shared \
     $(shim_sanitizer_nvcc_flags "$shim") \
     -I"$root/nvidia/third_party/nccl_include" "$root/nvidia/tests/conformance/nccl_collectives.cu" \
     -L"$shim" -lnccl -o "$out/group" || { echo "FAIL: compile"; exit 1; }

VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=256 VGPU_DEVICE_COUNT="$ranks" \
VGPU_NCCL_RANKS="$ranks" VGPU_NCCL_DIR="$out/rendezvous" VGPU_NCCL_TIMEOUT=120 \
LD_LIBRARY_PATH="$shim" "$out/group"
