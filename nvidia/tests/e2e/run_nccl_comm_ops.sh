#!/usr/bin/env bash
# NCCL communicator management -- split, shrink, non-blocking communicators,
# scalable init, windows -- and the newer collectives (pre-multiplied sums,
# all-to-all, gather, scatter), in the single-process form over several virtual
# devices. nccl_comm_ops.cu is self-verifying and also passes against NVIDIA's
# libnccl on two physical GPUs, which is where its error codes come from.
#
#   nvidia/tests/e2e/run_nccl_comm_ops.sh [nranks]     (default 4)
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
ranks="${1:-4}"
out="${TMPDIR:-/tmp}/vgpu-nccl-ops.$$"

command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libnccl.so.2" ]] || { echo "SKIP: libvgpunccl not built"; exit 0; }

mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
nvcc -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets -cudart shared \
     $(shim_sanitizer_nvcc_flags "$shim") \
     -I"$root/nvidia/third_party/nccl_include" "$root/nvidia/tests/e2e/nccl_comm_ops.cu" \
     -L"$shim" -lnccl -o "$out/ops" || { echo "FAIL: compile"; exit 1; }
require_shim_libs "$shim" "$out/ops" || exit 0

VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=256 VGPU_DEVICE_COUNT="$ranks" \
VGPU_NCCL_RANKS="$ranks" VGPU_NCCL_DIR="$out/rendezvous" VGPU_NCCL_TIMEOUT=120 \
LD_LIBRARY_PATH="$shim" "$out/ops"
