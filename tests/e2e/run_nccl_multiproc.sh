#!/usr/bin/env bash
# One rank per process against VirtualGPU's libnccl. Each rank is its own
# process with its own virtual device, so the collectives have to go through the
# file-backed transport rather than a shared heap.
#
#   tests/e2e/run_nccl_multiproc.sh [nranks]     (default 4)
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
nranks="${1:-4}"
out="${TMPDIR:-/tmp}/vgpu-nccl-mp.$$"

command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libnccl.so.2" ]] || { echo "SKIP: libvgpunccl not built"; exit 0; }

mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
nvcc -std=c++14 -arch=sm_86 -Wno-deprecated-gpu-targets -cudart shared \
     $(shim_sanitizer_nvcc_flags "$shim") \
     -I"$root/third_party/nccl_include" "$root/tests/e2e/nccl_multiproc.cu" \
     -L"$shim" -lnccl -o "$out/mp" || { echo "FAIL: compile"; exit 1; }

pids=()
for ((r = 0; r < nranks; ++r)); do
  VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=256 \
  VGPU_NCCL_DIR="$out/rendezvous" VGPU_NCCL_TIMEOUT=120 \
  LD_LIBRARY_PATH="$shim" "$out/mp" "$r" "$nranks" "$out/id" &
  pids+=($!)
done

fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[[ $fail -eq 0 ]] && echo "$nranks ranks, $nranks processes: allreduce + allgather + send/recv all correct"
exit $fail
