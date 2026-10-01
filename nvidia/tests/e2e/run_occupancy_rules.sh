#!/usr/bin/env bash
# cudaOccupancyMaxActiveBlocksPerMultiprocessor against CUDA's own occupancy
# calculator (cuda_occupancy.h) on several simulated GPUs, whose allocation
# rules differ by compute capability (occupancy_rules.cu; an RTX 3060 agrees
# with the calculator in every case).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-occupancy.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out" "$out.log"' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_75 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/occupancy_rules.cu" -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

rc=0
for gpu in nvidia/rtx3060 nvidia/a100 nvidia/h100 nvidia/t4 nvidia/b200; do
  VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" "$out" > "$out.log" 2>&1
  r=$?
  echo "  $gpu: $(tail -n 2 "$out.log" | head -n 1)"
  if [[ $r != 0 || "$(tail -n 1 "$out.log")" != PASS ]]; then sed 's/^/    /' "$out.log"; rc=1; fi
done
(( rc == 0 )) || echo "FAIL: occupancy differs from cuda_occupancy.h"
exit $rc
