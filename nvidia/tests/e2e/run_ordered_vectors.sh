#!/usr/bin/env bash
# Ordered vector accesses are single accesses across blocks; see ordered_vectors.cu.
#
# It needs the grid's blocks on more than one host thread, which is the default
# (VGPU_THREADS unset) on any machine with two or more cores.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/ordered_vectors.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_ordvec_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
# .relaxed needs sm_70; a T4 is sm_75.
nvcc -std=c++17 -cudart shared -arch=compute_75 -code=compute_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || {
  echo "$result"; rm -f "$out"; exit 1;
}
rm -f "$out"
echo "$result"
[[ "$result" == *PASS* ]]
