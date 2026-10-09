#!/usr/bin/env bash
# 64-bit tests chained through LOP3 predicates; see lop3_pred_chain.cu.
#
# Compiled to real SASS (sm_75), so the SASS interpreter runs it.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/lop3_pred_chain.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_lop3_pred_chain_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi

nvcc -std=c++17 -cudart shared -arch=sm_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || {
  echo "$result"; rm -f "$out"; exit 1;
}
rm -f "$out"
echo "$result"
[[ "$result" == *PASS* ]]
