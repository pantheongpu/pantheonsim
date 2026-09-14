#!/usr/bin/env bash
# A full 16x16x16 matmul through the public WMMA API, both B layouts.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/wmma_gemm.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_wmma_gemm_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
# WMMA with an f32 accumulator needs sm_80.
nvcc -std=c++14 -cudart shared -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out")"
rm -f "$out"
echo "wmma 16x16x16 gemm: $result"
[[ "$result" == "PASS" ]]
