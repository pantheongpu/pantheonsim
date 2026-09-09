#!/usr/bin/env bash
# __constant__ and __device__ variables, via cudaMemcpyToSymbol.
#
# The point is the compiler: the unit tests feed hand-written PTX, and nvcc
# emits forms a hand-written test never thinks to write. This is what catches
# the difference.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/tests/e2e/symbols.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_symbols_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
# bf16 arithmetic intrinsics need sm_80; half2 atomicAdd needs sm_60.
nvcc -std=c++17 -cudart shared -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || {
  echo "$result"; rm -f "$out"; exit 1;
}
rm -f "$out"
echo "$result"
[[ "$result" == *PASS* ]]
