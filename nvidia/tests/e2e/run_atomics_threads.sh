#!/usr/bin/env bash
# Device atomics from many blocks on every road the engines take (atomics_threads.cu): VGPU_THREADS 1, 2, 4 and 8, on the SASS and PTX engines.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/atomics_threads.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_atomics_threads_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
status=0
for sass in 1 0; do
  for threads in 1 2 4 8; do
    result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$sass VGPU_THREADS=$threads LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || true
    echo "sass=$sass threads=$threads: $result"
    [[ "$result" == *PASS* ]] || status=1
  done
done
rm -f "$out"
exit $status
