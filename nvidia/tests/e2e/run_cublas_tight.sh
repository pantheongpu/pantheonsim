#!/usr/bin/env bash
# A cuBLAS GEMM whose operands are allocated to exactly their extent, so an
# over-read of the last column's padding has nowhere to land.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cublas_tight.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_cublas_tight_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cublas_libs=("$shim"/libcublas.so.[0-9]*)
shopt -u nullglob
if (( ${#cublas_libs[@]} == 0 )); then
  echo "SKIP: libvgpucublas not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++14 -cudart shared -arch=compute_86 -code=compute_86 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     "$src" -o "$out" -lcublas
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Deliberately not quiet: when this fails it is because a read ran past the
# end of an allocation, and the diagnostic naming the address is the point.
# It goes to stderr, so it does not disturb the captured result.
result="$(VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" "$out")"
rm -f "$out"
echo "cuBLAS GEMM on exactly-sized operands: $result"
[[ "$result" == "PASS" ]]
