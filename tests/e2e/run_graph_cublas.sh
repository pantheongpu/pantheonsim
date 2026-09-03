#!/usr/bin/env bash
# End-to-end: a cuBLAS batched GEMM captured into a CUDA graph, compiled with
# nvcc and run against VirtualGPU's shims. Skips cleanly without a toolkit.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/tests/e2e/graph_cublas.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_graph_cublas_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
# Match whatever soname major the shim was built for, as run_e2e.sh does.
shopt -s nullglob
cublas_libs=("$shim"/libcublas.so.[0-9]*)
shopt -u nullglob
if (( ${#cublas_libs[@]} == 0 )); then
  echo "SKIP: libvgpucublas not built (CUDA ABI headers absent at build time)"; exit 0
fi
# compute_86 only: the kernel must reach the interpreter as PTX, not as SASS.
nvcc -std=c++14 -cudart shared -arch=compute_86 -code=compute_86 \
     -Wno-deprecated-gpu-targets "$src" -o "$out" -lcublas
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" "$out")"
rm -f "$out"
echo "captured cuBLAS batched GEMM: $result"
[[ "$result" == "PASS" ]]
