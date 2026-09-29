#!/usr/bin/env bash
# A program built with nvcc --default-stream per-thread; see per_thread_stream.cu.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/per_thread_stream.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_ptds_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
trap 'rm -f "$out"' EXIT
nvcc -std=c++17 -cudart shared --default-stream per-thread -arch=compute_75 -code=compute_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
require_shim_libs "$shim" "$out" || exit 0
# The build really does call the per-thread names, or the test proves nothing.
if ! nm -D "$out" | grep -q 'cudaMemcpy_ptds'; then
  echo "FAIL: the program does not call cudaMemcpy_ptds (is --default-stream per-thread honoured?)"; exit 1
fi
rc=0
result="$(VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || rc=$?
echo "$result"
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU|symbol lookup error' <<< "$result"; then
  echo "FAIL: the program reached an unimplemented entry point"; exit 1
fi
exit $rc
