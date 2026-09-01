#!/usr/bin/env bash
# End-to-end: compile an unmodified CUDA program with nvcc (shared cudart) and
# run it against VirtualGPU's libcudart. Skips cleanly if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
shim="$root/build/shim"
src="$root/tests/e2e/vector_add.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_vecadd_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ ! -e "$shim/libcudart.so.13" ]]; then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++14 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out"
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out")"
rm -f "$out"
echo "vectorAdd via libvgpucudart: $result"
[[ "$result" == "PASS" ]]
