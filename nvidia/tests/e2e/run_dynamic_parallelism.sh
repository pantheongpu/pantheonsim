#!/usr/bin/env bash
# Dynamic parallelism: kernels launching kernels, built as a program builds
# them (-rdc=true, linked with cudadevrt), checked exactly
# (dynamic_parallelism.cu). Skips when nvcc cannot target sm_90 (CDP2 needs
# CUDA 12).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
src="$root/nvidia/tests/e2e/dynamic_parallelism.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_dynamic_parallelism_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
if ! nvcc -std=c++17 -O1 -rdc=true -cudart shared -arch=compute_90 -code=compute_90 \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" 2>"$out.log"; then
  if grep -q "compute_90" "$out.log"; then
    rm -f "$out.log"; echo "SKIP: this nvcc cannot target sm_90 (needs CUDA 12.0 or later)"; exit 0
  fi
  cat "$out.log"; rm -f "$out.log"; exit 1
fi
rm -f "$out.log"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Kept out of a failing command substitution, so a failure prints its reason.
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
rm -f "$out"
echo "$result"
[[ "$(tail -n 1 <<<"$result")" == "PASS" ]] && ! grep -q '^FAIL' <<<"$result"
