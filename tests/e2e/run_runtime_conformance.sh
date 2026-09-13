#!/usr/bin/env bash
# The CUDA runtime API at its documented edges; see runtime_conformance.cu.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/tests/e2e/runtime_conformance.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_rtconf_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++17 -cudart shared -arch=compute_75 -code=compute_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
rc=0
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || rc=$?
rm -f "$out"
echo "$result"
exit $rc
