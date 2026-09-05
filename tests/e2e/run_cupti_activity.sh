#!/usr/bin/env bash
# A CUDA program that profiles itself through CUPTI: registers buffer
# callbacks, enables the activity kinds, runs work, flushes, and checks the
# records describe what it actually did.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/tests/e2e/cupti_activity.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_cupti_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cupti_libs=("$shim"/libcupti.so.[0-9]*)
shopt -u nullglob
if (( ${#cupti_libs[@]} == 0 )); then
  echo "SKIP: libvgpucupti not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++14 -cudart shared -arch=compute_86 -code=compute_86 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     "$src" -o "$out" -lcupti
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" "$out")"
rm -f "$out"
echo "CUPTI activity records: $result"
[[ "$result" == "PASS" ]]
