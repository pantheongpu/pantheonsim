#!/usr/bin/env bash
# Every thread of a block adding one to the same word in place must add one, not one per warp;
# see inplace_scalar_update.cu. Built for sm_75 and run on a T4, so the SASS executor runs it.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/inplace_scalar_update.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_inplace_scalar_update_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
if ! nvcc -std=c++17 -O2 -cudart shared -arch=compute_75 -code=compute_75 -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" 2>"$out.log"; then
  cat "$out.log"; rm -f "$out.log"; exit 1
fi
rm -f "$out.log"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
fail=0
# The default engine (SASS where the card has it), then the PTX interpreter.
for engine in default ptx; do
  extra=()
  [[ $engine == ptx ]] && extra=(VGPU_SASS=0)
  result="$(env VGPU_QUIET=1 VGPU_GPU=nvidia/t4 ${extra[@]+"${extra[@]}"} LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
  echo "engine $engine: $(head -n 1 <<<"$result")"
  [[ "$(tail -n 1 <<<"$result")" == "PASS" ]] || { echo "$result"; fail=1; }
done
rm -f "$out"
exit $fail
