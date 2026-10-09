#!/usr/bin/env bash
# tex.grad bit for bit against an RTX 3060 (texture_grad.cu), on both engines: SASS (TXD) and PTX (VGPU_SASS=0).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-texture-grad.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out" "$out.log"' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/texture_grad.cu" -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

log="$out.log"
for engine in 1 0; do
  echo "engine: $([[ $engine == 1 ]] && echo SASS || echo PTX)"
  VGPU_SASS=$engine VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" > "$log" 2>&1
  rc=$?
  sed 's/^/    /' "$log" | tail -n 8
  [[ $rc == 0 && "$(tail -n 1 "$log")" == PASS ]] || { echo "FAIL: tex.grad"; sed 's/^/    /' "$log" | grep -v " ok$" | head -20; exit 1; }
done
