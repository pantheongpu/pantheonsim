#!/usr/bin/env bash
# Texture and surface objects through the ordinary CUDA runtime API.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-textures.$$"
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
     "$root/nvidia/tests/e2e/textures.cu" -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

log="$out.log"
VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" "$out" > "$log" 2>&1
rc=$?
sed 's/^/    /' "$log"
fails=0
for want in "tex1Dfetch wrong: 0" "tex2D wrong: 0" "surface wrong: 0" \
            "linear filtering refused: yes"; do
  grep -q "$want" "$log" || { echo "FAIL: expected '$want'"; fails=1; }
done
(( rc == 0 && fails == 0 )) || { echo "FAIL: texture test"; exit 1; }
echo "RESULT: tex1Dfetch, tex2D over an array, surfaces, and the linear-filter refusal"
