#!/usr/bin/env bash
# Barriers inside device functions nvcc did not inline, built with __noinline__
# at -O3 and again with -G (device_function_barriers.cu; an RTX 3060 passes
# both).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-devfn-barriers.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out".*' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
rc=0
for opt in -O3 -G; do
  bin="$out${opt}"
  "$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
       $opt "$root/nvidia/tests/e2e/device_function_barriers.cu" -o "$bin" 2>"$bin.cc" ||
    { cat "$bin.cc"; echo "FAIL: does not compile with $opt"; exit 1; }
  if ! require_shim_libs "$shim" "$bin"; then exit 0; fi
  VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$bin" > "$bin.log" 2>&1
  r=$?
  echo "  $opt:"
  sed 's/^/    /' "$bin.log"
  [[ $r == 0 && "$(tail -n 1 "$bin.log")" == PASS ]] || { echo "FAIL: barriers in device functions ($opt)"; rc=1; }
done
exit $rc
