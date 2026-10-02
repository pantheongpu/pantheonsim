#!/usr/bin/env bash
# __managed__ variables in a module the driver API loads, from a cubin, a
# fatbin and PTX (managed_module_load.cpp), on a simulated RTX 3060 -- the card
# the expected answers come from, and the sm_86 the cubin is built for.
#
# Until SASS execution is in the tree (PR #255; include/vgpu/sass/cubin.hpp
# marks it), a cubin cannot be loaded, so VGPU_CUBIN_PENDING asks only that it
# be refused cleanly. Once it is, the cubin is held to every check.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
e2e="$root/nvidia/tests/e2e"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
have=("$shim"/libcuda.so.[0-9]*)
shopt -u nullglob
if (( ${#have[@]} == 0 )); then
  echo "SKIP: the driver shim is not built"; exit 0
fi
tmp="${TMPDIR:-/tmp}/vgpu_managed_module_$$"
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT

nvcc -cubin -arch=sm_86 "$e2e/managed_module.cu" -o "$tmp/managed.cubin"
nvcc -fatbin -arch=sm_86 "$e2e/managed_module.cu" -o "$tmp/managed.fatbin"
nvcc -ptx -arch=sm_86 "$e2e/managed_module.cu" -o "$tmp/managed.ptx"
nvcc -std=c++17 -cudart none $(shim_sanitizer_nvcc_flags "$shim") \
     "$e2e/managed_module_load.cpp" -o "$tmp/managed_module_load" -lcuda
require_shim_libs "$shim" "$tmp/managed_module_load" || exit 0

[[ -e "$root/include/vgpu/sass/cubin.hpp" ]] || export VGPU_CUBIN_PENDING=1
status=0
result="$(VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$tmp/managed_module_load" \
          "cubin:$tmp/managed.cubin" "fatbin:$tmp/managed.fatbin" "ptx:$tmp/managed.ptx" 2>&1)" ||
  status=$?
echo "$result"
if grep -qE 'is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: the program reached an unimplemented entry point"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
