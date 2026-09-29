#!/usr/bin/env bash
# libcuda and libcudart loaded privately, one after the other, share one
# simulated machine (nvidia/tests/e2e/private_loads.cpp). Needs only a C++
# compiler: the program looks everything up by name.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/private_loads.cpp"
out="${TMPDIR:-/tmp}/vgpu_e2e_private_loads_$$"
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )) || [[ ! -e "$shim/libcuda.so.1" ]]; then
  echo "SKIP: libvgpucudart or libvgpucuda not built"; exit 0
fi
# Both libraries carry the simulator's core, which a sanitizer build reports as
# an ODR violation when one program loads the two.
if [[ -n "$(shim_sanitizer "$shim")" ]]; then echo "SKIP: a sanitizer build loads two copies of the core"; exit 0; fi
"${CXX:-c++}" -std=c++17 -O1 "$src" -o "$out" -ldl
cudart="$(basename "${cudart_libs[0]}")"
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out" "$cudart" 2>&1 || true)"
rm -f "$out"
echo "private loads of libcuda and libcudart: $result"
[[ "$result" == "PASS" ]]
