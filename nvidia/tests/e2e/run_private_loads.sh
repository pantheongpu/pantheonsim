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
# Under a sanitizer build the program is instrumented the same way (the
# sanitizer's runtime has to come first), and loads both libraries, which
# carry a copy of the simulator's core each: both_shims_env.
case "$(shim_sanitizer "$shim")" in
  asan) san=(-fsanitize=address -fsanitize=undefined -fno-omit-frame-pointer) ;;
  tsan) san=(-fsanitize=thread) ;;
  *) san=() ;;
esac
"${CXX:-c++}" -std=c++17 -O1 "${san[@]}" "$src" -o "$out" -ldl
cudart="$(basename "${cudart_libs[0]}")"
result="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out" "$cudart" 2>&1 || true)"
rm -f "$out"
echo "private loads of libcuda and libcudart: $result"
[[ "$result" == "PASS" ]]
