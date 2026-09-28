#!/usr/bin/env bash
# The runtime and driver APIs in one program, on one simulated machine
# (nvidia/tests/e2e/mixed_apis.cu).
# Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/mixed_apis.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_mixed_apis_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
# The shim's soname major follows the installed toolkit (.so.12 under CUDA 12,
# .so.13 under CUDA 13), so match on whatever was built. Naming one major here
# made this skip -- and so report success without compiling anything -- on
# every host with the other one.
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++14 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Both libraries carry the simulator's core, which a sanitizer build reports as
# an ODR violation when one program loads the two.
if [[ -n "$(shim_sanitizer "$shim")" ]]; then rm -f "$out"; echo "SKIP: a sanitizer build loads two copies of the core"; exit 0; fi
# Kept out of a failing command substitution: with `set -e` the shell would
# exit before printing, and a CI log would show the failure with no reason in it.
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
rm -f "$out"
echo "runtime and driver APIs on one machine: $result"
[[ "$result" == "PASS" ]]
