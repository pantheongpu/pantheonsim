#!/usr/bin/env bash
# The runtime functions only the CUDA 13.2 header declares (runtime_132.cu), against the simulated runtime. SKIPs when the
# runtime shim was built with a toolkit older than 13.2 (it then has none of them); the driver's twins are checked against
# the card by e2e_exports_sweep_driver, and these against them.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/runtime_132.cu"
out="${TMPDIR:-/tmp}/vgpu-runtime-132.$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
trap 'rm -f "$out" "$out".*' EXIT
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets -w "${san_flags[@]}" "$src" -o "$out" -Xlinker --no-as-needed -lcuda -L"$shim" \
  || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=2 LD_LIBRARY_PATH="$shim" "$out" > "$out.txt" 2> "$out.err"
rc=$?
if head -1 "$out.txt" | grep -q '^SKIP'; then head -1 "$out.txt"; exit 0; fi
if (( rc != 0 )) || ! tail -1 "$out.txt" | grep -q '^PASS$'; then
  echo "FAIL: the 13.2 runtime functions do not do what they should"; cat "$out.txt" "$out.err"; exit 1
fi
echo "PASS"
