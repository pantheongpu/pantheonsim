#!/usr/bin/env bash
# What the runtime and driver remember between launches (the table of loaded kernels, module and kernel lookups,
# the function a kernel handle resolved to) must never show in a result or an error: launch_cache.cu launches
# kernels thousands of times with modules loaded and unloaded, devices reset and two devices in play, and checks
# the numbers against the arithmetic of the kernels. It is run with the memory on and with VGPU_LAUNCH_CACHE=0
# (nothing remembered), under both engines, and the three outputs must be the same. Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/launch_cache.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_launch_cache_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
status=0
ref=""
for mode in "VGPU_LAUNCH_CACHE=1 VGPU_SASS=1" "VGPU_LAUNCH_CACHE=0 VGPU_SASS=1" "VGPU_LAUNCH_CACHE=1 VGPU_SASS=0"; do
  result="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=2 VGPU_THREADS=2 \
                $mode LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
  echo "launch cache, $mode: $(tail -n 1 <<<"$result")"
  [[ "$(tail -n 1 <<<"$result")" == "PASS" ]] || { echo "$result"; status=1; }
  if [[ -z "$ref" ]]; then ref="$result"
  elif [[ "$result" != "$ref" ]]; then
    echo "FAIL: output with $mode differs from the first run's"; diff <(echo "$ref") <(echo "$result") || true; status=1
  fi
done
rm -f "$out"
exit "$status"
