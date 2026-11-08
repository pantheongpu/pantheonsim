#!/usr/bin/env bash
# The driver API's host functions, events and streams behind them; see async_host_events_driver.cu. One process per case, so a
# failure names what broke and a hang is bounded by `timeout`.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/async_host_events_driver.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_async_host_drv_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
trap 'rm -f "$out"' EXIT
nvcc -std=c++17 -cudart shared -arch=compute_75 -code=compute_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda -lpthread
require_shim_libs "$shim" "$out" || exit 0
rc=0
for c in not_ready elapsed sync order wait_event legacy hang idle; do
  echo "== $c"
  result="$(env $(both_shims_env "$shim") VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" timeout 120 "$out" "$c" 2>&1)" || rc=$?
  echo "$result"
  if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU|symbol lookup error' <<< "$result"; then
    echo "FAIL: the program reached an unimplemented entry point"; exit 1
  fi
done
exit $rc
