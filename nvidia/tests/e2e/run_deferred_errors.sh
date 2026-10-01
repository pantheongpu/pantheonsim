#!/usr/bin/env bash
# A kernel's fault is reported by the calls after its launch, not by the launch
# (deferred_errors.cu), with the codes an RTX 3060 gives. One process per case:
# the fault kills the context. Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/deferred_errors.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_deferred_errors_$$"
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
# The driver case loads both libraries, whose two copies of the core a
# sanitizer build reports as an ODR violation.
cases=(assert trap address launch-ex graph)
[[ -n "$(shim_sanitizer "$shim")" ]] || cases+=(driver)
status=0
for c in "${cases[@]}"; do
  # Kept out of a failing command substitution, so a failure prints its reason.
  # stdout only: the failed assert prints its message on stderr, as on the card.
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" "$c" 2>/dev/null || true)"
  echo "deferred kernel errors, $c: $result"
  [[ "$(tail -n 1 <<<"$result")" == "PASS" ]] || status=1
done
rm -f "$out"
exit "$status"
