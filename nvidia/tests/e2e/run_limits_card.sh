#!/usr/bin/env bash
# Device limits, the persisting-L2 limit and the access policy window on the t4 and a10g profiles against what a Tesla T4 and an A10G
# answered (nvidia/tests/data/card/limits.t4.txt, limits.a10g.txt; nvidia/tests/e2e/limits_probe.cu; driver 595.91, 2026-11-23).
# What this pins: the printf FIFO's default is 32 bytes for every thread the device holds (not 1 MiB), a limit past its largest is refused and
# changes nothing, the persisting-L2 limit starts at 3/16 of the L2 and moves in sixteenths of it, and a device without an access policy window
# (compute capability below 8.0) answers cudaErrorUnsupportedLimit to the limit and cudaErrorInvalidValue to a window.
# Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/limits_probe.cu"
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_limits_card.XXXXXX")"
trap 'rm -rf "$out"' EXIT
if ! command -v nvcc >/dev/null 2>&1; then echo "SKIP: nvcc not found"; exit 0; fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#cudart_libs[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_75 -w -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out/p"
require_shim_libs "$shim" "$out/p" || exit 0
fail=0
for p in t4 a10g; do
  VGPU_QUIET=1 VGPU_GPU=nvidia/$p LD_LIBRARY_PATH="$shim" "$out/p" > "$out/$p.txt" 2>&1 || true
  if diff "$root/nvidia/tests/data/card/limits.$p.txt" "$out/$p.txt" > "$out/$p.diff"; then echo "ok   $p"
  else echo "FAIL $p: differs from the card (card <, simulator >)"; head -12 "$out/$p.diff" | cut -c1-150; fail=1; fi
done
exit $fail
