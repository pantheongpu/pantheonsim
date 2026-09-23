#!/usr/bin/env bash
# Device memory shared between two processes. Two processes are the point: a
# handle that only works inside one process would pass any single-process test
# and be useless for what the API is for.
#
# Both run on the same simulated machine (the same VGPU_TELEMETRY_PATH), as two
# processes on one host share a GPU.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-ipc.$$"

command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#cudart_libs[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }

mkdir -p "$out"
trap 'rm -rf "$out"' EXIT
nvcc -std=c++14 -arch=sm_86 -Wno-deprecated-gpu-targets -cudart shared \
     $(shim_sanitizer_nvcc_flags "$shim") "$root/nvidia/tests/e2e/ipc.cu" \
     -o "$out/ipc" || { echo "FAIL: compile"; exit 1; }
if ! require_shim_libs "$shim" "$out/ipc"; then exit 0; fi

run() { VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=256 \
        VGPU_TELEMETRY_PATH="$out/run" VGPU_STATE_DIR="$out/state" \
        LD_LIBRARY_PATH="$shim" "$out/ipc" "$@"; }

run export "$out/handle" > "$out/export.log" 2>&1 &
exporter=$!
run import "$out/handle" > "$out/import.log" 2>&1 &
importer=$!
rc=0
wait "$exporter" || rc=1
wait "$importer" || rc=1
cat "$out/export.log" "$out/import.log"
grep -q "PASS export" "$out/export.log" || rc=1
grep -q "PASS import" "$out/import.log" || rc=1
[[ $rc -eq 0 ]] && echo "IPC between two processes: PASS"
exit $rc
