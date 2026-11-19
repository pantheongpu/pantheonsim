#!/usr/bin/env bash
# The resource check (VGPU_LEAK_CHECK) says "clean" when a program gives everything back and names the resource when it does not
# (leak_probe.cu). One process per case; the line is the simulator's own, from the file VGPU_LEAK_CHECK_FILE names.
# Also: VGPU_LEAK_CHECK=strict makes a leaking process exit with status 70 and leaves a clean one alone.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/leak_probe.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_leak_probe_$$"
log="$out.log"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
cublas_libs=("$shim"/libcublas.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
extra=()
if (( ${#cublas_libs[@]} )); then extra=(-DWITH_CUBLAS -lcublas); fi
trap 'rm -f "$out" "$log"' EXIT
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda "${extra[@]}"
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

# (a sanitizer's own leak detector would fail the processes that leak on purpose)
export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0"

status=0
run_case() {   # <case> <mode> <expected ending of the line> <expected exit status>
  local c="$1" mode="$2" want="$3" want_status="$4" got rc=0 line
  : > "$log"
  got="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_THREADS=1 VGPU_GPU=nvidia/rtx3060 VGPU_LEAK_CHECK="$mode" VGPU_LEAK_CHECK_FILE="$log" \
         LD_LIBRARY_PATH="$shim" "$out" "$c" 2>&1)" || rc=$?
  line="$(grep '^vgpu-leak-check: device_bytes' "$log" || true)"
  echo "leak probe, $c ($mode): $line [exit $rc]"
  if [[ "$line" != *"-> $want" ]]; then echo "  FAIL: wanted the line to end '-> $want'"; echo "$got" | sed 's/^/  | /'; status=1; fi
  if [[ "$rc" != "$want_status" ]]; then echo "  FAIL: exit status $rc, wanted $want_status"; echo "$got" | sed 's/^/  | /'; status=1; fi
}
run_case clean       1      "clean"                          0
run_case clean       strict "clean"                          0
run_case leak-device 1      "LEAK device_bytes,device_allocs" 0
run_case leak-driver 1      "LEAK device_bytes,device_allocs" 0
run_case leak-pinned 1      "LEAK pinned_allocs"             0
run_case leak-fd     1      "LEAK fds"                       0
run_case leak-thread 1      "LEAK threads"                   0
run_case leak-device strict "LEAK device_bytes,device_allocs" 70
# Off by default: no line, and nothing changes for a program that leaks.
: > "$log"
env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_LEAK_CHECK_FILE="$log" LD_LIBRARY_PATH="$shim" "$out" leak-device >/dev/null 2>&1
[[ ! -s "$log" ]] || { echo "FAIL: a line was written with VGPU_LEAK_CHECK unset"; status=1; }
[[ "$status" == 0 ]] && echo PASS
exit "$status"
