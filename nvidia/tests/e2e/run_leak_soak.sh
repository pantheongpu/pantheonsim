#!/usr/bin/env bash
# Growth over repeats (leak_soak.cu): the clean workload must not grow device memory, descriptors, threads, resident memory or
# mappings, and each deliberate leak must be named. One host thread runs the blocks (VGPU_THREADS=1): the worker pool grows its threads
# when the launches are long enough to want them, which is not a leak and moves with the speed of the machine. A check that has never failed proves nothing, so the leaks are the other half.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/leak_soak.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_leak_soak_$$"
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
trap 'rm -f "$out"' EXIT
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcuda "${extra[@]}"
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

# A sanitizer build keeps freed memory in a quarantine, so resident memory and mappings say nothing there; its leak detector would
# also fail the processes that leak on purpose.
export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_leaks=0"
export TSAN_OPTIONS="${TSAN_OPTIONS:+$TSAN_OPTIONS:}report_thread_leaks=0"
sanitized="$(shim_sanitizer "$shim")"
[[ -n "$sanitized" ]] && export SOAK_NO_RSS=1

status=0
run_case() {   # <leak flag or ""> <expected last line> <expected exit status>
  local flag="$1" want="$2" want_status="$3" got rc=0
  got="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_THREADS=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" $flag 2>&1)" || rc=$?
  echo "leak soak ${flag:-clean}: $(tail -n 2 <<<"$got" | tr '\n' ' ')[exit $rc]"
  # (a leaked thread holds a stack, so resident memory and mappings grow with it: the line names the resource, among others)
  if [[ "$(tail -n 1 <<<"$got")" != *"$want"* || "$rc" != "$want_status" ]]; then
    echo "  FAIL: wanted '$want' and exit $want_status"; echo "$got" | sed 's/^/  | /'; status=1
  fi
}
run_case ""               "PASS"                 0
run_case "--leak=device"  "LEAK device_free"     1
if [[ -z "$sanitized" ]]; then
  run_case "--leak=host"    "LEAK rss"             1
  run_case "--leak=pinned"  "LEAK rss"             1
fi
run_case "--leak=fd"      "LEAK fds"             1
run_case "--leak=thread"  "LEAK threads"         1
[[ "$status" == 0 ]] && echo PASS
exit "$status"
