#!/usr/bin/env bash
# How many host threads a launch gets (vgpu/exec/worker_pool.hpp, LaunchWidth): a small launch is kept to few threads
# because a worker's hand-off costs more than its blocks, but which thread runs a block, and how many there are, must
# change nothing. pool_cap.cu (thousands of small launches, atomics, other grid sizes of the same kernels, a heavy grid)
# runs at VGPU_THREADS 1, 2, 8 and 16 with the cap on (default) and off (VGPU_POOL_MIN_WORK_US=0), under both engines, with
# the performance counters on. Every run must pass and print the same output, counters included. Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/pool_cap.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_pool_cap_$$"
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
for engine in 1 0; do
  ref=""
  for cap in 100 0; do
    for t in 1 2 8 16; do
      result="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_COUNTERS=1 VGPU_SASS=$engine \
                    VGPU_THREADS=$t VGPU_POOL_MIN_WORK_US=$cap LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
      echo "pool cap, sass=$engine threads=$t min_work_us=$cap: $(grep -E '^(PASS|FAIL)' <<<"$result" | tail -n 1)"
      grep -q '^PASS' <<<"$result" || { echo "$result"; status=1; }
      if [[ -z "$ref" ]]; then ref="$result"
      elif [[ "$result" != "$ref" ]]; then
        echo "FAIL: output differs from the first run's (sass=$engine threads=$t min_work_us=$cap)"; diff <(echo "$ref") <(echo "$result") || true; status=1
      fi
    done
  done
done
rm -f "$out"
exit "$status"
