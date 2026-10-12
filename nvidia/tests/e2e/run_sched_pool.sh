#!/usr/bin/env bash
# Many launches on the pool of host threads (include/vgpu/exec/worker_pool.hpp);
# see sched_pool.cu. The program computes only integers, so it must print the
# same lines at every host-thread count and on both engines: SASS (the default)
# and PTX (VGPU_SASS=0). VGPU_THREADS=1 is the strictly ordered reference.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/sched_pool.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_sched_pool_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
trap 'rm -f "$out" "$out".*' EXIT
# sm_86: its SASS and its PTX, so both engines have something to run.
nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lpthread
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

run() {   # run <engine: sass|ptx> <threads or "">
  local sass=1; [[ "$1" == ptx ]] && sass=0
  local log="$out.$1.${2:-default}"
  local env_threads=()
  [[ -n "$2" ]] && env_threads=(VGPU_THREADS="$2")
  if ! env ${env_threads[@]+"${env_threads[@]}"} VGPU_SASS=$sass VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 \
        LD_LIBRARY_PATH="$shim" "$out" > "$log" 2>&1; then
    echo "FAIL: engine $1, VGPU_THREADS=${2:-default}"; cat "$log"; exit 1
  fi
}
rc=0
for engine in sass ptx; do
  run "$engine" 1
  tail -1 "$out.$engine.1" | grep -qx PASS || { echo "FAIL: $engine at VGPU_THREADS=1"; cat "$out.$engine.1"; exit 1; }
  for t in 2 7 ""; do
    run "$engine" "$t"
    if ! diff -u "$out.$engine.1" "$out.$engine.${t:-default}"; then
      echo "FAIL: $engine at VGPU_THREADS=${t:-default} differs from VGPU_THREADS=1"; rc=1
    fi
  done
done
# The two engines agree too: every value is an integer.
diff -u "$out.sass.1" "$out.ptx.1" || { echo "FAIL: SASS and PTX print different lines"; rc=1; }
(( rc == 0 )) && { cat "$out.sass.1"; echo "PASS: identical at 1, 2, 7 and the default number of host threads, on SASS and PTX"; }
exit $rc
