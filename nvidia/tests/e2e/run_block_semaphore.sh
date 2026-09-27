#!/usr/bin/env bash
# block_semaphore.cu: blocks that wait on each other through a global
# semaphore in CUTLASS's split-K shape, with the grid run on 1, 2, 4 and 8
# host threads. One thread never lets a block wait (the one it waits for has
# already run); more make the wait loop really go round.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
src="$root/nvidia/tests/e2e/block_semaphore.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_block_semaphore_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
if ! nvcc -std=c++17 -O2 -cudart shared -arch=compute_75 -code=compute_75 \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" 2>"$out.log"; then
  cat "$out.log"; rm -f "$out.log"; exit 1
fi
rm -f "$out.log"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
fail=0
for threads in 1 2 4 8; do
  # A small step budget: a wait that never ends fails in seconds, not an hour.
  result="$(VGPU_QUIET=1 VGPU_THREADS=$threads VGPU_MAX_STEPS=20000000 LD_LIBRARY_PATH="$shim" \
            "$out" 2>&1 || true)"
  echo "VGPU_THREADS=$threads: $(head -n 1 <<<"$result")"
  [[ "$(tail -n 1 <<<"$result")" == "PASS" ]] || fail=1
done
rm -f "$out"
exit $fail
