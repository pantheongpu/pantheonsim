#!/usr/bin/env bash
# atomicAdd with subnormal operands (tools/tcpapers/atomic_subnormal.cu): float adds in global memory flush subnormal inputs and
# results, float adds in shared memory and double adds anywhere do not, with and without -ftz. The pinned lines are an RTX 3060's
# (nvidia/tests/data/atomic_subnormal.rtx3060.txt, driver 610.88, nvcc 13.0.88, 2026-11-19); both engines must print them.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/tools/tcpapers/atomic_subnormal.cu"
want="$root/nvidia/tests/data/atomic_subnormal.rtx3060.txt"
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_atomic_subnormal.XXXXXX")"
trap 'rm -rf "$out"' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
fails=0
for ftz in false true; do
  nvcc -std=c++17 -O1 -arch=sm_86 -cudart shared -ftz=$ftz -w $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out/p_$ftz"
  if ! require_shim_libs "$shim" "$out/p_$ftz"; then exit 0; fi
  for eng in sass ptx; do
    VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$([[ $eng == sass ]] && echo 1 || echo 0) LD_LIBRARY_PATH="$shim" \
      "$out/p_$ftz" > "$out/o.txt" 2> "$out/e.txt" || { echo "FAIL -ftz=$ftz $eng: the program failed"; head -3 "$out/e.txt"; fails=1; continue; }
    if diff "$want" "$out/o.txt" > "$out/d.txt"; then echo "ok   -ftz=$ftz $eng"; else echo "FAIL -ftz=$ftz $eng: differs from the card's lines"; head -8 "$out/d.txt" | cut -c1-160; fails=1; fi
  done
done
exit $fails
