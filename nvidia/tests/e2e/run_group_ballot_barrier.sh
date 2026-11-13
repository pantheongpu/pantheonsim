#!/usr/bin/env bash
# Sub-warp groups at different copies of a group-masked vote.sync.ballot, then shfl.sync and bar.sync (#574); see
# group_ballot_barrier.cu. Built twice for a simulated H100: as PTX only (compute_90, the PTX interpreter runs it) and
# as SASS (sm_90, the SASS interpreter runs it).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/group_ballot_barrier.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_group_ballot_barrier_$$"
trap 'rm -f "$out.ptx" "$out.sass"' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi

nvcc -std=c++17 -O3 -cudart shared -arch=compute_90 -code=compute_90 \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out.ptx"
nvcc -std=c++17 -O3 -cudart shared -arch=sm_90 \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out.sass"
require_shim_libs "$shim" "$out.ptx" || exit 0
fail=0
for path in ptx sass; do
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out.$path" 2>&1)" || fail=1
  echo "== $path"
  echo "$result"
  [[ "$result" == *PASS* ]] || fail=1
done
exit $fail
