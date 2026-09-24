#!/usr/bin/env bash
# stmatrix: the warp stores the 8x8 matrices its registers hold to shared memory,
# in the layout ldmatrix reads them in. Checks where every element lands and that
# the two instructions are inverses.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/stmatrix.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_stmatrix_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
# stmatrix is sm_90 and up, so this one is compiled for Hopper and run on the
# H100 profile. A toolkit too old to know the instruction skips rather than fails.
probe="${TMPDIR:-/tmp}/vgpu_stmatrix_probe_$$.cu"
cat > "$probe" <<'PROBE'
__global__ void k(unsigned* o) {
  __shared__ unsigned short t[64];
  unsigned a = static_cast<unsigned>(__cvta_generic_to_shared(&t[0])), r = 1;
  asm volatile("stmatrix.sync.aligned.m8n8.x1.shared.b16 [%0], {%1};" ::"r"(a), "r"(r));
  if (threadIdx.x == 0) o[0] = t[0];
}
PROBE
if ! nvcc -arch=sm_90 -Wno-deprecated-gpu-targets -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc does not assemble stmatrix (needs a CUDA toolkit with sm_90)"; exit 0
fi
rm -f "$probe"
nvcc -std=c++14 -cudart shared -arch=compute_90 -code=compute_90 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Kept out of a failing command substitution: with `set -e` the shell would exit
# before printing, and a CI log would show the failure with no reason in it.
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
rm -f "$out"
echo "stmatrix layout and the ldmatrix round trip: $result"
[[ "$result" == "PASS" ]]
