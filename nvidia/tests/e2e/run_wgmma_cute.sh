#!/usr/bin/env bash
# Hopper's warpgroup MMA against CuTe: every swizzle mode, both majornesses,
# every element type, A from registers, two warpgroups, negation and scale-d.
# CuTe supplies the layouts and descriptors, so a misread of the ISA here would
# disagree with NVIDIA's own code rather than agree with a test written from the
# same misreading.
#
# CuTe is header-only and comes from a pinned CUTLASS release: $CUTLASS_DIR if
# set, otherwise downloaded once into the build directory. Skips when neither
# is possible, or when nvcc cannot target sm_90a.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
src="$root/nvidia/tests/e2e/wgmma_cute.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_wgmma_cute_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
probe="${TMPDIR:-/tmp}/vgpu_wgmma_probe_$$.cu"
echo '__global__ void k() { asm volatile("wgmma.fence.sync.aligned;"); }' > "$probe"
if ! nvcc -arch=compute_90a -code=compute_90a -c "$probe" -o /dev/null 2>/dev/null; then
  rm -f "$probe"
  echo "SKIP: this nvcc cannot target sm_90a (wgmma needs CUDA 12.0 or later)"; exit 0
fi
rm -f "$probe"
. "$root/nvidia/tests/e2e/cutlass_fetch.sh"
# PTX only: the simulator runs PTX, and skipping ptxas halves the compile.
nvcc -std=c++17 -O1 -cudart shared -arch=compute_90a -code=compute_90a --expt-relaxed-constexpr \
     -I "$cutlass/include" $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Kept out of a failing command substitution: with `set -e` the shell would exit
# before printing, and a CI log would show the failure with no reason in it.
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
rm -f "$out"
echo "$result"
[[ "$(tail -n 1 <<<"$result")" == "PASS" ]] && ! grep -q '^FAIL' <<<"$result"
