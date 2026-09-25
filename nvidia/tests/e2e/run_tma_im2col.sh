#!/usr/bin/env bash
# TMA's im2col mode against the definition of a convolution (tma_im2col.cu):
# descriptors from cuTensorMapEncodeIm2col made as CUTLASS makes them, loads
# as CuTe issues them, every tile compared exactly. Skips when nvcc cannot
# target sm_90a.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-tma-im2col.$$"
command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libcuda.so.1" ]] || { echo "SKIP: libvgpucuda not built"; exit 0; }
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#cudart_libs[@]} )) || { echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0; }
trap 'rm -f "$out" "$out.cu"' EXIT
echo '__global__ void k() { asm volatile("wgmma.fence.sync.aligned;"); }' > "$out.cu"
if ! nvcc -arch=compute_90a -code=compute_90a -c "$out.cu" -o /dev/null 2>/dev/null; then
  echo "SKIP: this nvcc cannot target sm_90a (needs CUDA 12.0 or later)"; exit 0
fi
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
nvcc -std=c++17 -O1 -cudart shared -arch=compute_90a -code=compute_90a "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/tma_im2col.cu" -L"$shim" -lcuda -o "$out" \
  || { echo "FAIL: tma_im2col.cu did not build"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
echo "$result"
[[ "$(tail -n 1 <<<"$result")" == "PASS" ]] && ! grep -q '^FAIL' <<<"$result"
