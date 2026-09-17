#!/usr/bin/env bash
# NVENC through the bare soname applications dlopen: libvgpunvenc in the shim
# directory, standing in for the driver's libnvidia-encode.so.1.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/nvenc_encode.cu"
inc="${NVENC_INCLUDE:-$root/nvidia/third_party/nvenc_include}"
out="${TMPDIR:-/tmp}/vgpu_e2e_nvenc_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ ! -e "$shim/libnvidia-encode.so.1" ]]; then
  echo "SKIP: libvgpunvenc not built (nvEncodeAPI.h absent at build time)"; exit 0
fi
# An RTX 3060: a card that has an encoder, unlike the datacenter A100.
nvcc -std=c++17 -cudart shared -arch=compute_86 -code=compute_86 -Wno-deprecated-gpu-targets \
     -I "$inc" $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -ldl
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || {
  echo "$result"; rm -f "$out"; exit 1;
}
rm -f "$out"
echo "$result"
[[ "$result" == *PASS* ]]
