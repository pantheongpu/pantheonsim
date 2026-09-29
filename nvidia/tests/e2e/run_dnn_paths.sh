#!/usr/bin/env bash
# The library paths PyTorch's CUDA build takes -- cuDNN's graph API,
# BatchNorm training and RNNs, cuBLASLt with a cuBLAS handle -- called
# directly and checked against host references (nvidia/tests/e2e/dnn_paths.cu).
# Skips if nvcc is unavailable.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/dnn_paths.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_dnn_paths_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
[[ -e "$shim/libcudnn.so" ]] || { echo "SKIP: libvgpucudnn not built"; exit 0; }
# cuDNN's headers are vendored (it ships outside the toolkit); the libraries
# linked are the shim's, under their own sonames.
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets \
     -I"$root/nvidia/third_party/cudnn_include" $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" \
     -L"$shim" -lcudnn -lcublasLt -lcublas
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
# Kept out of a failing command substitution: with `set -e` the shell would
# exit before printing, and a CI log would show the failure with no reason in it.
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/h100 LD_LIBRARY_PATH="$shim" "$out" 2>&1 || true)"
rm -f "$out"
echo "$result"
[[ "$(tail -1 <<< "$result")" == "PASS" ]]
