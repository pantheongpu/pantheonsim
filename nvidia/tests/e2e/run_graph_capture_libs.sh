#!/usr/bin/env bash
# cuBLASLt's matmul and cuDNN's graph-API convolution inside a captured CUDA
# graph (nvidia/tests/e2e/graph_capture_libs.cu), run against VirtualGPU's shims.
#
#   run_graph_capture_libs.sh            against the shims
#   run_graph_capture_libs.sh --card     against NVIDIA's own libraries on a real GPU: the
#                                        check that the program's expectations are what
#                                        NVIDIA's libraries do. Needs cuDNN 9 (the pip
#                                        wheel nvidia-cudnn-cu13 / CUDNN_LIB_DIR names it)
#                                        and a card; SKIPs without them.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/graph_capture_libs.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_graph_capture_libs_$$"
card=0
[[ "${1:-}" == --card ]] && card=1
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
cudnn_inc="$(cudnn_include_dir)" || { echo "SKIP: no cuDNN headers (see scripts/fetch-cudnn-headers.py)"; exit 0; }
trap 'rm -f "$out"' EXIT
if (( card )); then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  cudnn_lib="${CUDNN_LIB_DIR:-}"
  if [[ -z "$cudnn_lib" ]]; then
    for d in "$HOME"/.local/share/torch-cu13*/lib/python3*/site-packages/nvidia/cudnn/lib /usr/lib/x86_64-linux-gnu; do
      compgen -G "$d/libcudnn.so.9*" >/dev/null && cudnn_lib="$d" && break
    done
  fi
  [[ -n "$cudnn_lib" ]] || { echo "SKIP: no NVIDIA cuDNN 9 library (set CUDNN_LIB_DIR)"; exit 0; }
  blas_lib="$cuda_root/lib64"
  [[ -d "$blas_lib" ]] || blas_lib="$cuda_root/targets/x86_64-linux/lib"
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets -I"$cudnn_inc" "$src" -o "$out" \
       -L"$cudnn_lib" -L"$blas_lib" -l:libcudnn.so.9 -lcublasLt -lcuda
  result="$(LD_LIBRARY_PATH="$cudnn_lib:$blas_lib" "$out" 2>&1 || true)"
else
  shopt -s nullglob
  cudart_libs=("$shim"/libcudart.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cudart_libs[@]} == 0 )); then
    echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  [[ -e "$shim/libcudnn.so" && -e "$shim/libcublasLt.so" ]] || { echo "SKIP: libvgpucudnn or libvgpucublaslt not built"; exit 0; }
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets -I"$cudnn_inc" \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -L"$shim" -lcudnn -lcublasLt -lcuda
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  # libcuda and libcudart both load: both_shims_env is what a sanitizer build needs for that.
  result="$(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") "$out" 2>&1 || true)"
fi
echo "$result"
[[ "$(tail -1 <<< "$result")" == "PASS" ]]
