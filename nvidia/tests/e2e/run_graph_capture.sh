#!/usr/bin/env bash
# A graph_capture_<name>.cu program (see graph_capture_common.h): library calls recorded
# into a captured CUDA graph, their descriptors destroyed, and the graph launched with new
# inputs.
#
#   run_graph_capture.sh <name> <libs>            against VirtualGPU's shims
#   run_graph_capture.sh <name> <libs> --card     against NVIDIA's own libraries on a real
#                                                 GPU: the check that the program's
#                                                 expectations are what NVIDIA's libraries
#                                                 do. SKIPs without a card, nvcc, or the
#                                                 libraries (cuDNN 9: the pip wheel
#                                                 nvidia-cudnn-cu13 / CUDNN_LIB_DIR).
#
# <libs> is a comma-separated list of the libraries the program links (cudnn, cublasLt,
# cublas, cufft, cusolver, cusparse, curand, cutensor, nppc, ...), each given by the name
# after "lib". libcuda is always linked.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
name="$1"
libs_csv="${2:-}"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/graph_capture_$name.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_graph_capture_${name}_$$"
card=0
[[ "${3:-}" == --card ]] && card=1
IFS=, read -r -a libs <<< "$libs_csv"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
uses_cudnn=0
for l in "${libs[@]}"; do [[ "$l" == cudnn ]] && uses_cudnn=1; done
inc=()
if (( uses_cudnn )); then
  cudnn_inc="$(cudnn_include_dir)" || { echo "SKIP: no cuDNN headers (see scripts/fetch-cudnn-headers.py)"; exit 0; }
  inc=(-I"$cudnn_inc")
fi
trap 'rm -f "$out"' EXIT
if (( card )); then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  blas_lib="$cuda_root/lib64"
  [[ -d "$blas_lib" ]] || blas_lib="$cuda_root/targets/x86_64-linux/lib"
  link=(-L"$blas_lib")
  ld_path="$blas_lib"
  for l in "${libs[@]}"; do
    if [[ "$l" == cudnn ]]; then
      cudnn_lib="${CUDNN_LIB_DIR:-}"
      if [[ -z "$cudnn_lib" ]]; then
        for d in "$HOME"/.local/share/torch-cu13*/lib/python3*/site-packages/nvidia/cudnn/lib /usr/lib/x86_64-linux-gnu; do
          compgen -G "$d/libcudnn.so.9*" >/dev/null && cudnn_lib="$d" && break
        done
      fi
      [[ -n "$cudnn_lib" ]] || { echo "SKIP: no NVIDIA cuDNN 9 library (set CUDNN_LIB_DIR)"; exit 0; }
      link+=(-L"$cudnn_lib" -l:libcudnn.so.9)
      ld_path="$cudnn_lib:$ld_path"
    elif [[ -n "$l" ]]; then
      link+=(-l"$l")
    fi
  done
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "${inc[@]}" "$src" -o "$out" "${link[@]}" -lcuda
  result="$(LD_LIBRARY_PATH="$ld_path" "$out" 2>&1 || true)"
else
  shopt -s nullglob
  cudart_libs=("$shim"/libcudart.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cudart_libs[@]} == 0 )); then
    echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  link=()
  for l in "${libs[@]}"; do
    [[ -n "$l" ]] || continue
    [[ -e "$shim/lib$l.so" ]] || { echo "SKIP: the shim for lib$l is not built"; exit 0; }
    link+=(-l"$l")
  done
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "${inc[@]}" \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -L"$shim" "${link[@]}" -lcuda
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  # libcuda and libcudart both load: both_shims_env is what a sanitizer build needs for that.
  result="$(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" $(both_shims_env "$shim") "$out" 2>&1 || true)"
fi
echo "$result"
[[ "$(tail -1 <<< "$result")" == "PASS" ]]
