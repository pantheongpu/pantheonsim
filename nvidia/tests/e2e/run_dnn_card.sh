#!/usr/bin/env bash
# A cudnn-frontend test (dnn_attention.cpp, dnn_sdpa_mask.cpp, ...) against NVIDIA's own libcudnn.so.9 on a real
# GPU: the check that what the test expects is what the library does. SKIPs without a card, nvcc, the cuDNN
# headers or library (the pip wheel nvidia-cudnn-cu13, or CUDNN_LIB_DIR).
#
#   run_dnn_card.sh <name>        builds nvidia/tests/e2e/<name>.cpp and runs it on the card
#
# Serialise with the card lock:  flock ~/.cache/pantheonsim-card.lock nvidia/tests/e2e/run_dnn_card.sh dnn_sdpa_mask
set -euo pipefail
name="$1"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
src="$root/nvidia/tests/e2e/$name.cpp"
out="${TMPDIR:-/tmp}/vgpu_dnn_card_${name}_$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
cudnn_inc="$(cudnn_include_dir)" || { echo "SKIP: no cuDNN headers (see scripts/fetch-cudnn-headers.py)"; exit 0; }
fe="$("$root/nvidia/tests/e2e/fetch_cudnn_frontend.sh")" || { echo "SKIP: cudnn-frontend could not be fetched"; exit 0; }
cudnn_lib="${CUDNN_LIB_DIR:-}"
if [[ -z "$cudnn_lib" ]]; then
  for d in "$HOME"/.local/share/torch-cu13*/lib/python3*/site-packages/nvidia/cudnn/lib /usr/lib/x86_64-linux-gnu; do
    compgen -G "$d/libcudnn.so.9*" >/dev/null && cudnn_lib="$d" && break
  done
fi
[[ -n "$cudnn_lib" ]] || { echo "SKIP: no NVIDIA cuDNN 9 library (set CUDNN_LIB_DIR)"; exit 0; }
cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
lib="$cuda_root/lib64"
[[ -d "$lib" ]] || lib="$cuda_root/targets/x86_64-linux/lib"
trap 'rm -f "$out"' EXIT
nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets -Xcompiler -Wno-deprecated-declarations \
     -I"$cudnn_inc" -I"$fe" "$src" -o "$out" -L"$lib" -ldl
result="$(LD_LIBRARY_PATH="$cudnn_lib:$lib" "$out" 2>&1 || true)"
echo "$result"
[[ "$(tail -1 <<< "$result")" == SKIP:* ]] && exit 0
[[ "$(tail -1 <<< "$result")" == "PASS" ]]
