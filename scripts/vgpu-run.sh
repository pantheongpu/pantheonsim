#!/usr/bin/env bash
# vgpu-run.sh — run an unmodified CUDA program on a VirtualGPU device.
#
#   scripts/vgpu-run.sh --gpu nvidia/h100 ./my_cuda_app [args...]
#
# The application must be linked against the shared CUDA runtime (nvcc
# -cudart shared) so it resolves libcudart at load time; VirtualGPU's shim is
# then placed ahead of the real one via LD_LIBRARY_PATH. No GPU is used.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
shim="$here/build/shim"
gpu="nvidia/h100"
vram=""
args=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu) gpu="$2"; shift 2 ;;
    --vram-mb) vram="$2"; shift 2 ;;
    --) shift; args+=("$@"); break ;;
    *) args+=("$1"); shift ;;
  esac
done
if [[ ! -e "$shim/libcudart.so.13" ]]; then
  echo "error: VirtualGPU runtime shim not built. Run ./scripts/build.sh first" >&2
  echo "       (libvgpucudart requires the CUDA toolkit headers at build time)." >&2
  exit 1
fi
export VGPU_GPU="$gpu"
[[ -n "$vram" ]] && export VGPU_VRAM_MB="$vram"
export LD_LIBRARY_PATH="$shim${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "${args[@]}"
