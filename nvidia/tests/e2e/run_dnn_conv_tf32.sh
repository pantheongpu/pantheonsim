#!/usr/bin/env bash
# PyTorch's default FP32 convolution on a card with TF32 tensor cores (nvidia/tests/e2e/dnn_conv_tf32.cu) against what a real card printed: a Tesla T4
# (AWS g4dn.xlarge) and an A10G (AWS g5.xlarge), cuDNN 9.23 (CUDA 13.2 libraries), 2026-11-26, the same on two runs each
# (nvidia/tests/data/dnn_conv_tf32.{t4,a10g}.txt).
#   * the A10G: the default math type gives TF32-class results, FMA math FP32 ones; the heuristics list engines that note TENSOR_CORE (TF32 on FP32
#     inputs) first, and engines that do not. Every NVIDIA profile of compute capability 8.0 and later prints this file (a10g, a100, l4, h100, rtx3060 ...).
#   * the T4 (compute capability 7.5): no TF32 either way, no engine that notes TENSOR_CORE.
# What the files do not hold: the bits (cuDNN's TF32 engines sum in orders this does not model; the simulator rounds the inputs to TF32 and sums in
# double). nvidia/docs/libraries.md.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/dnn_conv_tf32.cu"
data="$root/nvidia/tests/data"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_e2e_dnn_conv_tf32.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
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
cudnn_inc="$(cudnn_include_dir)" || { echo "SKIP: no cuDNN headers (see scripts/fetch-cudnn-headers.py)"; exit 0; }
nvcc -std=c++17 -cudart shared --gpu-architecture=sm_86 -Wno-deprecated-gpu-targets -I"$cudnn_inc" $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$tmp/app" \
     -L"$shim" -lcudnn
if ! require_shim_libs "$shim" "$tmp/app"; then exit 0; fi
fails=0
# run <label> <profile> <pinned file> [env...]
run() {
  local label=$1 gpu=$2 want=$3; shift 3
  env VGPU_QUIET=1 VGPU_GPU="nvidia/$gpu" LD_LIBRARY_PATH="$shim" "$@" timeout 900 "$tmp/app" > "$tmp/out.txt" 2> "$tmp/err.txt" || true
  if diff "$want" "$tmp/out.txt" > "$tmp/d.txt"; then echo "ok   $label"; else
    echo "FAIL $label: output differs from $(basename "$want")"; head -6 "$tmp/d.txt" | cut -c1-240; head -3 "$tmp/err.txt" | cut -c1-200; fails=1
  fi
}
run "t4 profile = the Tesla T4 (no TF32)"                                 t4        "$data/dnn_conv_tf32.t4.txt"
run "a10g profile = the A10G (TF32 by default, FMA math without)"          a10g      "$data/dnn_conv_tf32.a10g.txt"
run "rtx3060 profile = the A10G (compute capability 8.6)"                  rtx3060   "$data/dnn_conv_tf32.a10g.txt"
run "h100 profile = the A10G's summary (compute capability 9.0)"           h100      "$data/dnn_conv_tf32.a10g.txt"
# NVIDIA_TF32_OVERRIDE=0 turns TF32 off for the library (it does for cuBLAS): the A10G profile then prints the T4's file.
run "a10g profile with NVIDIA_TF32_OVERRIDE=0 prints the T4's file"       a10g      "$data/dnn_conv_tf32.t4.txt" NVIDIA_TF32_OVERRIDE=0
(( fails == 0 ))
