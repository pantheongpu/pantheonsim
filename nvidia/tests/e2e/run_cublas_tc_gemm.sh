#!/usr/bin/env bash
# cuBLAS tensor-core GEMMs (FP16, BF16, TF32) on the simulator against real cards: nvidia/tests/e2e/cublas_tc_gemm.cu multiplies random
# matrices (fixed seed; a few shapes, transposes, alpha and beta, FP16/BF16 into FP16 and FP32, TF32 through the math mode) and prints a
# checksum of every result. The pinned files are what a real card printed, twice, the same both times (CUDA 13.0 toolkit, cuBLAS 13.x,
# 2026-11-23): cublas_tc_gemm.t4.txt (Tesla T4, AWS g4dn.xlarge, sm_75 build) and cublas_tc_gemm.a10g.txt (A10G, AWS g5.xlarge, sm_86
# build). A tensor core adds a chunk of products at once, so a half-precision product is not the sequential FP32 sum the host loops make
# (docs/matrix-numerics.md): the simulator's cuBLAS must print the card's bytes, not those of its own host loop.
#   * nvidia/t4 prints the T4's file: FP16 on the chunk-of-8 rule, BF16 as the ordinary FP32 kernel a Turing card runs (it has no BF16
#     tensor core), and a TF32 request answered with the plain FP32 product (it has no TF32 either).
#   * nvidia/a10g and nvidia/rtx3060 (sm_86) print the A10G's file.
#   * with VGPU_TC_EXACT=0 (the sequential sums of the releases before the tensor-core rules) the T4 profile does NOT print the T4's
#     file: the check is made by the rule and by nothing else.
# What the files do not hold, and why, is in the header of cublas_tc_gemm.cu (products cuBLAS splits over K).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cublas_tc_gemm.cu"
data="$root/nvidia/tests/data"
tmp="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_e2e_cublas_tc_gemm.XXXXXX")"
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
if [[ ! -e "$shim/libcublas.so" && -z "$(ls "$shim"/libcublas.so.* 2>/dev/null)" ]]; then
  echo "SKIP: libvgpucublas not built"; exit 0
fi
san=$(shim_sanitizer_nvcc_flags "$shim")
nvcc -std=c++14 -cudart shared -arch=sm_75 -Wno-deprecated-gpu-targets $san "$src" -o "$tmp/t4" -lcublas \
  || { echo "FAIL: cublas_tc_gemm.cu does not build for sm_75"; exit 1; }
nvcc -std=c++14 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets $san "$src" -o "$tmp/a10g" -lcublas \
  || { echo "FAIL: cublas_tc_gemm.cu does not build for sm_86"; exit 1; }
if ! require_shim_libs "$shim" "$tmp/t4"; then exit 0; fi
fails=0
# run <label> <binary> <profile> <pinned file> <expect: same|differ> [env...]
run() {
  local label=$1 bin=$2 gpu=$3 want=$4 expect=$5; shift 5
  env VGPU_QUIET=1 VGPU_GPU="nvidia/$gpu" LD_LIBRARY_PATH="$shim" "$@" timeout 900 "$bin" > "$tmp/out.txt" 2> "$tmp/err.txt" || true
  if diff "$want" "$tmp/out.txt" > "$tmp/d.txt"; then got=same; else got=differ; fi
  if [[ $got == "$expect" ]]; then echo "ok   $label"; else
    echo "FAIL $label: output $got, wanted $expect, against $(basename "$want")"; head -6 "$tmp/d.txt" | cut -c1-200; head -3 "$tmp/err.txt" | cut -c1-200; fails=1
  fi
}
t4_pin="$data/cublas_tc_gemm.t4.txt"
[[ -n "${CUBLAS_TC_T4_PIN:-}" ]] && t4_pin="$CUBLAS_TC_T4_PIN"    # for the mutation check: another pin must fail
run "cuBLAS tensor-core GEMMs, t4 profile = the Tesla T4"               "$tmp/t4"   t4      "$t4_pin"                          same
run "cuBLAS tensor-core GEMMs, a10g profile = the A10G"                 "$tmp/a10g" a10g    "$data/cublas_tc_gemm.a10g.txt"    same
run "cuBLAS tensor-core GEMMs, rtx3060 profile = the A10G (both sm_86)" "$tmp/a10g" rtx3060 "$data/cublas_tc_gemm.a10g.txt"    same
run "cuBLAS tensor-core GEMMs, t4 profile with VGPU_TC_EXACT=0 is not the T4" "$tmp/t4" t4  "$t4_pin"                          differ VGPU_TC_EXACT=0
(( fails == 0 ))
