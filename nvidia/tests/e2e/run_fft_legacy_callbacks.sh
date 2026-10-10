#!/usr/bin/env bash
# cuFFT's legacy callbacks (cufftXtSetCallback), the way NVIDIA documents them: the callbacks are
# device functions of the program itself (-rdc=true), their addresses read out of __device__
# variables and handed to the plan.
#
#   run_fft_legacy_callbacks.sh                 the program linked with libcufft.so's replacement (the shim):
#                                               it must see CUFFT_NOT_IMPLEMENTED, as with NVIDIA's libcufft.so
#   run_fft_legacy_callbacks.sh --card          linked with libcufft_static.a and libculibos.a, run on a real GPU:
#                                               the only route NVIDIA supports, and what the callbacks compute
#   run_fft_legacy_callbacks.sh --card --dynamic   linked with NVIDIA's libcufft.so on a real GPU: NOT_IMPLEMENTED
#
# The static route cannot run on the simulator: NVIDIA's libcufft_static.a carries a CUDA runtime of
# its own, which asks the driver for its internal export table (a simulated driver has none).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/fft_legacy_callbacks.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_fft_legacy_callbacks_$$"
card=0
dynamic=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --dynamic) dynamic=1 ;;
    *) echo "usage: $0 [--card [--dynamic]]" >&2; exit 2 ;;
  esac
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found"; exit 0
fi
trap 'rm -f "$out"' EXIT

if (( card )) && (( ! dynamic )); then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libdir=""
  for d in "$cuda_root/lib64" "$cuda_root/targets/x86_64-linux/lib" /usr/lib/x86_64-linux-gnu /usr/lib64; do
    if [[ -e "$d/libcufft_static.a" && -e "$d/libculibos.a" ]]; then libdir="$d"; break; fi
  done
  [[ -n "$libdir" ]] || { echo "SKIP: libcufft_static.a and libculibos.a not found beside nvcc"; exit 0; }
  nvcc -std=c++17 -rdc=true -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" \
       -L"$libdir" -lcufft_static -lculibos
  "$out"
  exit $?
fi
if (( card )); then
  nvcc -std=c++17 -rdc=true -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" -lcufft
  "$out"
  exit $?
fi

shopt -s nullglob
fft_libs=("$shim"/libcufft.so.[0-9]*)
shopt -u nullglob
(( ${#fft_libs[@]} )) || { echo "SKIP: libvgpucufft not built"; exit 0; }
nvcc -std=c++17 -rdc=true -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcufft
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
result="$(VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || status=$?
echo "$result" | grep -v '^\[vgpu\] .* plan created' || true
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: a kernel was refused or a stub reached"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
