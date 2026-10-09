#!/usr/bin/env bash
# cuFFT's LTO callbacks given as LTO-IR: the callbacks are compiled by nvcc -dlto into
# fatbins that hold NVVM bitcode (fft_lto_callback_loads.cu, fft_lto_callback_stores.cu), and
# fft_lto_callbacks.cpp hands them to cufftXtSetJITCallback.
#
#   run_fft_lto_callbacks.sh            run on the simulator, whose cuFFT turns the LTO-IR
#                                       into machine code with the CUDA toolkit's libnvJitLink
#   run_fft_lto_callbacks.sh --card     the same program against NVIDIA's cuFFT on a real GPU
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
dir="${TMPDIR:-/tmp}/vgpu_e2e_fft_lto_callbacks_$$"
card=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    *) echo "usage: $0 [--card]" >&2; exit 2 ;;
  esac
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found"; exit 0
fi
mkdir -p "$dir"
trap 'rm -rf "$dir"' EXIT
src="$root/nvidia/tests/e2e"
for part in loads stores; do
  nvcc -std=c++17 -dc -dlto -arch=lto_86 -fatbin -Wno-deprecated-gpu-targets "$src/fft_lto_callback_$part.cu" -o "$dir/$part.fatbin" \
    || { echo "SKIP: nvcc cannot write LTO-IR (-dlto -arch=lto_86)"; exit 0; }
done
if (( card )); then
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src/fft_lto_callbacks.cpp" -o "$dir/prog" -lcufft
  "$dir/prog" "$dir/loads.fatbin" "$dir/stores.fatbin"
  exit $?
fi
shopt -s nullglob
fft_libs=("$shim"/libcufft.so.[0-9]*)
shopt -u nullglob
(( ${#fft_libs[@]} )) || { echo "SKIP: libvgpucufft not built"; exit 0; }
nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     "$src/fft_lto_callbacks.cpp" -o "$dir/prog" -lcufft
if ! require_shim_libs "$shim" "$dir/prog"; then exit 0; fi
status=0
result="$(VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$dir/prog" "$dir/loads.fatbin" "$dir/stores.fatbin" 2>&1)" || status=$?
if skip="$(grep -m1 '^SKIP:' <<< "$result")"; then echo "$skip"; exit 0; fi
echo "$result" | grep -v '^\[vgpu\] .* plan created' || true
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: a kernel was refused or a stub reached"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
