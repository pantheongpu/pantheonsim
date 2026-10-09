#!/usr/bin/env bash
# cuFFT's legacy callbacks (cufftXtSetCallback), the way NVIDIA documents them:
# the program is linked with libcufft_static.a, so it carries NVIDIA's own cuFFT
# and the callbacks are device functions linked into it (-rdc=true). Run against
# a simulated GPU, it is NVIDIA's cuFFT kernels that VirtualGPU executes.
#
#   run_fft_legacy_callbacks.sh            build with the toolkit's static cuFFT, run on the simulator
#   run_fft_legacy_callbacks.sh --card     the same program on a real GPU: the check that the
#                                          expected results are what NVIDIA's library gives
#
# libcufft.so (what the shim stands in for) answers these calls NOT_IMPLEMENTED,
# which fft_callbacks.cu pins; this test is for the static route.
#
# NVIDIA's static cuFFT looks up the driver entry points it needs by name when
# the first plan is made and fails the plan (CUFFT_INTERNAL_ERROR) if one is
# missing. The test skips, naming them, while the driver shim lacks any.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/fft_legacy_callbacks.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_fft_legacy_callbacks_$$"
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
cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
libdir=""
for d in "$cuda_root/lib64" "$cuda_root/targets/x86_64-linux/lib" /usr/lib/x86_64-linux-gnu /usr/lib64; do
  if [[ -e "$d/libcufft_static.a" && -e "$d/libculibos.a" ]]; then libdir="$d"; break; fi
done
[[ -n "$libdir" ]] || { echo "SKIP: libcufft_static.a and libculibos.a not found beside nvcc"; exit 0; }
trap 'rm -f "$out"' EXIT

if (( card )); then
  nvcc -std=c++17 -rdc=true -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" \
       -L"$libdir" -lcufft_static -lculibos
  "$out"
  exit $?
fi

# The driver entry points NVIDIA's static cuFFT resolves and does not tolerate losing.
need=(cuStreamBeginCapture_v2 cuStreamEndCapture cuStreamWaitValue32 cuStreamBatchMemOp cuGraphInstantiate
      cuGraphLaunch cuGraphDestroy cuGraphExecDestroy cuGraphGetNodes cuGraphNodeGetType
      cuGraphKernelNodeGetParams cuGraphExecKernelNodeSetParams)
shopt -s nullglob
cuda_libs=("$shim"/libcuda.so.[0-9]*)
shopt -u nullglob
if (( ${#cuda_libs[@]} == 0 )); then echo "SKIP: libvgpucuda not built"; exit 0; fi
missing=()
exports="$(nm -D --defined-only "${cuda_libs[0]}" 2>/dev/null | awk '{print $3}' | sed 's/@.*//')"
for sym in "${need[@]}"; do
  grep -qx "$sym" <<< "$exports" || missing+=("$sym")
done
if (( ${#missing[@]} )); then
  echo "SKIP: NVIDIA's static cuFFT needs driver entry points the simulator lacks: ${missing[*]}"; exit 0
fi

nvcc -std=c++17 -rdc=true -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -L"$libdir" -lcufft_static -lculibos
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
result="$(VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || status=$?
echo "$result"
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: a kernel was refused or a stub reached"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
