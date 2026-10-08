#!/usr/bin/env bash
# NVTX markers and ranges a program makes while CUPTI listens -- the callbacks
# and the MARKER, MARKER_DATA and NAME records -- compared with what NVIDIA's own
# libcupti gave the same program on an RTX 3060.
#
#   run_cupti_nvtx.sh            run against the shims and compare with
#                                 nvidia/tests/data/cupti_nvtx.expected
#   run_cupti_nvtx.sh --card     run against NVIDIA's libraries on a real GPU
#                                 and compare with the same file: the check that
#                                 the expected output is what hardware prints
#   run_cupti_nvtx.sh --card --update
#                                 rewrite the expected file from the card
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cupti_nvtx.cu"
expected="$root/nvidia/tests/data/cupti_nvtx.expected"
out="${TMPDIR:-/tmp}/vgpu_e2e_cupti_nvtx_$$"
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 [--card [--update]]" >&2; exit 2 ;;
  esac
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
cuda_inc="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
if ! find "$cuda_inc/include" "$cuda_inc/targets/x86_64-linux/include" /usr/include -maxdepth 2 -name nvToolsExt.h 2>/dev/null | grep -q .; then
  echo "SKIP: NVTX headers not found"; exit 0
fi
trap 'rm -f "$out" "$out.txt"' EXIT

if (( card )); then
  # NVIDIA's libraries: the toolkit's own libcupti and the driver's.
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64" /usr/lib/x86_64-linux-gnu; do
    [[ -e "$d/libcupti.so" || -e "$d"/libcupti.so.[0-9]* ]] && libs="$d" && break
  done
  [[ -n "$libs" ]] || { echo "SKIP: no libcupti beside nvcc"; exit 0; }
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" -lcupti -L"$libs"
  NVTX_INJECTION64_PATH="$(ls "$libs"/libcupti.so.[0-9]* 2>/dev/null | head -1)" LD_LIBRARY_PATH="$libs" "$out" > "$out.txt"
else
  shopt -s nullglob
  cupti_libs=("$shim"/libcupti.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cupti_libs[@]} == 0 )); then
    echo "SKIP: libvgpucupti not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  if ! nm -D "${cupti_libs[0]}" 2>/dev/null | grep -q InitializeInjectionNvtx2; then
    echo "SKIP: libvgpucupti built without NVTX (no nvtx3 headers at build time)"; exit 0
  fi
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcupti
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  # The RTX 3060 profile: the trace names the device it ran on.
  VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 NVTX_INJECTION64_PATH="$(ls "$shim"/libcupti.so.[0-9]* | head -1)" LD_LIBRARY_PATH="$shim" "$out" > "$out.txt"
fi

if (( card && update )); then
  cp "$out.txt" "$expected"
  echo "wrote $expected"
  exit 0
fi
if diff -u "$expected" "$out.txt"; then
  echo "NVTX trace matches the card's"
else
  echo "FAIL: the NVTX trace differs from what NVIDIA's CUPTI printed for the same program"
  exit 1
fi
