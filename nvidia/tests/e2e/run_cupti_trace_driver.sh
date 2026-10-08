#!/usr/bin/env bash
# The trace a pure driver-API program gets of itself through CUPTI -- its
# driver calls entered and returned, the resource and synchronize callbacks, and
# the DRIVER, kernel, copy, fill, wait and stream activity records -- compared
# with the trace NVIDIA's own libcupti gave the same program on an RTX 3060.
# This is run_cupti_trace.sh for the driver domain.
#
#   run_cupti_trace_driver.sh     run against the shims and compare with
#                                 nvidia/tests/data/cupti_trace_driver.expected
#   run_cupti_trace_driver.sh --card
#                                 run against NVIDIA's libraries on a real GPU
#                                 and compare with the same file: the check that
#                                 the expected output is what hardware prints
#   run_cupti_trace_driver.sh --card --update
#                                 rewrite the expected file from the card
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cupti_trace_driver.cu"
expected="$root/nvidia/tests/data/cupti_trace_driver.expected"
out="${TMPDIR:-/tmp}/vgpu_e2e_cupti_trace_driver_$$"
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
trap 'rm -f "$out" "$out.txt"' EXIT

if (( card )); then
  # NVIDIA's libraries: the toolkit's own libcupti and the driver's.
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64" /usr/lib/x86_64-linux-gnu; do
    [[ -e "$d/libcupti.so" || -e "$d"/libcupti.so.[0-9]* ]] && libs="$d" && break
  done
  [[ -n "$libs" ]] || { echo "SKIP: no libcupti beside nvcc"; exit 0; }
  # Linked against the toolkit's stub libcuda; the loader finds the driver's.
  stubs="$libs/stubs"
  [[ -e "$stubs/libcuda.so" ]] || { echo "SKIP: no libcuda stub beside nvcc"; exit 0; }
  nvcc -x c++ -std=c++17 -cudart none -Wno-deprecated-gpu-targets "$src" -o "$out" -lcupti -L"$libs" -lcuda -L"$stubs"
  LD_LIBRARY_PATH="$libs" "$out" > "$out.txt"
else
  shopt -s nullglob
  cupti_libs=("$shim"/libcupti.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cupti_libs[@]} == 0 )); then
    echo "SKIP: libvgpucupti not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  nvcc -x c++ -std=c++17 -cudart none -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcupti -lcuda -L"$shim"
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  # The RTX 3060 profile: the trace names the device it ran on.
  VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out" > "$out.txt"
fi

if (( card && update )); then
  cp "$out.txt" "$expected"
  echo "wrote $expected"
  exit 0
fi
if diff -u "$expected" "$out.txt"; then
  echo "CUPTI driver trace matches the card's"
else
  echo "FAIL: the trace differs from what NVIDIA's CUPTI printed for the same program"
  exit 1
fi
