#!/usr/bin/env bash
# A program that traces itself through CUPTI, run against the shims and compared
# with what NVIDIA's own libcupti printed for the same program on an RTX 3060.
#
#   run_cupti_case.sh <case>                 run against the shims and compare
#                                            with nvidia/tests/data/cupti_<case>.expected
#   run_cupti_case.sh <case> --card          run against NVIDIA's libraries on a
#                                            real GPU and compare with the same
#                                            file: the check that the expected
#                                            output is what hardware prints
#   run_cupti_case.sh <case> --card --update rewrite the expected file from the card
#
# Cases (nvidia/tests/e2e/cupti_<case>.cu):
#   trace    runtime-API callbacks, resource and synchronize callbacks, and the
#            kernel, copy, fill, wait, stream, device and runtime records
#   nvtx     NVTX markers, ranges and domains (callbacks and MARKER records)
#   extcorr  external correlation ids pushed around runtime calls
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
case_name="${1:?usage: $0 <trace|nvtx|extcorr> [--card [--update]]}"
shift
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/cupti_${case_name}.cu"
expected="$root/nvidia/tests/data/cupti_${case_name}.expected"
out="${TMPDIR:-/tmp}/vgpu_e2e_cupti_${case_name}_$$"
[[ -f "$src" ]] || { echo "no such case: $case_name" >&2; exit 2; }
card=0; update=0
for a in "$@"; do
  case "$a" in
    --card) card=1 ;;
    --update) update=1 ;;
    *) echo "usage: $0 <case> [--card [--update]]" >&2; exit 2 ;;
  esac
done
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if [[ "$case_name" == nvtx ]]; then
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  if ! find "$cuda_root/include" "$cuda_root/targets/x86_64-linux/include" /usr/include -maxdepth 2 -name nvToolsExt.h 2>/dev/null | grep -q .; then
    echo "SKIP: NVTX headers not found"; exit 0
  fi
fi
trap 'rm -f "$out" "$out.txt"' EXIT

if (( card )); then
  # NVIDIA's libraries: the toolkit's own libcupti and the driver's.
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  libs=""
  for d in "$cuda_root/targets/x86_64-linux/lib" "$cuda_root/lib64" /usr/lib/x86_64-linux-gnu; do
    compgen -G "$d/libcupti.so*" >/dev/null && libs="$d" && break
  done
  [[ -n "$libs" ]] || { echo "SKIP: no libcupti beside nvcc"; exit 0; }
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out" -lcupti -L"$libs"
  # NVTX reaches a tool through the library named here, as a profiler sets it.
  NVTX_INJECTION64_PATH="$(ls "$libs"/libcupti.so.[0-9]* | head -1)" LD_LIBRARY_PATH="$libs" "$out" > "$out.txt"
else
  shopt -s nullglob
  cupti_libs=("$shim"/libcupti.so.[0-9]*)
  shopt -u nullglob
  if (( ${#cupti_libs[@]} == 0 )); then
    echo "SKIP: libvgpucupti not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  if [[ "$case_name" == nvtx ]] && ! nm -D "${cupti_libs[0]}" 2>/dev/null | grep InitializeInjectionNvtx2 >/dev/null; then
    echo "SKIP: libvgpucupti built without NVTX (no nvtx3 headers at build time)"; exit 0
  fi
  nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" -lcupti
  if ! require_shim_libs "$shim" "$out"; then exit 0; fi
  # The RTX 3060 profile: the trace names the device it ran on.
  VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 NVTX_INJECTION64_PATH="${cupti_libs[0]}" \
      LD_LIBRARY_PATH="$shim" "$out" > "$out.txt"
fi

if (( card && update )); then
  cp "$out.txt" "$expected"
  echo "wrote $expected"
  exit 0
fi
if diff -u "$expected" "$out.txt"; then
  echo "CUPTI $case_name trace matches the card's"
else
  echo "FAIL: the $case_name trace differs from what NVIDIA's CUPTI printed for the same program"
  exit 1
fi
