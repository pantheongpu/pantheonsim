#!/usr/bin/env bash
# sm_XYa cubins on the simulated GPUs that do and do not run them (arch_specific_cubins.cpp): a fatbin of an
# sm_100a cubin and plain compute_100 PTX runs as SASS on a B200 (10.0) and as PTX on a B300 (10.3) and a
# Vera Rubin (10.7); the same without the PTX has nothing to run there; an sm_100f cubin runs on all three;
# none of them runs on a Hopper. Needs an nvcc that targets sm_100a (CUDA 12.8 or later).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
e2e="$root/nvidia/tests/e2e"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
have=("$shim"/libcuda.so.[0-9]*)
shopt -u nullglob
if (( ${#have[@]} == 0 )); then
  echo "SKIP: the driver shim is not built"; exit 0
fi
tmp="${TMPDIR:-/tmp}/vgpu_arch_specific_$$"
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT

flags=(-Wno-deprecated-gpu-targets -fatbin)
if ! nvcc "${flags[@]}" -gencode arch=compute_100a,code=[sm_100a,compute_100a] "$e2e/arch_specific_cubin.cu" -o "$tmp/a_only.fatbin" 2>"$tmp/log"; then
  echo "SKIP: this nvcc does not target sm_100a ($(head -1 "$tmp/log"))"; exit 0
fi
nvcc "${flags[@]}" -gencode arch=compute_100a,code=sm_100a -gencode arch=compute_100,code=compute_100 \
     "$e2e/arch_specific_cubin.cu" -o "$tmp/a_plus_ptx.fatbin"
if ! nvcc "${flags[@]}" -gencode arch=compute_100f,code=[sm_100f,compute_100f] "$e2e/arch_specific_cubin.cu" -o "$tmp/f.fatbin" 2>/dev/null; then
  rm -f "$tmp/f.fatbin"
fi
nvcc -std=c++17 -cudart none $(shim_sanitizer_nvcc_flags "$shim") "$e2e/arch_specific_cubins.cpp" \
     -o "$tmp/arch_specific_cubins" -lcuda
require_shim_libs "$shim" "$tmp/arch_specific_cubins" || exit 0

fails=0
try() {   # <gpu> <fatbin> <run|fail>
  local out
  if out="$(VGPU_QUIET=1 VGPU_GPU="nvidia/$1" LD_LIBRARY_PATH="$shim" "$tmp/arch_specific_cubins" "$tmp/$2.fatbin" "$3" 2>&1)" &&
     [[ "$(tail -n 1 <<< "$out")" == PASS* ]]; then
    echo "ok   $2 on $1: $3"
  else
    echo "FAIL $2 on $1 (wanted $3): $out"; fails=1
  fi
}
for gpu in b200 b300 vr200; do
  try "$gpu" a_plus_ptx run
done
try b300 a_only fail
try vr200 a_only fail
try b200 a_only run
try h100 a_only fail
try h100 a_plus_ptx fail
if [[ -f "$tmp/f.fatbin" ]]; then
  for gpu in b200 b300 vr200; do try "$gpu" f run; done
  try h100 f fail
fi
(( fails == 0 )) && echo "PASS" || echo "FAIL: arch-specific cubins"
exit $fails
