#!/usr/bin/env bash
# nvJitLink and nvFatbin end to end: a self-checking program, built with nvcc
# and run against the shims on a simulated A100.
#
#   run_jit_link.sh nvjitlink_paths|nvfatbin_paths
#
# Beside the program this builds what only nvcc can make, from
# jitlink_lib.cu: a host object carrying relocatable PTX (-dc), a static
# library of it, and an LTO-IR fatbin -- the inputs nvJitLink takes from
# files and nvFatbinAddReloc reads -- and passes them on the command line.
#
# As in run_lib_check.sh, a call into a stub or a kernel the simulator
# refused fails the test even if the program's own checks passed.
set -euo pipefail
name="$1"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
e2e="$root/nvidia/tests/e2e"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
for lib in nvJitLink nvfatbin cuda; do
  shopt -s nullglob
  have=("$shim"/lib"$lib".so.[0-9]*)
  shopt -u nullglob
  if (( ${#have[@]} == 0 )); then
    echo "SKIP: the $lib shim is not built"; exit 0
  fi
done
tmp="${TMPDIR:-/tmp}/vgpu_jit_link_$$"
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT

nvcc -dc -gencode arch=compute_80,code=compute_80 -Wno-deprecated-gpu-targets \
     "$e2e/jitlink_lib.cu" -o "$tmp/jitlink_lib.o"
ar rcs "$tmp/libjitlink.a" "$tmp/jitlink_lib.o"
nvcc -fatbin -rdc=true -gencode arch=compute_80,code=lto_80 -Wno-deprecated-gpu-targets \
     "$e2e/jitlink_lib.cu" -o "$tmp/jitlink_lib_lto.fatbin"

# nvFatbin is linked against the shim's copy, CUDA 12.0's toolkit having
# none; nvJitLink against the toolkit's, as a program would be.
nvcc -std=c++17 -cudart none -Wno-deprecated-gpu-targets -Xcompiler -Wno-deprecated-declarations \
     $(shim_sanitizer_nvcc_flags "$shim") "$e2e/$name.cpp" -o "$tmp/$name" \
     "$shim/libnvfatbin.so" -lnvJitLink -lcuda
require_shim_libs "$shim" "$tmp/$name" || exit 0

case "$name" in
  nvjitlink_paths) args=("$tmp/jitlink_lib.o" "$tmp/libjitlink.a" "$tmp/jitlink_lib_lto.fatbin") ;;
  *) args=("$tmp/jitlink_lib.o") ;;
esac
status=0
result="$(VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$tmp/$name" "${args[@]}" 2>&1)" || status=$?
echo "$result"
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: the program reached an unimplemented entry point or a refused kernel"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
