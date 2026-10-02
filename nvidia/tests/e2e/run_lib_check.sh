#!/usr/bin/env bash
# A self-checking program that calls a CUDA library directly -- no framework --
# built with nvcc and run against the shims.
#
#   run_lib_check.sh <name> <lib>...
#
# compiles nvidia/tests/e2e/<name>.cu (or .cpp), links it with -l<lib> for each library,
# and runs it on a simulated GPU. The program prints "ok"/"FAIL" per check and
# ends with a PASS or FAIL line. These are the library paths PyTorch takes,
# pinned down where hosted CI can run them, which it cannot PyTorch itself.
#
# Deliberately not quiet: a call into a stub ("is not implemented by
# VirtualGPU") or a kernel the simulator refused fails the test even if the
# program's own checks happened to pass.
set -euo pipefail
name="$1"; shift
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/$name.cu"
# A host-only program is a .cpp: nvcc adds fatbinary registration, which
# needs the runtime, to every .cu, device code or none.
[[ -e "$src" ]] || src="${src%.cu}.cpp"
out="${TMPDIR:-/tmp}/vgpu_e2e_${name}_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
links=()
# A program on the driver API ("cuda" among its libraries) links no CUDA
# runtime: libcudart and libcuda each carry the simulator, and a sanitizer
# build refuses a process that loads both (an ODR violation on its globals).
cudart=shared
for lib in "$@"; do
  [[ "$lib" == cuda ]] && cudart=none
  shopt -s nullglob
  have=("$shim"/lib"$lib".so.[0-9]*)
  shopt -u nullglob
  if (( ${#have[@]} == 0 )); then
    echo "SKIP: the $lib shim is not built (CUDA ABI headers absent at build time)"; exit 0
  fi
  # cuStateVec, cuDSS, cuTENSOR, cuTensorNet and cuDNN are not part of the
  # toolkit, so nvcc has no copy to link against: link against the shim's,
  # which follows NVIDIA's ABI. cuDNN's headers are vendored.
  [[ "$lib" == custatevec || "$lib" == cudss || "$lib" == cutensor || "$lib" == cutensornet || "$lib" == cudnn ]] &&
    links+=("-L$shim")
  [[ "$lib" == cudnn ]] && links+=("-I$root/nvidia/third_party/cudnn_include")
  links+=("-l$lib")
done
trap 'rm -f "$out"' EXIT
nvcc -std=c++17 -cudart "$cudart" -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets -Xcompiler -Wno-deprecated-declarations \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" "${links[@]}"
require_shim_libs "$shim" "$out" || exit 0
status=0
result="$(VGPU_GPU=nvidia/a100 VGPU_E2E_DATA="$root/nvidia/tests/data" LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || status=$?
echo "$result" | grep -v '^\[vgpu\] .* plan created' || true
if grep -qE 'VirtualGPU error \[|is not implemented by VirtualGPU' <<< "$result"; then
  echo "FAIL: the program reached an unimplemented entry point or a refused kernel"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
