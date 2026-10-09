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
  # cuStateVec, cuDSS, cuSPARSELt, cuTENSOR, cuTensorNet, cuDNN, nvCOMP and NVSHMEM are not
  # part of the toolkit, and cuFile is not part of CUDA 12.0's, so nvcc has no
  # copy to link against: link against the shim's, which follows NVIDIA's ABI.
  # cuDNN's headers are fetched at configure time; the others' are the simulator's own
  # (nvidia/include/vgpu_*.h).
  [[ "$lib" == custatevec || "$lib" == cudss || "$lib" == cusparseLt || "$lib" == cutensor || "$lib" == cutensornet || "$lib" == cudnn ||
     "$lib" == cufile || "$lib" == nvcomp || "$lib" == nvcomp_cpu || "$lib" == nvshmem_host ]] && links+=("-L$shim")
  if [[ "$lib" == cudnn ]]; then
    cudnn_inc="$(cudnn_include_dir)" || { echo "SKIP: no cuDNN headers (see scripts/fetch-cudnn-headers.py)"; exit 0; }
    links+=("-I$cudnn_inc")
  fi
  links+=("-l$lib")
done
# A graph-API test built on NVIDIA's cudnn-frontend (header-only) gets it
# fetched; without network access it is skipped.
if grep -q '#include <cudnn_frontend.h>' "$src"; then
  fe="$("$root/nvidia/tests/e2e/fetch_cudnn_frontend.sh")" || { echo "SKIP: cudnn-frontend could not be fetched"; exit 0; }
  links+=("-I$fe")
fi
trap 'rm -f "$out"' EXIT
nvcc -std=c++17 -cudart "$cudart" -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets -Xcompiler -Wno-deprecated-declarations \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out" "${links[@]}"
require_shim_libs "$shim" "$out" || exit 0
# A program that opens cuDNN and the runtime itself (cudnn_dlhandle, as the
# cudnn-frontend does) opens them RTLD_GLOBAL, which puts libcuda and libcudart
# in one scope: each carries the simulator's globals, and a sanitizer build
# reports the pair as an ODR violation. They are the same code, by design.
# Where an NVIDIA runtime of another major is installed too, the frontend picks
# the lowest and runs on NVIDIA's own, which cannot reach a simulated driver;
# name the shim's.
if grep -q cudnn_dlhandle "$src"; then
  export ASAN_OPTIONS="${ASAN_OPTIONS:+$ASAN_OPTIONS:}detect_odr_violation=0"
  for rt in "$shim"/libcudart.so.[0-9]*; do
    if [[ "$rt" =~ \.so\.[0-9]+$ ]]; then export CUDNN_FRONTEND_CUDART_LIB_NAME="$rt"; fi
  done
fi
status=0
# VGPU_E2E_GPU picks another profile (the FP8 and block-scaled checks need a GPU that has them).
result="$(VGPU_GPU="${VGPU_E2E_GPU:-nvidia/a100}" VGPU_E2E_DATA="$root/nvidia/tests/data" LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || status=$?
echo "$result" | grep -v '^\[vgpu\] .* plan created' || true
# A program that tests the error paths on purpose (invalid arguments, a kernel that must fault) gets the
# library's "VirtualGPU error [...]" lines on stderr: it says VGPU_E2E_EXPECTS_REFUSALS in its source, and
# only a call into a stub still fails it. Its own checks decide whether each refusal was the right one.
refused='VirtualGPU error \[|is not implemented by VirtualGPU'
if grep -q 'VGPU_E2E_EXPECTS_REFUSALS' "$src"; then refused='is not implemented by VirtualGPU'; fi
if grep -qE "$refused" <<< "$result"; then
  echo "FAIL: the program reached an unimplemented entry point or a refused kernel"; exit 1
fi
[[ $status == 0 && "$(tail -1 <<< "$result")" == PASS* ]]
