#!/usr/bin/env bash
# Shared and local memory moved 1 to 16 bytes at a time; see sass_vector_memory.cu.
#
# The program compares every word with a host computation, so a pass on each engine is the engines agreeing with the
# host and with each other: the SASS interpreter (VGPU_TIER=0), the SASS tier (the default) and the PTX interpreter
# (VGPU_SASS=0), and the first and the last of these again without the fast paths. With --card it runs on the machine's real
# GPU and library instead (nothing is built against the shim).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/sass_vector_memory.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_sass_vector_memory_$$"
card=0
[[ "${1:-}" == "--card" ]] && card=1
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
if (( card )); then
  nvcc -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets "$src" -o "$out"
  rc=0; "$out" || rc=$?; rm -f "$out"; exit $rc
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
# A real sm_86 image, so that the SASS engines run it, and its PTX, for the PTX interpreter (VGPU_SASS=0).
nvcc -std=c++17 -cudart shared -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
fail=0
# (The "general" runs send every access down the executors' general path, not the fast paths: local memory is then read and
# written by Runner::mem_read / mem_write and the PTX interpreter's load_routed / store_routed.)
for engine in sass tier ptx sass-general ptx-general; do
  case "$engine" in
    sass) env_engine=(VGPU_TIER=0) ;;
    tier) env_engine=(VGPU_TIER=1) ;;
    ptx)  env_engine=(VGPU_SASS=0) ;;
    sass-general) env_engine=(VGPU_TIER=0 VGPU_FASTPATH=0) ;;
    ptx-general)  env_engine=(VGPU_SASS=0 VGPU_FASTPATH=0) ;;
  esac
  result="$(env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 "${env_engine[@]}" LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || {
    echo "== $engine: $result"; fail=1; continue; }
  echo "== $engine: $(echo "$result" | tr '\n' ' ')"
  [[ "$result" == *PASS* ]] || fail=1
done
rm -f "$out"
exit $fail
