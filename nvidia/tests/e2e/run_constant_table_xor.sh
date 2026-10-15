#!/usr/bin/env bash
# CUDA Samples' quasirandomGenerator kernel and its variations; see constant_table_xor.cu.
#
# Compiled to real SASS (sm_86), so the SASS interpreter runs it, and run again on
# the PTX interpreter (VGPU_SASS=0): a variant that differs on one engine only says
# which one is wrong.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/constant_table_xor.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_constant_table_xor_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi

# SASS for sm_86 and the PTX it came from, so both engines have something to run.
nvcc -std=c++17 -cudart shared -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
trap 'rm -f "$out"' EXIT
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
status=0
for engine in sass ptx; do
  sass=1; [[ $engine == ptx ]] && sass=0
  echo "== $engine"
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_SASS=$sass LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || status=1
  echo "$result"
  [[ "$(tail -1 <<< "$result")" == PASS ]] || status=1
done
if (( status )) && command -v cuobjdump >/dev/null 2>&1; then
  # What ptxas made of the two kernels that differ in one thing only: the row that the table is read at.
  for k in q_row_nostride q_row0; do
    echo "== SASS of $k"
    cuobjdump -sass "$out" | awk -v k="$k" '/Function : / { on = index($0, k) > 0 } on' | sed -E 's#/\*[0-9a-f]{4}\*/##; s#/\* 0x[0-9a-f]+ \*/##' | grep -v '^\s*$' | head -120
  done
fi
exit $status
