#!/usr/bin/env bash
# The packed integer instructions of PTX ISA 9.2 (packed_int_forms.cu) on a simulated RTX 5090, built for
# sm_120f and run on its SASS and on its PTX, each checked against a host loop. Skips when this nvcc's
# ptxas has no PTX ISA 9.2 (CUDA 13.2 and later).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-packed-int.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out" "$out.log" "$out.build"' EXIT
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
if ! "$nvcc_bin" -std=c++17 -gencode arch=compute_120f,code=[sm_120f,compute_120f] -cudart shared -w "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/packed_int_forms.cu" -o "$out" 2> "$out.build"; then
  if grep -q "not supported on .target\|Unexpected instruction types\|Feature '" "$out.build" ||
     grep -q "Unsupported gpu architecture\|compute_120f" "$out.build"; then
    echo "SKIP: this nvcc's ptxas does not assemble PTX ISA 9.2's packed integer forms for sm_120f"; exit 0
  fi
  cat "$out.build"; echo "FAIL: does not compile"; exit 1
fi
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
fails=0
for engine in default ptx; do
  env=(VGPU_QUIET=1 VGPU_GPU=nvidia/rtx5090 LD_LIBRARY_PATH="$shim" VGPU_SASS_LOG=1)
  [[ $engine == ptx ]] && env+=(VGPU_SASS=0)
  env "${env[@]}" "$out" > "$out.log" 2>&1
  rc=$?
  [[ $engine == default ]] && ! grep -q "running SASS" "$out.log" && { echo "FAIL: its SASS did not run"; sed 's/^/    /' "$out.log"; fails=1; }
  if [[ $rc != 0 || "$(tail -n 1 "$out.log")" != PASS ]]; then sed 's/^/    /' "$out.log"; echo "FAIL: packed integer forms ($engine)"; fails=1; fi
done
exit $fails
