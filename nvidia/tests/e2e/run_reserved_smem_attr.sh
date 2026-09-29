#!/usr/bin/env bash
# The driver's reserved shared memory per block, as every API reports it
# (reserved_smem_attr.cu), against what the cards report: an RTX 3060 says
# 1024 and 102400, a T4 0 and 65536.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-reserved-smem.$$"
[[ -e "$shim/libcuda.so.1" ]] || { echo "SKIP: libvgpucuda not built"; exit 0; }
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out"' EXIT
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_75 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/reserved_smem_attr.cu" -L"$shim" -lcuda -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi
# Both libraries carry the simulator's core, which a sanitizer build reports as
# an ODR violation when one program loads the two (as run_mixed_apis.sh skips).
if [[ -n "$(shim_sanitizer "$shim")" ]]; then echo "SKIP: a sanitizer build loads two copies of the core"; exit 0; fi
fail=0
check() {  # check <gpu> <expected line>
  local got
  got=$(VGPU_QUIET=1 VGPU_GPU=$1 LD_LIBRARY_PATH="$shim" "$out" 2>&1 | tail -n 1)
  if [[ "$got" == "$2" ]]; then echo "ok    $1: $got"; else echo "FAIL  $1: expected \"$2\", got \"$got\""; fail=1; fi
}
check nvidia/rtx3060 "reserved 1024 1024 1024 sm 102400 102400"   # the card's own line
check nvidia/t4 "reserved 0 0 0 sm 65536 65536"
exit $fail
