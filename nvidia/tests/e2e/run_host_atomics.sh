#!/usr/bin/env bash
# Device atomics on mapped host memory stay atomic against the host's own
# atomics racing them (host_atomics.cu). Run on the SASS by default and again
# with VGPU_SASS=0, on the PTX interpreter: each engine has its own atomics.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu_e2e_host_atomics.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out"' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
    "$root/nvidia/tests/e2e/host_atomics.cu" -o "$out" ||
  { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

fails=0
# run <name> <expect SASS: yes|no> [env...]
run() {
  local name=$1 sass=$2
  shift 2
  local log
  log="$(env VGPU_QUIET=1 VGPU_SASS_LOG=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$@" "$out" 2>&1)"
  local rc=$?
  echo "$log" | sed "s/^/    $name: /"
  if [[ $rc != 0 || "$(tail -n 1 <<< "$log")" != "PASS" ]] || grep -q '^FAIL' <<< "$log"; then
    echo "FAIL: $name"; fails=1
  fi
  if [[ $sass == yes ]] && ! grep -q "running SASS: an sm_86 cubin" <<< "$log"; then
    echo "FAIL: $name: the kernel did not run as SASS"; fails=1
  fi
  if [[ $sass == no ]] && grep -q "running SASS" <<< "$log"; then
    echo "FAIL: $name: expected PTX only"; fails=1
  fi
}
run sass yes
run ptx  no  VGPU_SASS=0
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: host atomics"
exit $fails
