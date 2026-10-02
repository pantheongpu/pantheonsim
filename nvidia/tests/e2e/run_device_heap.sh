#!/usr/bin/env bash
# malloc() and free() in a kernel: one heap per device, emptied by
# cudaDeviceReset, refunded by free(), shared by kernels and by both engines,
# and refused by the host's cudaFree, cuMemFree and cuMemGetAddressRange
# (device_heap.cu, against the values an RTX 3060 gives). Run on the SASS by
# default, where device_heap_peer.cu -- built to PTX only -- runs on the PTX
# interpreter beside it, and again with VGPU_SASS=0, all PTX.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu_e2e_device_heap.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out" "$out".*' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
e2e="$root/nvidia/tests/e2e"
# The test's kernels as SASS and PTX for sm_86; the peer's as PTX only.
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
    -c "$e2e/device_heap.cu" -o "$out.main.o" &&
"$nvcc_bin" -std=c++17 -gencode arch=compute_86,code=compute_86 -cudart shared -Wno-deprecated-gpu-targets \
    "${san_flags[@]}" -c "$e2e/device_heap_peer.cu" -o "$out.peer.o" &&
"$nvcc_bin" -arch=sm_86 -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
    "$out.main.o" "$out.peer.o" -o "$out" -lcuda ||
  { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

fails=0
# run <name> <expect SASS: yes|no> [env...]
run() {
  local name=$1 sass=$2
  shift 2
  local log
  # Kept out of a failing command substitution, so a failure prints its reason.
  # Both libcudart and libcuda (both_shims_env: what a sanitizer build needs).
  log="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_SASS_LOG=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$@" "$out" 2>&1)"
  local rc=$?
  echo "$log" | sed "s/^/    $name: /"
  if [[ $rc != 0 || "$(tail -n 1 <<< "$log")" != "PASS" ]] || grep -q '^FAIL' <<< "$log"; then
    echo "FAIL: $name"; fails=1
  fi
  if [[ $sass == yes ]] && ! grep -q "running SASS: an sm_86 cubin" <<< "$log"; then
    echo "FAIL: $name: the test's kernels did not run as SASS"; fails=1
  fi
  if [[ $sass == no ]] && grep -q "running SASS" <<< "$log"; then
    echo "FAIL: $name: expected PTX only"; fails=1
  fi
}
run sass+ptx yes
run ptx      no  VGPU_SASS=0
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: device heap"
exit $fails
