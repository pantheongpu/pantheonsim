#!/usr/bin/env bash
# Device-side graph launch and %current_graph_exec (device_graph_launch.cu,
# against the values an RTX 3060 gives), built -rdc=true with cudadevrt as a
# program using it is: run on the SASS by default and again with VGPU_SASS=0,
# on the PTX. Each run is the program twice: once through to the fault of a
# launch from outside any graph, once to the fault of a device launch of a
# graph not instantiated for device launch.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu_e2e_device_graph_launch.$$"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
trap 'rm -f "$out" "$out".*' EXIT

read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
if ! "$nvcc_bin" -std=c++17 -rdc=true -cudart shared -Wno-deprecated-gpu-targets "${san_flags[@]}" \
       -gencode arch=compute_86,code=sm_86 -gencode arch=compute_86,code=compute_86 \
       "$root/nvidia/tests/e2e/device_graph_launch.cu" -o "$out" -lcudadevrt 2> "$out.log"; then
  cat "$out.log"; echo "FAIL: does not compile"; exit 1
fi
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

fails=0
# run <name> <expect SASS: yes|no> [env...]
run() {
  local name=$1 sass=$2
  shift 2
  local arg log rc
  for arg in "" host-handle; do
    # Kept out of a failing command substitution, so a failure prints its reason.
    log="$(env VGPU_QUIET=1 VGPU_SASS_LOG=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$@" "$out" $arg 2>&1)"
    rc=$?
    echo "$log" | grep -v '^\[vgpu\]' | sed "s/^/    $name ${arg:-main}: /"
    if [[ $rc != 0 || "$(tail -n 1 <<< "$log")" != "PASS" ]] || grep -q '^FAIL' <<< "$log"; then
      echo "FAIL: $name ${arg:-main}"; fails=1
    fi
    if [[ $sass == yes ]] && ! grep -q "running SASS: an sm_86 cubin" <<< "$log"; then
      echo "FAIL: $name: the test's kernels did not run as SASS"; fails=1
    fi
    if [[ $sass == no ]] && grep -q "running SASS" <<< "$log"; then
      echo "FAIL: $name: expected PTX only"; fails=1
    fi
  done
}
run sass yes
run ptx  no  VGPU_SASS=0
[[ $fails == 0 ]] && echo "PASS" || echo "FAIL: device graph launch"
exit $fails
