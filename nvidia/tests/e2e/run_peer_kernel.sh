#!/usr/bin/env bash
# A kernel on one simulated device reaching another's allocations once peer access is enabled (peer_kernel.cu, issue #598): reads,
# stores and atomics, on the SASS path and on the PTX path (VGPU_SASS=0); without peer access, or after it is disabled again, the same
# read faults with cudaErrorIllegalAddress, as it does on a card (two RTX 3060s, measured: no peer path, and the kernel faults).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/peer_kernel.cu"
out="${TMPDIR:-/tmp}/vgpu-peer-kernel.$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
trap 'rm -f "$out" "$out".*' EXIT
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}" "$src" -o "$out" -L"$shim" \
  || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

fails=0
for engine in sass ptx; do
  sass=1; [[ $engine == ptx ]] && sass=0
  for mode in direct without disabled; do
    got="$(env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/h100 VGPU_DEVICE_COUNT=2 VGPU_SASS=$sass \
           LD_LIBRARY_PATH="$shim" timeout 300 "$out" $mode 2>&1)" || { echo "FAIL: $engine $mode: the program failed"; echo "$got" | sed 's/^/    /'; fails=1; continue; }
    if [[ $mode == direct ]]; then
      want="cudaDeviceCanAccessPeer(1, 0) = 1
the read of device 0's buffer from device 1: cudaSuccess
sum = 499500
atomic adds from device 1 into device 0's counter = 1024
words device 0 reads back wrong = 0"
    else
      want="cudaDeviceCanAccessPeer(1, 0) = 1
the read of device 0's buffer from device 1: cudaErrorIllegalAddress"
    fi
    if [[ "$got" != "$want" ]]; then
      echo "FAIL: $engine $mode:"; diff <(echo "$want") <(echo "$got") | sed 's/^/    /'; fails=1
    else
      echo "ok   $engine $mode"
    fi
  done
done
(( fails == 0 )) || exit 1
echo "PASS"
