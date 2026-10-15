#!/usr/bin/env bash
# The SASS executor's fast paths against its general path. sass_fastpath.cu is
# built for sm_86 and run twice on the SASS engine with the performance counters
# on: as it is, and with VGPU_FASTPATH=0, which sends every instruction down the
# general path. Both must pass the program's own checks and print byte for byte
# the same -- the answers and every counter.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
carts=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
(( ${#carts[@]} )) || { echo "SKIP: libvgpucudart not built"; exit 0; }
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }
nvcc_host_compiler_fix
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu-sass-fastpath.XXXXXX")"
trap 'rm -rf "$work"' EXIT
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"
"$nvcc_bin" -std=c++17 -arch=sm_86 -cudart shared -w -Wno-deprecated-gpu-targets "${san_flags[@]}" \
  "$root/nvidia/tests/e2e/sass_fastpath.cu" -o "$work/prog" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$work/prog"; then exit 0; fi
run() {   # <output file> [env...]
  local file=$1
  shift
  env VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 LD_LIBRARY_PATH="$shim" "$@" \
    "$work/prog" > "$file" 2>&1
}
fails=0
run "$work/fast.txt"; rc1=$?
run "$work/general.txt" VGPU_FASTPATH=0; rc0=$?
grep -q "running SASS" "$work/fast.txt" || { echo "FAIL: the program did not run on SASS"; fails=1; }
if (( rc1 != 0 )) || ! grep -q "^wrong: 0$" "$work/fast.txt"; then
  echo "FAIL: fast paths: wrong answer or failed run"; tail -20 "$work/fast.txt"; fails=1
fi
if (( rc0 != 0 )) || ! grep -q "^wrong: 0$" "$work/general.txt"; then
  echo "FAIL: general path: wrong answer or failed run"; tail -20 "$work/general.txt"; fails=1
fi
if ! diff -u "$work/general.txt" "$work/fast.txt" > "$work/diff.txt"; then
  echo "FAIL: the fast paths and the general path differ (answers or counters):"; head -40 "$work/diff.txt"; fails=1
fi
# An access that faults, with one lane of the warp outside: the fast paths decline it and the general path reports it, so the two
# runs print the same error (the fault message names the lane, the instruction and the address, so a difference would show here).
for fault in local gather scatter atomic shared sharedatomic; do
  env VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 LD_LIBRARY_PATH="$shim" "$work/prog" "$fault" > "$work/fault_fast_$fault.txt" 2>&1; rcf=$?
  env VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_FASTPATH=0 LD_LIBRARY_PATH="$shim" "$work/prog" "$fault" > "$work/fault_general_$fault.txt" 2>&1; rcg=$?
  if (( rcf != 0 || rcg != 0 )); then
    echo "FAIL: the $fault fault did not happen (rc fast $rcf, general $rcg)"; cat "$work/fault_fast_$fault.txt"; fails=1
  elif ! diff -u "$work/fault_general_$fault.txt" "$work/fault_fast_$fault.txt" > "$work/fault_diff.txt"; then
    echo "FAIL: the $fault fault is reported differently with the fast paths:"; head -20 "$work/fault_diff.txt"; fails=1
  fi
done
(( fails )) && exit 1
echo "OK: fast paths and general path agree ($(grep -c 'counters' "$work/fast.txt") launches' counters compared)"
