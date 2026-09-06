#!/usr/bin/env bash
# A grid-wide barrier through cooperative_groups, unmodified.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-coop-grid.$$"
[[ -e "$shim/libcudart.so."* ]] 2>/dev/null || true
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
     "$root/tests/e2e/cooperative_grid.cu" -o "$out" || { echo "FAIL: does not compile"; exit 1; }
if ! require_shim_libs "$shim" "$out"; then exit 0; fi

log="$out.log"
VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" "$out" > "$log" 2>&1
rc=$?
sed 's/^/    /' "$log"
fails=0
grep -q "cooperative launch advertised: yes" "$log" || { echo "FAIL: attribute not advertised"; fails=1; }
grep -q "grid barrier held, wrong values: 0" "$log" || { echo "FAIL: barrier did not hold"; fails=1; }
grep -q "oversized cooperative grid: cudaErrorCooperativeLaunchTooLarge" "$log" || \
  { echo "FAIL: an unresident grid was accepted"; fails=1; }
# An ordinary launch has no barrier object, so the generated code traps. What
# matters is that it fails rather than quietly producing different numbers.
grep -qE "ordinary launch of a grid-sync kernel: cudaError(IllegalInstruction|LaunchFailure)" "$log" || \
  { echo "FAIL: an ordinary launch of a grid-sync kernel did not fail loudly"; fails=1; }
rm -f "$log"
(( rc == 0 && fails == 0 )) || { echo "FAIL: cooperative grid test"; exit 1; }
echo "RESULT: cooperative launch, grid barrier, residency limit and the non-cooperative case"
