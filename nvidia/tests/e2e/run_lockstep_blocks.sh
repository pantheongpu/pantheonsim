#!/usr/bin/env bash
# Blocks that wait on one another (a grid barrier, a chain of flags, a producer and its consumers, patch
# locks), in a grid that fits the device; see lockstep_blocks.cu and "Blocks that wait on one another" in
# the README.
#
# Run as real SASS (sm_75) and as PTX alone (compute_75): the two engines each take turns between the
# blocks. The SASS run is repeated on one host thread (the order is then fixed) and with every launch in
# lockstep (VGPU_LOCKSTEP=all), where a kernel whose blocks do not wait on each other must give the same
# answers. Last, with the old behaviour (VGPU_LOCKSTEP=0, one thread) the patch locks must fail: the test
# would otherwise pass whatever this does.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/lockstep_blocks.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_lockstep_blocks_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
trap 'rm -f "$out".sass "$out".ptx' EXIT

nvcc -std=c++17 -cudart shared -gencode arch=compute_75,code=sm_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out.sass"
nvcc -std=c++17 -cudart shared -gencode arch=compute_75,code=compute_75 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out.ptx"
if ! require_shim_libs "$shim" "$out.sass"; then exit 0; fi

fails=0
run() {   # run <name> <binary> <expected: PASS|FAIL> [env...]
  local name=$1 bin=$2 want=$3
  shift 3
  local result
  result="$(env VGPU_QUIET=1 VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" "$@" "$bin" $ARGS 2>&1)" || true
  if [[ "$result" == *PASS* && $want == PASS ]] || [[ "$result" != *PASS* && $want == FAIL ]]; then
    echo "ok: $name"
  else
    echo "FAIL: $name (wanted $want)"; echo "$result"; fails=1
  fi
}
ARGS=
run "SASS"                 "$out.sass" PASS
run "SASS, one thread"     "$out.sass" PASS VGPU_THREADS=1
run "SASS, three threads"  "$out.sass" PASS VGPU_THREADS=3
run "SASS, all lockstep"   "$out.sass" PASS VGPU_LOCKSTEP=all
run "PTX"                  "$out.ptx"  PASS
run "PTX, one thread"      "$out.ptx"  PASS VGPU_THREADS=1
run "PTX, all lockstep"    "$out.ptx"  PASS VGPU_LOCKSTEP=all
ARGS=locks
run "old behaviour: the patch locks fail (SASS)" "$out.sass" FAIL VGPU_LOCKSTEP=0 VGPU_THREADS=1
run "old behaviour: the patch locks fail (PTX)"  "$out.ptx"  FAIL VGPU_LOCKSTEP=0 VGPU_THREADS=1
exit "$fails"
