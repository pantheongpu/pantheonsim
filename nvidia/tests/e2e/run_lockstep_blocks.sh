#!/usr/bin/env bash
# Blocks that wait on one another (a grid barrier, a chain of flags, a producer and its consumers, patch
# locks), in a grid that fits the device; see lockstep_blocks.cu and "Blocks that wait on one another" in
# the README.
#
# Run as real SASS (sm_75) and as PTX alone (compute_75): the two engines each take turns between the
# blocks. The SASS run is repeated on one host thread (the order is then fixed) and with every launch in
# lockstep (VGPU_LOCKSTEP=all), where a kernel whose blocks do not wait on each other must give the same
# answers. Last, with the old behaviour (VGPU_LOCKSTEP=0, one thread) the patch locks must fail: the test
# would otherwise pass whatever this does. Then the sort-shaped wait (a loop that stores and moves the address it
# polls on the turns that find work, as Barnes-Hut's SortKernel does) in the same four ways, and a check of which
# kernels took the lockstep path: the sort-shaped one must, a volatile loop that only walks an array must not.
# Last, how many host threads a small grid takes: the producer/consumer grid (10 blocks, waits on loads) spreads over them, the patch
# locks (12 blocks, a lock taken with an atomic) stay on one. Both engines.
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
trap 'rm -f "$out".sass "$out".ptx "$out".log' EXIT

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
ARGS=sortwait
# (a run that does not interleave the blocks spins until the step limit: kept short)
run "SASS, sort-shaped wait"               "$out.sass" PASS VGPU_MAX_STEPS=20000000
run "SASS, sort-shaped wait, one thread"   "$out.sass" PASS VGPU_MAX_STEPS=20000000 VGPU_THREADS=1
run "SASS, sort-shaped wait, three threads" "$out.sass" PASS VGPU_MAX_STEPS=20000000 VGPU_THREADS=3
run "old behaviour: the sort-shaped wait hits the step limit (SASS)" "$out.sass" FAIL VGPU_MAX_STEPS=20000000 VGPU_LOCKSTEP=0 VGPU_THREADS=1
rm -f "$out.log"
run "SASS, sort-shaped wait, logging the lockstep launches" "$out.sass" PASS VGPU_MAX_STEPS=20000000 VGPU_LOCKSTEP_LOG="$out.log"
if grep -q 'kernel _Z9sort_wait' "$out.log" 2>/dev/null; then echo "ok: sort_wait runs in lockstep"; else echo "FAIL: sort_wait did not run in lockstep"; fails=1; fi
if grep -q 'volatile_scan' "$out.log" 2>/dev/null; then echo "FAIL: volatile_scan was taken for a poller"; fails=1; else echo "ok: volatile_scan is not taken for a poller"; fi
# A grid of fewer than 16 blocks: waits on loads only give the same answer in any order, so it uses the host threads (two blocks to
# a thread at least); a kernel that takes locks keeps one thread and a fixed order.
host_threads() {   # host_threads <program arg> <kernel name fragment> [binary]: the host threads its lockstep launch reports
  rm -f "$out.log"
  env VGPU_QUIET=1 VGPU_GPU=nvidia/t4 LD_LIBRARY_PATH="$shim" VGPU_THREADS=4 VGPU_LOCKSTEP_LOG="$out.log" "${3:-$out.sass}" "$1" >/dev/null 2>&1 || true
  sed -n "/kernel $2/s/.* \([0-9][0-9]*\) host thread.*/\1/p" "$out.log" 2>/dev/null | head -n 1
}
t="$(host_threads queue _Z17producer_consumer)"
if [[ -n "$t" && "$t" -ge 2 ]]; then echo "ok: a 10-block grid that waits on loads uses $t host threads"; else echo "FAIL: producer_consumer (waits on loads, 10 blocks) ran on '${t:-no}' host threads, wanted 2 or more"; fails=1; fi
ARGS=queue
run "SASS, producer/consumer on several host threads" "$out.sass" PASS VGPU_THREADS=4
t="$(host_threads locks _Z11lock_kernel)"
if [[ "$t" == 1 ]]; then echo "ok: the 12-block patch-lock grid stays on one host thread"; else echo "FAIL: lock_kernel (takes locks, 12 blocks) ran on '${t:-no}' host threads, wanted 1"; fails=1; fi
ARGS=locks
run "SASS, patch locks with four host threads available" "$out.sass" PASS VGPU_THREADS=4
# The same on the PTX engine (the PTX-only build): it used to give every small grid one thread.
t="$(host_threads queue _Z17producer_consumer "$out.ptx")"
if [[ -n "$t" && "$t" -ge 2 ]]; then echo "ok: PTX, a 10-block grid that waits on loads uses $t host threads"; else echo "FAIL: PTX producer_consumer (waits on loads, 10 blocks) ran on '${t:-no}' host threads, wanted 2 or more"; fails=1; fi
ARGS=queue
run "PTX, producer/consumer on several host threads" "$out.ptx" PASS VGPU_THREADS=4
t="$(host_threads locks _Z11lock_kernel "$out.ptx")"
if [[ "$t" == 1 ]]; then echo "ok: PTX, the 12-block patch-lock grid stays on one host thread"; else echo "FAIL: PTX lock_kernel (takes locks, 12 blocks) ran on '${t:-no}' host threads, wanted 1"; fails=1; fi
ARGS=locks
run "PTX, patch locks with four host threads available" "$out.ptx" PASS VGPU_THREADS=4
exit "$fails"
