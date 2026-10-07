#!/usr/bin/env bash
# Memory-test patterns (walking ones and zeros, address in address,
# checkerboard, moving inversions, seeded random data, block copies, strides)
# over simulated device memory, and what a failing cell and a wild pointer look
# like to them; see memtest_patterns.cu. The faults are armed with `vgpu fault`,
# the simulator's fault-injection hook (see run_fault_datapath.sh).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
vgpu="$build/vgpu"
src="$root/nvidia/tests/e2e/memtest_patterns.cu"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
prog="$tmp/memtest_patterns"
nvcc -std=c++17 -cudart shared -arch=compute_75 -code=compute_75 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$prog" || exit 1
if ! require_shim_libs "$shim" "$prog"; then exit 0; fi

mkdir -p "$tmp/session"
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_SESSION="$tmp/session"
export VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=1 VGPU_QUIET=1
run() { LD_LIBRARY_PATH="$shim" "$prog" "$@" 2>&1; }
status() { LD_LIBRARY_PATH="$shim" "$prog" "$@" >/dev/null 2>&1; echo $?; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
# The count a test printed, by its name.
errors() { sed -n "s/^test $1 *errors: \([0-9]*\).*/\1/p" <<<"$2" | head -1; }

out=$(run)
expect "a clean machine passes every pattern" "total errors: 0" "$(grep '^total' <<<"$out")"
expect "and reports PASS" "PASS" "$(tail -1 <<<"$out")"
expect "all ten tests ran" "10" "$(grep -c '^test ' <<<"$out")"

expect "mapped host memory passes them too" "total errors: 0" "$(run mapped | grep -E '^(total|SKIP)')"

expect "a bit flipped by the host is found exactly, in each of three bit positions" \
  "flip: PASS" "$(run flip | tail -1)"

# A cell stuck at 1: the first allocation is the buffer, at the start of device
# memory. Word 4 (offset 0x10), bit 0, stuck at 1.
"$vgpu" fault stuck --offset 0x10 --bit 0 --value 1 >/dev/null
out=$(run)
expect "a stuck-at-1 cell fails the suite" "yes" \
  "$([[ "$out" != *"total errors: 0"* ]] && echo yes || echo no)"
expect "the walking-zeros test sees it" "yes" \
  "$([[ "$(errors 'walking zeros' "$out")" -ge 1 ]] && echo yes || echo no)"
expect "the walking-ones test sees it in the patterns with bit 0 clear" "yes" \
  "$([[ "$(errors 'walking ones' "$out")" -ge 1 ]] && echo yes || echo no)"
expect "the exit status is 1" "1" "$(status)"
"$vgpu" fault stuck --clear >/dev/null
expect "cleared, the machine passes again" "total errors: 0" "$(run | grep '^total')"

"$vgpu" fault arm --bitflip --count 5 --on store >/dev/null
out=$(run)
expect "bit flips armed on stores are caught by the verify reads" "yes" \
  "$([[ "$out" != *"total errors: 0"* && "$out" != *"launch failed"* ]] && echo yes || echo no)"
"$vgpu" fault arm --bitflip --rate 0 >/dev/null
expect "and stop when disarmed" "total errors: 0" "$(run | grep '^total')"

expect "a pointer to nothing fails the verify kernel with an illegal address" \
  "launch failed: 700 cudaErrorIllegalAddress (wild)" "$(run wild | head -1)"
expect "and the program's exit status says so" "3" "$(status wild)"
exit $fail
