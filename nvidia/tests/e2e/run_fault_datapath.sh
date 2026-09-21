#!/usr/bin/env bash
# Faults armed with `vgpu fault arm`, taken by a running kernel's loads: silent
# bit flips the program sees as wrong data, corrected errors it never sees, and
# an uncorrectable error that fails its kernel.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="${VGPU_BUILD_DIR:-$root/build}"
shim="$build/shim"
vgpu="$build/vgpu"
src="$root/nvidia/tests/e2e/fault_datapath.cu"
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
prog="$tmp/fault_datapath"
nvcc -std=c++17 -cudart shared -arch=compute_75 -code=compute_75 -Wno-deprecated-gpu-targets \
     $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$prog" || exit 1
if ! require_shim_libs "$shim" "$prog"; then exit 0; fi

# A machine of its own, which the program and `vgpu fault` both describe.
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
export VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=1 VGPU_QUIET=1
run() { LD_LIBRARY_PATH="$shim" "$prog" 2>&1; }
q() { "$vgpu" smi --query-gpu="$1" --format=csv,noheader 2>&1; }
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}

expect "a clean run copies every value" "mismatches: 0" "$(run)"

"$vgpu" fault arm --bitflip --count 3 >/dev/null
expect "three armed bit flips corrupt three values, silently" "mismatches: 3" "$(run)"
expect "nothing is counted for a flip ECC did not see" "0, 0" \
  "$(q ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total)"
expect "fault show tells what was delivered" "yes" \
  "$("$vgpu" fault show | grep -q 'bit flips 0 (bit flips delivered: 3)' && echo yes || echo no)"

"$vgpu" fault arm --ecc corrected --count 2 >/dev/null
expect "corrected errors change nothing the program sees" "mismatches: 0" "$(run)"
expect "and are counted" "2" "$(q ecc.errors.corrected.volatile.total)"

"$vgpu" fault arm --ecc uncorrected >/dev/null
expect "an uncorrectable error fails the kernel with error 214" \
  "launch failed: 214 cudaErrorECCUncorrectable" "$(run)"
expect "counted, with a page pending retirement" "1, 1, Yes" \
  "$(q ecc.errors.uncorrected.volatile.total,retired_pages.dbe,retired_pages.pending)"
expect "nothing is left armed" "mismatches: 0" "$(run)"

"$vgpu" fault arm --hang --seconds 2 >/dev/null
start=$(date +%s.%N)
got=$(run)
took=$(echo "$(date +%s.%N) - $start" | bc)
expect "a hung launch fails with a launch timeout" "launch failed: 702 cudaErrorLaunchTimeout" "$got"
expect "after the time it was armed for" "yes" "$( (( $(echo "$took >= 2" | bc) )) && echo yes || echo no)"
expect "and the next run is clean" "mismatches: 0" "$(run)"
exit $fail
