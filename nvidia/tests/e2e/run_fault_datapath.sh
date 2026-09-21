#!/usr/bin/env bash
# Faults armed with `vgpu fault arm`, taken by a running kernel's loads, stores
# and shared-memory loads: silent bit flips the program sees as wrong data,
# corrected errors it never sees, and an uncorrectable error that fails its
# kernel.
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

# A machine of its own, which the program and `vgpu fault` both describe, and a
# session whose dmesg the driver's Xid lines go to.
mkdir -p "$tmp/session"
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_SESSION="$tmp/session"
export VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=1 VGPU_QUIET=1
run() { LD_LIBRARY_PATH="$shim" "$prog" "$@" 2>&1; }
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
  "$("$vgpu" fault show | grep -qE '^  bit flips delivered +3$' && echo yes || echo no)"

"$vgpu" fault arm --ecc corrected --count 2 >/dev/null
expect "corrected errors change nothing the program sees" "mismatches: 0" "$(run)"
expect "and are counted" "2" "$(q ecc.errors.corrected.volatile.total)"

"$vgpu" fault arm --ecc uncorrected >/dev/null
expect "an uncorrectable error fails the kernel with error 214" \
  "launch failed: 214 cudaErrorECCUncorrectable" "$(run)"
expect "counted, with a page pending retirement" "1, 1, Yes" \
  "$(q ecc.errors.uncorrected.volatile.total,retired_pages.dbe,retired_pages.pending)"
expect "nothing is left armed" "mismatches: 0" "$(run)"

"$vgpu" fault arm --bitflip --count 2 --on store >/dev/null
expect "flips armed on stores are what memory holds when it is copied back" "mismatches: 2" "$(run)"
expect "and nothing new is counted" "2, 1" \
  "$(q ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total)"
expect "an ECC error cannot be armed on stores" "2" \
  "$("$vgpu" fault arm --ecc uncorrected --on store >/dev/null 2>&1; echo $?)"

"$vgpu" fault arm --bitflip --on shared >/dev/null
expect "a copy that does not use shared memory never takes a shared fault" "mismatches: 0" "$(run)"
expect "one staged through it does" "mismatches: 1" "$(run shared)"
"$vgpu" fault arm --ecc corrected --count 3 --on shared >/dev/null
expect "corrected shared-memory errors change nothing" "mismatches: 0" "$(run shared)"
expect "and count as the L1 cache's" "2, 3" \
  "$(q ecc.errors.corrected.volatile.device_memory,ecc.errors.corrected.volatile.l1_cache)"
"$vgpu" fault arm --ecc uncorrected --on shared >/dev/null
expect "an uncorrectable one fails the kernel with error 214" \
  "launch failed: 214 cudaErrorECCUncorrectable" "$(run shared)"
expect "and retires no page of device memory" "1, 2, 1" \
  "$(q retired_pages.dbe,ecc.errors.uncorrected.volatile.total,ecc.errors.uncorrected.volatile.l1_cache)"

expect "a clean floating-point pass changes nothing" "mismatches: 0" "$(run float)"
"$vgpu" fault arm --bitflip --on alu --count 2 >/dev/null
expect "integer arithmetic -- indexes, addresses -- never takes an ALU fault" "mismatches: 0" "$(run)"
expect "two flips in floating-point results corrupt two values, silently" "mismatches: 2" "$(run float)"
expect "and nothing counts it" "5, 2" \
  "$(q ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total)"
expect "an ECC error cannot be armed on arithmetic results" "2" \
  "$("$vgpu" fault arm --ecc corrected --on alu >/dev/null 2>&1; echo $?)"

# The program's first allocation is its input, at the start of device memory:
# element 4 is 0x5A5A0004, element 5 is 0x5A5A0005.
"$vgpu" fault stuck --offset 0x10 --bit 0 --value 1 >/dev/null
expect "a bit stuck at 1 under a 0 corrupts the one value read through it" "mismatches: 1" "$(run)"
"$vgpu" fault stuck --offset 20 --bit 0 --value 1 >/dev/null
expect "one stuck at what the value already holds changes nothing more" "mismatches: 1" "$(run)"
"$vgpu" fault reset --volatile >/dev/null
expect "a driver reload does not mend them" "mismatches: 1" "$(run)"
expect "fault show lists them" "2" "$("$vgpu" fault show | grep -c '^  stuck cell ')"
"$vgpu" fault stuck --clear >/dev/null
expect "until they are cleared" "mismatches: 0" "$(run)"
expect "a cell past the end of memory is refused" "2" \
  "$("$vgpu" fault stuck --offset 0x1000000000 --bit 0 --value 1 >/dev/null 2>&1; echo $?)"

xids() { grep -o 'NVRM: Xid (PCI:[^)]*): [0-9]*, .*' "$tmp/session/dmesg.log" | sed -n "$1"; }
expect "a wild read fails the kernel with an illegal address" \
  "launch failed: 700 cudaErrorIllegalAddress" "$(run wild)"
expect "and the MMU fault is logged as Xid 31, a read of an unmapped page" "yes" \
  "$(xids '$p' | grep -qE ': 31, pid=[0-9]+, name=fault_datapath, Ch 00000008, intr 00000000\. MMU Fault: ENGINE GRAPHICS GPC0 GPCCLIENT_T1_0 faulted @ 0x[0-9a-f]+_[0-9a-f]{5}000\. Fault is of type FAULT_PDE ACCESS_TYPE_VIRT_READ$' && echo yes || echo no)"
expect "a misaligned read fails the kernel" \
  "launch failed: 716 cudaErrorMisalignedAddress" "$(run misaligned)"
expect "and the SM's exception is logged as Xid 13" "yes" \
  "$(xids '$p' | grep -qE ': 13, pid=[0-9]+, name=fault_datapath, Graphics SM Warp Exception on \(GPC 0, TPC 0, SM 0\): Misaligned Address$' && echo yes || echo no)"

"$vgpu" fault arm --bitflip --on copy >/dev/null
expect "a flip armed on copies corrupts what the copy back delivers" "mismatches: 1" "$(run)"
expect "and leaves memory as it was: the next copy is clean" "mismatches: 0" "$(run)"
"$vgpu" fault arm --ecc uncorrected --on copy >/dev/null
expect "an uncorrectable error fails the copy with error 214" \
  "copy failed: 214 cudaErrorECCUncorrectable" "$(run)"
expect "counted (since the reload above), logged as Xid 48 and 63" "1 yes" \
  "$(q ecc.errors.uncorrected.volatile.total) $(grep -q 'Xid (PCI:0000:01:00): 63' "$tmp/session/dmesg.log" && echo yes || echo no)"

"$vgpu" fault lose >/dev/null
expect "a launch on a GPU that has fallen off the bus fails" \
  "launch failed: 719 cudaErrorLaunchFailure" "$(run)"
"$vgpu" fault lose --clear >/dev/null
expect "and runs again once it is back" "mismatches: 0" "$(run)"

"$vgpu" fault arm --hang --seconds 2 >/dev/null
start=$(date +%s.%N)
got=$(run)
took=$(echo "$(date +%s.%N) - $start" | bc)
expect "a hung launch fails with a launch timeout" "launch failed: 702 cudaErrorLaunchTimeout" "$got"
expect "after the time it was armed for" "yes" "$( (( $(echo "$took >= 2" | bc) )) && echo yes || echo no)"
expect "and the next run is clean" "mismatches: 0" "$(run)"
exit $fail
