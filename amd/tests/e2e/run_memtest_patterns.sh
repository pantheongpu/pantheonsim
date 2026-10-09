#!/usr/bin/env bash
# Memory-test patterns (walking ones and zeros, address in address,
# checkerboard, moving inversions, seeded random data, block copies, strides)
# run by a hipcc-built program over simulated AMD device memory, and what a
# failing cell, bit flips on stores, an uncorrectable HBM error and a wild
# pointer look like to them. The program is amd/tests/hipcc/memtest.cpp, built
# by amd/tests/hipcc/build.sh for each architecture and checked in, so this
# needs no ROCm. The faults are armed with `vgpu fault`, the simulator's
# fault-injection hook.
#
#   amd/tests/e2e/run_memtest_patterns.sh
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
vgpu="$build/vgpu"
[[ -e "$shim/libamdhip64.so.6" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
[[ -x "$vgpu" ]] || { echo "SKIP: no vgpu at $vgpu"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
yes_if() { [[ "$2" == "yes" ]] && echo "ok    $1" || { echo "FAIL  $1"; fail=1; }; }

# The programs were built with the HIP runtime Ubuntu packages (ROCm 5.7's
# hipcc, the compiler that supports every architecture below), so they ask for
# libamdhip64.so.5; the shim stands in for it under that name, in a directory
# of its own so nothing else in the shim's directory is renamed.
libs="$tmp/libs"; mkdir -p "$libs"
for f in "$shim"/*; do ln -s "$(cd "$shim" && realpath "$f")" "$libs/$(basename "$f")"; done
ln -sf "$(cd "$shim" && realpath libamdhip64.so.6)" "$libs/libamdhip64.so.5"
# A sanitizer-instrumented shim needs its runtime loaded first.
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libamdhip64.so.6" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi
export LD_PRELOAD="$preload"

mkdir -p "$tmp/session"
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_SESSION="$tmp/session"
export VGPU_DEVICE_COUNT=1 VGPU_QUIET=1
v() { "$vgpu" "$@" 2>&1; }
# The count a test printed, by its name.
errors() { sed -n "s/^test $1 *errors: \([0-9]*\).*/\1/p" <<<"$2" | head -1; }

# gpu arch wave-size ecc(yes: HBM with ECC) device-name
while read -r gpu arch wave ecc name; do
  exe="$root/amd/tests/hipcc/memtest.$arch"
  echo "--- $gpu ($arch, wave $wave)"
  if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    expect "the hipcc-built program is in the repository" "yes" \
      "$(git -C "$root" ls-files --error-unmatch "amd/tests/hipcc/memtest.$arch" >/dev/null 2>&1 && echo yes || echo no)"
  fi
  export VGPU_GPU=$gpu
  run() { LD_LIBRARY_PATH="$libs" "$exe" "$@" 2>&1; }
  status() { LD_LIBRARY_PATH="$libs" "$exe" "$@" >/dev/null 2>&1; echo $?; }
  v fault stuck --clear >/dev/null; v fault arm --bitflip --rate 0 >/dev/null

  out=$(run)
  expect "it names the device and its wave size" "device $name warp $wave" "$(grep '^device ' <<<"$out")"
  expect "a clean machine passes every pattern" "total errors: 0" "$(grep '^total' <<<"$out")"
  expect "and reports PASS" "PASS" "$(tail -1 <<<"$out")"
  expect "all ten tests ran" "10" "$(grep -c '^test ' <<<"$out")"
  expect "mapped host memory passes them too" "total errors: 0" "$(run mapped | grep -E '^(total|SKIP)')"
  expect "a bit flipped by the host is found exactly, in each of three bit positions" \
    "flip: PASS" "$(run flip | tail -1)"

  # A cell stuck at 1: the first allocation is the buffer, at the start of
  # device memory. Word 4 (offset 0x10), bit 0, stuck at 1.
  v fault stuck --offset 0x10 --bit 0 --value 1 >/dev/null
  out=$(run)
  yes_if "a stuck-at-1 cell fails the suite" "$([[ "$out" != *"total errors: 0"* ]] && echo yes || echo no)"
  yes_if "the walking-zeros test sees it" "$([[ "$(errors 'walking zeros' "$out")" -ge 1 ]] && echo yes || echo no)"
  yes_if "the walking-ones test sees it in the patterns with bit 0 clear" \
    "$([[ "$(errors 'walking ones' "$out")" -ge 1 ]] && echo yes || echo no)"
  expect "the first bad word is word 4" "yes" \
    "$(grep -q '^test walking zeros .*first at word 4)' <<<"$out" && echo yes || echo no)"
  expect "the exit status is 1" "1" "$(status)"
  v fault stuck --clear >/dev/null
  expect "cleared, the machine passes again" "total errors: 0" "$(run | grep '^total')"

  # Bit flips armed on stores. The first stores the counter sees belong to the
  # runtime's own bookkeeping, not to the program, so enough are armed to
  # reach the program's.
  v fault arm --bitflip --count 1000 --on store >/dev/null
  out=$(run)
  yes_if "bit flips armed on stores are caught by the verify reads" \
    "$([[ "$out" != *"total errors: 0"* && "$out" != *"launch failed"* ]] && echo yes || echo no)"
  v fault arm --bitflip --rate 0 >/dev/null
  expect "and stop when disarmed" "total errors: 0" "$(run | grep '^total')"

  if [[ $ecc == yes ]]; then
    # Uncorrectable HBM ECC on a load: the kernel fails, and the row is
    # remapped. A corrected error is counted and changes nothing.
    v fault arm --ecc corrected --count 3 --on load >/dev/null
    expect "corrected ECC errors on loads change nothing" "total errors: 0" "$(run | grep '^total')"
    expect "and are counted" "corrected 3" \
      "$(v fault show | awk '/^    corrected/ {print "corrected " $2}')"
    v fault arm --ecc uncorrected --count 1 --on load >/dev/null
    out=$(run)
    expect "an uncorrectable error on a load fails the kernel" "launch failed: 719 hipErrorLaunchFailure (walking)" \
      "$(grep '^launch failed' <<<"$out" | head -1)"
    expect "and the card counts it and remaps a row" "yes" \
      "$([[ "$(v fault show)" == *'uncorrectable 1, pending yes'* ]] && echo yes || echo no)"
    v fault arm --ecc uncorrected --count 1 --on load >/dev/null
    expect "the exit status says so" "3" "$(status)"
    expect "the next run, with nothing armed, passes" "total errors: 0" "$(run | grep '^total')"
    v fault reset --volatile >/dev/null
  else
    expect "a card without ECC refuses an ECC fault" "yes" \
      "$([[ "$(v fault arm --ecc uncorrected --on load)" == *'has no ECC'* ]] && echo yes || echo no)"
  fi

  expect "a pointer to nothing fails the verify kernel" "launch failed: 719 hipErrorLaunchFailure (wild)" \
    "$(run wild | grep '^launch failed')"
  expect "and the program's exit status says so" "3" "$(status wild)"
done <<'GPUS'
amd/mi300x gfx942 64 yes AMD Instinct MI300X
amd/mi250x gfx90a 64 yes AMD Instinct MI250X
amd/rx6900xt gfx1030 32 no AMD Radeon RX 6900 XT
amd/rx6800 gfx1030 32 no AMD Radeon RX 6800
amd/rx6700xt gfx1031 32 no AMD Radeon RX 6700 XT
amd/rx7900xtx gfx1100 32 no AMD Radeon RX 7900 XTX
GPUS
exit $fail
