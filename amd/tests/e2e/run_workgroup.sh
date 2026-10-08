#!/usr/bin/env bash
# Work-groups of more than one dimension and of several waves, run by a
# hipcc-built program on four simulated AMD GPUs: every work-item gets its own
# ids and the waves see each other's LDS writes. The programs were built by
# amd/tests/hipcc/build.sh with the Ubuntu 24.04 hipcc (ROCm 5.7.1), whose
# code objects are version 4: gfx90a, gfx942 and gfx1100 pack a work-item's
# ids into v0 whatever the version (gfx1030 has a register per id), and a
# simulator that decided by the version ran every wave of a 64x2 block as y = 0.
#
#   amd/tests/e2e/run_workgroup.sh
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
root="$(cd "$(dirname "$0")/../../.." && pwd)"
shim="$build/shim"
[[ -e "$shim/libamdhip64.so.6" ]] || { echo "SKIP: no HIP shim in $shim"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
# The programs ask for libamdhip64.so.5 (ROCm 5.7's hipcc); the shim stands in for it under that name.
libs="$tmp/libs"; mkdir -p "$libs"
for f in "$shim"/*; do ln -s "$(cd "$shim" && realpath "$f")" "$libs/$(basename "$f")"; done
ln -sf "$(cd "$shim" && realpath libamdhip64.so.6)" "$libs/libamdhip64.so.5"
preload=""
if command -v objdump >/dev/null; then
  preload=$(objdump -p "$shim/libamdhip64.so.6" 2>/dev/null | awk '/NEEDED/ && /lib(asan|tsan)\.so/ {print $2}')
fi
export LD_PRELOAD="$preload"
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_DEVICE_COUNT=1 VGPU_QUIET=1

# gpu arch wave-size device-name
while read -r gpu arch wave name; do
  exe="$root/amd/tests/hipcc/workgroup.$arch"
  echo "--- $gpu ($arch, wave $wave)"
  if git -C "$root" rev-parse --git-dir >/dev/null 2>&1; then
    expect "the hipcc-built program is in the repository" "yes" \
      "$(git -C "$root" ls-files --error-unmatch "amd/tests/hipcc/workgroup.$arch" >/dev/null 2>&1 && echo yes || echo no)"
  fi
  out=$(VGPU_GPU=$gpu LD_LIBRARY_PATH="$libs" "$exe" 2>&1)
  expect "it names the device and its wave size" "device $name warp $wave" "$(grep '^device ' <<<"$out")"
  expect "every block shape gives every work-item its own ids" "0" \
    "$(grep '^block' <<<"$out" | grep -vc 'wrong ids 0 of')"
  expect "and the waves of a work-group see each other's LDS writes" "0" \
    "$(grep '^block' <<<"$out" | grep -vc 'LDS exchange wrong 0$')"
  expect "all nine block shapes ran" "9" "$(grep -c '^block' <<<"$out")"
  expect "the program reports PASS" "PASS" "$(tail -1 <<<"$out")"
done <<'GPUS'
amd/mi300x gfx942 64 AMD Instinct MI300X
amd/mi250x gfx90a 64 AMD Instinct MI250X
amd/rx6900xt gfx1030 32 AMD Radeon RX 6900 XT
amd/rx7900xtx gfx1100 32 AMD Radeon RX 7900 XTX
GPUS
exit $fail
