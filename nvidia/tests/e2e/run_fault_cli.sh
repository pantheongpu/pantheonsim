#!/usr/bin/env bash
# `vgpu fault`: errors injected into the machine and read back through every
# surface a health tool reads -- nvidia-smi's fields and table, nvidia-smi -p,
# and rocm-smi.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
# A machine of its own: nothing here may touch the user's reliability state.
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
t4() { VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=2 "$vgpu" "$@" 2>&1; }
q() { t4 smi -i "$1" --query-gpu="$2" --format=csv,noheader; }

expect "a machine nobody injected into is clean" "0, 0, 0, No" \
  "$(q 0 ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.aggregate.total,retired_pages.dbe,retired_pages.pending)"
t4 fault inject --gpu 0 --ecc corrected --count 3 >/dev/null
expect "inject corrected errors" "0" "$?"
expect "counted in both lifetimes, on the injected GPU only" "3, 3, 0" \
  "$(q 0 ecc.errors.corrected.volatile.device_memory,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.volatile.total)"
expect "the other GPU is untouched" "0" "$(q 1 ecc.errors.corrected.volatile.total)"
t4 fault inject --gpu 0 --ecc uncorrected --location dram >/dev/null
expect "an uncorrected DRAM error retires a page on a GDDR card, pending" "1, 1, Yes" \
  "$(q 0 ecc.errors.uncorrected.volatile.total,retired_pages.dbe,retired_pages.pending)"
expect "the table's ECC column counts it" "yes" \
  "$(t4 smi | grep -q 'Off |                    1 |' && echo yes || echo no)"
t4 fault inject --gpu 0 --ecc corrected --location l2_cache --count 2 >/dev/null
expect "per-location counts and the total" "2, 5" \
  "$(q 0 ecc.errors.corrected.volatile.l2_cache,ecc.errors.corrected.volatile.total)"

t4 smi -i 0 -p 0 >/dev/null
expect "nvidia-smi -p 0 succeeds" "0" "$?"
expect "-p 0 zeroes volatile counts, keeps aggregate ones and the pending page" "0, 5, Yes" \
  "$(q 0 ecc.errors.corrected.volatile.total,ecc.errors.corrected.aggregate.total,retired_pages.pending)"
t4 fault reset --gpu 0 --volatile >/dev/null
expect "a driver reload completes the pending retirement" "1, No" "$(q 0 retired_pages.dbe,retired_pages.pending)"
t4 smi -i 0 -p 1 >/dev/null
expect "-p 1 zeroes aggregate counts; retired pages stay" "0, 1" \
  "$(q 0 ecc.errors.corrected.aggregate.total,retired_pages.dbe)"

h100() { VGPU_GPU=nvidia/h100 VGPU_DEVICE_COUNT=1 "$vgpu" "$@" 2>&1; }
h100 fault inject --ecc uncorrected >/dev/null
expect "an uncorrected error remaps a row on an HBM card" "1, Yes, [N/A]" \
  "$(h100 smi --query-gpu=remapped_rows.uncorrectable,remapped_rows.pending,retired_pages.dbe --format=csv,noheader)"

out=$(VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=1 "$vgpu" fault inject --ecc corrected 2>&1); rc=$?
expect "a card without ECC refuses ECC injection" "2 yes" \
  "$rc $(grep -q 'has no ECC' <<< "$out" && echo yes || echo no)"

t4 fault inject --gpu 1 --pcie replay --count 4 >/dev/null
expect "fault show lists PCIe errors" "yes" \
  "$(t4 fault show --gpu 1 | grep -q 'replay 4' && echo yes || echo no)"

amd() { VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=1 "$vgpu" "$@" 2>/dev/null; }
amd fault inject --ecc uncorrected --count 2 >/dev/null
amd fault inject --ecc corrected --location register_file >/dev/null
if command -v python3 >/dev/null; then
  got=$(amd smi --rocm --showrasinfo --json | python3 -c '
import json, sys
c = json.load(sys.stdin)["card0"]
print(c["UMC uncorrectable errors"], c["GFX correctable errors"])' 2>&1)
  expect "rocm-smi puts memory errors on UMC and on-chip ones on GFX" "2 1" "$got"
fi

t4 fault inject --ecc sideways >/dev/null; expect "an unknown ECC kind is refused" "2" "$?"
t4 fault inject --gpu 9 --ecc corrected >/dev/null; expect "a GPU that is not there is refused" "2" "$?"
t4 fault inject --pcie replay --location l2_cache >/dev/null; expect "--location with --pcie is refused" "2" "$?"
t4 fault reset >/dev/null; expect "reset needs a lifetime" "2" "$?"
exit $fail
