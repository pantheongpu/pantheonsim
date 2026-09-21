#!/usr/bin/env bash
# `vgpu fault`: errors injected into the machine and read back through every
# surface a health tool reads -- nvidia-smi's fields and table, nvidia-smi -p,
# and rocm-smi.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
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

# Inside a session the driver and the kernel log what happened, in their forms.
sess="$tmp/session"
mkdir -p "$sess"
printf '[    3.141593] nvidia 0000:01:00.0: enabling device (0000 -> 0003)\n' > "$sess/dmesg.log"
VGPU_SESSION="$sess" t4 fault inject --gpu 0 --ecc uncorrected >/dev/null
VGPU_SESSION="$sess" t4 fault inject --gpu 1 --pcie fatal >/dev/null
VGPU_SESSION="$sess" t4 fault inject --gpu 1 --pcie naks_sent >/dev/null
expect "an uncorrectable error is logged as Xid 48, with the page as Xid 63" "1 1" \
  "$(grep -c "NVRM: Xid (PCI:0000:01:00): 48, pid='<unknown>', name=<unknown>, An uncorrectable double bit error" "$sess/dmesg.log") $(grep -c 'NVRM: Xid (PCI:0000:01:00): 63, ' "$sess/dmesg.log")"
expect "a fatal PCIe error is logged by AER; a NAK is not" "1 4" \
  "$(grep -c 'pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received: 0000:02:00.0' "$sess/dmesg.log") $(wc -l < "$sess/dmesg.log" | tr -d ' ')"
expect "logged lines are stamped after the boot log" "yes" \
  "$(awk -F'[][]' 'NR > 1 && $2 + 0 <= 3.141593 { bad = 1 } END { print bad ? "no" : "yes" }' "$sess/dmesg.log")"

out=$(VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=1 "$vgpu" fault arm --ecc uncorrected 2>&1); rc=$?
expect "ECC faults cannot be armed on a card without ECC" "2 yes" "$rc $(grep -q 'has no ECC' <<< "$out" && echo yes || echo no)"
VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=1 "$vgpu" fault arm --bitflip >/dev/null
expect "a bit flip can be armed on any card" "0" "$?"
t4 fault inject --bitflip >/dev/null; expect "a bit flip is armed, not injected" "2" "$?"
t4 fault arm --ecc corrected --bitflip >/dev/null; expect "arm takes one kind at a time" "2" "$?"

# Clock-event reasons: every reading agrees with an injected throttle.
t4 fault throttle --gpu 0 --reason sw_thermal_slowdown,sw_power_cap >/dev/null
expect "throttle starts" "0" "$?"
expect "the reasons are active, with GPU idle, in the driver's mask" \
  "0x0000000000000025, Active, Active, Not Active" \
  "$(q 0 clocks_event_reasons.active,clocks_event_reasons.sw_thermal_slowdown,clocks_event_reasons.sw_power_cap,clocks_event_reasons.hw_slowdown)"
expect "a thermal slowdown reads at the T4's slowdown threshold, a power cap at the limit" "93, 70.00 W" \
  "$(q 0 temperature.gpu,power.draw)"
expect "the other GPU is not throttled" "0x0000000000000001" "$(q 1 clocks_event_reasons.active)"
expect "fault show names the reasons" "yes" \
  "$(t4 fault show --gpu 0 | grep -q 'clock-event reasons    sw_power_cap, sw_thermal_slowdown (until cleared)' && echo yes || echo no)"
# ctypes loads the NVML shim into python, which a sanitizer runtime refuses.
if [[ -e "$build/shim/libnvidia-ml.so.1" ]] && command -v python3 >/dev/null &&
   [[ -z "$(shim_sanitizer "$build/shim")" ]]; then
  got=$(VGPU_GPU=nvidia/t4 VGPU_DEVICE_COUNT=2 python3 -c '
import ctypes, sys
lib = ctypes.CDLL(sys.argv[1])
lib.nvmlInit_v2()
h = ctypes.c_void_p()
lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(h))
r = ctypes.c_ulonglong()
lib.nvmlDeviceGetCurrentClocksThrottleReasons(h, ctypes.byref(r))
print(hex(r.value))' "$build/shim/libnvidia-ml.so.1" 2>&1)
  expect "NVML reports the same reasons" "0x25" "$got"
fi
sleep 0.2
t4 fault throttle --gpu 0 --clear >/dev/null
expect "clearing ends them; the time they were active is kept" "0x0000000000000001 yes" \
  "$(q 0 clocks_event_reasons.active) $( [[ "$(t4 smi -i 0 --query-gpu=clocks_event_reasons_counters.sw_thermal_slowdown --format=csv,noheader,nounits)" -ge 100000 ]] && echo yes || echo no)"
t4 fault throttle --gpu 1 --reason hw_slowdown --seconds 1 >/dev/null
expect "a timed throttle is active" "Active" "$(q 1 clocks_event_reasons.hw_slowdown)"
sleep 1.3
expect "and ends by itself" "Not Active" "$(q 1 clocks_event_reasons.hw_slowdown)"
t4 fault throttle --reason gpu_idle >/dev/null; expect "GPU idle cannot be injected" "2" "$?"
t4 fault throttle --reason hw_slowdown --clear >/dev/null; expect "throttle takes --reason or --clear" "2" "$?"
t4 fault inject --ecc corrected --seconds 3 >/dev/null; expect "--seconds does not belong to inject" "2" "$?"

t4 fault inject --ecc sideways >/dev/null; expect "an unknown ECC kind is refused" "2" "$?"
t4 fault inject --gpu 9 --ecc corrected >/dev/null; expect "a GPU that is not there is refused" "2" "$?"
t4 fault inject --pcie replay --location l2_cache >/dev/null; expect "--location with --pcie is refused" "2" "$?"
t4 fault reset >/dev/null; expect "reset needs a lifetime" "2" "$?"
exit $fail
