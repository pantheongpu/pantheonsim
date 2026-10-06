#!/usr/bin/env bash
# NVML and nvidia-smi as a GPU health tool uses them: ECC counts, retired pages,
# remapped rows, clock-event reasons and their times, PCIe replays, NVLink and
# accounting, on cards that differ in what they have. nvml_health.py drives
# NVML through ctypes; the -q sections of nvidia-smi are checked here against
# the same injected errors.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "SKIP: vgpu not built"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
# A machine of its own per card: nothing here may touch the user's reliability
# state, and no live process publishes telemetry, so the profile answers.
machine() {  # machine <profile> [count]
  rm -rf "$tmp/run" "$tmp/state"
  export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_BIN="$vgpu"
  export VGPU_GPU="$1" VGPU_DEVICE_COUNT="${2:-1}"
}
smi() { "$vgpu" smi "$@" 2>&1; }
count() { grep -cE "$1" <<< "$2"; }
# "    Double Bit ECC     : 1" lines as "Double Bit ECC=1", joined with |, so a
# check does not depend on how far the report pads its colons.
kv() { grep -E "$1" | sed -E 's/^ +//; s/ +: +/=/' | paste -sd'|'; }
# The value of the first line named exactly <key>.
val() { awk -F' +: ' -v k="$1" '{ sub(/^ +/, "", $1) } $1 == k { print $2; exit }'; }

# ---- NVML ------------------------------------------------------------------
# ctypes loads the shim into python, which a sanitizer runtime refuses.
if [[ ! -e "$build/shim/libnvidia-ml.so.1" ]]; then echo "SKIP: NVML shim not built (NVML checks)"
elif ! command -v python3 >/dev/null; then echo "SKIP: no python3 (NVML checks)"
elif [[ -n "$(shim_sanitizer "$build/shim")" ]]; then echo "SKIP: sanitizer shim cannot load into python (NVML checks)"
else
  for card in t4 l4 h100 a100 rtx3060; do
    machine "nvidia/$card"
    python3 "$root/nvidia/tests/e2e/nvml_health.py" "$build/shim" "$card" || fail=1
  done
fi

# ---- nvidia-smi -q sections --------------------------------------------------
# T4: ECC, retires pages, no NVLink.
machine nvidia/t4 2
q=$(smi -i 0 -q)
expect "-q has the ECC Errors, Retired Pages, Remapped Rows and Clocks Event Reasons sections" "1 1 1 1" \
  "$(count '^    ECC Errors$' "$q") $(count '^    Retired Pages$' "$q") $(count '^    Remapped Rows +: N/A$' "$q") $(count '^    Clocks Event Reasons$' "$q")"
expect "a clean T4's retired pages" "Single Bit ECC=0|Double Bit ECC=0|Pending Page Blacklist=No" \
  "$(smi -i 0 -q -d PAGE_RETIREMENT | kv 'Single|Double|Pending')"
expect "a T4 has no remapped rows" "Remapped Rows=N/A" "$(smi -i 0 -q -d ROW_REMAPPER | kv Remapped)"
"$vgpu" fault inject --gpu 0 --ecc corrected --location l2_cache --count 2 >/dev/null
"$vgpu" fault inject --gpu 0 --ecc uncorrected --location dram >/dev/null
"$vgpu" fault inject --gpu 0 --pcie replay --count 3 >/dev/null
ecc=$(smi -i 0 -q -d ECC)
expect "ECC Errors: L2 corrected twice and device memory uncorrected once, volatile and aggregate" "2 2 4" \
  "$(count 'L2 Cache +: 2$' "$ecc") $(count 'Device Memory +: 1$' "$ecc") $(count 'Total +: [12]$' "$ecc")"
expect "ECC Errors: single and double bit, volatile and aggregate" "2 2 2" \
  "$(count 'Single Bit$' "$ecc") $(count 'Double Bit$' "$ecc") $(count '^        (Volatile|Aggregate)$' "$ecc")"
expect "-d ECC prints no identity lines" "0" "$(count 'Product Name' "$ecc")"
expect "the other GPU's ECC Errors are clean" "0" "$(count ': [1-9]' "$(smi -i 1 -q -d ECC | sed -n '/ECC Errors/,$p')")"
expect "an uncorrected DRAM error retires a page, pending" "Double Bit ECC=1|Pending Page Blacklist=Yes" \
  "$(smi -i 0 -q -d PAGE_RETIREMENT | kv 'Double|Pending')"
expect "PCIe replays are in the PCI block" "3" "$(smi -i 0 -q | val 'Replays Since Reset')"
smi -i 0 -r >/dev/null; rc=$?
expect "-r resets the GPU" "0" "$rc"
expect "a reset completes the pending retirement; the retired page stays" \
  "Double Bit ECC=1|Pending Page Blacklist=No" "$(smi -i 0 -q -d PAGE_RETIREMENT | kv 'Double|Pending')"
ecc=$(smi -i 0 -q -d ECC)
expect "a reset zeroes the volatile counts and keeps the aggregate ones" "0 1" \
  "$(count 'L2 Cache +: [1-9]' "$(sed -n '/Volatile/,/Aggregate/p' <<< "$ecc")") $(count 'L2 Cache +: 2$' "$(sed -n '/Aggregate/,$p' <<< "$ecc")")"
out=$(smi -r); rc=$?
expect "-r with no GPU selected resets them all" "0 GPU 00000000:01:00.0 was successfully reset.|GPU 00000000:02:00.0 was successfully reset.|All done." \
  "$rc $(paste -sd'|' <<< "$out")"
smi -i 5 -r >/dev/null; rc=$?
expect "-r on a GPU that is not there" "6" "$rc"

# Clock-event reasons and their times.
cer=$(smi -i 0 -q -d PERFORMANCE)
expect "an idle GPU: P8, idle the only reason, no time held down" "P8 Active Not Active 0 us" \
  "$(val 'Performance State' <<< "$cer") $(val Idle <<< "$cer") $(val 'SW Power Cap' <<< "$cer") $(val 'SW Power Capping' <<< "$cer")"
"$vgpu" fault throttle --gpu 0 --reason sw_power_cap,hw_thermal_slowdown >/dev/null
sleep 0.3
cer=$(smi -i 0 -q -d PERFORMANCE)
expect "an injected throttle is Active in the section" "Active Active Not Active" \
  "$(val 'SW Power Cap' <<< "$cer") $(val 'HW Thermal Slowdown' <<< "$cer") $(val 'SW Thermal Slowdown' <<< "$cer")"
"$vgpu" fault throttle --gpu 0 --clear >/dev/null
cer=$(smi -i 0 -q -d PERFORMANCE)
held=$(val 'SW Power Capping' <<< "$cer" | awk '{print $1}')
expect "cleared: not active, and the time it was active is kept" "Not Active yes" \
  "$(val 'SW Power Cap' <<< "$cer") $([[ $held -ge 200000 ]] && echo yes || echo no)"
expect "Clocks Event Reasons Counters is in the full report too" "1" \
  "$(count '^    Clocks Event Reasons Counters$' "$(smi -i 0 -q)")"

# Power and temperature limits.
pw=$(smi -i 0 -q -d POWER)
expect "T4 power limits: 70 W, and half of it as the least a limit can be" \
  "Current Power Limit=70.00 W|Requested Power Limit=70.00 W|Default Power Limit=70.00 W|Min Power Limit=35.00 W|Max Power Limit=70.00 W" \
  "$(kv 'Power Limit' <<< "$pw")"
expect "temperature thresholds: the T4's slowdown, and shutdown above it" "93 98" \
  "$(smi -i 0 -q -d TEMPERATURE | val 'GPU Slowdown Temp' | awk '{print $1}') $(smi -i 0 -q -d TEMPERATURE | val 'GPU Shutdown Temp' | awk '{print $1}')"
expect "accounting is off by default" "Disabled 4000" \
  "$(smi -i 0 -q -d ACCOUNTING | val 'Accounting Mode') $(smi -i 0 -q -d ACCOUNTING | val 'Accounting Mode Buffer Size')"

# H100: HBM, remaps rows, NVLink 4 x 18.
machine nvidia/h100
expect "an H100 has no retired pages" "Single Bit ECC=N/A|Double Bit ECC=N/A|Pending Page Blacklist=N/A" \
  "$(smi -q -d PAGE_RETIREMENT | kv 'Single|Double|Pending')"
expect "a clean H100's remapped rows" \
  "Correctable Error=0|Uncorrectable Error=0|Pending=No|Remapping Failure Occurred=No" \
  "$(smi -q -d ROW_REMAPPER | kv '(Correctable|Uncorrectable) Error|Pending|Failure')"
"$vgpu" fault inject --ecc uncorrected >/dev/null
expect "an uncorrected error remaps a row, pending" "Uncorrectable Error=1|Pending=Yes" \
  "$(smi -q -d ROW_REMAPPER | kv 'Uncorrectable Error|Pending')"
expect "the bank histogram is N/A: no profile has a bank count" "5" \
  "$(smi -q -d ROW_REMAPPER | grep -cE '(Max|High|Partial|Low|None) +: N/A')"
smi -r >/dev/null
expect "a reset applies the remap, and the row stays remapped" "Uncorrectable Error=1|Pending=No" \
  "$(smi -q -d ROW_REMAPPER | kv 'Uncorrectable Error|Pending')"
nv=$(smi nvlink -s)
expect "nvlink -s: eighteen links at NVLink 4's speed" "GPU 0: NVIDIA H100 80GB HBM3|18|18" \
  "$(head -1 <<< "$nv" | sed 's/ (UUID.*//')|$(count 'Link [0-9]+: 26.562 GB/s' "$nv")|$(count 'Link [0-9]+:' "$nv")"
nv=$(smi nvlink -e)
expect "nvlink -e: replay, recovery and CRC errors per link, all zero" "54 54" \
  "$(count 'Link [0-9]+: (Replay|Recovery|CRC) Errors: 0$' "$nv") $(count 'Link [0-9]+:' "$nv")"
smi nvlink -x >/dev/null; expect "nvlink takes -s and -e only" "2" "$?"

# A100: NVLink 3 x 12 at 25 GB/s.
machine nvidia/a100
expect "nvlink -s on an A100: twelve links at 25 GB/s" "12" "$(count 'Link [0-9]+: 25 GB/s' "$(smi nvlink -s)")"

# L4: GDDR, but Ada remaps rows.
machine nvidia/l4
"$vgpu" fault inject --ecc uncorrected >/dev/null
expect "an L4 remaps rows although its memory is GDDR, and retires no page" \
  "Uncorrectable Error=1|Pending=Yes Double Bit ECC=N/A" \
  "$(smi -q -d ROW_REMAPPER | kv 'Uncorrectable Error|Pending') $(smi -q -d PAGE_RETIREMENT | kv Double)"
expect "and has no NVLink: its GPU is named, no link is listed" "1 0" \
  "$(count '^GPU 0' "$(smi nvlink -s)") $(count 'Link' "$(smi nvlink -s)")"

# An RTX 3060: no ECC, so nothing to count, retire or remap.
machine nvidia/rtx3060
ecc=$(smi -q -d ECC)
expect "no ECC: Current and Pending are N/A, and so is every count" "N/A N/A 0" \
  "$(val Current <<< "$ecc") $(val Pending <<< "$ecc") $(sed -n '/ECC Errors/,$p' <<< "$ecc" | grep -cE ': [0-9]')"
expect "no ECC: retired pages and remapped rows are N/A" \
  "Single Bit ECC=N/A|Double Bit ECC=N/A|Pending Page Blacklist=N/A Remapped Rows=N/A" \
  "$(smi -q -d PAGE_RETIREMENT | kv 'Single|Double|Pending') $(smi -q -d ROW_REMAPPER | kv Remapped)"
out=$(smi -p 0)
expect "no ECC: -p says so" "yes" "$(grep -q 'not supported' <<< "$out" && echo yes || echo no)"

# A GPU off the bus cannot be reset.
machine nvidia/t4 2
"$vgpu" fault lose --gpu 1 >/dev/null
smi -i 1 -r >/dev/null; rc=$?
expect "-r on a lost GPU: exit 15, as nvidia-smi's other lost-GPU paths" "15" "$rc"

# The -d sections that stay refused are refused by name.
smi -q -d ENCODER_STATS >/dev/null; expect "-d ENCODER_STATS is still a section with no data" "2" "$?"

exit $fail
