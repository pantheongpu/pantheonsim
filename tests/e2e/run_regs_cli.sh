#!/usr/bin/env bash
# `vgpu regs` and the configuration space it serves: shared across processes,
# logged, and -- where lspci is installed -- rendered by the real lspci from
# `vgpu smi --lspci-dump`, capabilities, degraded link and AER status included.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
h100() { VGPU_GPU=nvidia/h100 VGPU_DEVICE_COUNT=2 "$vgpu" "$@" 2>&1; }
val() { h100 regs read "$@" | head -1 | awk '{print $4}'; }

expect "the database lists every register, each done or a model" "yes" \
  "$(h100 regs list | awk 'NR > 1 { n++; if ($NF != "done" && $NF != "model") bad = 1 } END { print (n > 40 && !bad) ? "yes" : "no" }')"
expect "a read names and decodes the register" "0x08a link_status = 0x1105" \
  "$(h100 regs read link_status | head -1 | awk '{print $1, $2, $3, $4}')"
expect "and its fields" "16" "$(h100 regs read link_status | awk '$2 == "negotiated_link_width" {print $3}')"
h100 regs write --gpu 1 command 0x0002 >/dev/null
expect "a write in one process is what the next one reads" "0x0002 0x0006" "$(val --gpu 1 command) $(val --gpu 0 command)"
h100 regs write bar0 0xffffffff >/dev/null
expect "a BAR answers a sizing probe with its size (16 MiB)" "0xff000000" "$(val bar0)"
h100 fault inject --pcie bad_tlp >/dev/null
expect "an injected error sets its AER status bit" "0x00000040" "$(val aer_correctable_status)"
h100 regs write aer_correctable_status 0x40 >/dev/null
expect "and writing 1 clears it" "0x00000000" "$(val aer_correctable_status)"
expect "the log names each access and its process" "write 0x110 aer_correctable_status vgpu" \
  "$(h100 regs log --last 2 | head -1 | awk '{print $5, $6, $7, $2}')"
h100 regs read nothing >/dev/null; expect "an unknown register is refused" "2" "$?"
h100 regs read 0x001 --size 2 >/dev/null; expect "a misaligned access is refused" "2" "$?"
h100 regs read --gpu 7 link_status >/dev/null; expect "a GPU that is not there is refused" "2" "$?"

# A machine something publishes, as a session's is: the PCI device's sysfs
# files are written from the registers, and change with them.
sess="$tmp/session"; mkdir -p "$sess"
VGPU_GPU=nvidia/h100 VGPU_QUIET=1 "$vgpu" serve --gpu nvidia/h100 --count 2 >/dev/null 2>&1 &
serve=$!
trap 'kill $serve 2>/dev/null; wait $serve 2>/dev/null; rm -rf "$tmp"' EXIT
for _ in $(seq 50); do [[ -n "$(ls "$tmp/run" 2>/dev/null | grep -v '^ras-\|^regs-')" ]] && break; sleep 0.1; done
uuid=$(h100 smi -i 1 --query-gpu=uuid --format=csv,noheader)
files="$sess/sysfs/$uuid"
VGPU_SESSION="$sess" h100 fault link --gpu 1 --width 8 >/dev/null
expect "the device's sysfs files come from its registers" "0x10de 0x030200 32.0 GT/s PCIe 8 16" \
  "$(cat "$files/vendor" "$files/class" "$files/current_link_speed" "$files/current_link_width" "$files/max_link_width" | paste -sd' ')"
expect "config is the whole space, byte for byte" "4096 de10" \
  "$(stat -c %s "$files/config") $(od -An -tx1 -N2 "$files/config" | tr -d ' ')"
expect "resource lists the BARs as the kernel does" "0x00000000e2000000 0x00000000e2ffffff 0x0000000000040200" \
  "$(head -1 "$files/resource")"
VGPU_SESSION="$sess" h100 regs write --gpu 1 command 0x0002 >/dev/null
expect "a register write reaches the config file" "02 00" "$(od -An -tx1 -j4 -N2 "$files/config" | sed 's/^ //')"
"$build/bin/nvidia-smi" -i 1 --query-gpu=pcie.link.width.current --format=csv,noheader >/dev/null 2>&1
expect "nvidia-smi reads the link through the registers, and the log names it" \
  "nvidia-smi link_capabilities nvidia-smi link_status" \
  "$(h100 regs log --gpu 1 --last 2 | awk '{printf "%s%s %s", sep, $2, $7; sep = " "}')"
VGPU_SESSION="$sess" h100 fault link --gpu 1 --clear >/dev/null
kill $serve 2>/dev/null; wait $serve 2>/dev/null

# An AMD GPU's MMIO registers: engine status, and the SMU mailbox.
mi() { VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=1 "$vgpu" "$@" 2>&1; }
mval() { mi regs read --space mmio "$@" | head -1 | awk '{print $4}'; }
expect "mmio lists the registers with where each offset comes from" "0x08010 grbm_status" \
  "$(mi regs list --space mmio | awk '$4 == "grbm_status" {print $1, $4}')"
expect "an idle GPU's GRBM_STATUS: FIFOs available, DB and CB clean" "0x00003028" "$(mval grbm_status)"
mi regs write --space mmio smu_response 0 >/dev/null
mi regs write --space mmio smu_argument 41 >/dev/null
mi regs write --space mmio smu_message 0x1 >/dev/null
expect "the SMU answers a test message with its argument plus one" "0x00000001 0x0000002a" \
  "$(mval smu_response) $(mval smu_argument)"
mi regs write --space mmio smu_message 0x77 >/dev/null
expect "and refuses a message it does not know" "0x000000fe" "$(mval smu_response)"
h100 regs read --space mmio grbm_status >/dev/null; expect "an NVIDIA GPU has no MMIO modelled yet" "2" "$?"
mi regs read --space mmio 0x8012 >/dev/null; expect "an MMIO access is a whole aligned dword" "2" "$?"
VGPU_QUIET=1 "$vgpu" serve --gpu amd/mi300x --count 1 --load 0.9 >/dev/null 2>&1 &
loaded=$!
busy=""
for _ in $(seq 50); do busy=$(mval grbm_status); [[ "$busy" == 0x8* || "$busy" == 0xe* ]] && break; sleep 0.1; done
kill $loaded 2>/dev/null; wait $loaded 2>/dev/null
expect "a busy GPU's GRBM_STATUS has GUI_ACTIVE and CP busy set" "yes" \
  "$( (( (busy >> 31) & 1 && (busy >> 29) & 1 )) && echo yes || echo no)"

if command -v lspci >/dev/null; then
  h100 fault link --gpu 0 --gen 3 --width 8 >/dev/null
  h100 fault inject --gpu 0 --pcie replay >/dev/null
  h100 smi --lspci-dump > "$tmp/dump"
  out=$(lspci -F "$tmp/dump" -vvv -s 01:00.0 2>&1)
  expect "lspci renders the capability chain" "3" \
    "$(grep -cE 'Capabilities: \[(60|68|78)\] (Power Management|MSI|Express)' <<< "$out")"
  expect "a degraded link, as lspci reports one" "LnkSta: Speed 8GT/s (downgraded), Width x8 (downgraded)" \
    "$(grep -o 'LnkSta:.*' <<< "$out" | tr -s '\t ' ' ')"
  expect "and the replay timeout in AER" "yes" "$(grep -q 'CESta:.*Timeout+' <<< "$out" && echo yes || echo no)"
  expect "the second GPU is untouched" "yes" \
    "$(lspci -F "$tmp/dump" -vvv -s 02:00.0 | grep -q 'LnkSta:.*Speed 32GT/s, Width x16$' && echo yes || echo no)"
fi
exit $fail
