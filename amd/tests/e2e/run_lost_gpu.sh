#!/usr/bin/env bash
# An AMD GPU that falls off the bus, as amdgpu loses one: the kernel log has
# the fatal AER error, the failed slot reset and amdgpu's permanent-failure
# callback; rocm-smi, amd-smi and rocm_agent_enumerator no longer list it and
# the GPU after it moves up; its registers read as all ones and writes go
# nowhere; the other GPU is untouched; and --clear brings it back.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state" VGPU_SESSION="$tmp/session"
export VGPU_GPU=amd/mi300x VGPU_DEVICE_COUNT=2
mkdir -p "$VGPU_SESSION"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
v() { "$vgpu" "$@" 2>&1; }
reg() { v regs read "$@" | head -1 | awk '{print $4}'; }

before_bdf=$(v smi --amd list | awk '/BDF:/ {print $2}' | paste -sd' ')
expect "amd-smi lists both GPUs" "0000:01:00.0 0000:02:00.0" "$before_bdf"

expect "fault lose takes an AMD GPU off the bus" "GPU 0 (AMD Instinct MI300X) has fallen off the bus." \
  "$(v fault lose --gpu 0)"
expect "the kernel log has the PCI core's recovery and amdgpu's answers, in order" \
  "pcieport 0000:00:01.0: AER: Uncorrected (Fatal) error received: 0000:01:00.0|amdgpu 0000:01:00.0: amdgpu: PCI error: detected callback!!|amdgpu 0000:01:00.0: amdgpu: pci_channel_io_frozen: state(2)!!|pcieport 0000:00:01.0: AER: subordinate device reset failed|amdgpu 0000:01:00.0: amdgpu: PCI error: detected callback!!|amdgpu 0000:01:00.0: amdgpu: pci_channel_io_perm_failure: state(3)!!|pcieport 0000:00:01.0: AER: device recovery failed" \
  "$(sed 's/^\[[^]]*\] //' "$VGPU_SESSION/dmesg.log" | paste -sd'|')"
expect "and no NVIDIA Xid" "0" "$(grep -c NVRM "$VGPU_SESSION/dmesg.log")"
expect "losing it again logs nothing more" "7" "$(v fault lose --gpu 0 >/dev/null; wc -l < "$VGPU_SESSION/dmesg.log" | tr -d ' ')"

expect "amd-smi lists only the GPU left, as GPU 0" "GPU: 0|BDF: 0000:02:00.0" \
  "$(v smi --amd list | grep -E 'GPU:|BDF:' | sed 's/^ *//' | paste -sd'|')"
expect "rocm-smi's table has one row" "1" "$(v smi --rocm | grep -cE '^[0-9]+ ')"
expect "rocm-smi -d 1 finds no second GPU" "2" "$(v smi --rocm -d 1 >/dev/null 2>&1; echo $?)"
expect "rocm_agent_enumerator lists one GPU agent" "gfx000|gfx942" "$(v smi --agents | paste -sd'|')"

expect "its configuration space reads as all ones" "0xffff 0xffffffff" \
  "$(reg --gpu 0 vendor_id) $(reg --gpu 0 --space mmio grbm_status)"
v regs write --gpu 0 command 0x0002 >/dev/null
expect "a write goes nowhere" "0xffff" "$(reg --gpu 0 command)"
expect "the other GPU still answers" "0x1002" "$(reg --gpu 1 vendor_id)"
expect "fault show says why" "yes" \
  "$(v fault show --gpu 0 | grep -q 'fallen off (PCIe recovery failed)' && echo yes || echo no)"

expect "--clear brings it back" "GPU 0 (AMD Instinct MI300X) is back on the bus." "$(v fault lose --gpu 0 --clear)"
expect "amd-smi lists both again" "$before_bdf" "$(v smi --amd list | awk '/BDF:/ {print $2}' | paste -sd' ')"
expect "and its registers answer" "0x1002 0x0406" "$(reg --gpu 0 vendor_id) $(reg --gpu 0 command)"
exit $fail
