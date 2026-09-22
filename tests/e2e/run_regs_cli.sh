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
