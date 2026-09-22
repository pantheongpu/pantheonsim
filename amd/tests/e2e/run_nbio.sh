#!/usr/bin/env bash
# An AMD GPU's NBIO registers behind BAR5 -- the strap amdgpu reads the
# revision from, the VRAM size it sizes memory by, and the compute and memory
# partition modes -- and the partition files amdgpu publishes from them in
# sysfs (current_compute_partition, current_memory_partition,
# available_memory_partition), in a session's sysfs.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
tmp=$(mktemp -d)
serve=""
trap '[[ -n "$serve" ]] && kill $serve 2>/dev/null; wait 2>/dev/null; rm -rf "$tmp"' EXIT
export VGPU_TELEMETRY_PATH="$tmp/run" VGPU_STATE_DIR="$tmp/state"
fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
reg() { VGPU_GPU=$1 "$vgpu" regs read --space mmio "$2" 2>&1 | head -1 | awk '{print $4}'; }
field() { VGPU_GPU=$1 "$vgpu" regs read --space mmio "$2" 2>&1 | awk -v f="$3" '$2 == f {print $3}'; }

expect "MI300X's strap: its device ID, the function enabled" "0x100074a1" "$(reg amd/mi300x nbio_strap0)"
expect "and decoded by field" "29857 1" \
  "$(field amd/mi300x nbio_strap0 strap_device_id) $(field amd/mi300x nbio_strap0 strap_func_en)"
expect "CONFIG_MEMSIZE is the VRAM in MiB: 192 GiB, 288 GiB" "196608 294912" \
  "$(( $(reg amd/mi300x nbio_config_memsize) )) $(( $(reg amd/mi350x nbio_config_memsize) ))"
expect "VGPU_VRAM_MB is what the register says too" "0x00004000" \
  "$(VGPU_VRAM_MB=16384 VGPU_GPU=amd/mi300x "$vgpu" regs read --space mmio nbio_config_memsize | head -1 | awk '{print $4}')"
expect "NPS1 and NPS4 on CDNA3, NPS1 and NPS2 on CDNA4" "0x00000009 0x00000009 0x00000003" \
  "$(reg amd/mi300x nbio_partition_mem_cap) $(reg amd/mi325x nbio_partition_mem_cap) $(reg amd/mi350x nbio_partition_mem_cap)"
expect "SPX and NPS1, as a card ships" "0x00000000 0x00000010" \
  "$(reg amd/mi300x nbio_partition_compute_status) $(reg amd/mi300x nbio_partition_mem_status)"

# A session's sysfs, written while a GPU is published.
sess="$tmp/session"; mkdir -p "$sess"
VGPU_QUIET=1 "$vgpu" serve --gpu amd/mi300x --count 1 >/dev/null 2>&1 &
serve=$!
for _ in $(seq 50); do [[ -n "$(ls "$tmp/run" 2>/dev/null | grep -v '^ras-\|^regs-')" ]] && break; sleep 0.1; done
uuid=$(VGPU_GPU=amd/mi300x "$vgpu" smi --query-gpu=uuid --format=csv,noheader)
VGPU_SESSION="$sess" VGPU_GPU=amd/mi300x "$vgpu" fault link --gpu 0 --clear >/dev/null
files="$sess/sysfs/$uuid"
expect "amdgpu's partition files, from those registers" "SPX|NPS1|NPS1, NPS4" \
  "$(cat "$files/current_compute_partition" "$files/current_memory_partition" "$files/available_memory_partition" 2>&1 | paste -sd'|')"
expect "and mem_info_vram_total agrees with CONFIG_MEMSIZE" "$((196608 * 1024 * 1024))" \
  "$(cat "$files/mem_info_vram_total" 2>&1)"
exit $fail
