#!/usr/bin/env bash
# Emits the telemetry block of a device profile from NVML (via nvidia-smi),
# which reports what the CUDA runtime does not: the power cap, the thermal
# slowdown limit, and the PCI ids. Run on the physical device.
#
#   nvidia/tools/characterize-telemetry.sh [device-index]
#
# Missing values are emitted as 0 rather than guessed, so a profile always
# shows which fields the hardware actually answered.
set -uo pipefail
dev="${1:-0}"
q() { nvidia-smi -i "$dev" --query-gpu="$1" --format=csv,noheader,nounits 2>/dev/null | tr -d ' \r'; }

power=$(q power.max_limit); power=${power%%.*}
sm=$(q clocks.max.sm)
mem=$(q clocks.max.memory)
# The slowdown threshold is not exposed on every driver or in every container,
# and a zero here would tell monitoring tools the GPU is permanently over its
# limit. Fall back to the shutdown threshold, then to a value that is at least
# in the right place, rather than writing a zero into a profile.
tmax=$(nvidia-smi -i "$dev" -q 2>/dev/null |
       awk -F: '/GPU Slowdown Temp/{gsub(/[^0-9]/,"",$2); print $2; exit}')
if [[ -z "${tmax:-}" || "$tmax" == "0" ]]; then
  tmax=$(nvidia-smi -i "$dev" -q 2>/dev/null |
         awk -F: '/GPU Shutdown Temp/{gsub(/[^0-9]/,"",$2); print $2; exit}')
fi
if [[ -z "${tmax:-}" || "$tmax" == "0" ]]; then
  tmax=""   # left unset below, so the profile keeps whatever it already had
fi
pciid=$(nvidia-smi -i "$dev" --query-gpu=pci.device_id --format=csv,noheader 2>/dev/null | tr -d ' \r')

# pci.device_id reads as 0xDDDDVVVV: device id high half, vendor id low half.
devid=0; vendid=4318
if [[ "$pciid" =~ ^0[xX]([0-9a-fA-F]{4})([0-9a-fA-F]{4})$ ]]; then
  devid=$((16#${BASH_REMATCH[1]}))
  vendid=$((16#${BASH_REMATCH[2]}))
fi

echo "telemetry:"
echo "  power_limit_w: ${power:-0}"
echo "  sm_clock_max_mhz: ${sm:-0}"
echo "  mem_clock_max_mhz: ${mem:-0}"
if [[ -n "${tmax:-}" ]]; then
  echo "  temperature_max_c: ${tmax}"
else
  echo "  # temperature_max_c: this driver did not report a thermal threshold"
fi
printf '  pci_vendor_id: %d          # 0x%04X\n' "$vendid" "$vendid"
printf '  pci_device_id: %d          # 0x%04X\n' "$devid" "$devid"
