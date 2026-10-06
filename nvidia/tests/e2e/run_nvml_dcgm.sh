#!/usr/bin/env bash
# NVML as NVIDIA's DCGM and NVVS (`dcgmi diag`) read it; see nvml_dcgm.py.
# DCGM itself is not run here: this checks the NVML answers its fields are built from.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
[[ -x "$build/vgpu" ]] || { echo "SKIP: vgpu not built"; exit 0; }
[[ -e "$build/shim/libnvidia-ml.so.1" ]] || { echo "SKIP: NVML shim not built"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: $san shim cannot load into python"; exit 0; fi
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
rc=0
# A GDDR card with ECC (retires pages), an HBM card (remaps rows) and a card
# without ECC, each on a machine of its own: nothing here may touch the user's
# reliability state.
for gpu in nvidia/t4 nvidia/h100 nvidia/rtx3060; do
  d="$tmp/${gpu//\//_}"
  VGPU_TELEMETRY_PATH="$d/run" VGPU_STATE_DIR="$d/state" VGPU_GPU=$gpu VGPU_DEVICE_COUNT=2 \
    python3 "$root/nvidia/tests/e2e/nvml_dcgm.py" "$build/shim" "$build/vgpu" "$gpu" || rc=1
done
exit $rc
