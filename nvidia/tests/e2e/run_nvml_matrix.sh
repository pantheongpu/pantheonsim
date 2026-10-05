#!/usr/bin/env bash
# The simulator's NVML getters against the RTX 3060 return-code matrix.
#
#   run_nvml_matrix.sh            probe the shim (VGPU_GPU=nvidia/rtx3060, two devices)
#   VGPU_NVML_CARD=<libnvidia-ml.so.1> run_nvml_matrix.sh
#                                 probe NVIDIA's library on a machine with the card
#                                 (device index VGPU_NVML_CARD_INDEX, default 1) and
#                                 compare with the card column. Getters only.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
cc="$(command -v cc || command -v gcc || true)"
[[ -n "$cc" ]] || { echo "SKIP: no C compiler"; exit 0; }
inc=""
for d in "${VGPU_NVML_INCLUDE:-}" /usr/local/cuda/include /usr/include; do
  [[ -n "$d" && -f "$d/nvml.h" ]] && { inc="$d"; break; }
done
[[ -n "$inc" ]] || { echo "SKIP: no nvml.h"; exit 0; }
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
python3 "$root/nvidia/tools/nvml_card_probe_gen.py" "$inc/nvml.h" > "$work/probe.c" || { echo "FAIL: probe generator"; exit 1; }
"$cc" -O0 -w -o "$work/probe" "$work/probe.c" -I"$inc" -ldl || { echo "FAIL: probe does not compile against $inc/nvml.h"; exit 1; }
matrix="$root/nvidia/tests/data/nvml_rtx3060_matrix.tsv"
if [[ -n "${VGPU_NVML_CARD:-}" ]]; then
  exec python3 "$root/nvidia/tests/e2e/nvml_matrix.py" "$work/probe" "$VGPU_NVML_CARD" "${VGPU_NVML_CARD_INDEX:-1}" "$matrix" --card
fi
[[ -e "$build/shim/libnvidia-ml.so.1" ]] || { echo "SKIP: NVML shim not built"; exit 0; }
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: $san shim cannot be loaded by an uninstrumented program"; exit 0; fi
# An empty telemetry directory, so no other test's live process answers instead.
mkdir "$work/telemetry" "$work/state"
VGPU_TELEMETRY_PATH="$work/telemetry" VGPU_STATE_DIR="$work/state" VGPU_GPU=nvidia/rtx3060 VGPU_DEVICE_COUNT=2 \
  python3 "$root/nvidia/tests/e2e/nvml_matrix.py" "$work/probe" "$build/shim/libnvidia-ml.so.1" 1 "$matrix"
