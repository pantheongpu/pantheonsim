#!/usr/bin/env bash
# Every function the toolchain's nvml.h declares is exported by the NVML shim.
# VGPU_NVML_INCLUDE is the toolkit include directory the shim was built with
# (CMake sets it); without it the usual places are searched.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
[[ -e "$build/shim/libnvidia-ml.so.1" ]] || { echo "SKIP: NVML shim not built"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
hdr=""
for d in "${VGPU_NVML_INCLUDE:-}" /usr/local/cuda/include /usr/include; do
  [[ -n "$d" && -f "$d/nvml.h" ]] && { hdr="$d/nvml.h"; break; }
done
[[ -n "$hdr" ]] || { echo "SKIP: no nvml.h"; exit 0; }
. "$root/tests/shim_guard.sh"
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: $san shim cannot load into python"; exit 0; fi
python3 "$root/nvidia/tests/e2e/nvml_coverage.py" "$build/shim" "$hdr"
