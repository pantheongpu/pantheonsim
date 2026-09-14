#!/usr/bin/env bash
# NVML as monitoring tools call it; see nvml_api.py.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
[[ -x "$build/vgpu" ]] || { echo "SKIP: vgpu not built"; exit 0; }
[[ -e "$build/shim/libnvidia-ml.so.1" ]] || { echo "SKIP: NVML shim not built"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
# ctypes loads the shim into python, which a sanitizer runtime refuses.
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: $san shim cannot load into python"; exit 0; fi
VGPU_QUIET=1 "$build/vgpu" shell -y --gpu nvidia/t4 --count 2 --no-isolate \
  -c "python3 '$root/nvidia/tests/e2e/nvml_api.py' '$build/shim'"
