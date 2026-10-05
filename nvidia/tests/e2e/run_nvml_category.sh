#!/usr/bin/env bash
# One category of NVML checks (nvml_<category>.py) against the shim, once per
# profile named, inside `vgpu shell` so that the machine publishes telemetry
# and the settings overlay lives in a session of its own.
#
#   run_nvml_category.sh <category> <profile> [<profile> ...]
#
# The python file gets the shim directory, the profile id and a count of 2 as
# arguments. VGPU_NVML_ROOT=1 lets the checks exercise the setters (nvml_common.inc).
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
cat="$1"; shift
[[ -x "$build/vgpu" ]] || { echo "SKIP: vgpu not built"; exit 0; }
[[ -e "$build/shim/libnvidia-ml.so.1" ]] || { echo "SKIP: NVML shim not built"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3"; exit 0; }
san="$(shim_sanitizer "$build/shim")"
if [[ -n "$san" ]]; then echo "SKIP: $san shim cannot load into python"; exit 0; fi
py="$root/nvidia/tests/e2e/nvml_$cat.py"
[[ -f "$py" ]] || { echo "FAIL: $py is missing"; exit 1; }
rc=0
for profile in "$@"; do
  echo "== nvml_$cat on $profile"
  state="$(mktemp -d)"
  VGPU_STATE_DIR="$state" VGPU_NVML_ROOT=1 VGPU_QUIET=1 \
    "$build/vgpu" shell -y --gpu "$profile" --count 2 --no-isolate \
    -c "python3 '$py' '$build/shim' '$profile'" || rc=1
  rm -rf "$state"
done
exit $rc
