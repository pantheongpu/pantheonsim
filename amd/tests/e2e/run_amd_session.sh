#!/usr/bin/env bash
# `vgpu shell` on an AMD machine looks like an AMD machine: the ROCm tools are
# there and NVIDIA's driver is not.
#
# The session used to supply nvidia-smi, nvcc and /proc/driver/nvidia whatever
# GPU it simulated, so a simulated MI300X answered nvidia-smi as an NVIDIA
# card. Tools pick a vendor by what is installed -- pantheon.py builds for CUDA
# when it finds nvidia-smi and nvcc -- and on a workstation with an NVIDIA card
# the host's own nvidia-smi showed through as well.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
build="$(cd "${VGPU_BUILD_DIR:-$root/build}" && pwd)"
vgpu="$build/vgpu"
[[ -x "$vgpu" ]] || { echo "no vgpu at $vgpu"; exit 1; }
work="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_amd_session.XXXXXX")"
trap 'rm -rf "$work"' EXIT
unset VGPU_GPU VGPU_DEVICE_COUNT VGPU_SESSION
export VGPU_QUIET=1 VGPU_TELEMETRY_PATH="$work/telemetry"
mkdir -p "$VGPU_TELEMETRY_PATH"

fail=0
expect() {  # expect <name> <expected> <actual>
  if [[ "$3" == "$2" ]]; then echo "ok    $1"; else
    echo "FAIL  $1"; echo "      expected: $2"; echo "      actual:   $3"; fail=1; fi
}
sess() { timeout 60 "$vgpu" shell -y --gpu amd/mi300x --count 2 "$@" </dev/null 2>/dev/null; }

# --- the session's own tools, with or without isolation ---
expect "the session supplies rocm-smi, amd-smi and rocm_agent_enumerator" "3" \
  "$(sess --no-isolate -c 'for t in rocm-smi amd-smi rocm_agent_enumerator; do [ -x "$VGPU_SESSION/bin/$t" ] && echo "$t"; done' | wc -l)"
expect "and no nvidia-smi or nvcc of its own" "" \
  "$(sess --no-isolate -c 'ls "$VGPU_SESSION/bin/nvidia-smi" "$VGPU_SESSION/bin/nvcc" 2>/dev/null')"
expect "and no NVIDIA driver files among the ones it overlays" "no" \
  "$(sess --no-isolate -c '[ -e "$VGPU_SESSION/root/proc/driver/nvidia" ] && echo yes || echo no')"
expect "rocm_agent_enumerator names the card's target" "gfx942" \
  "$(sess --no-isolate -c 'rocm_agent_enumerator' | grep -v gfx000 | sort -u)"

# --- isolated, the host's NVIDIA driver is out of sight too ---
if unshare --user --map-root-user true >/dev/null 2>&1; then
  # What pantheon.py asks: is there an nvidia-smi, is there a rocm-smi.
  probe='import shutil; print(shutil.which("nvidia-smi") is None, shutil.which("rocm-smi") is not None)'
  expect "a PATH search finds rocm-smi and no nvidia-smi" "True True" \
    "$(sess -c "python3 -c '$probe'")"
  expect "/proc/driver/nvidia is gone" "no" \
    "$(sess -c '[ -e /proc/driver/nvidia ] && echo yes || echo no')"
else
  echo "skip  isolation checks: unprivileged user namespaces are unavailable here"
fi

# --- an NVIDIA machine keeps its driver ---
expect "an NVIDIA session still has nvidia-smi and /proc/driver/nvidia" "yes yes" \
  "$(timeout 60 "$vgpu" shell -y --no-isolate --gpu nvidia/a10 -c \
      '[ -x "$VGPU_SESSION/bin/nvidia-smi" ] && printf "yes "; [ -e "$VGPU_SESSION/root/proc/driver/nvidia/version" ] && echo yes' \
      </dev/null 2>/dev/null)"

exit $fail
