#!/usr/bin/env bash
# vgpu-run.sh — kept for the invocation that predates `vgpu run`.
#
# `vgpu run` does everything this did and adds the pre-flight checks that catch
# the two ways a program silently fails to reach the simulator (a CUDA soname
# the shim does not carry, and a DT_RPATH that outranks LD_LIBRARY_PATH). Prefer
# it directly:
#
#   build/vgpu run --gpu nvidia/h100 ./my_cuda_app [args...]
#
# This forwards there, so existing scripts keep working.
set -euo pipefail
here="$(cd "$(dirname "$0")/.." && pwd)"
vgpu="${VGPU_BUILD_DIR:-$here/build}/vgpu"
if [[ ! -x "$vgpu" ]]; then
  echo "error: $vgpu not built. Run ./scripts/build.sh first." >&2
  exit 1
fi
exec "$vgpu" run "$@"
