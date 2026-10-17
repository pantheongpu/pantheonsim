#!/usr/bin/env bash
# Which error each refused launch configuration reports (launch_refusals.cu): the runtime's and the driver's code for a block or grid
# past the limits, a zero dimension, too much dynamic shared memory, a block past __launch_bounds__, a block whose registers do not
# fit, and a cluster launch on a part without clusters -- against an RTX 3060's CUDA 13.0 (nvidia/tests/data/launch_refusals.rtx3060.txt).
#
#   run_launch_refusals.sh                    simulator vs the card file
#   run_launch_refusals.sh --card [--update]  NVIDIA's runtime on a real GPU vs the card file (or rewrite it)
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/launch_refusals.cu"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_launch_refusals.XXXXXX")"
trap 'rm -rf "$out"' EXIT
build() { nvcc -std=c++17 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets "$@" "$src" -lcuda -o "$out/app"; }
if [[ "${1:-}" == --card ]]; then
  build
  cuda_root="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
  LD_LIBRARY_PATH="$cuda_root/lib64" "$out/app" > "$out/card.txt"
  if [[ "${2:-}" == --update ]]; then cp "$out/card.txt" "$data/launch_refusals.rtx3060.txt"; echo "wrote launch_refusals.rtx3060.txt"; exit 0; fi
  diff "$data/launch_refusals.rtx3060.txt" "$out/card.txt" && { echo "PASS (card)"; exit 0; }
  echo "FAIL: the card disagrees with launch_refusals.rtx3060.txt"; exit 1
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
build $(shim_sanitizer_nvcc_flags "$shim")
if ! require_shim_libs "$shim" "$out/app"; then exit 0; fi
env $(both_shims_env "$shim") VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 LD_LIBRARY_PATH="$shim" "$out/app" > "$out/sim.txt" 2>&1 || true
if diff "$data/launch_refusals.rtx3060.txt" "$out/sim.txt" > "$out/d.txt"; then
  echo "PASS"
else
  echo "FAIL: the simulator's launch refusals differ from the card's:"; head -30 "$out/d.txt"; exit 1
fi
