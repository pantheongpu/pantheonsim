#!/usr/bin/env bash
# Compile against NVIDIA's cuda.h, link against VirtualGPU's libcuda. The shim
# is written against a clean-room header, so this is what checks the two have
# not drifted apart.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-driver-abi.$$"
command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libcuda.so.1" ]] || { echo "SKIP: libvgpucuda not built"; exit 0; }
trap 'rm -f "$out"' EXIT

# The vendor header has to come from the same toolkit the shim was built
# against; that is the comparison this test exists to make.
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }

nvcc_host_compiler_fix

# And the program has to carry whatever sanitizer the shim was built with: an
# uninstrumented binary that loads an instrumented library aborts with "ASan
# runtime does not come first in initial library list".
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"

"$nvcc_bin" -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/tests/e2e/driver_abi.cu" \
     -L"$shim" -lcuda -o "$out" || { echo "FAIL: does not link against the shim"; exit 1; }
VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=64 LD_LIBRARY_PATH="$shim" "$out"
