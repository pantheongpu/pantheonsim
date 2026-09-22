#!/usr/bin/env bash
# The virtual memory management API through a real program: reserve, create,
# map, grant access, run a kernel, grow the buffer in place, unmap. Compiled
# against NVIDIA's cuda.h, so the structs it passes are the vendor header's.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
out="${TMPDIR:-/tmp}/vgpu-driver-abi.$$"
command -v nvcc >/dev/null || { echo "SKIP: nvcc not found"; exit 0; }
[[ -e "$shim/libcuda.so.1" ]] || { echo "SKIP: libvgpucuda not built"; exit 0; }
trap 'rm -f "$out"' EXIT

# The vendor header has to come from the same toolkit the shim was built
# against, so the structs are laid out as the shim expects.
nvcc_bin="$(pick_nvcc_for_shim "$shim")"
[[ -n "$nvcc_bin" ]] || { echo "SKIP: no nvcc matching this shim's toolkit"; exit 0; }

nvcc_host_compiler_fix

# And the program has to carry whatever sanitizer the shim was built with: an
# uninstrumented binary that loads an instrumented library aborts with "ASan
# runtime does not come first in initial library list".
read -r -a san_flags <<< "$(shim_sanitizer_nvcc_flags "$shim")"

"$nvcc_bin" -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets "${san_flags[@]}" \
     "$root/nvidia/tests/e2e/vmm.cu" \
     -L"$shim" -lcuda -o "$out" || { echo "FAIL: does not link against the shim"; exit 1; }
VGPU_QUIET=1 VGPU_GPU=nvidia/a10 VGPU_VRAM_MB=64 LD_LIBRARY_PATH="$shim" "$out"
