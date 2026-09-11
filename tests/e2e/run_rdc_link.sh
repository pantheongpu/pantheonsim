#!/usr/bin/env bash
# The same programs, built with -rdc=true (separate compilation).
#
# This is a different loading path, not a different program. A -rdc build
# leaves the primary fatbin empty -- 16 bytes, just a header -- and hangs the
# real device code off the wrapper's fourth field as a list of *relocatable*
# fatbins. Device linking would normally consume them; with a PTX-only -code
# there is nothing for nvlink to link, so the pieces arrive still separate and
# the runtime has to put them together.
#
# Running the existing tests through it is the point: any of them failing here
# and passing normally means the linking is wrong, not the program.
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
tmp="${TMPDIR:-/tmp}/vgpu_rdc_$$"
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT
failures=0
for src in device_functions symbols device_intrinsics; do
  out="$tmp/$src"
  nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 \
       -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
       "$root/tests/e2e/$src.cu" -o "$out"
  if ! require_shim_libs "$shim" "$out"; then echo "SKIP: $src"; continue; fi
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || true
  echo "rdc $src: $(printf '%s' "$result" | tail -1)"
  [[ "$result" == *PASS* ]] || failures=$((failures + 1))
done
# A genuine two-unit build: device functions and a __constant__ defined in one
# translation unit, used from a kernel in another. Concatenating the pieces is
# only a link if references actually resolve across them, and this is what says
# whether they do.
mtu="$root/tests/e2e/multitu"
nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     -I"$mtu" -c "$mtu/lib.cu" -o "$tmp/lib.o"
nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     -I"$mtu" -c "$mtu/main.cu" -o "$tmp/main.o"
nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     "$tmp/lib.o" "$tmp/main.o" -o "$tmp/multitu"
if require_shim_libs "$shim" "$tmp/multitu"; then
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$tmp/multitu" 2>&1)" || true
  echo "rdc two translation units: $(printf '%s' "$result" | tail -1)"
  [[ "$result" == *PASS* ]] || failures=$((failures + 1))
fi

exit $failures
