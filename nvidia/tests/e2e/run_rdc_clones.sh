#!/usr/bin/env bash
# Device functions the device link cloned, one copy per kernel that reaches them, under one section name.
#
# nvlink gives each kernel its own copy of libdevice's helpers (pow's __internal_accurate_pow, the trigonometric
# reduction, the f64 divide), each reading the constants of the kernel it was cloned for from bank 2 at offsets
# of its own, and names every copy's section the same. A function whose address a table holds (GooFit's PDFs:
# pdf_argus here) calls the copy its own symbol table names, and a loader that finds the copy by section name
# finds the last one, whose constants sit elsewhere: pow read the wrong table, and the PDF returned inf, then NaN
# in GooFit's GenArgusTest. Compiled to real SASS, with the call through a table of pointers checked against the
# host, on a Turing GPU and an Ampere one; the same program passes on an RTX 3060.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/rdc_clones"
tmp="${TMPDIR:-/tmp}/vgpu_e2e_rdc_clones_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
mkdir -p "$tmp"
trap 'rm -rf "$tmp"' EXIT
failures=0
for target in sm_75:nvidia/t4 sm_86:nvidia/a10; do
  arch="${target%%:*}" gpu="${target#*:}"
  nvcc -std=c++17 -cudart shared -rdc=true -arch="$arch" -Wno-deprecated-gpu-targets \
       $(shim_sanitizer_nvcc_flags "$shim") "$src/lib.cu" "$src/main.cu" -o "$tmp/$arch"
  if ! require_shim_libs "$shim" "$tmp/$arch"; then continue; fi
  # The program only means something if the link cloned pow, as it does for these kernels.
  if command -v cuobjdump >/dev/null 2>&1; then
    clones="$(cuobjdump -sass "$tmp/$arch" 2>/dev/null | grep -c 'Function : __internal_accurate_pow' || true)"
    if (( clones < 2 )); then echo "SKIP: $arch: the link made $clones copies of pow, not the 2 or more this checks"; continue; fi
  fi
  result="$(VGPU_QUIET=1 VGPU_GPU="$gpu" LD_LIBRARY_PATH="$shim" "$tmp/$arch" 2>&1)" || true
  echo "$arch on $gpu: $(printf '%s' "$result" | tail -1)"
  [[ "$result" == *PASS* ]] || { printf '%s\n' "$result"; failures=$((failures + 1)); }
done
(( failures == 0 ))
