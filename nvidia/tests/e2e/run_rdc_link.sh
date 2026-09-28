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
root="$(cd "$(dirname "$0")/../../.." && pwd)"
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
for src in device_functions symbols device_intrinsics rdc_strings rdc_shared; do
  out="$tmp/$src"
  nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 \
       -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
       "$root/nvidia/tests/e2e/$src.cu" -o "$out"
  if ! require_shim_libs "$shim" "$out"; then echo "SKIP: $src"; continue; fi
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$out" 2>&1)" || true
  echo "rdc $src: $(printf '%s' "$result" | tail -1)"
  [[ "$result" == *PASS* ]] || failures=$((failures + 1))
done
# A genuine two-unit build: device functions and a __constant__ defined in one
# translation unit, used from a kernel in another. Concatenating the pieces is
# only a link if references actually resolve across them, and this is what says
# whether they do.
mtu="$root/nvidia/tests/e2e/multitu"
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

# More translation units than the runtime used to link: it stopped at 64
# pieces, so a kernel in the 65th unit or later was missing. AMReX's programs
# have about 320. Each unit holds one kernel and its launcher.
many="$tmp/many"
mkdir -p "$many"
for i in $(seq 0 69); do
  printf '__global__ void k%d(int* o) { o[%d] = %d; }\nvoid launch%d(int* o) { k%d<<<1, 1>>>(o); }\n' \
    "$i" "$i" "$((i * 3 + 1))" "$i" "$i" > "$many/u$i.cu"
done
{
  echo '#include <cstdio>'
  for i in $(seq 0 69); do echo "void launch$i(int*);"; done
  echo 'int main() {'
  echo '  int* d; cudaMalloc(&d, 70 * sizeof(int)); cudaMemset(d, 0, 70 * sizeof(int));'
  for i in $(seq 0 69); do echo "  launch$i(d);"; done
  echo '  int h[70]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);'
  echo '  int bad = cudaGetLastError() != cudaSuccess;'
  echo '  for (int i = 0; i < 70; ++i) if (h[i] != i * 3 + 1) { std::printf("FAIL unit %d: %d\n", i, h[i]); ++bad; }'
  echo '  std::printf(bad ? "FAILED\n" : "PASS\n"); return bad; }'
} > "$many/main.cu"
ls "$many"/u*.cu "$many/main.cu" | xargs -P 4 -I{} sh -c \
  'nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 -Wno-deprecated-gpu-targets '"$(shim_sanitizer_nvcc_flags "$shim")"' -c "$1" -o "${1%.cu}.o"' _ {}
nvcc -std=c++17 -cudart shared -rdc=true -arch=compute_80 -code=compute_80 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") \
     "$many"/*.o -o "$many/prog"
if require_shim_libs "$shim" "$many/prog"; then
  result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a100 LD_LIBRARY_PATH="$shim" "$many/prog" 2>&1)" || true
  echo "rdc 71 translation units: $(printf '%s' "$result" | tail -1)"
  [[ "$result" == *PASS* ]] || failures=$((failures + 1))
fi

exit $failures
