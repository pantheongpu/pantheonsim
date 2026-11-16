#!/usr/bin/env bash
# BMMA (1-bit tensor MMA, wmma experimental::precision::b1, #661) on the simulator against an RTX 3060: nvidia/tests/e2e/bmma_gemm.cu
# multiplies random bit matrices (fixed seed, six shapes, three kernels with different tile arrangements, XOR+popc and AND+popc) and prints a
# checksum per case plus one over all of them, after comparing every result with a host popcount reference. The pinned file
# nvidia/tests/data/bmma_gemm.rtx3060.txt is the RTX 3060's output (measured, CUDA 13.0, sm_86 build, 36 cases, all equal to the host
# reference). The simulator must print the same bytes (the results are integers) from the sm_86 SASS (BMMA.88128.XOR.POPC and .AND.POPC),
# from the same cubin on the PTX engine, and from a compute_80 PTX-only build.
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/bmma_gemm.cu"
want="$root/nvidia/tests/data/bmma_gemm.rtx3060.txt"
[[ -n "${BMMA_PIN:-}" ]] && want="$BMMA_PIN"    # for the mutation check: another pin must fail
tmp="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_e2e_bmma_gemm.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
san=$(shim_sanitizer_nvcc_flags "$shim")
nvcc -std=c++14 -cudart shared -arch=sm_86 -Wno-deprecated-gpu-targets $san "$src" -o "$tmp/sass" \
  || { echo "FAIL: bmma_gemm.cu does not build for sm_86"; exit 1; }
nvcc -std=c++14 -cudart shared -arch=compute_80 -code=compute_80 -Wno-deprecated-gpu-targets $san "$src" -o "$tmp/ptx" \
  || { echo "FAIL: bmma_gemm.cu does not build for compute_80"; exit 1; }
if ! require_shim_libs "$shim" "$tmp/sass"; then exit 0; fi
fails=0
run() {   # <label> <binary> <VGPU_SASS value> <profile>
  VGPU_QUIET=1 VGPU_GPU="nvidia/$4" VGPU_SASS="$3" LD_LIBRARY_PATH="$shim" timeout 600 "$2" > "$tmp/out.txt" 2> "$tmp/err.txt" || true
  if diff "$want" "$tmp/out.txt" > "$tmp/d.txt"; then echo "ok   $1"; else echo "FAIL $1: differs from $(basename "$want")"; head -6 "$tmp/d.txt" | cut -c1-200; head -3 "$tmp/err.txt" | cut -c1-200; fails=1; fi
}
run "bmma sm_86 SASS, rtx3060"               "$tmp/sass" 1 rtx3060
run "bmma sm_86 cubin on the PTX engine"     "$tmp/sass" 0 rtx3060
run "bmma compute_80 PTX, rtx3060"           "$tmp/ptx"  0 rtx3060
run "bmma sm_86 SASS, a10g (same sm_86)"     "$tmp/sass" 1 a10g
(( fails == 0 ))
