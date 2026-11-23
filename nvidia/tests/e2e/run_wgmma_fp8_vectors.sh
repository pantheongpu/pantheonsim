#!/usr/bin/env bash
# The e4m3 / e5m2 sum of wgmma on the h100 profile (include/vgpu/exec/tc_fp8.hpp, docs/numerics/fp8-vectors.md): 24 vectors each, recorded on an H100
# and an H200 through wgmma.mma_async m64n8k32 by Khattak and Mikaitis (nvidia/tests/data/tcpapers/fp8-wgmma/, BSD-2, see ../data/tcpapers/NOTICE), through
# a real wgmma on both engines (the SASS engine runs ptxas's QGMMA, VGPU_SASS=0 the PTX engine). The simulator must reproduce them bit for bit, and
# the sequential sum (VGPU_TC_EXACT=0) must not.
#   VGPU_KHATTAK_VECTORS=<dir>   the full sets (tools/tcpapers/khattak-fetch.sh <x> then <x>/vectors): 5,000 vectors a file.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data/tcpapers/fp8-wgmma"
src="$root/tools/tcpapers/khattak_fp8_wgmma_vectors.cu"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
if ! command -v nvcc >/dev/null 2>&1; then echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0; fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_wgmma_fp8.XXXXXX")"
trap 'rm -rf "$out"' EXIT
probe="$out/probe.cu"
echo '__global__ void k() { asm volatile("wgmma.fence.sync.aligned;"); }' > "$probe"
if ! nvcc -arch=compute_90a -code=compute_90a -c "$probe" -o /dev/null 2>/dev/null; then
  echo "SKIP: this nvcc cannot target sm_90a (wgmma needs CUDA 12.0 or later)"; exit 0
fi
if ! nvcc -std=c++17 -O1 -gencode "arch=compute_90a,code=[sm_90a,compute_90a]" -cudart shared -w $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out/prog" 2> "$out/b.txt"; then
  echo "FAIL: the program does not build for sm_90a"; cat "$out/b.txt"; exit 1
fi
if ! require_shim_libs "$shim" "$out/prog"; then exit 0; fi

fail=0
mism() {   # <engine: 1 SASS, 0 PTX> <fmt> <file> [ENV=value]   the mismatch count of one file
  local eng="$1" fmt="$2" f="$3"; shift 3
  env VGPU_QUIET=1 VGPU_GPU=nvidia/h100 VGPU_SASS="$eng" LD_LIBRARY_PATH="$shim" "$@" "$out/prog" "$fmt" "$f" --show 0 2>&1 | sed -n 's/.*mismatches=\([0-9]*\)$/\1/p'
}
check() {   # <label> <fmt> <file>
  local label="$1" fmt="$2" f="$3" eng m
  for eng in 1 0; do
    m="$(mism $eng "$fmt" "$f")"
    if [[ "$m" == 0 ]]; then echo "ok   wgmma $label $(basename "$f" .txt) $([[ $eng == 1 ]] && echo sass || echo ptx): 0 mismatches"
    else echo "FAIL wgmma $label $(basename "$f" .txt) $([[ $eng == 1 ]] && echo sass || echo ptx): mismatches=${m:-none}"; fail=1; fi
    m="$(mism $eng "$fmt" "$f" VGPU_TC_EXACT=0)"
    if [[ "$m" =~ ^[0-9]+$ ]] && (( m >= 1 )); then echo "ok   wgmma $label $(basename "$f" .txt) $([[ $eng == 1 ]] && echo sass || echo ptx) (VGPU_TC_EXACT=0): $m mismatches, as the sequential sum must"
    else echo "FAIL wgmma $label $(basename "$f" .txt) $([[ $eng == 1 ]] && echo sass || echo ptx) (VGPU_TC_EXACT=0): mismatches=${m:-none}, the sequential sum must not reproduce them"; fail=1; fi
  done
}
for f in "$data"/H100.e4m3.fp32.txt "$data"/H200.e4m3.fp32.txt; do check "sample" e4m3 "$f"; done
for f in "$data"/H100.e5m2.fp32.txt "$data"/H200.e5m2.fp32.txt; do check "sample" e5m2 "$f"; done
if [[ -n "${VGPU_KHATTAK_VECTORS:-}" ]]; then   # the full sets: the rule alone, both engines
  for f in "$VGPU_KHATTAK_VECTORS"/H100.e4m3.fp32.txt "$VGPU_KHATTAK_VECTORS"/H200.e4m3.fp32.txt; do
    for eng in 1 0; do m="$(mism $eng e4m3 "$f")"; if [[ "$m" == 0 ]]; then echo "ok   wgmma full $(basename "$f" .txt) eng=$eng: 0 mismatches"; else echo "FAIL wgmma full $(basename "$f" .txt) eng=$eng: mismatches=${m:-none}"; fail=1; fi; done
  done
  for f in "$VGPU_KHATTAK_VECTORS"/H100.e5m2.fp32.txt "$VGPU_KHATTAK_VECTORS"/H200.e5m2.fp32.txt; do
    for eng in 1 0; do m="$(mism $eng e5m2 "$f")"; if [[ "$m" == 0 ]]; then echo "ok   wgmma full $(basename "$f" .txt) eng=$eng: 0 mismatches"; else echo "FAIL wgmma full $(basename "$f" .txt) eng=$eng: mismatches=${m:-none}"; fail=1; fi; done
  done
fi
(( fail == 0 )) && echo PASS
exit $fail
