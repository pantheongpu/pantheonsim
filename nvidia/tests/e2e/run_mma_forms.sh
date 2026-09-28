#!/usr/bin/env bash
# Every mma.sync and mma.sp form an RTX 3060 (sm_86) runs -- 123 of them:
# Volta's m8n8k4, f64 in each rounding mode, s8/u8/s4/u4 with .satfinite,
# .b1 .and/.xor.popc, tf32's dropped mantissa bits, and every sparse type,
# shape and selector -- hashed and compared with the hashes the GPU gave
# (nvidia/tools/gen_mma_forms.py generates the program and records them).
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
src="$root/nvidia/tests/e2e/mma_forms.cu"
out="${TMPDIR:-/tmp}/vgpu_e2e_mma_forms_$$"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0
fi
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
nvcc -std=c++17 -O1 -cudart shared -arch=compute_86 -code=compute_86 \
     -Wno-deprecated-gpu-targets $(shim_sanitizer_nvcc_flags "$shim") "$src" -o "$out"
if ! require_shim_libs "$shim" "$out"; then rm -f "$out"; exit 0; fi
set +e
result="$(VGPU_QUIET=1 VGPU_GPU=nvidia/a10 LD_LIBRARY_PATH="$shim" "$out")"
rc=$?
set -e
rm -f "$out"
echo "$result" | tail -20
[[ $rc -eq 0 ]]
