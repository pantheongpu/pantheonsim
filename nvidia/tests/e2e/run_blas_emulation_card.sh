#!/usr/bin/env bash
# blas_emulation_paths.cu against NVIDIA's own libcublas on a real GPU (compute
# capability 8 or later, CUDA 13).
#
#   run_blas_emulation_card.sh            check the program passes on the card
#                                         and that nvidia/tests/data/
#                                         blas_emulation_paths.card.txt is what
#                                         the card's results hash to
#   run_blas_emulation_card.sh --update   rewrite the data file from the card
#
# Serialise with the card lock:
#   flock ~/.cache/pantheonsim-card.lock nvidia/tests/e2e/run_blas_emulation_card.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
data="$root/nvidia/tests/data/blas_emulation_paths.card.txt"
src="$root/nvidia/tests/e2e/blas_emulation_paths.cu"
out="${TMPDIR:-/tmp}/vgpu_emu_card_$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
trap 'rm -f "$out" "$out.txt"' EXIT
nvcc -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets -Wno-deprecated-declarations "$src" -o "$out" -lcublas
if [[ "${1:-}" == --update ]]; then
  EMU_PRINT=1 "$out" > "$out.txt"
  { grep -E '^#' "$data" || true; cat "$out.txt"; } > "$data.new"
  mv "$data.new" "$data"
  echo "wrote $data ($(wc -l < "$out.txt") results)"
else
  VGPU_E2E_DATA="$root/nvidia/tests/data" "$out"
fi
