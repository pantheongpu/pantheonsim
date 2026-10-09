#!/usr/bin/env bash
# solver_metis_paths.cu against NVIDIA's own libcusolver on a real GPU.
#
#   run_metis_card.sh            compare the card's permutations with
#                                nvidia/tests/data/solver_metis_paths.card.txt
#                                (the check that the data file is what the
#                                hardware's library returns)
#   run_metis_card.sh --update   rewrite the data file from the card (the
#                                "#" comment lines and the "known" lines are
#                                kept)
#
# Needs a GPU and the toolkit's libcusolver; serialise with the card lock:
#   flock ~/.cache/pantheonsim-card.lock nvidia/tests/e2e/run_metis_card.sh
set -euo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
data="$root/nvidia/tests/data/solver_metis_paths.card.txt"
src="$root/nvidia/tests/e2e/solver_metis_paths.cu"
out="${TMPDIR:-/tmp}/vgpu_metis_card_$$"
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc not found"; exit 0; }
trap 'rm -f "$out" "$out.txt"' EXIT
nvcc -std=c++17 -arch=sm_86 -Wno-deprecated-gpu-targets -Wno-deprecated-declarations "$src" -o "$out" -lcusolver -lcusparse
if [[ "${1:-}" == --update ]]; then
  METIS_PRINT=1 "$out" > "$out.txt"
  { grep -E '^(#|known )' "$data"; cat "$out.txt"; } > "$data.new"
  mv "$data.new" "$data"
  echo "wrote $data ($(wc -l < "$out.txt") matrices)"
else
  VGPU_E2E_DATA="$root/nvidia/tests/data" "$out"
fi
