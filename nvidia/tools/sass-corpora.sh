#!/usr/bin/env bash
# Regenerates nvidia/tests/data/sass/<arch>.txt, the decoder test's corpora:
# one instruction of every shape found in
#   - full local corpora, when given (build them with sass-corpus.py over
#     this repo's tests and PyTorch's cubins: cuobjdump -xelf all on
#     libtorch_cuda.so etc.), and
#   - the probes in nvidia/tests/data/sass/probes, compiled for the arch.
# Existing lines are kept, so a regenerated corpus only grows.
#
#   sass-corpora.sh [full-corpus-dir]      (needs nvcc, cuobjdump, nvdisasm)
set -euo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
full="${1:-}"
data="$root/nvidia/tests/data/sass"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
nvcc="${CUDA_HOME:-/usr/local/cuda}/bin/nvcc"
for arch in sm_75 sm_80 sm_86 sm_89 sm_90 sm_90a sm_100 sm_100a sm_103a sm_120 sm_120a; do
  inputs=()
  [[ -n "$full" && -f "$full/$arch.txt" ]] && inputs+=("$full/$arch.txt")
  for p in "$data"/probes/*.cu; do
    out="$tmp/$arch.$(basename "$p" .cu).cubin"
    # A probe an architecture cannot compile (wgmma below sm_90a) is skipped.
    "$nvcc" -std=c++17 -arch="$arch" -cubin -w -o "$out" "$p" 2>/dev/null && inputs+=("$out") || true
  done
  python3 "$root/nvidia/tools/sass-corpus.py" --per-shape 1 "$arch" "$data/$arch.txt" "${inputs[@]}"
done
