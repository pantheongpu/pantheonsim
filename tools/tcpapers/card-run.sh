#!/usr/bin/env bash
# Runs the three programs of tools/tcpapers on the card the CUDA runtime finds (an NVIDIA GPU, nvcc in PATH or CUDA_HOME) and, with
# --update, rewrites the card's pinned files (nvidia/tests/data/tcpapers/{fprev,tc_features}.rtx3060.txt: run it on an sm_86 card only).
# Makes no cloud call.
#
#   tools/tcpapers/card-run.sh <out-dir> [--update] [<khattak vectors dir>]
#
# <out-dir>/fprev.txt, tc_features.txt; and, if a vectors dir (tools/tcpapers/khattak-fetch.sh <x>, then <x>/vectors) is given,
# <out-dir>/khattak.txt: one line a vector file with its mismatch count against that card.
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
out="$(mkdir -p "${1:?usage: $0 <out-dir> [--update] [vectors dir]}" && cd "$1" && pwd)"; shift
upd=0; [[ "${1:-}" == --update ]] && { upd=1; shift; }
vec="${1:-}"
command -v nvidia-smi > /dev/null && nvidia-smi -L > /dev/null 2>&1 || { echo "no card"; exit 1; }
nvcc_bin="${NVCC:-$(command -v nvcc || ls /usr/local/cuda*/bin/nvcc 2>/dev/null | tail -1)}"
[[ -n "$nvcc_bin" ]] || { echo "no nvcc"; exit 1; }
cuda_root="$(dirname "$(dirname "$(readlink -f "$nvcc_bin")")")"
export LD_LIBRARY_PATH="$cuda_root/lib64:${LD_LIBRARY_PATH:-}"
cap="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1 | tr -d '. ')"
no80=(); (( cap < 80 )) && no80=(-DNO80)
b() { "$nvcc_bin" -std=c++17 -O1 -arch="sm_$cap" -cudart shared -w "$@"; }
if (( cap >= 80 )); then b -DNO_CUBLAS "$root/tools/tcpapers/fprev_probe.cu" -o "$out/fprev_probe" || exit 1; else echo "fprev_probe needs sm_80 (its bf16 and tf32 loops)"; fi
b "${no80[@]}" "$root/tools/tcpapers/tc_features.cu" -o "$out/tc_features" || exit 1
(( cap >= 80 )) && { b "$root/tools/tcpapers/khattak_vectors.cu" -o "$out/khattak_vectors" || exit 1; }
[[ -x "$out/fprev_probe" ]] && "$out/fprev_probe" > "$out/fprev.txt"; "$out/tc_features" > "$out/tc_features.txt"
{ nvidia-smi --query-gpu=name,driver_version,compute_cap --format=csv,noheader | head -1; "$nvcc_bin" --version | tail -1; } > "$out/gpu.txt"
cat "$out/gpu.txt"
if [[ -n "$vec" && -x "$out/khattak_vectors" ]]; then
  : > "$out/khattak.txt"
  for f in "$vec"/*.txt; do
    IFS=. read -r card fmt o < <(basename "$f" .txt)
    k=$(( ($(head -1 "$f" | wc -w) - 2) / 2 ))
    echo "$(basename "$f" .txt) $("$out/khattak_vectors" "$fmt" "$k" "$o" "$f" --show 0 | tail -1)" | tee -a "$out/khattak.txt"
  done
fi
if (( upd )); then
  (( cap == 86 )) || { echo "--update rewrites the RTX 3060's files: run it on an sm_86 card (this one is sm_$cap)"; exit 1; }
  cp "$out/fprev.txt" "$root/nvidia/tests/data/tcpapers/fprev.rtx3060.txt"; cp "$out/tc_features.txt" "$root/nvidia/tests/data/tcpapers/tc_features.rtx3060.txt"
  echo "wrote the RTX 3060 pins"
fi
