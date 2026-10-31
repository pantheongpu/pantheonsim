#!/usr/bin/env bash
# Run the binaries trace-aws.sh collected from rented GPUs against the simulator
# and compare with what NVIDIA's CUPTI printed on the real card.
#
#   nvidia/tools/trace-compare.sh <collected dir> [build dir]
#
# <collected dir> holds one directory per instance type (g4dn.xlarge/, ...), each
# with <case>.txt (the card), <case>.bin (the program) and info.txt. The program is
# the same bytes on both sides; only the libraries behind it change. Writes
# <collected dir>/<type>/<case>.sim and .diff, and a summary table to stdout.
set -uo pipefail
dir="$(cd "${1:?usage: trace-compare.sh <collected dir> [build dir]}" && pwd)"
root="$(cd "$(dirname "$0")/../.." && pwd)"
shim="${2:-$root/build}/shim"
# Instance type (or provider directory) -> simulator profile. Types without a profile are listed, not guessed.
profile_for() {
  case "$1" in
    gpu_1x_a10) echo a10 ;;  gpu_1x_a100_sxm4) echo a100-sxm4-40gb ;;  gpu_1x_h100_pcie) echo h100-pcie ;;
    runpod-a40) echo a40 ;;  runpod-a100pcie) echo a100-80gb-pcie ;;  runpod-a100sxm) echo a100 ;;
    runpod-h100sxm) echo h100 ;;  runpod-h100nvl) echo h100-nvl ;;  runpod-h200sxm) echo h200 ;;
    runpod-rtxpro6000) echo rtx-pro-6000-server ;;  runpod-b300) echo b300 ;;  runpod-l40) echo l40 ;;
    rtx4090) echo rtx4090 ;;  rtx5090) echo rtx5090 ;;  rtx4080) echo rtx4080 ;;
    g4dn.*) echo t4 ;;     g5.*) echo a10g ;;  g6.*) echo l4 ;;     g6e.*) echo l40s ;;
    g7.*) echo rtx-pro-4500 ;;  g7e.*) echo rtx-pro-6000 ;;  p5.*|p5e.*|p5en.*) echo h100 ;;  p4d.*) echo a100-sxm4-40gb ;;
    p4de.*) echo a100 ;;   p6-b200.*) echo b200 ;;  p6-b300.*) echo b300 ;;  *) echo "" ;;
  esac
}
printf '| instance | profile | case | card lines | diff lines | result |\n|---|---|---|---|---|---|\n'
for t in "$dir"/*/; do
  type="$(basename "$t")"; [[ -s "$t/info.txt" ]] || continue
  prof="$(profile_for "$type")"
  if [[ -z "$prof" || ! -f "$root/nvidia/profiles/$prof.yaml" ]]; then
    printf '| %s | (none) | all | - | - | no simulator profile for this card |\n' "$type"; continue
  fi
  ndev=1
  for bin in "$t"/*.bin; do
    c="$(basename "$bin" .bin)"
    card="$t/$c.txt"
    if head -1 "$card" | grep -q '^SKIP'; then printf '| %s | %s | %s | - | - | card: %s |\n' "$type" "$prof" "$c" "$(head -1 "$card")"; continue; fi
    cubin=""; [[ "$c" == resource ]] && cubin="$t/resource_module.cubin"
    [[ "$c" == misc || "$c" == peer ]] && ndev=2 || ndev=1
    cupti=$(ls "$shim"/libcupti.so.[0-9]* 2>/dev/null | head -1)
    env VGPU_QUIET=1 VGPU_GPU="nvidia/$prof" VGPU_DEVICE_COUNT=$ndev CUPTI_TEST_CUBIN="$cubin" \
        NVTX_INJECTION64_PATH="$cupti" LD_LIBRARY_PATH="$shim" timeout 600 "$bin" > "$t/$c.sim" 2>"$t/$c.sim.err" || echo "EXIT $?" >> "$t/$c.sim"
    diff -u "$card" "$t/$c.sim" > "$t/$c.diff"
    n=$(grep -c '^[-+][^-+]' "$t/$c.diff")
    printf '| %s | %s | %s | %s | %s | %s |\n' "$type" "$prof" "$c" "$(wc -l < "$card")" "$n" "$([[ $n == 0 ]] && echo identical || echo differs)"
  done
done
