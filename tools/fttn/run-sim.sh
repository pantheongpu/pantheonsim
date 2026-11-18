#!/usr/bin/env bash
# Runs the FTTN programs (tools/fttn/build.sh) on the simulator, each profile with the binaries of its architecture, NVIDIA on both
# engines, and prints one line a run: FTTN's summary row, or the motivating example's D.
#
#   tools/fttn/run-sim.sh <dir> <build-dir> <out-dir> [ENV=value ...]
#
# e.g. VGPU_TC_EXACT=0 (the sequential sums), VGPU_TC_EXACT=published (the sm_120 Blackwell profiles take the published model; b200 and gb200 have it by default),
# VGPU_MFMA_CDNA2=fttn (the opt-in MI250X rule). The AMD runs need ROCm's libraries on LD_LIBRARY_PATH.
# FTTN's summary prints the block width in hexadecimal after a hex dump earlier in the run: "10" in a bf16 row is 16.
set -uo pipefail
dir="${1:?usage: $0 <fttn-dir> <build-dir> <out-dir> [ENV=value ...]}"; shim="$(cd "${2:?}" && pwd)/shim"; out="${3:?}"; shift 3
extra=("$@"); mkdir -p "$out"
row() {   # <file>
  case "$1" in
    *motivating*) grep -a MOTIVATING "$1" | sed 's/.*D=/D=/; s/ elements.*//' ;;
    *) grep -a -A2 '^GPU  ' "$1" | tail -1 | tr -s ' ' ;;
  esac
}
nv() {   # <profile> <sm>
  for t in fp16in bf16in tf-fp32in fp64in motivating; do
    [[ -x "$dir/bin/sm_$2/$t" ]] || continue
    for e in sass ptx; do
      f="$out/nvidia.$1.sm_$2.$t.$e.txt"
      env "${extra[@]}" VGPU_QUIET=1 VGPU_GPU="nvidia/$1" VGPU_SASS=$([[ $e == sass ]] && echo 1 || echo 0) LD_LIBRARY_PATH="$shim" timeout 600 "$dir/bin/sm_$2/$t" > "$f" 2> "$f.err"
      echo "nvidia/$1 sm_$2 $t $e | $(row "$f") rc=$?"
    done
  done
}
amd() {   # <profile> <gfx>
  for t in fp16in bf16in tf-fp32in fp64in motivating; do
    [[ -x "$dir/bin-amd/$2/$t" ]] || continue
    f="$out/amd.$1.$2.$t.txt"
    env "${extra[@]}" VGPU_QUIET=1 VGPU_GPU="amd/$1" LD_LIBRARY_PATH="$shim:${LD_LIBRARY_PATH:-}" timeout 600 "$dir/bin-amd/$2/$t" > "$f" 2> "$f.err"
    echo "amd/$1 $2 $t | $(row "$f") rc=$?"
  done
}
nv t4 75; nv rtx2080ti 75; nv rtx3060 86; nv rtx3060 80; nv rtx3060 75; nv a10g 86; nv a100 80; nv l4 89; nv l40s 89
nv h100 90; nv h100-pcie 90; nv h200 90; nv gh200-480gb 90
nv b200 100; nv gb200 100; nv b300 100; nv vr200 100; nv thor 110
nv rtx-pro-6000 120; nv rtx-pro-6000-server 120; nv rtx5090 120; nv gb10 120
amd mi250x gfx90a; amd mi300x gfx942; amd mi325x gfx942; amd rx7900xtx gfx1100
