#!/usr/bin/env bash
# The tensor-core numerics papers on the simulator (tools/tcpapers, docs/matrix-numerics.md, "The three papers"): what the simulator
# prints is held to what an RTX 3060 prints, and to the results recorded on other cards in the published validation vectors.
#
#   nvidia/tests/data/tcpapers/fprev.rtx3060.txt        FPRev's summation trees (Xie et al., arXiv 2411.00442) for the f32 kernels and the
#                                                       wmma f16 / bf16 / tf32 loops: an RTX 3060's lines, measured. The simulator agrees on
#                                                       both engines, and from an sm_80 cubin on the a100 profile.
#   nvidia/tests/data/tcpapers/fprev.h100.txt           the same programs on the h100 profile (the Hopper rule: sums of 16 products); the
#                                                       simulator's lines, no H100 of ours has run the program. The b200 profile prints
#                                                       them too, by default since 2026-11-16 (RE-RECORDED: it needed VGPU_TC_EXACT=published).
#   nvidia/tests/data/tcpapers/tc_features.*.txt        the features of Fasi et al. (PeerJ CS 2021) and Khattak and Mikaitis (arXiv 2512.07004):
#                                                       rtx3060 measured, h100 the simulator's.
#   nvidia/tests/data/tcpapers/<card>.<fmt>.<out>.txt   24 recorded test vectors each of an A100, an H100 and a B200 (Khattak and Mikaitis's
#                                                       model_validation, BSD-2, see NOTICE): the simulator must reproduce them bit for bit with the
#                                                       profile of the card (b200 by default since 2026-11-16; before, with VGPU_TC_EXACT=published), and the wrong rule must not.
#   VGPU_KHATTAK_VECTORS=<dir>   the full sets (tools/tcpapers/khattak-fetch.sh <x> then <x>/vectors): 5,000 vectors a file, 28 files.
#   VGPU_TCPAPERS_BIN=<dir>      programs already built: <dir>/sm<NN>/{fprev_probe,tc_features,khattak_vectors} (skips nvcc).
# The sum is the device's, not the cubin's architecture, so each profile runs the program built for its own architecture.
set -uo pipefail
root="$(cd "$(dirname "$0")/../../.." && pwd)"
. "$root/tests/shim_guard.sh"
data="$root/nvidia/tests/data/tcpapers"
tool="$root/tools/tcpapers"
shim="${VGPU_BUILD_DIR:-$root/build}/shim"
shopt -s nullglob
cudart_libs=("$shim"/libcudart.so.[0-9]*)
shopt -u nullglob
if (( ${#cudart_libs[@]} == 0 )); then
  echo "SKIP: libvgpucudart not built (CUDA ABI headers absent at build time)"; exit 0
fi
out="$(mktemp -d "${TMPDIR:-/tmp}/vgpu_tcpapers.XXXXXX")"
trap 'rm -rf "$out"' EXIT
bin="${VGPU_TCPAPERS_BIN:-$out/bin}"
if [[ -z "${VGPU_TCPAPERS_BIN:-}" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then echo "SKIP: nvcc not found (e2e needs the CUDA toolkit to compile the app)"; exit 0; fi
  for a in 86 80 90 100 120; do
    mkdir -p "$bin/sm$a"
    ok=1
    nvcc -std=c++17 -O1 -arch=sm_$a -cudart shared -w -DNO_CUBLAS $(shim_sanitizer_nvcc_flags "$shim") "$tool/fprev_probe.cu" -o "$bin/sm$a/fprev_probe" 2> "$out/b.txt" || ok=0
    nvcc -std=c++17 -O1 -arch=sm_$a -cudart shared -w $(shim_sanitizer_nvcc_flags "$shim") "$tool/tc_features.cu" -o "$bin/sm$a/tc_features" 2>> "$out/b.txt" || ok=0
    nvcc -std=c++17 -O1 -arch=sm_$a -cudart shared -w $(shim_sanitizer_nvcc_flags "$shim") "$tool/khattak_vectors.cu" -o "$bin/sm$a/khattak_vectors" 2>> "$out/b.txt" || ok=0
    if (( ! ok )); then
      if [[ $a == 86 ]]; then echo "FAIL: the programs do not build for sm_86"; cat "$out/b.txt"; exit 1; fi
      echo "SKIP sm_$a: this toolkit does not build it ($(head -1 "$out/b.txt"))"; rm -rf "$bin/sm$a"
    fi
  done
fi
if ! require_shim_libs "$shim" "$bin/sm86/tc_features"; then exit 0; fi
fails=0
# run <program> <sm> <profile> <engine> <outfile> [ENV=value] [args...]   (the environment words are those with an =, before the first -- )
run() {
  local prog="$1" a="$2" gpu="$3" eng="$4" of="$5"; shift 5
  local envs=() args=()
  while (( $# )); do if [[ "$1" == *=* && ${#args[@]} == 0 ]]; then envs+=("$1"); else args+=("$1"); fi; shift; done
  env ${envs[@]+"${envs[@]}"} VGPU_QUIET=1 VGPU_GPU="nvidia/$gpu" VGPU_SASS=$([[ $eng == sass ]] && echo 1 || echo 0) LD_LIBRARY_PATH="$shim" \
    timeout 600 "$bin/sm$a/$prog" ${args[@]+"${args[@]}"} > "$of" 2> "$of.err" || true
}
expect() {   # <label> <want file> <got file>
  if diff "$2" "$3" > "$out/d.txt"; then echo "ok   $1"; else echo "FAIL $1: differs from $(basename "$2")"; head -6 "$out/d.txt" | cut -c1-200; fails=1; fi
}
# 1 and 2: the trees and the features.
for prog in fprev_probe tc_features; do
  name="${prog/_probe/}"; [[ $prog == tc_features ]] && name=tc_features
  for eng in sass ptx; do
    run $prog 86 rtx3060 $eng "$out/o.txt";            expect "$name sm_86 rtx3060 $eng" "$data/$name.rtx3060.txt" "$out/o.txt"
    [[ -x "$bin/sm80/$prog" ]] && { run $prog 80 a100 $eng "$out/o.txt"; expect "$name sm_80 a100 $eng (the RTX 3060's lines)" "$data/$name.rtx3060.txt" "$out/o.txt"; }
    [[ -x "$bin/sm90/$prog" ]] && { run $prog 90 h100 $eng "$out/o.txt"; expect "$name sm_90 h100 $eng" "$data/$name.h100.txt" "$out/o.txt"; }
    [[ -x "$bin/sm100/$prog" ]] && { run $prog 100 b200 $eng "$out/o.txt"; expect "$name sm_100 b200 (default: the Hopper rule, inferred from a B200's recorded vectors) $eng (the h100's lines)" "$data/$name.h100.txt" "$out/o.txt"; }
    [[ -x "$bin/sm120/$prog" ]] && { run $prog 120 rtx-pro-6000 $eng "$out/o.txt" VGPU_TC_EXACT=published; expect "$name sm_120 rtx-pro-6000 VGPU_TC_EXACT=published $eng (the h100's lines)" "$data/$name.h100.txt" "$out/o.txt"; }
  done
done
# the switches reach the programs: the sequential sum is not the card's, and b200 with VGPU_TC_EXACT=0 (the way back to the earlier
# sequential results) is not Hopper's
for prog in fprev_probe tc_features; do
  name="${prog/_probe/}"; [[ $prog == tc_features ]] && name=tc_features
  run $prog 86 rtx3060 sass "$out/o.txt" VGPU_TC_EXACT=0
  if cmp -s "$data/$name.rtx3060.txt" "$out/o.txt"; then echo "FAIL $name: VGPU_TC_EXACT=0 prints the card's lines (the switch does not reach the program)"; fails=1; else echo "ok   $name VGPU_TC_EXACT=0 differs from the card's lines"; fi
  if [[ -x "$bin/sm100/$prog" ]]; then
    run $prog 100 b200 sass "$out/o.txt" VGPU_TC_EXACT=0
    if cmp -s "$data/$name.h100.txt" "$out/o.txt"; then echo "FAIL $name: b200 prints the Hopper lines with VGPU_TC_EXACT=0"; fails=1; else echo "ok   $name b200 with VGPU_TC_EXACT=0 is not the Hopper rule"; fi
  fi
done
# FPRev's non-tensor kernels do not depend on the matrix rule: the RTX 3060's lines for them are the same with the sequential sum
grep -v '^wmma' "$data/fprev.rtx3060.txt" > "$out/want.txt"
run fprev_probe 86 rtx3060 sass "$out/o.txt" VGPU_TC_EXACT=0; grep -v '^wmma' "$out/o.txt" > "$out/got.txt"
expect "fprev non-wmma kernels with VGPU_TC_EXACT=0" "$out/want.txt" "$out/got.txt"
# 3. The recorded vectors.
mism() { sed -n 's/.*mismatches=\([0-9]*\).*/\1/p' "$1" | tail -1; }
kv() {   # <sm> <profile> <engine> <file> [ENV=value]   prints the mismatch count of one vector file
  local a="$1" gpu="$2" eng="$3" f="$4"; shift 4
  local base; base="$(basename "$f" .txt)"; local card fmt o; IFS=. read -r card fmt o <<< "$base"
  local k=$(( ($(head -1 "$f" | wc -w) - 2) / 2 ))
  run khattak_vectors "$a" "$gpu" "$eng" "$out/kv.txt" "$@" "$fmt" "$k" "$o" "$f" --show 0
  mism "$out/kv.txt"
}
for eng in sass ptx; do
  for f in "$data"/A100.*.txt; do
    for pair in 86:rtx3060 80:a100; do
      a="${pair%%:*}"; gpu="${pair##*:}"; [[ -x "$bin/sm$a/khattak_vectors" ]] || continue
      m="$(kv "$a" "$gpu" $eng "$f")"; if [[ "$m" == 0 ]]; then echo "ok   $(basename "$f" .txt) on $gpu $eng: 0 mismatches"; else echo "FAIL $(basename "$f" .txt) on $gpu $eng: mismatches=${m:-?}"; fails=1; fi
    done
  done
  if [[ -x "$bin/sm90/khattak_vectors" ]]; then
    for f in "$data"/H100.*.txt; do
      m="$(kv 90 h100 $eng "$f")"; if [[ "$m" == 0 ]]; then echo "ok   $(basename "$f" .txt) on h100 $eng: 0 mismatches"; else echo "FAIL $(basename "$f" .txt) on h100 $eng: mismatches=${m:-?}"; fails=1; fi
      m="$(kv 86 rtx3060 $eng "$f")"; if [[ "$m" =~ ^[0-9]+$ ]] && (( m >= 16 )); then echo "ok   $(basename "$f" .txt) on rtx3060 $eng: $m mismatches (the Ampere rule is not Hopper's)"; else echo "FAIL $(basename "$f" .txt) on rtx3060 $eng: expected at least 16 mismatches, got ${m:-?}"; fails=1; fi
    done
    for f in "$data"/A100.fp16.fp32.txt "$data"/A100.bf16.fp32.txt; do
      m="$(kv 90 h100 $eng "$f")"; if [[ "$m" =~ ^[0-9]+$ ]] && (( m >= 1 )); then echo "ok   $(basename "$f" .txt) on h100 $eng: $m mismatches (the Hopper rule is not Ampere's)"; else echo "FAIL $(basename "$f" .txt) on h100 $eng: expected mismatches, got ${m:-?}"; fails=1; fi
    done
  fi
  if [[ -x "$bin/sm100/khattak_vectors" ]]; then
    for f in "$data"/B200.*.txt; do
      m="$(kv 100 b200 $eng "$f")"; if [[ "$m" == 0 ]]; then echo "ok   $(basename "$f" .txt) on b200 (default, the Hopper rule) $eng: 0 mismatches"; else echo "FAIL $(basename "$f" .txt) on b200 (default, the Hopper rule) $eng: mismatches=${m:-?}"; fails=1; fi
      [[ "$f" == */B200.fp16.fp16.txt ]] && continue   # the sequential sum is the B200's on 4,998 of its 5,000 binary16-accumulator vectors
      m="$(kv 100 b200 $eng "$f" VGPU_TC_EXACT=0)"; if [[ "$m" =~ ^[0-9]+$ ]] && (( m >= 16 )); then echo "ok   $(basename "$f" .txt) on b200 (VGPU_TC_EXACT=0) $eng: $m mismatches (sequential sum, not the B200's)"; else echo "FAIL $(basename "$f" .txt) on b200 (VGPU_TC_EXACT=0) $eng: expected at least 16 mismatches, got ${m:-?}"; fails=1; fi
    done
  fi
done
# 4. Optionally the full sets.
if [[ -n "${VGPU_KHATTAK_VECTORS:-}" ]]; then
  for f in "$VGPU_KHATTAK_VECTORS"/*.txt; do
    base="$(basename "$f" .txt)"; card="${base%%.*}"
    case $card in A100) a=80; gpu=a100;; A2) a=86; gpu=rtx3060;; Ada) a=89; gpu=rtx6000-ada;; L40S) a=89; gpu=l40s;; H100) a=90; gpu=h100;; H200) a=90; gpu=h200;; B200) a=100; gpu=b200;; *) continue;; esac
    [[ -x "$bin/sm$a/khattak_vectors" ]] || { echo "skip $base: no sm_$a build"; continue; }
    m="$(kv "$a" "$gpu" sass "$f")"; if [[ "$m" == 0 ]]; then echo "ok   full $base on $gpu: 0 mismatches"; else echo "FAIL full $base on $gpu: mismatches=${m:-?}"; fails=1; fi
  done
fi
(( fails == 0 )) && echo PASS || { echo FAIL; exit 1; }
