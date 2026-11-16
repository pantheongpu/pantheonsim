#!/usr/bin/env bash
# Builds the numerics programs (mathfn, fp_examples, varity_cases) for one architecture.
#
#   tools/numerics/build.sh nvidia <bin-dir> <sm> [more sm ...]     nvcc -arch=sm_<sm>          -> <bin-dir>/sm_<sm>/
#   tools/numerics/build.sh amd    <bin-dir> [gfx ...]              hipcc --offload-arch=<gfx>  -> <bin-dir>/<gfx>/  (none given: hipcc's
#                                                                   own choice, the card's, -> <bin-dir>/native/)
# In each directory:
#   mathfn              the toolkit's defaults (hipcc: -O3, which is what nvcc gives device code unasked; clang's own default is -O0)
#   mathfn_fast         -use_fast_math (nvcc) / -ffast-math (hipcc): the short forms of the math functions; on NVIDIA they are the MUFU
#                       instruction with little or nothing around it, which is where a card and the simulator differ most
#   fp_examples         the toolkit's defaults (multiplies and adds may fuse)
#   fp_examples_nofma   -fmad=false (nvcc) / -ffp-contract=off (hipcc)
#   varity_cases.O0, varity_cases.O3
# A program that does not build leaves <name>.build.err and is reported; the others are still built. NUMERICS_FLAGS adds flags.
set -uo pipefail
kind="${1:?usage: $0 nvidia|amd <bin-dir> [arch ...]}"; mkdir -p "${2:?}"; bin="$(cd "$2" && pwd)"; shift 2
here="$(cd "$(dirname "$0")" && pwd)"
archs=("$@"); (( ${#archs[@]} )) || archs=(native)
read -r -a extra <<< "${NUMERICS_FLAGS:-}"   # more compiler flags, for every program (a sanitizer build's, in the simulator's tests)
fails=0
for a in "${archs[@]}"; do
  case "$kind" in
    nvidia) d="$bin/sm_$a"; cc=(nvcc -std=c++17 -cudart shared -arch="sm_$a" -Wno-deprecated-gpu-targets); fast=(-use_fast_math); nofma=(-fmad=false) ;;
    amd)    d="$bin/$a"; cc=(hipcc -std=c++17 -w -O3); [[ "$a" != native ]] && cc+=(--offload-arch="$a"); fast=(-ffast-math); nofma=(-ffp-contract=off) ;;
    *) echo "nvidia or amd"; exit 2 ;;
  esac
  mkdir -p "$d"
  b() {   # <output name> <source> [flags]
    local o="$1" s="$2"; shift 2
    if "${cc[@]}" "$@" "${extra[@]}" -I "$here" "$here/$s" -o "$d/$o" > "$d/$o.build.err" 2>&1; then rm -f "$d/$o.build.err"; else echo "does not build: $d/$o ($(head -1 "$d/$o.build.err"))"; fails=$((fails + 1)); fi
  }
  b mathfn mathfn.cu
  b mathfn_fast mathfn.cu "${fast[@]}"
  b fp_examples fp_examples.cu
  b fp_examples_nofma fp_examples.cu -DNO_FMAD "${nofma[@]}"
  b varity_cases.O0 varity_cases.cu -O0
  b varity_cases.O3 varity_cases.cu -O3
  echo "$d: $(ls "$d" | grep -v -c '\.err$') programs"
done
exit $(( fails > 0 ))
