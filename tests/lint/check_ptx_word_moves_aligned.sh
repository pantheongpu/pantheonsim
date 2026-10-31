#!/usr/bin/env bash
# The PTX interpreter's register-word moves (Interpreter::gather_words and scatter_words, which the tensor-core instructions use around
# every mma) start on a 64-byte boundary.
#
# Their cost moves by 2 to 6% at equal instructions with where the linker puts them: a build where scatter_words started 96 bytes before a
# 4 KB page boundary (so that it crossed it) ran the mma_ptx workload on the PTX engine 2.4% (Zen 4) and 6.5% (Zen 5) slower than the build
# before, with every counter the same. The functions carry [[gnu::aligned(64)]] (src/exec/interpreter.cpp); this reads the symbols of the
# built shim and fails when one is elsewhere. (A clone of a function for a constant argument or a cold part is not the function.)
set -uo pipefail
root="$(cd "$(dirname "$0")/../.." && pwd)"
build="${VGPU_BUILD_DIR:-$root/build}"
shopt -s nullglob
libs=("$build"/shim/libcudart.so.*)
shopt -u nullglob
(( ${#libs[@]} )) || { echo "SKIP: no libcudart shim built"; exit 0; }
command -v nm >/dev/null || { echo "SKIP: no nm"; exit 0; }
lib="$(readlink -f "${libs[0]}")"
rc=0
found=0
for fn in gather_words scatter_words; do
  syms="$(nm -C --defined-only "$lib" 2>/dev/null | grep -F "Interpreter::$fn(" | grep -v -e 'clone .cold' -e 'clone .part' -e 'clone .constprop' -e 'clone .isra' || true)"
  if [[ -z "$syms" ]]; then
    echo "SKIP: $fn is not a symbol of $(basename "$lib") (inlined, or stripped)"; continue
  fi
  while read -r addr _ name; do
    found=$((found + 1))
    if (( 0x$addr % 64 != 0 )); then
      echo "FAIL: $name is at 0x$addr, not on a 64-byte boundary"; rc=1
    fi
  done <<< "$syms"
done
(( found )) || { echo "SKIP: neither function is a symbol of $(basename "$lib")"; exit 0; }
(( rc == 0 )) && echo "ok: $found Interpreter word-move symbols start on a 64-byte boundary"
exit $rc
