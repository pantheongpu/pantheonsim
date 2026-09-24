#!/usr/bin/env bash
# Rebuilds the gfx942 code object the code-object tests read. Checked in
# because no ROCm is needed to read one, and a test should not depend on a
# compiler being installed: clang with the amdgcn target builds it.
#
#   amd/tests/data/build.sh [clang]
#
# The kernel is written against clang's AMDGPU builtins rather than HIP, so it
# builds with no ROCm headers present.
set -euo pipefail
cd "$(dirname "$0")"
clang=${1:-clang}
sources="vector_add ops math memory globals grid bytes int64 atomics mixed half"
for src in $sources; do
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx942 -nogpulib -O2 -c "$src.c" -o "$src.gfx942.o"
  echo "wrote $(pwd)/$src.gfx942.o"
done

# The same code as the assembler writes it, which the decoder's test compares
# against instruction by instruction. llvm-objdump ships with clang; without
# it the listing already checked in stays as it is.
objdump=${2:-$(dirname "$(readlink -f "$(command -v "$clang")")")/llvm-objdump}
if [[ -x "$objdump" ]]; then
  for src in $sources; do
    "$objdump" -d --mcpu=gfx942 "$src.gfx942.o" |
      sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > "$src.gfx942.dis"
    echo "wrote $(pwd)/$src.gfx942.dis ($(wc -l < "$src.gfx942.dis") instructions)"
  done
else
  echo "no llvm-objdump beside $clang: keeping the listing as it is"
fi
