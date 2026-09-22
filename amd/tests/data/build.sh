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
"$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx942 -nogpulib -O2 -c vector_add.c -o vector_add.gfx942.o
echo "wrote $(pwd)/vector_add.gfx942.o"

# The same code as the assembler writes it, which the decoder's test compares
# against instruction by instruction. llvm-objdump ships with clang; without
# it the listing already checked in stays as it is.
objdump=${2:-$(dirname "$(readlink -f "$(command -v "$clang")")")/llvm-objdump}
if [[ -x "$objdump" ]]; then
  "$objdump" -d --mcpu=gfx942 vector_add.gfx942.o |
    sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > vector_add.gfx942.dis
  echo "wrote $(pwd)/vector_add.gfx942.dis ($(wc -l < vector_add.gfx942.dis) instructions)"
else
  echo "no llvm-objdump beside $clang: keeping the listing as it is"
fi
