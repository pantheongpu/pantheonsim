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
sources="vector_add ops math memory globals grid bytes int64 atomics mixed half calls builtins packed spill crosslane doubles narrow lds idioms counters"
for src in $sources; do
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx942 -nogpulib -O2 -c "$src.c" -o "$src.gfx942.o"
  echo "wrote $(pwd)/$src.gfx942.o"
done

# Kernels written in assembly, for instructions a compiler emits only now and
# then (test_amd_gcn_asm).
for src in asm_sopk asm_scalar asm_memory asm_vector asm_libs; do
  "$clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c "$src.s" -o "$src.gfx942.o"
  echo "wrote $(pwd)/$src.gfx942.o"
done

# The same code as the assembler writes it, which the decoder's test compares
# against instruction by instruction. llvm-objdump ships with clang; without
# it the listing already checked in stays as it is.
objdump=${2:-$(dirname "$(readlink -f "$(command -v "$clang")")")/llvm-objdump}
if [[ -x "$objdump" ]]; then
  for src in $sources asm_sopk asm_scalar asm_memory asm_vector asm_libs; do
    "$objdump" -d --mcpu=gfx942 "$src.gfx942.o" |
      sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > "$src.gfx942.dis"
    echo "wrote $(pwd)/$src.gfx942.dis ($(wc -l < "$src.gfx942.dis") instructions)"
  done
else
  echo "no llvm-objdump beside $clang: keeping the listing as it is"
fi

# Offload bundles as ROCm's libraries carry device code: one code object for
# each of three targets, plain, compressed with zstd by clang's bundler, and
# compressed with zlib in the format clang wrote before (version 2: sizes and
# a hash ahead of the stream). And two kernels linked into one object, as
# Tensile links its libraries, each keeping its own metadata note.
bundler=$(dirname "$(readlink -f "$(command -v "$clang")")")/clang-offload-bundler
lld=$(dirname "$(readlink -f "$(command -v "$clang")")")/ld.lld
if [[ -x "$bundler" && -x "$lld" ]]; then
  targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx90a
  targets=$targets,hipv4-amdgcn-amd-amdhsa--gfx942:xnack+,hipv4-amdgcn-amd-amdhsa--gfx942:xnack-
  inputs=(--input=/dev/null --input=asm_sopk.gfx942.o --input=asm_scalar.gfx942.o --input=asm_vector.gfx942.o)
  "$bundler" --type=o --targets=$targets "${inputs[@]}" --output=bundle.bin
  "$bundler" --type=o --targets=$targets "${inputs[@]}" --output=bundle_zstd.bin --compress
  python3 -c '
import hashlib, struct, zlib
raw = open("bundle.bin", "rb").read()
packed = zlib.compress(raw, 9)
head = b"CCOB" + struct.pack("<HHII", 2, 0, 24 + len(packed), len(raw)) + hashlib.md5(raw).digest()[:8]
open("bundle_zlib.bin", "wb").write(head + packed)'
  "$lld" -shared asm_scalar.gfx942.o asm_vector.gfx942.o -o linked.gfx942.hsaco
  # And the kernels an HSA program loads (amd/tests/hsa), linked as ROCm's
  # loader wants them.
  "$lld" -shared vector_add.gfx942.o -o vector_add.gfx942.hsaco
  echo "wrote the bundles, linked.gfx942.hsaco and vector_add.gfx942.hsaco"
fi
