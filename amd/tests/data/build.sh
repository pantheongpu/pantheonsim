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

# The executor's kernels for gfx1100, gfx1201 and gfx1030 too (RDNA3, RDNA4
# and RDNA2, wave32), which the same tests run with VGPU_TEST_TARGET. builtins, crosslane, idioms and counters
# use CDNA's own instructions (s_memtime, row broadcasts, sdot4, MFMA) and are
# not built for it.
for src in vector_add ops math memory globals grid bytes int64 atomics mixed half calls packed spill doubles narrow lds; do
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx1100 -nogpulib -O2 -c "$src.c" -o "$src.gfx1100.o"
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx1201 -nogpulib -O2 -c "$src.c" -o "$src.gfx1201.o"
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx1030 -nogpulib -O2 -c "$src.c" -o "$src.gfx1030.o"
  echo "wrote $(pwd)/$src.gfx1100.o, $src.gfx1201.o and $src.gfx1030.o"
done

# vector_add for the RDNA generic targets (test_amd_gcn_asm), which need a
# clang that knows them (ROCm's; LLVM 19 and later).
for t in gfx11-generic gfx12-generic; do
  "$clang" -x c -target amdgcn-amd-amdhsa -mcpu=$t -nogpulib -O2 -c vector_add.c -o vector_add.$t.o ||
    echo "this clang has no $t: keeping vector_add.$t.o as it is"
done

# The scratch kernel for gfx90a too, which reaches its private memory through
# a buffer resource rather than flat scratch (test_amd_gcn_memory).
"$clang" -x c -target amdgcn-amd-amdhsa -mcpu=gfx90a -nogpulib -O2 -c memory.c -o memory.gfx90a.o
echo "wrote $(pwd)/memory.gfx90a.o"

# Kernels written in assembly, for instructions a compiler emits only now and
# then (test_amd_gcn_asm).
for src in asm_sopk asm_scalar asm_memory asm_vector asm_libs asm_logic asm_atomics asm_bcast asm_wait asm_realtime asm_isa_gaps asm_lds64 asm_ldsf32 asm_cvt_ubyte asm_dot_clamp asm_hwid asm_quad asm_mulhi asm_xf32 asm_permute_exec asm_barrier_order; do
  "$clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx942 -c "$src.s" -o "$src.gfx942.o"
  echo "wrote $(pwd)/$src.gfx942.o"
done

# And for gfx950, for what only it has (test_amd_gcn_asm).
for src in asm_lds_dma asm_ds_tr; do
  "$clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx950 -c "$src.s" -o "$src.gfx950.o"
  echo "wrote $(pwd)/$src.gfx950.o"
done

# And for gfx1100 (RDNA3, wave32), for what only RDNA has (test_amd_gcn_asm).
for src in asm_permlane asm_images asm_wave asm_literal asm_fminmax11 asm_hwid11; do
  "$clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1100 -c "$src.s" -o "$src.gfx1100.o"
  echo "wrote $(pwd)/$src.gfx1100.o"
done

# And for gfx1201 (RDNA4), for what only it has (test_amd_gcn_asm).
for src in asm_ttmp asm_cycles asm_fminmax12 asm_salu_f16 asm_hwid12; do
  "$clang" -x assembler -target amdgcn-amd-amdhsa -mcpu=gfx1201 -c "$src.s" -o "$src.gfx1201.o"
  echo "wrote $(pwd)/$src.gfx1201.o"
done

# The same code as the assembler writes it, which the decoder's test compares
# against instruction by instruction. llvm-objdump ships with clang; without
# it the listing already checked in stays as it is.
objdump=${2:-$(dirname "$(readlink -f "$(command -v "$clang")")")/llvm-objdump}
if [[ -x "$objdump" ]]; then
  for src in $sources asm_sopk asm_scalar asm_memory asm_vector asm_libs asm_logic asm_atomics asm_bcast asm_wait asm_isa_gaps asm_lds64 asm_ldsf32 asm_cvt_ubyte asm_dot_clamp; do
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
  # A bundle with generic code as well as a processor's own: gfx9-4-generic
  # (asm_sopk) and gfx942 with XNACK off (asm_vector), compressed.
  targets=host-x86_64-unknown-linux-gnu,hipv4-amdgcn-amd-amdhsa--gfx9-4-generic
  targets=$targets,hipv4-amdgcn-amd-amdhsa--gfx942:xnack-
  "$bundler" --type=o --targets=$targets --input=/dev/null --input=asm_sopk.gfx942.o --input=asm_vector.gfx942.o \
    --output=bundle_generic.bin --compress
  "$lld" -shared asm_scalar.gfx942.o asm_vector.gfx942.o -o linked.gfx942.hsaco
  # And the kernels an HSA program loads (amd/tests/hsa), linked as ROCm's
  # loader wants them.
  "$lld" -shared vector_add.gfx942.o -o vector_add.gfx942.hsaco
  echo "wrote the bundles, linked.gfx942.hsaco and vector_add.gfx942.hsaco"
fi

# devmalloc.gfx942.hsaco: device-side malloc and free (devmalloc.cpp), which needs hipcc. Run by
# test_amd_runtime_gaps. hipcc --genco writes an offload bundle; the code object is its gfx942 entry.
#   hipcc --genco --offload-arch=gfx942 -O1 devmalloc.cpp -o devmalloc.co
#   clang-offload-bundler --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
#       --input=devmalloc.co --output=devmalloc.gfx942.hsaco
