#!/usr/bin/env bash
# Every instruction rocBLAS carries for gfx942 in the routines this runs,
# decoded as ROCm's llvm-objdump prints it: the kernels built into the library
# (a compressed offload bundle per source file, inside librocblas) and
# Tensile's float and double GEMM kernels (compressed bundles and plain code
# objects beside it). Some seventeen million instructions, so the few hundred
# distinct forms a program reaching them can run are all covered before it
# runs them.
#
# Runs wherever ROCm with rocBLAS is installed and skips everywhere else.
set -uo pipefail
build="${VGPU_BUILD_DIR:-build}"
test_bin="$build/test_amd_gcn"
[[ -x "$test_bin" ]] || { echo "SKIP: no $test_bin"; exit 0; }
rocm=""
for c in "${VGPU_ROCM_PATH:-}" "${ROCM_PATH:-}" /opt/rocm $(ls -d "$HOME"/.local/share/rocm-*/opt/rocm-* 2>/dev/null | sort -rV); do
  [[ -n "$c" && -x "$c/lib/llvm/bin/llvm-objdump" && -x "$c/lib/llvm/bin/clang-offload-bundler" &&
     -d "$c/lib/rocblas/library" ]] && ls "$c"/lib/librocblas.so.* >/dev/null 2>&1 && { rocm=$c; break; }
done
[[ -n "$rocm" ]] || { echo "SKIP: no ROCm with rocBLAS found (set VGPU_ROCM_PATH)"; exit 0; }
command -v python3 >/dev/null || { echo "SKIP: no python3 to split rocBLAS's device code"; exit 0; }
tmp=$(mktemp -d)
trap 'rm -rf "$tmp"' EXIT
bundler="$rocm/lib/llvm/bin/clang-offload-bundler"
objdump="$rocm/lib/llvm/bin/llvm-objdump"

# The library's device code: its .hip_fatbin section is one compressed bundle
# after another, each saying in its header how long it is.
lib=$(ls "$rocm"/lib/librocblas.so.* | head -1)
read -r off size < <("$rocm/lib/llvm/bin/llvm-readelf" -SW "$lib" | awk '$2==".hip_fatbin" {print $5, $6}')
python3 - "$lib" "$off" "$size" "$tmp" <<'PY'
import struct, sys
lib, off, size, out = sys.argv[1], int(sys.argv[2], 16), int(sys.argv[3], 16), sys.argv[4]
section = open(lib, "rb").read()[off:off + size]
at = n = 0
while True:
    at = section.find(b"CCOB", at)
    if at < 0: break
    total = struct.unpack_from("<Q", section, at + 8)[0]   # format 3: the whole bundle's size
    open(f"{out}/lib{n}.bundle", "wb").write(section[at:at + total])
    at += total
    n += 1
PY
for b in "$tmp"/lib*.bundle; do
  "$bundler" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942:xnack- --input="$b" \
    --output="${b%.bundle}.o" 2>/dev/null
done
# Tensile's float and double GEMMs (the types this runs and checks): a bundle
# per problem, and a plain object for the fallbacks. The half, bfloat16, int8
# and FP8 ones are not yet decoded.
n=0
for f in "$rocm"/lib/rocblas/library/TensileLibrary_Type_{SS,DD}_*_gfx942.co; do
  "$bundler" --unbundle --type=o --targets=hipv4-amdgcn-amd-amdhsa--gfx942 --input="$f" \
    --output="$tmp/tensile$((n++)).o" 2>/dev/null
done
for f in "$rocm"/lib/rocblas/library/TensileLibrary_Type_{SS,DD}_*gfx942-xnack-.hsaco "$rocm"/lib/rocblas/library/Kernels.so-*-gfx942-xnack-.hsaco; do
  cp "$f" "$tmp/tensile$((n++)).o"
done
rm -f "$tmp"/*.bundle
find "$tmp" -name '*.o' -size 0 -delete

# Each object against its listing, as many at once as there are cores. The
# listing keeps the zeroes between kernels (-z), which this decodes too, and
# the objects lose their local symbols first, so that a branch prints where
# it goes rather than the name of a label there.
strip="$rocm/lib/llvm/bin/llvm-strip"
export objdump test_bin strip
check() {
  "$strip" --discard-all "$1"
  "$objdump" -d -z --mcpu=gfx942 "$1" | sed -n 's/^\t\(.*\)\/\/ .*/\1/p' | sed 's/[[:space:]]*$//; s/  */ /g' > "$1.dis"
  VGPU_GCN_OBJECT="$1" VGPU_GCN_LISTING="$1.dis" "$test_bin" > "$1.out" 2>&1
  grep -q '^\[ PASS \] every_instruction_decodes_as_the_assembler_wrote_it' "$1.out" ||
    { grep -m1 -A1 '^\[ FAIL \] every_instruction' "$1.out" | tail -1; }
}
export -f check
failures=$(ls "$tmp"/*.o | xargs -P "$(nproc)" -I{} bash -c 'check {}')
objects=$(ls "$tmp"/*.o | wc -l)
instructions=$(cat "$tmp"/*.dis | wc -l)
if [[ -z "$failures" && "$objects" -gt 0 ]]; then
  echo "ok    all $instructions instructions in $objects of rocBLAS's gfx942 code objects decode as llvm-objdump prints them"
  exit 0
fi
echo "FAIL  rocBLAS's device code, against $rocm's llvm-objdump:"
echo "$failures" | sed 's/^/      /' | head -20
exit 1
