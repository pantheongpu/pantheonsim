#!/bin/bash
# pchprobe.sh <dir with a kept .cpp and rt-*/> [cpu]: what a precompiled header of the runtime would save. The source's first lines
# (<cstddef>, <cstdint>, the VGPU_SIMD_AVX2 define, the runtime header) become one #include of a header that has them, compiled once.
d=$1; cpu=${2:-13}
rt=$(ls -d "$d"/rt-* | head -1)
f=$(ls "$d"/*.cpp | head -1)
fl="-std=c++20 -O2 -funroll-loops -fPIC -fexceptions -ffp-contract=off -fvisibility=hidden -w -fno-stack-protector -fno-stack-clash-protection -fcf-protection=none -mavx2 -mfma -mf16c -mpopcnt"
o=$(mktemp -d -p "$d")
cp "$rt"/* "$o/"
printf '#include <cstddef>\n#include <cstdint>\n#define VGPU_SIMD_AVX2 1\n#include "vgpu_jit_rt.h"\n' > "$o/vgpu_jit_pch.h"
{ echo '#include "vgpu_jit_pch.h"'; grep -v -E '^#include <cstddef>$|^#include <cstdint>$|^#define VGPU_SIMD_AVX2 1$|^#include "vgpu_jit_rt.h"$' "$f"; } > "$o/src.cpp"
t() { local s=$(date +%s.%N); taskset -c $cpu nice -n 19 "$@" > /dev/null 2>"$o/err"; local rc=$?; local e=$(date +%s.%N); printf "%.3f%s" "$(echo "$e - $s" | bc)" "$([ $rc = 0 ] || echo '!')"; }
printf "%s bytes=%s plain=" "$(basename $f | cut -c1-12)" "$(stat -c %s $f)"; t c++ $fl -I$rt -shared -o $o/a.so "$f"
printf " make-pch="; t c++ $fl -I$o -x c++-header -o "$o/vgpu_jit_pch.h.gch" "$o/vgpu_jit_pch.h"
printf " gch-bytes=%s with-pch=" "$(stat -c %s $o/vgpu_jit_pch.h.gch 2>/dev/null)"; t c++ $fl -I$o -Winvalid-pch -shared -o $o/b.so "$o/src.cpp"
printf " with-pch-again="; t c++ $fl -I$o -Winvalid-pch -shared -o $o/b.so "$o/src.cpp"
printf " pch-used=%s\n" "$(c++ $fl -I$o -H -fsyntax-only "$o/src.cpp" 2>&1 | grep -c '^! ')"
rm -rf "$o"
