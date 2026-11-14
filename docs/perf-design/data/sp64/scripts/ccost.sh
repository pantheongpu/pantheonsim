#!/bin/bash
# ccost.sh <dir with kept .cpp files and rt-*/> : times the compiler on each generated source with several flag sets (one CPU, taskset).
# Run under the machine lock. Prints: file bytes, then seconds per variant.
d=$1; cpu=${2:-13}
rt=$(ls -d "$d"/rt-* | head -1)
base="-std=c++20 -funroll-loops -I$rt -fPIC -shared -fexceptions -ffp-contract=off -fvisibility=hidden -w -fno-stack-protector -fno-stack-clash-protection -fcf-protection=none -mavx2 -mfma -mf16c -mpopcnt"
t() { local s=$(date +%s.%N); taskset -c $cpu nice -n 19 "$@" > /dev/null 2>&1; local rc=$?; local e=$(date +%s.%N); printf "%.3f%s" "$(echo "$e - $s" | bc)" "$([ $rc = 0 ] || echo '!')"; }
o=$(mktemp -d -p "$d")
printf '#include "vgpu_jit_rt.h"\nextern "C" int vgpu_jit_meta_len = 0;\n' > "$o/empty.cpp"
echo "file bytes version [hdr-only-O2] E syntax O0 O1 O2 O2-no-unroll"
for f in "$o/empty.cpp" "$d"/*.cpp; do
  printf "%s %s " "$(basename "$f" | cut -c1-12)" "$(stat -c %s "$f")"
  t c++ --version; printf " "
  [ "$f" = "$o/empty.cpp" ] || { t c++ $base -O2 -o $o/e.so "$o/empty.cpp"; printf " "; }
  t c++ $base -E -o $o/x.ii "$f"; printf " "
  t c++ $base -fsyntax-only "$f"; printf " "
  t c++ $base -O0 -o $o/x0.so "$f"; printf " "
  t c++ $base -O1 -o $o/x1.so "$f"; printf " "
  t c++ $base -O2 -o $o/x2.so "$f"; printf " "
  t c++ $base -O2 -fno-unroll-loops -o $o/x3.so "$f"; echo
done
rm -rf "$o"
