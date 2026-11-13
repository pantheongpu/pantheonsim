#!/bin/bash
# ccost2.sh <dir> [cpu] "<flags A>" "<flags B>" ...: the compile of the kept source with the JIT's command line and the given -O flags instead of -O2.
d=$1; cpu=$2; shift 2
rt=$(ls -d "$d"/rt-* | head -1); f=$(ls "$d"/*.cpp | head -1)
base="-std=c++20 -funroll-loops -I$rt -fPIC -shared -fexceptions -ffp-contract=off -fvisibility=hidden -w -fno-stack-protector -fno-stack-clash-protection -fcf-protection=none -mavx2 -mfma -mf16c -mpopcnt"
o=$(mktemp -d -p "$d")
printf "%s" "$(basename $f | cut -c1-12)"
for fl in "$@"; do s=$(date +%s.%N); taskset -c $cpu nice -n 19 c++ $base $fl -o $o/x.so "$f" 2>/dev/null; e=$(date +%s.%N); printf " [%s]=%.3f" "$fl" "$(echo "$e - $s" | bc)"; done; echo
rm -rf "$o"
