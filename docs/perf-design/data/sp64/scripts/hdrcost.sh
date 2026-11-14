#!/bin/bash
# hdrcost.sh <rt dir> [cpu]: -fsyntax-only time of the runtime header's parts (median of 5).
rt=$1; cpu=${2:-13}; o=$(mktemp -d)
fl="-std=c++20 -O2 -fPIC -w -mavx2 -mfma -mf16c -mpopcnt -I$rt"
m() { printf '%s\n' "$@" > $o/x.cpp; for i in 1 2 3 4 5; do s=$(date +%s%N); taskset -c $cpu nice -n 19 c++ $fl -fsyntax-only $o/x.cpp 2>/dev/null; e=$(date +%s%N); echo $(( (e-s)/1000000 )); done | sort -n | sed -n 3p; }
echo "nothing $(m '')"
echo "cstdint $(m '#include <cstdint>' '#include <cstddef>')"
echo "immintrin $(m '#include <immintrin.h>')"
echo "algorithm $(m '#include <algorithm>')"
echo "cmath $(m '#include <cmath>')"
echo "bit+cstring $(m '#include <bit>' '#include <cstring>')"
echo "std-all $(m '#include <algorithm>' '#include <bit>' '#include <cmath>' '#include <cstring>' '#include <immintrin.h>')"
echo "rt.h $(m '#include <cstddef>' '#include <cstdint>' '#define VGPU_SIMD_AVX2 1' '#include "vgpu_jit_rt.h"')"
rm -rf $o
