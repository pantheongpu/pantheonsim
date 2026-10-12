#!/usr/bin/env bash
# Throwaway: builds the harness twice (baseline ISA, and the engine with AVX2/FMA) against build/libvgpu.a.
set -e
S=${S:-$HOME/.cache/pantheonsim-r7-design}
O=${O:-/tmp/claude-1002/r7d}
H=$(cd "$(dirname "$0")" && pwd)
INC="-I$S/include -I$S/nvidia/include -I$S/amd/include -I$S/build/generated"
cd "$H"
run() { nice -n 10 bash -c "ulimit -v 12582912; $*"; }
run g++ -O3 -DNDEBUG -ffp-contract=off -std=gnu++20 -c engine.cpp $INC -o $O/engine_base.o
run g++ -O3 -DNDEBUG -ffp-contract=off -std=gnu++20 -c codegen.cpp $INC -o $O/codegen.o
run g++ -O3 -DNDEBUG -mavx2 -mfma -mpopcnt -mbmi2 -ffp-contract=off -std=gnu++20 -c engine.cpp $INC -o $O/engine_avx2.o
run g++ -O3 -DNDEBUG -std=gnu++20 -flto=auto $INC r7proto.cpp $O/engine_base.o $O/codegen.o $S/build/libvgpu.a -lpthread -ldl -o $O/r7proto
run g++ -O3 -DNDEBUG -std=gnu++20 -flto=auto $INC r7proto.cpp $O/engine_avx2.o $O/codegen.o $S/build/libvgpu.a -lpthread -ldl -o $O/r7proto_avx2
