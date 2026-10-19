#!/bin/bash
# The PGO measurement's builds (inside drx.sh, repo root /w, 4 CPUs): a timed plain Release build, a timed instrumented build,
# the timed training run, the profile's size, and a timed "use" build. Logs in /results/small-pgo/.
set -u
L=/results/small-pgo; mkdir -p $L
cd /w
t() { local s=$SECONDS; "$@"; echo "SECONDS $((SECONDS - s)) :: $*" >> $L/times.txt; }
nvcc -std=c++17 -O2 -arch=sm_86 -cudart shared -Xcompiler -ffp-contract=off -Wno-deprecated-gpu-targets tools/bench/kernels/bench.cu -o /results/small-mb/bench_kernels
echo "== plain" >> $L/times.txt
t bash -c 'cmake -S . -B build-plain -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build-plain -j4 >$0/plain.log 2>&1' $L
echo "== generate" >> $L/times.txt
t bash -c 'cmake -S . -B build-pgo -DCMAKE_BUILD_TYPE=Release -DVGPU_PGO=generate >/dev/null && cmake --build build-pgo -j4 >$0/generate.log 2>&1' $L
echo "== train" >> $L/times.txt
t bash docs/perf-design/data/round10/r10small-pgo-train.sh build-pgo /results/small-mb/bench_kernels "${PGO_GROUPS:-}" >$L/train.log 2>&1
du -sh build-pgo/pgo >> $L/times.txt; find build-pgo -name '*.gcda' | wc -l >> $L/times.txt
echo "== use" >> $L/times.txt
t bash -c 'cmake -S . -B build-pgo -DCMAKE_BUILD_TYPE=Release -DVGPU_PGO=use >/dev/null && cmake --build build-pgo -j4 >$0/use.log 2>&1' $L
echo "DONE" >> $L/times.txt
