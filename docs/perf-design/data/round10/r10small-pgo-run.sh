#!/bin/bash
# The PGO measurement's builds (inside drx.sh, repo root /w, 4 CPUs). Logs in /results/small-pgo/.
#   r10small-pgo-run.sh [name [groups]]
# name (default pgo): build directory build-<name>; groups: PyTorch sweep groups to train on besides the harness workloads.
# Steps, each timed: a plain Release build (build-plain, only when name is "pgo"), the instrumented build, the training run, the
# profile's size, and the "use" build.
set -u
name=${1:-pgo} groups=${2:-}
L=/results/small-pgo; mkdir -p $L
cd /w
t() { local s=$SECONDS; "$@"; echo "SECONDS $((SECONDS - s)) :: $*" >> $L/times-$name.txt; }
nvcc -std=c++17 -O2 -arch=sm_86 -cudart shared -Xcompiler -ffp-contract=off -Wno-deprecated-gpu-targets tools/bench/kernels/bench.cu -o /results/small-mb/bench_kernels
if [ "$name" = pgo ]; then
  echo "== plain" >> $L/times-$name.txt
  t bash -c 'cmake -S . -B build-plain -DCMAKE_BUILD_TYPE=Release >/dev/null && cmake --build build-plain -j4 >$0/plain.log 2>&1' $L
fi
echo "== generate" >> $L/times-$name.txt
t bash -c 'cmake -S . -B build-'$name' -DCMAKE_BUILD_TYPE=Release -DVGPU_PGO=generate >/dev/null && cmake --build build-'$name' -j4 >$0/generate-'$name'.log 2>&1' $L
echo "== train" >> $L/times-$name.txt
t bash docs/perf-design/data/round10/r10small-pgo-train.sh build-$name /results/small-mb/bench_kernels "$groups" >$L/train-$name.log 2>&1
du -sh build-$name/pgo >> $L/times-$name.txt; find build-$name -name '*.gcda' | wc -l >> $L/times-$name.txt
echo "== use" >> $L/times-$name.txt
t bash -c 'cmake -S . -B build-'$name' -DCMAKE_BUILD_TYPE=Release -DVGPU_PGO=use >/dev/null && cmake --build build-'$name' -j4 >$0/use-'$name'.log 2>&1' $L
echo "DONE" >> $L/times-$name.txt
