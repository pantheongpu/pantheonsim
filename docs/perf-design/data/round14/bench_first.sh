#!/bin/bash
# Round 14: the first-launch rule. Wall seconds of a whole process with the mode off, on without the rule (VGPU_HOST_GPU_NO_FIRST_LAUNCH=1: the
# history-only behaviour) and on with it (the default), median of REP runs each (default 3); output compared with the mode off.
# usage: bench_first.sh <rows...>   (PolyBench names from ~/corp/poly)
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=${SHIM:-$HOME/sim/build/shim} VGPU_THREADS=4
d=$(mktemp -d); cd $d
now() { date +%s.%N; }
f='time\|^[0-9.]*$\|^GPU\|elapsed\|Elapsed\|seconds\|[0-9] ms\|[0-9] us\|GB/s\|MB/s\|Throughput\|throughput'
med() { sort -n | awk '{a[NR]=$1} END{print a[int((NR+1)/2)]}'; }
for n in "$@"; do P=~/corp/poly/$n
  for i in $(seq ${REP:-3}); do
    a=$(now); env VGPU_QUIET=1 $P > $d/o0.txt 2>/dev/null; b=$(now); echo "$b - $a" | bc >> $d/t0
    env VGPU_HOST_GPU=exact VGPU_HOST_GPU_NO_FIRST_LAUNCH=1 VGPU_HOST_GPU_LOG=1 $P > $d/o1.txt 2> $d/e1.txt; c=$(now); echo "$c - $b" | bc >> $d/t1
    env VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 $P > $d/o2.txt 2> $d/e2.txt; e=$(now); echo "$e - $c" | bc >> $d/t2
  done
  s1=$(grep "^host_gpu: [0-9]* launches" $d/e1.txt | sed 's/host_gpu: //; s/ (.*//'); s2=$(grep "^host_gpu: [0-9]* launches" $d/e2.txt | sed 's/host_gpu: //; s/ (.*//')
  fl=$(grep -c "first launch decided" $d/e2.txt)
  same=$(diff <(grep -v -i "$f" $d/o0.txt) <(grep -v -i "$f" $d/o2.txt) > /dev/null && echo same || echo DIFF)
  printf "%-8s off %7.2f | no rule %7.2f (%s) | rule %7.2f (%s) first-launch-offloads=%s out %s\n" $n $(med < $d/t0) $(med < $d/t1) "$s1" $(med < $d/t2) "$s2" $fl $same
  mkdir -p ${OUTDIR:-$HOME/out}; cp $d/e2.txt ${OUTDIR:-$HOME/out}/first.$n.log; rm -f $d/t0 $d/t1 $d/t2
done
echo FINISHED
