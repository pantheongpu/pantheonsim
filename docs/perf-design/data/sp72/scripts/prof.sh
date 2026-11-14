#!/bin/bash
# usage: prof.sh <builddir> <tag> <workload> <reps> [ENV=..]   -> out/prof-<tag>-<workload>.txt (classified)
b=$1; tag=$2; wl=$3; reps=$4; shift 4
cd /tmp/sp72-amd-tier; mkdir -p out/samples
rm -f out/samples/$tag-$wl.*.txt
for i in $(seq 1 $reps); do
  ./run1.sh $b $wl LD_PRELOAD=/tmp/sp72-amd-tier/libsigprof.so SIGPROF_OUT=/tmp/sp72-amd-tier/out/samples/$tag-$wl.$i.txt SIGPROF_HZ=${HZ:-5000} "$@" > /dev/null 2>&1
done
python3 -I classify.py $b/libamdhip64.so.6.4 out/samples/$tag-$wl.*.txt > out/prof-$tag-$wl.txt 2>&1
rm -f out/samples/$tag-$wl.*.txt
