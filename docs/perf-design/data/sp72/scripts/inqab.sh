#!/bin/bash
# usage: inqab.sh <tag> <passes> <threads|default> <test>...   the tests with blocks off and on (work/build), alternating, -> out/inq-<tag>.txt
tag=$1; passes=$2; th=$3; shift 3
cd /tmp/sp72-amd-tier
T=""; [ "$th" != default ] && T="VGPU_THREADS=$th"
for p in $(seq 1 $passes); do
  for t in "$@"; do
    if [ $((p % 2)) = 1 ]; then order="off on"; else order="on off"; fi
    for side in $order; do
      if [ $side = on ]; then taskset -c ${CPUS:-0-15} ./inq.sh $side$p work/build $t $T VGPU_AMD_BLOCKS=1 >> out/inq-$tag.txt 2>&1
      else taskset -c ${CPUS:-0-15} ./inq.sh $side$p work/build $t $T >> out/inq-$tag.txt 2>&1; fi
    done
  done
done
echo DONE >> out/inq-$tag.txt
