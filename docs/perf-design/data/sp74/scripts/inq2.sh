#!/bin/bash
# inq2.sh <tag> <passes> <threads|default> <specA> <specB> -- <test>...   two builds (spec = tree[:ENV=v,...]) alternating -> out/inq-<tag>.txt
tag=$1; passes=$2; th=$3; A=$4; B=$5; shift 6
cd /tmp/sp72-amd-tier
T=""; [ "$th" != default ] && T="VGPU_THREADS=$th"
for p in $(seq 1 $passes); do
  for t in "$@"; do
    if [ $((p % 2)) = 1 ]; then order="$A $B"; else order="$B $A"; fi
    for spec in $order; do
      b=${spec%%:*}; e=""; [ "$spec" != "$b" ] && e=$(echo ${spec#*:} | tr , " ")
      taskset -c ${CPUS:-0-15} ./inq.sh $spec$p $b/build $t $T $e >> out/inq-$tag.txt 2>&1
    done
  done
done
echo DONE >> out/inq-$tag.txt
