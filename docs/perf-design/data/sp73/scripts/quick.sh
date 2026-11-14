#!/bin/bash
# usage: quick.sh <runs> <workload> <gfx942|gfx1030> <gpu> <build[:ENV=..]>...   best device CPU seconds of each build, alternating
cd /tmp/sp72-amd-tier; runs=$1; wl=$2; tg=$3; gpu=$4; shift 4
declare -A best
for r in $(seq 1 $runs); do
  for spec in "$@"; do
    b=${spec%%:*}; e=""; [ "$spec" != "$b" ] && e=$(echo ${spec#*:} | tr , " ")
    t=$(TG=$tg GPU=$gpu taskset -c 14,15 ./run1.sh $b/build $wl $e 2>/dev/null | grep BENCH | sed "s/.*device_cpu_s=\([0-9.]*\).*/\1/")
    best[$spec]=$(python3 -c "print(min(${best[$spec]:-999}, ${t:-999}))")
  done
done
printf "%-10s %-8s" $wl $tg; for spec in "$@"; do printf " %s=%s" $spec ${best[$spec]}; done; echo
