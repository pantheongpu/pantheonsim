#!/bin/bash
# run1.sh <outprefix> <tier 0|1> <cpu> <stats 0|1> -- cmd...   (host; wall/user/sys/maxrss in <outprefix>.time, stdout+stderr in <outprefix>.out)
out=$1 tier=$2 cpu=$3 stats=$4; shift 5
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VGPU_THREADS=${VGPU_THREADS:-1} VGPU_TIER=$tier
if [ "$stats" = 1 ] && [ "$tier" = 1 ]; then export VGPU_TIER_STATS=1 VGPU_TIER_STATS_FILE=$out.stats; rm -f $out.stats; fi
/usr/bin/time -f "%e %U %S %M" -o $out.time taskset -c $cpu "$@" > $out.out 2>&1
echo $? > $out.rc
