#!/bin/bash
# panpairs.sh <reps> <dur> <workloads...>
reps=$1 dur=$2; shift 2
cd ~/sim/m; mkdir -p pa
for r in $(seq 1 $reps); do for w in "$@"; do
  if (( r % 2 )); then c0=1; c1=2; else c0=2; c1=1; fi
  ./pan.sh $w 0 $c0 $dur pa/${w}_r${r}_t0 & ./pan.sh $w 1 $c1 $dur pa/${w}_r${r}_t1 & wait
  echo "$w r$r $(tail -n 1 pa/${w}_r${r}_t0.time) | $(tail -n 1 pa/${w}_r${r}_t1.time) rc=$(cat pa/${w}_r${r}_t0.rc)/$(cat pa/${w}_r${r}_t1.rc) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> pa/summary.txt
done; done
echo PAN_DONE >> pa/summary.txt
