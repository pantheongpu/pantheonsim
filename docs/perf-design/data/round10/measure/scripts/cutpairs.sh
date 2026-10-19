#!/bin/bash
# cutpairs.sh <reps> <cases...>: tier0/tier1 concurrently on cpu 1 and 2 (swapped each rep)
reps=$1; shift
cd ~/sim/m; mkdir -p ct
for r in $(seq 1 $reps); do for c in "$@"; do
  if (( r % 2 )); then c0=1; c1=2; else c0=2; c1=1; fi
  ./cut.sh $c 0 $c0 0 ct/${c}_r${r}_t0 & ./cut.sh $c 1 $c1 1 ct/${c}_r${r}_t1 & wait
  echo "$c r$r $(cat ct/${c}_r${r}_t0.time|tail -1) | $(cat ct/${c}_r${r}_t1.time|tail -1) rc=$(cat ct/${c}_r${r}_t0.rc)/$(cat ct/${c}_r${r}_t1.rc) load=$(cut -d' ' -f1-3 /proc/loadavg)" >> ct/pairs_summary.txt
done; done
echo PAIRS_DONE >> ct/pairs_summary.txt
