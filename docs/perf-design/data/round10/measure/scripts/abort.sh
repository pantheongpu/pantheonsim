#!/bin/bash
# abort.sh <reps> <cpu> <groups...>: serial alternating, abort frequency
reps=$1 cpu=$2; shift 2
cd /tmp; PY=~/.local/share/torch-cu130/bin/python; mkdir -p ~/sim/m/ab
for r in $(seq 1 $reps); do for g in "$@"; do for t in 0 1; do
  export VGPU_SWEEP_TIER=full VGPU_SWEEP_GROUPS=$g
  ~/sim/m/run1.sh ~/sim/m/ab/${g}_r${r}_t$t $t $cpu 0 -- ~/sim/build/vgpu run --gpu nvidia/rtx5090 --preload $PY ~/sim/nvidia/tests/pytorch/sweep.py
  echo "$g r$r t$t rc=$(cat ~/sim/m/ab/${g}_r${r}_t$t.rc) wall=$(tail -1 ~/sim/m/ab/${g}_r${r}_t$t.time|cut -d' ' -f1)" >> ~/sim/m/ab/summary.txt
done; done; done
echo DONE >> ~/sim/m/ab/summary.txt
