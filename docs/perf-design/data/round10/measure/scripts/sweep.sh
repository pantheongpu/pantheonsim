#!/bin/bash
# sweep.sh <tier quick|full> <reps> <stats 0|1> <groups...>  : pairs tier0/tier1 concurrently on cpu 1 and 2 (swapped each rep)
tier=$1 reps=$2 stats=$3; shift 3
cd ~/sim
PY=~/.local/share/torch-cu130/bin/python
dir=m/sw/$tier; mkdir -p $dir
for r in $(seq 1 $reps); do
 for g in "$@"; do
  if (( r % 2 )); then c0=1; c1=2; else c0=2; c1=1; fi
  export VGPU_SWEEP_TIER=$tier VGPU_SWEEP_GROUPS=$g
  for t in 0 1; do cpu=$c0; [ $t = 1 ] && cpu=$c1
    ( cd /tmp && TRITON_CACHE_DIR=/tmp/m_triton_$t INDUCTOR=0 ~/sim/m/run1.sh ~/sim/$dir/${g}_r${r}_t$t $t $cpu $stats -- ~/sim/build/vgpu run --gpu nvidia/rtx5090 --preload $PY ~/sim/nvidia/tests/pytorch/sweep.py ) &
  done
  wait
  echo "$tier $g r$r $(cat ~/sim/$dir/${g}_r${r}_t0.time) | $(cat ~/sim/$dir/${g}_r${r}_t1.time) uptime=$(cut -d' ' -f1-3 /proc/loadavg)" >> ~/sim/$dir/summary.txt
 done
done
echo SWEEP_DONE >> ~/sim/$dir/summary.txt
