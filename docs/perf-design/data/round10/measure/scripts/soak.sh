#!/bin/bash
# soak.sh <suite...>: runs each suite under tier 0 (cpus 0,8) and tier 1 (cpus 3,11) in parallel
cd ~/sim/m; mkdir -p soak
for s in "$@"; do
  for t in 0 1; do
    if [ $t = 0 ]; then cpus=0,8; else cpus=3,11; fi
    ( export VGPU_TIER=$t VGPU_THREADS=1 SUITE_TIMEOUT=${SUITE_TIMEOUT:-300}; taskset -c $cpus ./ext/suites.sh run $s ~/sim/m/suites/$s ~/sim/build/shim ~/sim/m/soak/${s}_t$t > ~/sim/m/soak/${s}_t$t.txt 2>&1 ) &
  done
  wait
  echo "$s done $(date +%T)" >> soak/done.txt
done
