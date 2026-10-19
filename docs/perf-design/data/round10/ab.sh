#!/bin/bash
# usage: ab.sh <cpu> <runs> <program> <buildA> <buildB> [ENV=V ...]   (run inside drx.sh; alternates A B A B, prints user+sys CPU seconds per run)
cpu=$1 runs=$2 prog=$3 A=$4 B=$5; shift 5
for i in $(seq $runs); do
  for b in $A $B; do
    t=$( { env LD_LIBRARY_PATH=$b/shim VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 "$@" /usr/bin/time -f "%U %S" taskset -c $cpu $prog >/tmp/out.$$ ; } 2>&1 | tail -1)
    grep -q PASS /tmp/out.$$ || echo "NO PASS $b" >&2
    echo "$b $(echo $t | awk '{print $1+$2}')"
  done
done
