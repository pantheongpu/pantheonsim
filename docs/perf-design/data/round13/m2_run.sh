#!/bin/bash
# m2_run.sh <program-path> <name> [args...]: capture the first launches of a program under the simulator (A10G profile), then replay on the card.
prog=$1; name=$2; shift 2
cap=$HOME/cap/$name; rm -rf $cap; mkdir -p $cap
wd=$HOME/cap/$name.wd; rm -rf $wd; mkdir -p $wd
(cd $wd && VGPU_QUIET=1 VGPU_GPU=nvidia/a10g VGPU_THREADS=4 VGPU_HOSTGPU_CAPTURE=$cap VGPU_HOSTGPU_CAPTURE_MAX=${CAPMAX:-2} VGPU_HOSTGPU_CAPTURE_STOP=${CAPSTOP:-0} \
  LD_LIBRARY_PATH=$HOME/sim/build/shim timeout ${SIMTIMEOUT:-900} $prog "$@" > $cap/stdout.txt 2>&1; echo "sim rc $?" > $cap/rc.txt)
/tmp/m2_replay $prog $cap
