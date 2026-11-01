#!/bin/bash
# Round 14: VGPU_HOST_GPU_VERIFY=1 (every launch the mode would offload also runs in the simulator and the memory is compared) with the traffic ban built in.
# usage: verify_ban.sh <rows...>; T = seconds per program (default 100)
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=${SHIM:-$HOME/sim/build/shim} VGPU_THREADS=4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 VGPU_HOST_GPU_LOG=1 VGPU_QUIET=1
T=${T:-100}; d=$(mktemp -d); cd $d
for n in "$@"; do case $n in *-cuda) p=~/corp/hec/$n/main ;; *) p=~/corp/poly/$n ;; esac
  timeout $T $p > /dev/null 2> $d/err.txt; l=$d/err.txt
  echo "$n | in-set identical $(grep -c 'in the exact set): identical' $l) different $(grep -c 'in the exact set): DIFFERENT' $l) | outside-set identical $(grep -c 'outside the exact set): identical' $l) different $(grep -c 'outside the exact set): DIFFERENT' $l) | traffic-bans $(grep -c '(traffic)' $l)"
  grep 'DIFFERENT' $l | sed 's/^/    /' | sort | uniq -c | head -3; done
echo FINISHED
