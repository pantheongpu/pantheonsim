#!/bin/bash
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$HOME/sim/build/shim VGPU_THREADS=4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 VGPU_QUIET=1
cd $(mktemp -d)
for p in ~/corp/poly/* ; do [ -x $p ] && [ ! -d $p ] && case $p in *.log) ;; *) timeout 8 $p 2>&1 | grep "refused" ;; esac; done
R=~/corp/rod
for t in "backprop 16384" "gaussian -s 64" "lud -s 256 -v" "needle 256 10" "pathfinder 1000 50 20" "hotspot 64 2 20 $R/data/temp_64 $R/data/power_64 o.out" "bfs $R/data/graph4096.txt"; do set -- $t; n=$1; shift; timeout 10 $R/$n "$@" 2>&1 | grep refused; done
