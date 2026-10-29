#!/bin/bash
# step A: a VERIFY run of every corpus program with the event trace on (100 s per program at most)
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$HOME/sim/build/shim VGPU_THREADS=4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=1 VGPU_QUIET=1
T=${T:-100}; mkdir -p ~/trace; d=$(mktemp -d); cd $d
one() { n=$1; shift; s=$(date +%s.%N); VGPU_HOST_GPU_TRACE=$HOME/trace/$n.trace timeout $T "$@" > /dev/null 2>&1; rc=$?; e=$(date +%s.%N)
  echo "$n rc=$rc wall=$(echo "$e - $s" | bc) launches=$(grep -c '^L ' $HOME/trace/$n.trace)"; }
for p in ~/corp/poly/*; do [ -x $p ] && [ ! -d $p ] && case $p in *.log) ;; *) one poly_$(basename $p) $p ;; esac; done
R=~/corp/rod
one rod_backprop $R/backprop 16384; one rod_gaussian $R/gaussian -s 64; one rod_lud $R/lud -s 256 -v; one rod_needle $R/needle 256 10; one rod_pathfinder $R/pathfinder 1000 50 20
one rod_srad_v2 $R/srad_v2 64 64 0 31 0 31 0.5 2; one rod_pf $R/particlefilter_float -x 64 -y 64 -z 5 -np 1000; one rod_sc $R/sc_gpu 10 20 16 1024 1024 1000 none out.txt 1
one rod_hotspot $R/hotspot 64 2 20 $R/data/temp_64 $R/data/power_64 o.out; one rod_bfs $R/bfs $R/data/graph4096.txt; one rod_lavaMD $R/lavaMD -boxes1d 4
while IFS='|' read dd a; do one hec_$dd ~/corp/hec/$dd/main $a; done < /tmp/hec_list.txt
echo FINISHED
