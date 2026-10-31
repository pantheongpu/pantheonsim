#!/bin/bash
# VGPU_HOST_GPU_VERIFY=all over the three corpora: every launch the card can run is run on both and compared (60 s per program, per-launch lines).
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$HOME/sim/build/shim VGPU_THREADS=4 VGPU_HOST_GPU=exact VGPU_HOST_GPU_VERIFY=all VGPU_HOST_GPU_LOG=1 VGPU_QUIET=1
T=${T:-60}
d=$(mktemp -d); cd $d
one() { n=$1; shift; timeout $T "$@" > /dev/null 2> $d/err.txt; l=$d/err.txt
  echo "$n | in-set identical $(grep -c 'in the exact set): identical' $l) different $(grep -c 'in the exact set): DIFFERENT' $l) | outside-set identical $(grep -c 'outside the exact set): identical' $l) different $(grep -c 'outside the exact set): DIFFERENT' $l)"
  grep 'DIFFERENT' $l | sed 's/^/    /' | sort | uniq -c | head -3; }
for p in ~/corp/poly/*; do [ -x $p ] && [ ! -d $p ] && case $p in *.log) ;; *) one poly/$(basename $p) $p ;; esac; done
R=~/corp/rod
one rod/backprop $R/backprop 16384; one rod/gaussian $R/gaussian -s 64; one rod/lud $R/lud -s 256 -v; one rod/needle $R/needle 256 10; one rod/pathfinder $R/pathfinder 1000 50 20
one rod/srad_v2 $R/srad_v2 64 64 0 31 0 31 0.5 2; one rod/pf $R/particlefilter_float -x 64 -y 64 -z 5 -np 1000; one rod/sc $R/sc_gpu 10 20 16 1024 1024 1000 none out.txt 1
one rod/hotspot $R/hotspot 64 2 20 $R/data/temp_64 $R/data/power_64 o.out; one rod/bfs $R/bfs $R/data/graph4096.txt; one rod/lavaMD $R/lavaMD -boxes1d 4
while IFS='|' read dd a; do one hec/$dd ~/corp/hec/$dd/main $a; done < /tmp/hec_list.txt
echo FINISHED
