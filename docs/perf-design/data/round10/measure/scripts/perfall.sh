#!/bin/bash
cd ~/sim/m; P=$(ls /usr/lib/linux-tools/*/perf|head -1)
for c in "$@"; do
  sudo env HOME=/home/ubuntu $P record -F 300 -o ct/perf_$c.data -- ./cut.sh $c 1 0 0 ct/perf_$c > ct/perf_$c.log 2>&1
  echo "$c done" >> ct/perf_done.txt
done
