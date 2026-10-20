#!/bin/bash
# profgrp.sh <group> <tier> : perf with dwarf call graph on cpu 3
g=$1 t=$2; cd /tmp; P=$(ls /usr/lib/linux-tools/*/perf|head -1)
export VGPU_SWEEP_TIER=full VGPU_SWEEP_GROUPS=$g
sudo -E env HOME=/home/ubuntu $P record -F 250 --call-graph dwarf,12288 -o ~/sim/m/prof/${g}${t}.data -- ~/sim/m/run1.sh ~/sim/m/prof/${g}${t} $t 3 0 -- ~/sim/build/vgpu run --gpu nvidia/rtx5090 --preload ~/.local/share/torch-cu130/bin/python ~/sim/nvidia/tests/pytorch/sweep.py > ~/sim/m/prof/${g}${t}.log 2>&1
sudo $P script -i ~/sim/m/prof/${g}${t}.data -F comm,ip,sym,dso 2>/dev/null > ~/sim/m/prof/${g}${t}.script
echo done
