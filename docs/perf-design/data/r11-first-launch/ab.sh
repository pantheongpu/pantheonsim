#!/usr/bin/env bash
# usage: ab.sh tag corpora programs runs [extra run.py args]
tag=$1; corp=$2; progs=$3; runs=$4; shift 4
export PATH=/usr/local/cuda-13.0/bin:$PATH
mkdir -p ~/r11-ab2; cd ~/a656
flock /tmp/rig-card.lock python3 tools/bench/hostgpu/run.py --build ~/a656-build --corp ~/corp --gpu nvidia/a10g --corpora $corp --programs $progs --forms sim,turborules,turbof0 --runs $runs --long-secs 30 --long-runs 3 --timeout 300 --no-exact-stats --json ~/r11-ab2/$tag.json --md ~/r11-ab2/$tag.md "$@" > ~/r11-ab2/$tag.txt 2>&1
echo done $tag >> ~/r11-ab2/progress
