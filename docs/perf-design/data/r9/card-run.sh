#!/usr/bin/env bash
# Round r9 card run (A10G): why launches stay in the simulator (#659) and what silu/s3d pay (#662). From ~/src, corpora ~/corp, build ~/build.
# Stages (run: decisions samples others silu gram, in that order). The first run of the last three failed (the packed corpus listed aidw without its directory; the GRAMSCHM stage had no sim form); this is the script that made the tables.
set -u
out=$HOME/r9-out; build=$HOME/build; corp=$HOME/corp
mkdir -p "$out"; cd ~/src
export PATH=/usr/local/cuda-13.0/bin:$PATH
H="python3 tools/bench/hostgpu/run.py --build $build --corp $corp --gpu nvidia/a10g"
S=$corp/samples
sample_dir() { dirname "$(find "$S" -type f -name "$1" -perm -u+x | head -1)"; }
logged() {   # tag secs cwd cmd...
  local tag=$1 secs=$2 cwd=$3; shift 3
  ( cd "$cwd" && env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 timeout -s KILL "$secs" "$@" > /dev/null 2> "$out/log-$tag.err" )
  grep -E '^host_gpu' "$out/log-$tag.err" > "$out/log-$tag.txt"
  echo "== $tag $(grep -c 'host_gpu_kept' "$out/log-$tag.txt") kept"
}
echo "START $(date -u +%T)" > "$out/progress"
# 1. the decisions: every kept launch with its reason
for p in segmentationTreeThrust histogram mergeSort; do logged $p 120 "$(sample_dir $p)" ./$p; done
logged radixsort2 60 "$corp/hec/src/radixsort2-cuda" ./main -float -keysonly -n=65536
logged silu 60 "$corp/hec/src/silu-cuda" ./main 8 8 64 64 1
logged s3d 60 "$corp/hec/src/s3d-cuda" ./main -q -n 1 -s 1
echo "decisions done $(date -u +%T)" >> "$out/progress"
# 2. timings, sim/exact/turbo and the rule before sp76 (turbod0)
$H --corpora samples --programs segmentationTreeThrust,histogram,mergeSort --forms sim,exact,turbo,turbod0 --runs 10 --long-secs 20 --long-runs 3 --json "$out/samples.json" --md "$out/samples.md" > "$out/samples.txt" 2>&1
echo "samples done $(date -u +%T)" >> "$out/progress"
$H --corpora polybench,rodinia,hecbench --programs 3MM,lud,radixsort2-cuda --forms sim,exact,turbo,turbod0 --runs 5 --long-secs 20 --long-runs 3 --json "$out/others.json" --md "$out/others.md" > "$out/others.txt" 2>&1
echo "others done $(date -u +%T)" >> "$out/progress"
$H --corpora hecbench --programs silu-cuda,s3d-cuda --forms sim,exact,turbo,turbod0,turbo0 --runs 15 --long-secs 20 --long-runs 3 --json "$out/silu.json" --md "$out/silu.md" > "$out/silu.txt" 2>&1
echo "silu done $(date -u +%T)" >> "$out/progress"
# GRAMSCHM last (about 130 s a run): one run turbo and one exact
$H --corpora polybench --programs GRAMSCHM --forms sim,exact,turbo --runs 1 --long-secs 20 --long-runs 1 --timeout 400 --json "$out/gram.json" --md "$out/gram.md" > "$out/gram.txt" 2>&1
echo "ALL DONE $(date -u +%T)" >> "$out/progress"
