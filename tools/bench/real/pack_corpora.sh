#!/usr/bin/env bash
# pack_corpora.sh <outdir>: collects the prebuilt corpora from the dev machine
# (~/.cache/{polybenchGpu,gpu-rodinia,HeCBench,cuda-samples}) into the layout
# real.py expects.  Only copies files; nothing is run.  Then
#   tar -C <outdir> -cf - . | ssh server3 'mkdir -p /tmp/corp && tar -C /tmp/corp -xf -'
set -e
out=${1:?outdir}; c=$HOME/.cache
mkdir -p "$out"/{polybench/bin,rodinia,hec/src,samples}
for p in $(ls "$c/polybenchGpu/bin" | grep -v -E '\.log|out_'); do cp -p "$c/polybenchGpu/bin/$p" "$out/polybench/bin/"; done
tar -C "$c/gpu-rodinia" -cf - --exclude='out*' --exclude='*_card*' --exclude='*_sim*' --exclude='*.log' bin | tar -C "$out/rodinia" -xf -
cp "$c/HeCBench/reduced.txt" "$out/hec/"
dirs=$(cut -d'|' -f1 "$c/HeCBench/reduced.txt")
(cd "$c/HeCBench/src" && tar -cf - --exclude='*.o' --exclude='*.log' --exclude='fpdc-cuda/*.bin' --exclude='redrc.*' --exclude='rc.*' $dirs) | tar -C "$out/hec/src" -xf -
# samples that passed on the simulator (rc 0 in sim_main.txt)
sd=$(awk '$1==0{print $2}' "$c/cuda-samples/build/sim_main.txt" | xargs -n1 dirname | sort -u)
(cd "$c/cuda-samples/build" && tar -cf - --exclude=CMakeFiles --exclude='*.log' $sd) | tar -C "$out/samples" -xf -
cp "$c/cuda-samples/build/sim_main.txt" "$out/samples/"
