#!/bin/bash
# sp67 #570 item 3: the division idiom recogniser (VGPU_HOST_GPU_DIV=1) on the programs built by nvcc 13.0 and 13.2; no card, every launch refused for its size
# (VGPU_HOST_GPU_MAX_MIB=0) so that every kernel is classified and nothing starts a driver. Prints the per-kernel lines of the log.
cd ~/.cache/sp67/tk
for v in 13.0 13.2; do
  for p in ADI LU GRAMSCHM CORR COVAR host_gpu_lazy:div host_gpu_lazy:divexp host_gpu_lazy:rcp; do
    b=${p%%:*}; a=${p#*:}; [[ $a == $p ]] && a=""
    echo "== $v $p"
    (cd $v && flock /tmp/pantheonsim-builder.lock nice -n 19 env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$HOME/.cache/sp67/build/shim VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_MAX_MIB=0 timeout 20 ./$b $a 2>&1 >/dev/null) \
      | grep -E "refused:|SFU forms|not the division idiom|stays in the simulator|admitted" | sed -E "s/ \(cached\)//; s/\(sim .*/(size rule: admitted, kept for its size)/" | sort | uniq -c | sed "s/^ */  /"
  done
done
