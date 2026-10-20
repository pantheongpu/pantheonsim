#!/bin/bash
# pan.sh <workload> <tier> <cpu> <dur> <outprefix>
w=$1 tier=$2 cpu=$3 dur=$4 out=$5
export LD_LIBRARY_PATH=~/sim/build/shim VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_VRAM_MB=1024 VGPU_MAX_STEPS=17179869184 VGPU_COUNTERS=1 VGPU_SASS_LOG=1
extra=""; case $w in mma_virus|omni_virus) extra="--gemm_size 256";; esac
exec ~/sim/m/run1.sh $out $tier $cpu 1 -- ~/sim/m/pantheon/build-vgpu-86/$w 0 $dur 2 --kernel_loops 2 --grid_size 4 $extra
