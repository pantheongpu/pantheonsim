#!/bin/bash
# sp68 offline table (wall: /usr/bin/time inside the lock, so the wait for it is not counted): per program, the plain simulator wall time and the exact-mode log (square root idiom on, division on, no card: MAX_MIB=0)
cd ~/.cache/sp68
C=$PWD/corp
mkdir -p off
run() {   # name timeout dir cmd...
  local n=$1 t=$2 dir=$3; shift 3
  (cd $dir && flock /tmp/pantheonsim-builder.lock nice -n 19 /usr/bin/time -f "%e" -o $HOME/.cache/sp68/off/$n.wall env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$HOME/.cache/sp68/build/shim timeout $t "$@" > $HOME/.cache/sp68/off/$n.plain.out 2>&1)
  echo "$n plain rc=$? wall=$(tail -1 off/$n.wall)" >> off/summary.txt
  (cd $dir && flock /tmp/pantheonsim-builder.lock nice -n 19 env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$HOME/.cache/sp68/build/shim VGPU_HOST_GPU=exact VGPU_HOST_GPU_SQRT=1 VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_LOG=1 VGPU_HOST_GPU_MAX_MIB=0 timeout $t "$@" > $HOME/.cache/sp68/off/$n.log.out 2> $HOME/.cache/sp68/off/$n.log.err)
  echo "$n log rc=$?" >> off/summary.txt
}
: > off/summary.txt
run CORR 200 $C/polybenchGpu/bin ./CORR
run rod_nn 100 $C/gpu-rodinia/bin ./nn $C/filelist4 -r 5 -lat 30 -lng 90
run asyncAllocation 100 $C/hec/src/asyncAllocation-cuda ./main 200 5
run attention-paged 100 $C/hec/src/attention-paged-cuda ./main 8 32 128 4096 128 100
run hellinger 100 $C/hec/src/hellinger-cuda ./main 1
run knn 100 $C/hec/src/knn-cuda ./main 1
run hec_nn 100 $C/hec/src/nn-cuda ./main $C/filelist4 -r 5 -lat 30 -lng 90 -i 10 -t
run GRAMSCHM 250 $C/polybenchGpu/bin ./GRAMSCHM
echo DONE >> off/summary.txt
