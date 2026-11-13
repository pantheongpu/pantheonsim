#!/bin/bash
# Round sp68 on the A10G rig: the short programs, 20 interleaved runs per side, timed to the microsecond (the 3-run A/B rounds to the millisecond, and
# the recogniser's own cost, paid once per kernel, is what a 20 ms program could show); then knn's output with its timing lines dropped, per side.
cd ~/w
R=corp/gpu-rodinia/bin H=corp/hec/src
X0="VGPU_HOST_GPU=exact;VGPU_HOST_GPU_DIV=1;VGPU_HOST_GPU_SQRT=0"
X1="VGPU_HOST_GPU=exact;VGPU_HOST_GPU_DIV=1;VGPU_HOST_GPU_SQRT=1"
python3 ab.py --out res/ab-tiny.json --runs 20 --prog "rod_nn=$R:./nn $HOME/w/corp/filelist4 -r 5 -lat 30 -lng 90" \
  --prog "hec_nn=$H/nn-cuda:./main $HOME/w/corp/filelist4 -r 5 -lat 30 -lng 90 -i 10 -t" --prog "ADI=corp/polybenchGpu/bin:./ADI" \
  --side "x0=b68:$X0" --side "x1=b68:$X1" > res/ab-tiny.txt 2>&1
for s in sim x0 x1; do
  case $s in sim) e="";; x0) e="VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_SQRT=0";; x1) e="VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_SQRT=1";; esac
  (cd $H/knn-cuda && env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$HOME/w/b68/shim $e ./main 1 2>/dev/null | grep -v " done in " | sha256sum | cut -c1-16) >> res/knn-hash.txt
done
echo done > res/short.done
