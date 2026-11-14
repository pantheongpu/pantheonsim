#!/bin/bash
# Round sp68 on the A10G rig: one VGPU_HOST_GPU_VERIFY=1 run per program with the square root idiom on (every launch the mode sends to the card also runs
# in the simulator and the memory is compared to the bit); GRAMSCHM on the merged build, cut at 600 s and counted from the per-launch lines.
cd ~/w
P=corp/polybenchGpu/bin R=corp/gpu-rodinia/bin H=corp/hec/src
V="VGPU_HOST_GPU=exact;VGPU_HOST_GPU_DIV=1;VGPU_HOST_GPU_SQRT=1;VGPU_HOST_GPU_VERIFY=1;VGPU_HOST_GPU_LOG=1"
python3 ab.py --out res/verify.json --runs 1 --timeout 600 --prog "CORR=$P:./CORR" --prog "hellinger=$H/hellinger-cuda:./main 1" --prog "knn=$H/knn-cuda:./main 1" \
  --prog "rod_nn=$R:./nn $HOME/w/corp/filelist4 -r 5 -lat 30 -lng 90" --prog "hec_nn=$H/nn-cuda:./main $HOME/w/corp/filelist4 -r 5 -lat 30 -lng 90 -i 10 -t" \
  --side "v=b68:$V" > res/verify.txt 2>&1
# GRAMSCHM: stderr kept to count the per-launch verify lines
(cd $P && env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$HOME/w/b65/shim VGPU_HOST_GPU=exact VGPU_HOST_GPU_DIV=1 VGPU_HOST_GPU_SQRT=1 \
  VGPU_HOST_GPU_VERIFY=1 VGPU_HOST_GPU_LOG=1 timeout 600 ./GRAMSCHM > $HOME/w/res/gram-verify.out 2> $HOME/w/res/gram-verify.err)
echo "rc=$?" >> res/verify.txt
grep -oE "^host_gpu: verify \S+ \([^)]*\): (identical|DIFFERENT)" res/gram-verify.err | sort | uniq -c >> res/verify.txt
grep "^host_gpu_verify" res/gram-verify.err >> res/verify.txt
xz -f res/gram-verify.err
echo done > res/verify.done
