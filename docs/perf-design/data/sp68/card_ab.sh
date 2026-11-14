#!/bin/bash
# Round sp68 on the A10G rig: the A/B of the square root idiom, 3 interleaved runs per side. b68 = sp68-sqrt-idiom, b65 = sp68-with-sp65 (the scratch
# merge with Round sp65's streak rule). sim = mode off; x0 = exact, VGPU_HOST_GPU_DIV=1, VGPU_HOST_GPU_SQRT=0; x1 = the same with VGPU_HOST_GPU_SQRT=1.
cd ~/w
P=corp/polybenchGpu/bin R=corp/gpu-rodinia/bin H=corp/hec/src
X0="VGPU_HOST_GPU=exact;VGPU_HOST_GPU_DIV=1;VGPU_HOST_GPU_SQRT=0"
X1="VGPU_HOST_GPU=exact;VGPU_HOST_GPU_DIV=1;VGPU_HOST_GPU_SQRT=1"
python3 ab.py --out res/ab-short.json --runs 3 --prog "CORR=$P:./CORR" --prog "COVAR=$P:./COVAR" --prog "ADI=$P:./ADI" \
  --prog "hellinger=$H/hellinger-cuda:./main 1" --prog "knn=$H/knn-cuda:./main 1" --prog "rod_nn=$R:./nn $HOME/w/corp/filelist4 -r 5 -lat 30 -lng 90" \
  --prog "hec_nn=$H/nn-cuda:./main $HOME/w/corp/filelist4 -r 5 -lat 30 -lng 90 -i 10 -t" \
  --side sim=b68 --side "x0=b68:$X0" --side "x1=b68:$X1" > res/ab-short.txt 2>&1
python3 ab.py --out res/ab-gram.json --runs 3 --prog "GRAMSCHM=$P:./GRAMSCHM" --side sim=b65 --side "x0=b65:$X0" --side "x1=b65:$X1" > res/ab-gram.txt 2>&1
python3 ab.py --out res/ab-gram68.json --runs 1 --prog "GRAMSCHM=$P:./GRAMSCHM" --side "x0=b68:$X0" --side "x1=b68:$X1" > res/ab-gram68.txt 2>&1
echo done > res/ab.done
