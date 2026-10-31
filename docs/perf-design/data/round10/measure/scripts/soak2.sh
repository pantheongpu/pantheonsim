#!/bin/bash
# soak2.sh <label> <threads> <counters 0|1> : correctness pairs, tier0 on cpus 1,9 and tier1 on cpus 2,10
label=$1 th=$2 cn=$3
cd ~/sim/m; mkdir -p s2/$label
export VGPU_THREADS=$th
[ "$cn" = 1 ] && export VGPU_COUNTERS=1
PY=~/.local/share/torch-cu130/bin/python
for c in tma_load sm120_f4_f6 sm120_bs_mxf4 sm100_blockscaled gemm_sm80_sparse_f16 sm100_f8; do
  for t in 0 1; do if [ $t = 0 ]; then cpu=1,9; else cpu=2,10; fi
    ./cut.sh $c $t $cpu 1 s2/$label/$c.t$t $th & done; wait
done
for g in amp attention fft graphs linalg losses nn ops optim rnn sdpa sparse; do
  for t in 0 1; do if [ $t = 0 ]; then cpu=1,9; else cpu=2,10; fi
    ( export VGPU_SWEEP_TIER=full VGPU_SWEEP_GROUPS=$g; cd /tmp; ~/sim/m/run1.sh ~/sim/m/s2/$label/sw_$g.t$t $t $cpu 1 -- ~/sim/build/vgpu run --gpu nvidia/rtx5090 --preload $PY ~/sim/nvidia/tests/pytorch/sweep.py ) & done; wait
done
echo S2_DONE >> s2/$label/done.txt
