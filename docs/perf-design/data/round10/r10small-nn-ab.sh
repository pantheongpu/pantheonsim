#!/bin/bash
# usage (inside drx.sh, repo root /w): r10small-nn-ab.sh <cpu> <runs> <buildA> <buildB> [group]
# Alternates A B A B ...: the PyTorch quick-sweep group (default nn) under `vgpu run --preload`, one thread, pinned to <cpu>.
# Prints "<build> <wall s> <user+sys s> <ok lines>" per run.
cpu=$1 runs=$2 A=$3 B=$4 group=${5:-nn}
py=/root/.local/share/torch-cu130/bin/python
sweep=/w/nvidia/tests/pytorch/sweep.py
tmp=$(mktemp -d); cd "$tmp"
for i in $(seq "$runs"); do
  for b in "$A" "$B"; do
    s=$(date +%s.%N)
    out=$( { OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VGPU_SWEEP_GROUPS=$group VGPU_SWEEP_TIER=quick TRITON_CACHE_DIR=$tmp/t TORCHINDUCTOR_CACHE_DIR=$tmp/i \
      /usr/bin/time -f "TIME %e %U %S" taskset -c "$cpu" /w/$b/vgpu run --gpu nvidia/rtx5090 --preload "$py" "$sweep" 2>&1; } )
    t=$(grep '^TIME' <<<"$out" | tail -1)
    echo "$b $(awk '{print $2, $3+$4}' <<<"$t") ok=$(grep -c '^ok ' <<<"$out") fail=$(grep -c '^FAIL ' <<<"$out")"
  done
done
