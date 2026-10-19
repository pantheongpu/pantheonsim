#!/bin/bash
# The training run of this round's PGO measurement (inside drx.sh, repo root /w).
#   r10small-pgo-train.sh <instrumented build dir> <bench_kernels binary> [extra: pytorch group names to train on, e.g. "tensor linalg"]
# Every harness workload at the quick size on the SASS engine (tier off and on) and on the PTX interpreter, one host thread.
b=$1 bk=$2 groups=${3:-}
export LD_LIBRARY_PATH=/w/$b/shim VGPU_QUIET=1 VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1
for w in $("$bk" --list); do
  for mode in "VGPU_SASS=1" "VGPU_SASS=1 VGPU_TIER=1" "VGPU_SASS=0"; do
    env $mode "$bk" "$w" quick >/dev/null 2>&1 || echo "train: $w ($mode) failed" >&2
  done
done
if [ -n "$groups" ]; then
  py=/root/.local/share/torch-cu130/bin/python tmp=$(mktemp -d)
  for g in $groups; do
    ( cd "$tmp" && OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VGPU_SWEEP_GROUPS=$g VGPU_SWEEP_TIER=quick \
        /w/$b/vgpu run --gpu nvidia/rtx5090 --preload "$py" /w/nvidia/tests/pytorch/sweep.py >/dev/null 2>&1 ) || echo "train: pytorch group $g failed" >&2
  done
fi
echo "train: $(find /w/$b -name '*.gcda' | wc -l) profile files"
