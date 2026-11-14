#!/bin/bash
# usage: bench70.sh <tag> <buildA> <buildB> [extra bench args]
cd /tmp/sp69-inq
tag=$1; A=$2; B=$3; shift 3
export PATH=/tmp/sp69-inq/rocm/bin:/tmp/sp69-inq/rocm/lib/llvm/bin:$PATH ROCM_PATH=/tmp/sp69-inq/rocm HIP_PATH=/tmp/sp69-inq/rocm
python3 -I sim/tools/bench-amd/bench.py --build $A --build $B --hipcc /tmp/sp69-inq/rocm/bin/hipcc --cache /tmp/sp69-inq/bench-cache --json /tmp/sp69-inq/bench-$tag.json "$@" > bench-$tag.log 2>&1
echo BENCH_EXIT $? >> bench-$tag.log
