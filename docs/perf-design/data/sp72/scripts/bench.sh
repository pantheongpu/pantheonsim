#!/bin/bash
# usage: bench.sh <tag> <args to bench.py...>   (run from base tree; json + log in /tmp/sp72-amd-tier/out)
cd /tmp/sp72-amd-tier; mkdir -p out
tag=$1; shift
export PATH=/tmp/sp69-inq/rocm/bin:/tmp/sp69-inq/rocm/lib/llvm/bin:$PATH ROCM_PATH=/tmp/sp69-inq/rocm HIP_PATH=/tmp/sp69-inq/rocm
python3 -I ${BENCHSRC:-base}/tools/bench-amd/bench.py --hipcc /tmp/sp69-inq/rocm/bin/hipcc --cache /tmp/sp72-amd-tier/bench-cache --json out/bench-$tag.json "$@" > out/bench-$tag.log 2>&1
echo BENCH_EXIT $? >> out/bench-$tag.log
