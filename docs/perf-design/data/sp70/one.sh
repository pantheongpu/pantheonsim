#!/bin/bash
# usage: one70.sh <label> <shimdir> "<tags>" [ENV=...] -> prints "label wall user sys rc"
label=$1; sim=$2; tags=$3; shift 3
. /tmp/sp69-inq/env.sh
export PSEUDOPOD_SHARE_PATH=/tmp/sp69-inq/inq-src/external_libs/pseudopod/share/ INQ_SHARE_PATH=/tmp/sp69-inq/inq-src/share/
d=/tmp/sp69-inq/runs/w_$label; rm -rf $d; mkdir -p $d; cd $d
/usr/bin/time -f "%e %U %S" -o time.txt env "$@" VGPU_GPU=amd/mi300x LD_LIBRARY_PATH=$sim/build/shim:$LD_LIBRARY_PATH /tmp/sp69-inq/build-hip/src/all_unit_tests "$tags" > out.txt 2> err.txt
rc=$?
echo "$label $(tail -1 time.txt) rc=$rc pass=$(grep -c "All tests passed" out.txt)"
cd /tmp/sp69-inq/runs; rm -rf $d
