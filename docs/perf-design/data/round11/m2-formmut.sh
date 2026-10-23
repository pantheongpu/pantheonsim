#!/bin/bash
# in the container, cwd /w: for every generated form, the code generated for it made wrong in one lane; sass_tier must then differ from the tier
cd /w/build-work
export LD_LIBRARY_PATH=/w/build-mut/shim VGPU_GPU=nvidia/rtx3060 VGPU_THREADS=1 VGPU_COUNTERS=1 VGPU_SASS_LOG=1 VGPU_QUIET=1
ref=$(VGPU_TIER=1 ./sass_tier.sm_86 2>&1)
base=$(VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2 VGPU_JIT_CACHE=/w/build-work/jc-mut0 ./sass_tier.sm_86 2>&1)
[ "$ref" == "$base" ] && echo "unmutated JIT build: identical to the tier" || echo "UNMUTATED DIFFERS"
for f in mov iadd3 iadd3x imad imad32 imadwide imadhi lop3 lea shf isetp fsetp fsel ffma fadd fmul prmt imnmx i2fp maskless; do
  out=$(VGPU_JIT_MUTANT=$f VGPU_TIER=1 VGPU_JIT=1 VGPU_JIT_THRESHOLD=1 VGPU_JIT_SYNC=1 VGPU_JIT_HOT_PERMILLE=0 VGPU_JIT_MAX_OPS=100000000 VGPU_JIT_MIN_REGION=2 VGPU_JIT_CACHE=/w/build-work/jc-mut-$f ./sass_tier.sm_86 2>&1)
  if [ "$out" == "$ref" ]; then echo "$f: SURVIVED (output identical)"; else echo "$f: caught ($(diff <(echo "$ref") <(echo "$out") | grep -c '^>') lines differ)"; fi
  rm -rf /w/build-work/jc-mut-$f
done
