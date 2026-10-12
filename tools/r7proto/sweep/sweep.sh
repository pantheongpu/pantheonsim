#!/usr/bin/env bash
cd /tmp/claude-1002/r7d
run() { # binary isa scenario args...
  local bin=$1 isa=$2; shift 2
  echo "### $bin isa=$isa :: $*"; echo "load before: $(cut -d' ' -f1-3 /proc/loadavg)"
  ./$bin "$@" reps=9 variants=base,proto,genr,genf opt=2 isa=$isa 2>&1 | grep -v "^warp" | sed 's/  */ /g' | cut -c1-210
  echo "load after: $(cut -d' ' -f1-3 /proc/loadavg)"
}
for pair in "r7proto base" "r7proto_avx2 avx2"; do set -- $pair
  run $1 $2 elf/int_virus.2.sm_86.cubin int_virus grid=64 block=256 iters=200
  run $1 $2 k.cubin ffma_loop grid=16 block=256 iters=1000
  run $1 $2 k.cubin int_loop grid=16 block=256 iters=1000
  run $1 $2 k.cubin vecadd grid=2048 block=256 bulk=0
  run $1 $2 k.cubin vecadd grid=2048 block=256 bulk=1
  run $1 $2 k.cubin saxpy grid=64 block=256 stride_iters=16 bulk=0
  run $1 $2 k.cubin saxpy grid=64 block=256 stride_iters=16 bulk=1
done
