#!/bin/bash
# wall time of a program with the mode off and on; mode flags in $MODEENV (e.g. VGPU_HOST_GPU_STATIC=1)
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$HOME/sim/build/shim VGPU_THREADS=4
d=$(mktemp -d); cd $d
now() { date +%s.%N; }
row() { n=$1; shift
  a=$(now); env VGPU_QUIET=1 "$@" > $d/o0.txt 2>/dev/null; b=$(now)
  env VGPU_HOST_GPU=exact $MODEENV "$@" > $d/o1.txt 2> $d/e1.txt; c=$(now)
  s=$(grep "^host_gpu: [0-9]* launches" $d/e1.txt | sed 's/host_gpu: //; s/ (.*//')
  same=$(diff <(grep -v -i "time\|^[0-9.]*$" $d/o0.txt) <(grep -v -i "time\|^[0-9.]*$" $d/o1.txt) > /dev/null && echo same || echo DIFF)
  g0=$(grep -A1 "GPU Time" $d/o0.txt | tail -1); g1=$(grep -A1 "GPU Time" $d/o1.txt | tail -1)
  printf "%-9s process off %6.2f s on %6.2f s | GPU phase off %s on %s | output %s | %s\n" $n $(echo "$b - $a" | bc) $(echo "$c - $b" | bc) "$g0" "$g1" $same "$s"; }
P=~/corp/poly
for n in GEMM SYRK SYR2K 3MM 2DCONV 3DCONV GEMVER MVT FDTD-2D; do row $n $P/$n; done
echo FINISHED
