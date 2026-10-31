#!/bin/bash
# wall time of a program with the mode off, write-through and persistent (the default); output compared (lines with a time dropped)
export VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH=$HOME/sim/build/shim VGPU_THREADS=4
d=$(mktemp -d); cd $d
now() { date +%s.%N; }
row() { n=$1; shift
  a=$(now); env VGPU_QUIET=1 "$@" > $d/o0.txt 2>/dev/null; b=$(now)
  env VGPU_HOST_GPU=exact VGPU_HOST_GPU_WRITE_THROUGH=1 "$@" > $d/o1.txt 2> $d/e1.txt; c=$(now)
  env VGPU_HOST_GPU=exact "$@" > $d/o2.txt 2> $d/e2.txt; e=$(now)
  s1=$(grep "^host_gpu: [0-9]* launches" $d/e1.txt | sed 's/host_gpu: //; s/ (.*//'); s2=$(grep "^host_gpu: [0-9]* launches" $d/e2.txt | sed 's/host_gpu: //; s/ (.*//')
  m1=$(grep "MiB went" $d/e1.txt | sed 's/host_gpu: //'); m2=$(grep "MiB went" $d/e2.txt | sed 's/host_gpu: //')
  f='time\|^[0-9.]*$\|^GPU\|elapsed\|Elapsed\|seconds\|[0-9] ms\|[0-9] us\|GB/s\|MB/s\|Throughput\|throughput'
  same1=$(diff <(grep -v -i "$f" $d/o0.txt) <(grep -v -i "$f" $d/o1.txt) > /dev/null && echo same || echo DIFF)
  same2=$(diff <(grep -v -i "$f" $d/o0.txt) <(grep -v -i "$f" $d/o2.txt) > /dev/null && echo same || echo DIFF)
  printf "%-28s off %7.2f s | write-through %7.2f s (%s; %s) out %s | persistent %7.2f s (%s; %s) out %s\n" $n $(echo "$b - $a" | bc) $(echo "$c - $b" | bc) "$s1" "$m1" $same1 $(echo "$e - $c" | bc) "$s2" "$m2" $same2; }
P=~/corp/poly
for n in FDTD-2D GRAMSCHM LU JACOBI2D 3DCONV ADI GEMM SYRK; do row poly/$n $P/$n; done
R=~/corp/rod
row rod/needle $R/needle 256 10; row rod/sc $R/sc_gpu 10 20 16 1024 1024 1000 none out.txt 1; row rod/gaussian $R/gaussian -s 64
for dd in aligned-types-cuda babelstream-cuda base64e-cuda axpby-cuda; do a=$(grep "^$dd|" /tmp/hec_list.txt | cut -d'|' -f2); row hec/$dd ~/corp/hec/$dd/main $a; done
echo FINISHED
