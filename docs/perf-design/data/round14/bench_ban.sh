#!/bin/bash
# Round 14: persistent host-GPU mode before (SHIM_BASE) and after (SHIM_BAN) the per-kernel traffic ban, with the mode off and write-through as controls
# (both run on SHIM_BASE: the ban only acts in the persistent mode). Wall seconds; output compared with the mode off (lines with a time dropped).
# usage: bench_ban.sh <rows...>   rows: LU GRAMSCHM FDTD-2D JACOBI2D babelstream-cuda base64e-cuda; REP=n repeats (default 1); CTRL=0 skips off/write-through
export VGPU_GPU=nvidia/a10g VGPU_THREADS=4
SHIM_BASE=${SHIM_BASE:-$HOME/shim-base}; SHIM_BAN=${SHIM_BAN:-$HOME/sim/build/shim}
d=$(mktemp -d); cd $d
now() { date +%s.%N; }
f='time\|^[0-9.]*$\|^GPU\|elapsed\|Elapsed\|seconds\|[0-9] ms\|[0-9] us\|GB/s\|MB/s\|Throughput\|throughput'
one() { local shim=$1 out=$2 err=$3; shift 3
  a=$(now); env LD_LIBRARY_PATH=$shim "$@" > $out 2> $err; b=$(now); T=$(echo "$b - $a" | bc); }
row() { local n=$1; shift
  [ "${CTRL:-1}" = 1 ] && {
    one $SHIM_BASE $d/o0.txt $d/e0.txt VGPU_QUIET=1 "$@"; t0=$T
    one $SHIM_BASE $d/o1.txt $d/e1.txt VGPU_HOST_GPU=exact VGPU_HOST_GPU_WRITE_THROUGH=1 "$@"; t1=$T; }
  for i in $(seq ${REP:-1}); do
    one $SHIM_BASE $d/o2.txt $d/e2.txt VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 "$@"; t2=$T
    one $SHIM_BAN $d/o3.txt $d/e3.txt VGPU_HOST_GPU=exact VGPU_HOST_GPU_LOG=1 "$@"; t3=$T
    s2=$(grep "^host_gpu: [0-9]* launches" $d/e2.txt | sed 's/host_gpu: //; s/ (.*//'); s3=$(grep "^host_gpu: [0-9]* launches" $d/e3.txt | sed 's/host_gpu: //; s/ (.*//')
    m2=$(grep "MiB went" $d/e2.txt | sed 's/host_gpu: //'); m3=$(grep "MiB went" $d/e3.txt | sed 's/host_gpu: //')
    k3=$(grep -c "(traffic)" $d/e3.txt)
    ref=$d/o0.txt; [ -s $d/o0.txt ] || ref=$d/o2.txt
    sa=$(diff <(grep -v -i "$f" $ref) <(grep -v -i "$f" $d/o2.txt) >/dev/null && echo same || echo DIFF)
    sb=$(diff <(grep -v -i "$f" $ref) <(grep -v -i "$f" $d/o3.txt) >/dev/null && echo same || echo DIFF)
    printf "%-22s off %7.2f | wt %7.2f | base %7.2f (%s; %s) %s | ban %7.2f (%s; %s) %s traffic-bans=%s\n" $n ${t0:-0} ${t1:-0} $t2 "$s2" "$m2" $sa $t3 "$s3" "$m3" $sb $k3
    nn=${n//\//_}; cp $d/e3.txt $OUTDIR/$nn.ban.$i.log 2>/dev/null; cp $d/e2.txt $OUTDIR/$nn.base.$i.log 2>/dev/null
    CTRL=0; done; CTRL=1; }
OUTDIR=${OUTDIR:-$HOME/out}; mkdir -p $OUTDIR
P=~/corp/poly
for n in "$@"; do case $n in
  *-cuda) row hec/$n ~/corp/hec/$n/main ;;
  *) row poly/$n $P/$n ;; esac; done
echo FINISHED
