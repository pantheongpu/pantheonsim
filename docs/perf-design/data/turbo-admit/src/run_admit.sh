#!/bin/bash
# run_admit.sh <list> <outdir> : per program, native (sm_86 build), simulator, turbo before (PTX variables = all), turbo after, turbo verify=all
export PATH=/usr/local/cuda-13.0/bin:$PATH
R=$HOME/par-admit
SH=$R/build/shim
list=$1; out=$2; mkdir -p $out
T=${T:-300}
while IFS='|' read -r p args; do
  [ -z "$p" ] && continue
  for form in native sim before after verify; do
    d=$out/$p.$form; mkdir -p $d; cd $d
    case $form in
      native) bin=$R/bin86/$p; env=("env");;
      sim) bin=$R/bin90/$p; env=(env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$SH);;
      before) bin=$R/bin90/$p; env=(env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$SH VGPU_HOST_GPU=turbo VGPU_HOST_GPU_PTX_VARIABLES=all);;
      after) bin=$R/bin90/$p; env=(env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$SH VGPU_HOST_GPU=turbo);;
      verify) bin=$R/bin90/$p; env=(env VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$SH VGPU_HOST_GPU=turbo VGPU_HOST_GPU_VERIFY=all);;
    esac
    [ -x $bin ] || { echo "$p $form nobin" > $d/status; continue; }
    s=$(date +%s.%N)
    timeout $T "${env[@]}" $bin $args > $d/out.txt 2> $d/err.txt; rc=$?
    e=$(date +%s.%N)
    echo "rc=$rc wall=$(echo "$e - $s" | bc)" > $d/status
  done
  echo "$p done"
done < $list
echo ALLDONE
