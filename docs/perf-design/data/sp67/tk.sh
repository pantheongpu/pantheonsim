#!/bin/bash
# tk.sh: compile the division probes and PolyBench programs with nvcc 13.0 and 13.2 (explicit paths), dump SASS with each toolkit's cuobjdump.
# Layout: ~/sp67-tk/src (sources), ~/sp67-tk/<ver>/<prog> (binary), <prog>.sass
set -u
T=~/sp67-tk
cd $T/src
for v in 13.0 13.2; do
  NV=/usr/local/cuda-$v/bin/nvcc; OD=/usr/local/cuda-$v/bin/cuobjdump
  mkdir -p $T/$v
  nice -n 19 $NV -std=c++20 -O2 -arch=sm_86 -I include tools/probes/hostgpu_sfu_exhaustive.cu -o $T/$v/sfu_exhaustive 2> $T/$v/sfu.err
  nice -n 19 $NV -std=c++17 -arch=sm_86 -cudart shared -w -Wno-deprecated-gpu-targets nvidia/tests/e2e/host_gpu_lazy.cu nvidia/tests/e2e/host_gpu_lazy_heap.cu -o $T/$v/host_gpu_lazy 2> $T/$v/lazy.err
  for p in ADI LU GRAMSCHM CORR COVAR 2DCONV; do
    ( cd polybench/CUDA/$p && nice -n 19 $NV -O3 -arch=sm_86 -cudart shared -w -DcudaThreadSynchronize=cudaDeviceSynchronize *.cu -o $T/$v/$p ) 2> $T/$v/$p.err
  done
  for b in $T/$v/*; do [ -x "$b" ] && [ -f "$b" ] && $OD -sass "$b" > "$b.sass" 2>&1; done
done
ls -la $T/13.0 $T/13.2
