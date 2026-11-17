cd $WORK/corp/sp22-corp/rodinia/bin
B=$WORK/build
t() { local s=$(date +%s.%N); "$@" >/dev/null 2>&1; local e=$(date +%s.%N); printf "%.3f " $(echo "$e - $s" | bc); }
E="VGPU_GPU=nvidia/a10g VGPU_THREADS=4 LD_LIBRARY_PATH=$B/shim VGPU_QUIET=1 VGPU_HOST_GPU=turbo"
for mc in unset 1 2 4 8 32; do
  echo -n "bfs turbo, CUDA_DEVICE_MAX_CONNECTIONS=$mc: "
  for i in 1 2 3 4 5 6; do if [ $mc = unset ]; then t env $E ./bfs data/graph4096.txt; else t env $E CUDA_DEVICE_MAX_CONNECTIONS=$mc ./bfs data/graph4096.txt; fi; done; echo
done
cd ../../polybench/bin
for mc in unset 1; do
  echo -n "FDTD-2D turbo (4.2 s), MAX_CONNECTIONS=$mc: "
  for i in 1 2; do if [ $mc = unset ]; then t env $E ./FDTD-2D; else t env $E CUDA_DEVICE_MAX_CONNECTIONS=$mc ./FDTD-2D; fi; done; echo
done
