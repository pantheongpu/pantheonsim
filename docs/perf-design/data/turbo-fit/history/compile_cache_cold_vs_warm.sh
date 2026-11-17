cd $WORK/pb
B=$WORK/build
E="VGPU_GPU=nvidia/h100 VGPU_THREADS=4 LD_LIBRARY_PATH=$B/shim VGPU_QUIET=1 VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1"
t() { local s=$(date +%s.%N); env $E "$@" ./GEMM_ptx90 > /tmp/c2.out 2>&1; local e=$(date +%s.%N); printf "wall %.3f  %s\n" $(echo "$e - $s" | bc) "$(grep -E '^host_gpu_turbo' /tmp/c2.out | sed -E 's/.*launches_card=([0-9]+).*card_s=([0-9.]+).*/cards=\1 card_s=\2/')"; }
echo "after drop_caches, CACHE_DISABLE=1:"; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; t CUDA_CACHE_DISABLE=1
echo "again, warm, CACHE_DISABLE=1:"; t CUDA_CACHE_DISABLE=1
echo "after drop_caches, fresh CACHE_PATH:"; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; rm -rf /tmp/cc_x; t CUDA_CACHE_PATH=/tmp/cc_x
echo "second (cache hit, warm):"; t CUDA_CACHE_PATH=/tmp/cc_x
echo "after drop_caches, cache hit (cache files cold):"; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; t CUDA_CACHE_PATH=/tmp/cc_x
echo "after drop_caches, native-style non-turbo sim run (no card):"; sync; echo 3 | sudo tee /proc/sys/vm/drop_caches >/dev/null; s=$(date +%s.%N); env VGPU_GPU=nvidia/h100 VGPU_THREADS=4 LD_LIBRARY_PATH=$B/shim VGPU_QUIET=1 ./GEMM_ptx90 >/dev/null 2>&1; e=$(date +%s.%N); echo "wall $(echo "$e - $s" | bc)"
