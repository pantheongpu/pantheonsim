cd $WORK/pb
B=$WORK/build
E="VGPU_GPU=nvidia/h100 VGPU_THREADS=4 LD_LIBRARY_PATH=$B/shim VGPU_QUIET=1 VGPU_HOST_GPU=turbo"
one() { # label prog [env...]
  local lab=$1 prog=$2; shift 2
  local s=$(date +%s.%N)
  local out=$(env $E "$@" ./${prog}_ptx90 2>&1 | grep -E "^host_gpu_turbo" | sed -E 's/.*launches_card=([0-9]+).*card_s=([0-9.]+).*/cards=\1 card_s=\2/')
  local e=$(date +%s.%N)
  printf "%-8s %-28s wall %.3f %s\n" $prog "$lab" $(echo "$e - $s" | bc) "$out"
}
for prog in GEMM ATAX; do
  echo "== $prog"
  rm -rf $HOME/.nv/ComputeCache
  for i in 1 2 3; do one "default cache, run $i" $prog; done
  du -sh $HOME/.nv/ComputeCache | cut -f1 | sed 's/^/default cache size after: /'
  for rep in 1 2; do
    d=/tmp/cc_$$_$rep; rm -rf $d
    for i in 1 2; do one "CACHE_PATH fresh dir $rep, run $i" $prog CUDA_CACHE_PATH=$d; done
    du -sh $d | cut -f1 | sed 's/^/  that dir: /'
  done
  for i in 1 2; do one "CUDA_CACHE_DISABLE=1, run $i" $prog CUDA_CACHE_DISABLE=1; done
  one "CUDA_CACHE_MAXSIZE=4096 (tiny), run 1" $prog CUDA_CACHE_PATH=/tmp/cc_tiny_$prog CUDA_CACHE_MAXSIZE=4096
  one "CUDA_CACHE_MAXSIZE=4096 (tiny), run 2" $prog CUDA_CACHE_PATH=/tmp/cc_tiny_$prog CUDA_CACHE_MAXSIZE=4096
done
