#!/usr/bin/env bash
# #660, the part only a card can decide: what does one very short launch cost the card path, and so what is a launch of 20-120 us in the simulator
# worth? Run on the rig (one A10G, otherwise idle), from the root of the tree, corpora in ~/corp, build as card-run.sh (SP76_BUILD).
# Not run when written (no card). About 10 minutes.
#   docs/perf-design/data/turbo-left/card-short-launch.sh [out dir]
# Why a card measurement: include/vgpu/host_gpu.hpp prices a launch with Costs{launch_s=7 us, kernel_s_default=100 us, min_sim_s=200 us, margin=2}. A kernel
# whose launch the simulator takes under 200 us (JACOBI1D: 20,000 launches in 2.3 s, 115 us each; sc_gpu 0.6 ms; LargeKernelParameter 0.2-0.3 ms) is never sent,
# so the card never runs it and its card time stays the 100 us prior. If the card time of such a kernel is 5 to 10 us, that prior is the whole reason. Nothing
# on the simulator side shows it: the prior is a constant, not a bug.
# Four readings, in out/summary.txt:
#   1. native: each program's own wall time on the card (no simulator) divided by its launch count: an upper bound of the card's cost per launch with the
#      program's own host work. JACOBI1D 20,000 launches, sc_gpu 1,432, LargeKernelParameter 2,002.
#   2. micro: an empty kernel launched n times, (a) asynchronously with one sync at the end, (b) with a sync after each (what a host that waits for each
#      launch pays), (c) a 4-byte device-to-host copy after each; and 1 MiB host<->device copies (the rates Costs assumes: push 12 GB/s, pull 10 GB/s).
#   3. nsys: the kernels' own durations in the native JACOBI1D (`nsys` needed; skipped if absent): the card kernel time to put in place of the 100 us prior.
#   4. the mode today: the three programs in turbo with VGPU_HOST_GPU_LOG=1 (every launch kept: "the estimate does not beat the card path") and the summary line.
# Decision rule once the numbers are in: let k be the nsys kernel time and l the (b) latency. A launch can pay only if sim_s > 2 * (l + k + moves); for JACOBI1D
# sim_s = 115 us, so it needs l + k under 57 us with the arrays kept on the card (streak rule, sp65). If (b) is over 50 us, #660 is closed as "no gain"
# (the issue allows that outcome); if (b) is 10 to 30 us, the lever is a measured kernel_s prior (from the first launch the mode sends as a probe) and the 200 us floor
# lowered to about 4 * l; either is a rule change that needs the census (data/sp75/census.py) and a card A/B on every program with launches under 200 us.
set -u
out=${1:-$HOME/turboleft-out}
build=${SP76_BUILD:-$HOME/sp76-build}
corp=${SP76_CORP:-$HOME/corp}
mkdir -p "$out"
export PATH=/usr/local/cuda-13.0/bin:$PATH
cat > "$out/micro.cu" <<'CU'
#include <cstdio>
#include <chrono>
#include <cuda_runtime.h>
__global__ void k(int* p) { if (threadIdx.x == 0 && blockIdx.x == 0 && p) p[0] += 1; }
static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }
int main() {
  int* d; cudaMalloc(&d, 1 << 20); void* h; cudaMallocHost(&h, 1 << 20); cudaMemset(d, 0, 1 << 20);
  for (int i = 0; i < 100; ++i) k<<<32, 128>>>(d); cudaDeviceSynchronize();
  const int n = 20000;
  double t = now(); for (int i = 0; i < n; ++i) k<<<32, 128>>>(d); cudaDeviceSynchronize(); double a = (now() - t) / n;
  t = now(); for (int i = 0; i < n; ++i) { k<<<32, 128>>>(d); cudaDeviceSynchronize(); } double b = (now() - t) / n;
  t = now(); for (int i = 0; i < n; ++i) { k<<<32, 128>>>(d); cudaMemcpy(h, d, 4, cudaMemcpyDeviceToHost); } double c = (now() - t) / n;
  t = now(); for (int i = 0; i < 200; ++i) cudaMemcpy(d, h, 1 << 20, cudaMemcpyHostToDevice); double push = (now() - t) / 200;
  t = now(); for (int i = 0; i < 200; ++i) cudaMemcpy(h, d, 1 << 20, cudaMemcpyDeviceToHost); double pull = (now() - t) / 200;
  printf("micro: async launch %.2f us, launch+sync %.2f us, launch+4B copy %.2f us, 1 MiB push %.1f us (%.1f GB/s), pull %.1f us (%.1f GB/s)\n",
         a * 1e6, b * 1e6, c * 1e6, push * 1e6, 1.048576e-3 / push, pull * 1e6, 1.048576e-3 / pull);
}
CU
nvcc -O2 -arch=sm_86 -w -o "$out/micro" "$out/micro.cu" && "$out/micro" | tee "$out/summary.txt"
# the programs: native (no simulator), then the mode
py='import sys; sys.path.insert(0, "tools/bench/real"); import real
for c, n, cwd, argv, cp in real.manifest(sys.argv[1], ["polybench", "rodinia", "samples"]):
    if n in ("JACOBI1D", "sc_gpu", "LargeKernelParameter"): print(n, cwd, " ".join(argv))'
python3 -I -c "$py" "$corp" > "$out/programs.txt"
while read -r name cwd argv; do
  best=999999
  for i in 1 2 3 4 5; do
    s=$(date +%s.%N); ( cd "$cwd" && timeout 300 $argv > /dev/null 2>&1 ); e=$(date +%s.%N)
    best=$(python3 -I -c "print(min($best, $e - $s))")
  done
  echo "native $name: best of 5 $best s" | tee -a "$out/summary.txt"
  ( cd "$cwd" && env VGPU_GPU=nvidia/a10g LD_LIBRARY_PATH="$build/shim" VGPU_HOST_GPU=turbo VGPU_HOST_GPU_LOG=1 timeout -s KILL 300 $argv > /dev/null 2> "$out/log-$name.err" )
  grep -E '^host_gpu(_[a-z]+)?: ' "$out/log-$name.err" | grep -E 'launches ran|host_gpu_turbo|host_gpu_refusals|too_small' | tail -3 | cut -c1-300 | tee -a "$out/summary.txt"
  grep -c 'stays in the simulator' "$out/log-$name.err" | sed "s/^/$name launches kept by the size rule: /" | tee -a "$out/summary.txt"
  grep 'stays in the simulator' "$out/log-$name.err" | head -2 | cut -c1-300 | tee -a "$out/summary.txt"
done < "$out/programs.txt"
if command -v nsys > /dev/null; then
  read -r name cwd argv < <(grep '^JACOBI1D' "$out/programs.txt")
  ( cd "$cwd" && nsys profile -o "$out/jacobi1d" --force-overwrite true $argv > /dev/null 2>&1 && nsys stats --report cuda_gpu_kern_sum --format csv "$out/jacobi1d.nsys-rep" 2> /dev/null | head -6 | tee -a "$out/summary.txt" )
else
  echo "nsys not found: skip reading 3 (use ncu --metrics gpu__time_duration.sum on one launch instead)" | tee -a "$out/summary.txt"
fi
echo "out: $out/summary.txt"
