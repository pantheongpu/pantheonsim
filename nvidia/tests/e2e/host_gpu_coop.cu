// A cooperative launch with a grid-wide barrier, for turbo mode's cooperative launches (#661, nvidia/tests/e2e/run_host_gpu_stub.sh): the kernel is the
// one cg::this_grid().sync() makes (an atomic counter, a spin, and a trap for a launch that is not cooperative). The stand-in card computes nothing, so a
// verify run is the one whose output is the simulator's; FIT_COOP_BLOCKS sets the grid (4 blocks of 32 threads by default).
#include <cooperative_groups.h>
#include <cstdio>
#include <cstdlib>
namespace cg = cooperative_groups;

__global__ void coop(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i + 1;
  cg::this_grid().sync();
  if (i < n) p[i] += p[(i + 1) % n];
}

int main() {
  const int blocks = std::getenv("FIT_COOP_BLOCKS") ? std::atoi(std::getenv("FIT_COOP_BLOCKS")) : 4;
  int np = blocks * 32;
  int* d = nullptr;
  if (cudaMalloc(&d, np * sizeof(int)) != cudaSuccess) return 2;
  cudaMemset(d, 0, np * sizeof(int));
  void* args[] = {&d, &np};
  std::printf("launch=%d\n", static_cast<int>(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(coop), dim3(blocks), dim3(32), args, 0, 0)));
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  int* h = static_cast<int*>(std::malloc(np * sizeof(int)));
  cudaMemcpy(h, d, np * sizeof(int), cudaMemcpyDeviceToHost);
  long long sum = 0;
  for (int i = 0; i < np; ++i) sum += h[i];
  std::printf("sum=%lld\n", sum);
  return 0;
}
