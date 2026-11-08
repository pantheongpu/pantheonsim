// The second translation unit of the program run_host_gpu_stub.sh builds (host_gpu_lazy.cu): one kernel that takes a block of the device heap and never gives it
// back, so that device memory outside the allocation table exists while the other module's kernels launch. It is a module of its own because a kernel that
// calls malloc keeps every kernel of its module in the simulator (and this one is meant to).
#include <cstdio>

__global__ void leak(unsigned char** out) {
  unsigned char* p = static_cast<unsigned char*>(malloc(64));
  if (p) p[0] = 7;
  *out = p;
}

void leak_heap() {
  unsigned char** slot = nullptr;
  cudaMalloc(&slot, sizeof *slot);
  leak<<<1, 1>>>(slot);
  cudaDeviceSynchronize();
  cudaFree(slot);
}
