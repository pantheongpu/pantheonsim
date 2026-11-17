// Turbo mode's fit (docs/perf-design/12-host-gpu-turbo.md "Turbo fit", nvidia/tests/e2e/run_host_gpu_stub.sh): launches whose fate depends on the card the
// stand-in driver reports, not on the simulated profile: a launch too small for the performance rules, a block of 1024 threads, 64 KiB of dynamic shared
// memory, a long grid, an allocation, and (built for sm_90) a kernel of one instruction that no turbo class names (ELECT).
// Every mode prints a checksum of what the kernels wrote, which is the program's whole output: the stand-in card computes nothing, so a launch that went to
// it either left the simulator's result (a verify run) or never reached it (the card refused the launch or the module).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

__global__ void tiny(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i * 3 + 1;
}

__global__ void __launch_bounds__(1024) wide(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = i * 5 + 2;
}

extern __shared__ int dyn[];
__global__ void shared_user(int* p) {
  dyn[threadIdx.x] = static_cast<int>(threadIdx.x) * 7;
  __syncthreads();
  p[threadIdx.x] = dyn[(threadIdx.x + 1) % blockDim.x];
}

// One instruction nothing in include/vgpu/host_gpu.hpp names: elect.sync is ELECT on sm_90 (the stand-in is told its card is an sm_90 one).
__global__ void elect(int* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  unsigned is = 0;
  asm volatile("{ .reg .pred %%p; elect.sync _|%%p, 0xffffffff; selp.u32 %0, 1, 0, %%p; }" : "=r"(is));
  p[threadIdx.x] = static_cast<int>(is);
#else
  p[threadIdx.x] = 1;
#endif
}

static long long sum(const int* a, int n) {
  long long s = 0;
  for (int i = 0; i < n; ++i) s += a[i];
  return s;
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "tiny";
  const int n = 1 << 14;
  int* d = nullptr;
  if (cudaMalloc(&d, (std::strcmp(mode, "mem") == 0 ? 8u << 20 : n * sizeof(int))) != cudaSuccess) return 2;
  cudaMemset(d, 0, n * sizeof(int));
  if (!std::strcmp(mode, "tiny") || !std::strcmp(mode, "mem")) {
    for (int i = 0; i < 3; ++i) tiny<<<1, 32>>>(d, 32);
  } else if (!std::strcmp(mode, "wide")) {
    wide<<<2, 1024>>>(d, 2048);
  } else if (!std::strcmp(mode, "shared")) {
    const int bytes = 64 * 1024;
    cudaFuncSetAttribute(shared_user, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    shared_user<<<1, 256, bytes>>>(d);
  } else if (!std::strcmp(mode, "grid")) {
    tiny<<<4096, 4>>>(d, n);
  } else if (!std::strcmp(mode, "elect")) {
    elect<<<1, 32>>>(d);
  }
  if (cudaDeviceSynchronize() != cudaSuccess) {
    std::fprintf(stderr, "launch failed: %s\n", cudaGetErrorString(cudaGetLastError()));
    return 3;
  }
  int* h = static_cast<int*>(std::malloc(n * sizeof(int)));
  cudaMemcpy(h, d, n * sizeof(int), cudaMemcpyDeviceToHost);
  std::printf("mode=%s sum=%lld\n", mode, sum(h, n));
  std::free(h);
  cudaFree(d);
  return 0;
}
