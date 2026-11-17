// Turbo mode's fit (docs/perf-design/12-host-gpu-turbo.md "Turbo fit", nvidia/tests/e2e/run_host_gpu_stub.sh): launches whose fate depends on the card the
// stand-in driver reports, not on the simulated profile: a launch too small for the performance rules, a block of 1024 threads, 64 KiB of dynamic shared
// memory, a long grid, an allocation, and (built for sm_90) a kernel of one instruction that no turbo class names (QMMA, sm_89) and one that is class never (ELECT, sm_90).
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

// 40 KiB of static shared memory beside the dynamic kind: a launch whose TOTAL is what a profile's per-block limit is judged on (the runtime API checks the dynamic part alone).
__global__ void shared_both(int* p) {
  __shared__ int tile[10240];
  tile[threadIdx.x] = static_cast<int>(threadIdx.x) * 3;
  dyn[threadIdx.x] = tile[threadIdx.x] + 1;
  __syncthreads();
  p[threadIdx.x] = dyn[(threadIdx.x + 1) % blockDim.x];
}

// A kernel compiled for at most 128 threads a block, launched with 256: the simulator refuses it (launch bounds are checked by its launch validation, not by the runtime API).
__global__ void __launch_bounds__(128) bounded(int* p) { p[threadIdx.x] = static_cast<int>(threadIdx.x) + 1; }

// One instruction nothing in include/vgpu/host_gpu.hpp names: elect.sync is ELECT on sm_90 (the stand-in is told its card is an sm_90 one).
#ifndef FIT_NO_ELECT
__global__ void elect(int* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  unsigned is = 0;
  asm volatile("{ .reg .pred %%p; elect.sync _|%%p, 0xffffffff; selp.u32 %0, 1, 0, %%p; }" : "=r"(is));
  p[threadIdx.x] = static_cast<int>(is);
#else
  p[threadIdx.x] = 1;
#endif
}


// An FP8 tensor-core multiply (QMMA on sm_89): an instruction no turbo class names, which runs on an sm_89 card.
__global__ void qmma(int* p) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
  unsigned a0 = threadIdx.x, a1 = 1, a2 = 2, a3 = 3, b0 = 4, b1 = 5;
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  p[threadIdx.x] = static_cast<int>(d0 + d1 + d2 + d3);
#else
  p[threadIdx.x] = 1;
#endif
}

#endif

// A float atomic: class atomic-float (turbo keeps it out with VGPU_HOST_GPU_ATOMICS=0), also in PTX the card compiles.
__global__ void fatom(float* p) { atomicAdd(p, 1.0f); }

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
    if (!std::strcmp(mode, "tiny") && std::getenv("FIT_ATOM")) fatom<<<1, 32>>>(reinterpret_cast<float*>(d) + 4096);   // (a launch in the module with the atomic)
    for (int i = 0; i < 3; ++i) tiny<<<1, 32>>>(d, 32);
  } else if (!std::strcmp(mode, "wide")) {
    wide<<<2, 1024>>>(d, 2048);
  } else if (!std::strcmp(mode, "shared")) {
    const char* kb = std::getenv("FIT_SHARED_KB");   // (the profile's limit is the simulator's to enforce: tests ask for more or less than a profile allows)
    const int bytes = (kb ? std::atoi(kb) : 64) * 1024;
    const cudaError_t sa = cudaFuncSetAttribute(shared_user, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
    shared_user<<<1, 256, bytes>>>(d);
    std::printf("set_attribute=%d launch=%d\n", static_cast<int>(sa), static_cast<int>(cudaGetLastError()));
  } else if (!std::strcmp(mode, "bounds")) {
    const int t = std::getenv("FIT_THREADS") ? std::atoi(std::getenv("FIT_THREADS")) : 256;
    bounded<<<1, t>>>(d);
    std::printf("launch=%d\n", static_cast<int>(cudaGetLastError()));
  } else if (!std::strcmp(mode, "shared2")) {
    const char* kb = std::getenv("FIT_SHARED_KB");
    const int bytes = (kb ? std::atoi(kb) : 30) * 1024;
    shared_both<<<1, 256, bytes>>>(d);
    std::printf("launch=%d\n", static_cast<int>(cudaGetLastError()));
  } else if (!std::strcmp(mode, "grid")) {
    tiny<<<4096, 4>>>(d, n);
  } else if (!std::strcmp(mode, "hostmap")) {   // the kernel writes pinned host memory the program mapped (zero-copy)
    int *h = nullptr, *dh = nullptr;
    if (cudaHostAlloc(&h, 32 * sizeof(int), cudaHostAllocMapped) != cudaSuccess || cudaHostGetDevicePointer(&dh, h, 0) != cudaSuccess) return 2;
    std::memset(h, 0, 32 * sizeof(int));
    cudaFree(d);   // (no device memory at all, as in a zero-copy program)
    tiny<<<1, 32>>>(dh, 32);
    cudaDeviceSynchronize();
    std::printf("hostsum=%lld\n", sum(h, 32));
    cudaFreeHost(h);
    return 0;
  } else if (!std::strcmp(mode, "qmma")) {
#ifndef FIT_NO_ELECT
    qmma<<<1, 32>>>(d);
#endif
  } else if (!std::strcmp(mode, "elect")) {
#ifndef FIT_NO_ELECT
    elect<<<1, 32>>>(d);
#endif
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
