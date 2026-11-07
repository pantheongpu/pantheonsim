// A zero-byte allocation (pantheonsim #515). Checked on an RTX 3060 with CUDA 13.0:
//   runtime  cudaMalloc, cudaMallocHost, cudaHostAlloc, cudaMallocManaged, cudaMallocAsync, cudaMallocFromPoolAsync
//            -> cudaSuccess with a NULL pointer; cudaFree/cudaFreeHost/cudaFreeAsync of that NULL are no-ops.
//   driver   cuMemAlloc and cuMemAllocManaged -> CUDA_ERROR_INVALID_VALUE with *dptr set to 0;
//            cuMemAllocPitch(0, 0) likewise, with *pitch 0;
//            cuMemAllocHost, cuMemHostAlloc, cuMemAllocAsync, cuMemAllocFromPoolAsync -> CUDA_SUCCESS with NULL.
// Code such as Gpufit allocates the buffers it does not use with size 0 and tells them apart by the NULL.
#include <cuda.h>
#include <cuda_runtime.h>

#include "vtest.hpp"

namespace {
void* const kSentinel = reinterpret_cast<void*>(0x1);
}

VTEST(runtime_zero_byte_allocations_succeed_with_null) {
  void* p = kSentinel;
  VCHECK_EQ(cudaMalloc(&p, 0), cudaSuccess);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaFree(p), cudaSuccess);

  p = kSentinel;
  VCHECK_EQ(cudaMallocHost(&p, 0), cudaSuccess);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaFreeHost(p), cudaSuccess);

  p = kSentinel;
  VCHECK_EQ(cudaHostAlloc(&p, 0, cudaHostAllocDefault), cudaSuccess);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaFreeHost(p), cudaSuccess);

  p = kSentinel;
  VCHECK_EQ(cudaMallocManaged(&p, 0), cudaSuccess);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaFree(p), cudaSuccess);
}

VTEST(runtime_zero_byte_stream_ordered_allocations_succeed_with_null) {
  cudaStream_t s;
  VCHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
  void* p = kSentinel;
  VCHECK_EQ(cudaMallocAsync(&p, 0, s), cudaSuccess);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaFreeAsync(p, s), cudaSuccess);

  cudaMemPool_t pool;
  VCHECK_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);
  p = kSentinel;
  VCHECK_EQ(cudaMallocFromPoolAsync(&p, 0, pool, s), cudaSuccess);
  VCHECK(p == nullptr);
  VCHECK_EQ(cudaFreeAsync(p, s), cudaSuccess);
  VCHECK_EQ(cudaStreamSynchronize(s), cudaSuccess);
  VCHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
}

VTEST(runtime_nonzero_allocations_are_still_distinct_and_non_null) {
  void *a = nullptr, *b = nullptr;
  VCHECK_EQ(cudaMalloc(&a, 1), cudaSuccess);
  VCHECK_EQ(cudaMalloc(&b, 1), cudaSuccess);
  VCHECK(a != nullptr && b != nullptr && a != b);
  VCHECK_EQ(cudaFree(a), cudaSuccess);
  VCHECK_EQ(cudaFree(b), cudaSuccess);
  VCHECK_EQ(cudaFree(nullptr), cudaSuccess);
}

VTEST(driver_zero_byte_allocations_match_the_card) {
  VCHECK_EQ(cuInit(0), CUDA_SUCCESS);
  CUdevice dev;
  VCHECK_EQ(cuDeviceGet(&dev, 0), CUDA_SUCCESS);
  CUcontext ctx;
  VCHECK_EQ(cuDevicePrimaryCtxRetain(&ctx, dev), CUDA_SUCCESS);
  VCHECK_EQ(cuCtxSetCurrent(ctx), CUDA_SUCCESS);

  CUdeviceptr d = 0x1;
  VCHECK_EQ(cuMemAlloc(&d, 0), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(d, CUdeviceptr{0});
  d = 0x1;
  VCHECK_EQ(cuMemAllocManaged(&d, 0, CU_MEM_ATTACH_GLOBAL), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(d, CUdeviceptr{0});
  size_t pitch = 5;
  d = 0x1;
  VCHECK_EQ(cuMemAllocPitch(&d, &pitch, 0, 0, 4), CUDA_ERROR_INVALID_VALUE);
  VCHECK_EQ(d, CUdeviceptr{0});
  VCHECK_EQ(pitch, size_t{0});
  VCHECK_EQ(cuMemFree(0), CUDA_SUCCESS);

  void* h = kSentinel;
  VCHECK_EQ(cuMemAllocHost(&h, 0), CUDA_SUCCESS);
  VCHECK(h == nullptr);
  h = kSentinel;
  VCHECK_EQ(cuMemHostAlloc(&h, 0, 0), CUDA_SUCCESS);
  VCHECK(h == nullptr);
  VCHECK_EQ(cuMemFreeHost(h), CUDA_SUCCESS);

  d = 0x1;
  VCHECK_EQ(cuMemAllocAsync(&d, 0, nullptr), CUDA_SUCCESS);
  VCHECK_EQ(d, CUdeviceptr{0});
  VCHECK_EQ(cuMemAlloc(&d, 16), CUDA_SUCCESS);
  VCHECK(d != 0);
  VCHECK_EQ(cuMemFree(d), CUDA_SUCCESS);
  VCHECK_EQ(cuDevicePrimaryCtxRelease(dev), CUDA_SUCCESS);
}

VTEST_MAIN
