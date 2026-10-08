// A block that asks for the most dynamic shared memory the part allows, as
// CUTLASS's Hopper kernels do: the per-block opt-in maximum less the kernel's
// static shared memory. It must launch, fill and read back all of it, and
// report the same static size and occupancy whichever of its codes runs --
// from sm_90 a cubin keeps the driver's reserved shared memory in the
// kernel's own section, and counting it as the kernel's static memory as
// well left no room for such a block.
#include <cstdio>
#include <cuda_runtime.h>

__shared__ int fixed[64];

__global__ void fill(unsigned* out, unsigned words) {
  extern __shared__ unsigned dyn[];
  for (unsigned i = threadIdx.x; i < words; i += blockDim.x) dyn[i] = i * 2654435761u;
  if (threadIdx.x < 64) fixed[threadIdx.x] = static_cast<int>(threadIdx.x);
  __syncthreads();
  unsigned sum = 0;
  for (unsigned i = threadIdx.x; i < words; i += blockDim.x) sum += dyn[words - 1 - i] ^ i;
  atomicAdd(out, sum + static_cast<unsigned>(fixed[threadIdx.x % 64]));
}

int main() {
  int dev = 0, optin = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  cudaFuncAttributes attr{};
  if (cudaFuncGetAttributes(&attr, fill) != cudaSuccess) { std::printf("FAIL: cudaFuncGetAttributes\n"); return 1; }
  const int dynamic = optin - static_cast<int>(attr.sharedSizeBytes);
  if (cudaFuncSetAttribute(fill, cudaFuncAttributeMaxDynamicSharedMemorySize, dynamic) != cudaSuccess) {
    std::printf("FAIL: cudaFuncSetAttribute(%d)\n", dynamic);
    return 1;
  }
  int blocks = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fill, 256, dynamic);
  std::printf("static shared %zu, dynamic %d of %d, %d block(s) per multiprocessor\n", attr.sharedSizeBytes,
              dynamic, optin, blocks);

  unsigned* out = nullptr;
  cudaMalloc(&out, sizeof(unsigned));
  cudaMemset(out, 0, sizeof(unsigned));
  const unsigned words = static_cast<unsigned>(dynamic) / 4;
  fill<<<2, 256, dynamic>>>(out, words);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: launch: %s\n", cudaGetErrorString(e)); return 1; }
  unsigned got = 0;
  cudaMemcpy(&got, out, sizeof got, cudaMemcpyDeviceToHost);
  unsigned want = 0;
  for (unsigned i = 0; i < words; ++i) want += ((words - 1 - i) * 2654435761u) ^ i;
  unsigned fixed_sum = 0;
  for (unsigned t = 0; t < 256; ++t) fixed_sum += t % 64;
  want = 2 * (want + fixed_sum);
  std::printf("%s: sum %u\n", got == want ? "ok" : "FAIL", got);
  cudaFree(out);
  return got == want ? 0 : 1;
}
