// Kernels with a known dispatch geometry: a 3D grid, static plus dynamic LDS,
// and a private (scratch) segment from a local array indexed at run time.
#include "common.h"

__global__ void lds_kernel(int* out) {
  __shared__ int tile[512];          // 2048 bytes of static LDS
  extern __shared__ int dyn[];       // the launch's dynamic LDS
  int t = threadIdx.x;
  tile[t] = t;
  dyn[t] = 2 * t;
  __syncthreads();
  if (t < 64) out[blockIdx.x * 64 + t] = tile[(t + 1) % 64] + dyn[(t + 2) % 64];
}

__global__ void scratch_kernel(int* out, int k) {
  int a[256];                         // indexed at run time: lives in scratch
  for (int i = 0; i < 256; i++) a[i] = i * k;
  int s = 0;
  for (int i = 0; i < 64; i++) s += a[(threadIdx.x * 7 + i * k) & 255];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

__global__ void cube_kernel(int* out) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;
  out[(z * 8 + y) * 16 + x] = 1;
}

int main() {
  int* d;
  CHECK(hipMalloc(&d, 4096 * sizeof(int)));
  CHECK(hipMemset(d, 0, 4096 * sizeof(int)));
  hipLaunchKernelGGL(lds_kernel, dim3(3), dim3(128), 128 * sizeof(int), 0, d);
  hipLaunchKernelGGL(scratch_kernel, dim3(2), dim3(64), 0, 0, d, 3);
  hipLaunchKernelGGL(cube_kernel, dim3(2, 2, 2), dim3(8, 4, 2), 0, 0, d);
  CHECK(hipDeviceSynchronize());
  int* h = new int[4096];
  CHECK(hipMemcpy(h, d, 4096 * sizeof(int), hipMemcpyDeviceToHost));
  int ones = 0;
  for (int i = 0; i < 4 * 16 * 8; i++) ones += h[i] == 1;
  CHECK(hipFree(d));
  std::printf("geometry ones %d (want 512)\n", ones);
  return ones != 512;
}
