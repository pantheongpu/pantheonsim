// A completely ordinary CUDA program. Compiled with nvcc, unaware of VirtualGPU.
// The e2e test runs it against libvgpucudart and checks the result.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void vecAdd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 4096;
  size_t bytes = n * sizeof(float);
  float *ha = (float*)malloc(bytes), *hb = (float*)malloc(bytes), *hc = (float*)malloc(bytes);
  for (int i = 0; i < n; ++i) { ha[i] = i * 0.5f; hb[i] = i * 0.25f; }
  float *da, *db, *dc;
  if (cudaMalloc(&da, bytes) || cudaMalloc(&db, bytes) || cudaMalloc(&dc, bytes)) { printf("FAIL malloc\n"); return 1; }
  cudaMemcpy(da, ha, bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(db, hb, bytes, cudaMemcpyHostToDevice);
  vecAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { printf("FAIL launch: %s\n", cudaGetErrorString(e)); return 1; }
  cudaMemcpy(hc, dc, bytes, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) {
    float want = ha[i] + hb[i];
    if (hc[i] != want) { printf("FAIL c[%d]=%g want %g\n", i, hc[i], want); return 1; }
  }
  cudaFree(da); cudaFree(db); cudaFree(dc);
  printf("PASS\n");
  return 0;
}
