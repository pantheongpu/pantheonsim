// A CUDA program that checks itself: the example docs/docker.md builds in an
// image that has nvcc and runs on the simulator.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 16;
  float *a, *b, *c;
  cudaMallocManaged(&a, n * sizeof(float));
  cudaMallocManaged(&b, n * sizeof(float));
  cudaMallocManaged(&c, n * sizeof(float));
  for (int i = 0; i < n; ++i) { a[i] = float(i); b[i] = 2.0f * i; }
  add<<<(n + 255) / 256, 256>>>(a, b, c, n);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("FAIL: launch"); return 1; }
  int bad = 0;
  for (int i = 0; i < n; ++i) bad += c[i] != 3.0f * i;
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  std::printf("%s: %s, %d of %d wrong\n", bad ? "FAIL" : "PASS", p.name, bad, n);
  return bad != 0;
}
