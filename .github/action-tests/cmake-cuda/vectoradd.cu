#include <cstdio>
__global__ void add(float* c, const float* a, const float* b, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
int main() {
  const int n = 4096;
  static float a[n], b[n], c[n];
  for (int i = 0; i < n; ++i) { a[i] = i; b[i] = 2 * i; }
  float *da, *db, *dc;
  if (cudaMalloc(&da, sizeof a) || cudaMalloc(&db, sizeof b) || cudaMalloc(&dc, sizeof c)) { std::puts("cudaMalloc failed"); return 2; }
  cudaMemcpy(da, a, sizeof a, cudaMemcpyHostToDevice);
  cudaMemcpy(db, b, sizeof b, cudaMemcpyHostToDevice);
  add<<<(n + 255) / 256, 256>>>(dc, da, db, n);
  if (cudaDeviceSynchronize() != cudaSuccess) { std::puts("kernel failed"); return 2; }
  cudaMemcpy(c, dc, sizeof c, cudaMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) if (c[i] != 3 * i) { std::printf("wrong at %d\n", i); return 1; }
  std::printf("ok: %d elements\n", n);
  return 0;
}
