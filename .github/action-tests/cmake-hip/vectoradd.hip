#include <hip/hip_runtime.h>
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
  if (hipMalloc(&da, sizeof a) || hipMalloc(&db, sizeof b) || hipMalloc(&dc, sizeof c)) { std::puts("hipMalloc failed"); return 2; }
  hipMemcpy(da, a, sizeof a, hipMemcpyHostToDevice);
  hipMemcpy(db, b, sizeof b, hipMemcpyHostToDevice);
  add<<<(n + 255) / 256, 256>>>(dc, da, db, n);
  if (hipDeviceSynchronize() != hipSuccess) { std::puts("kernel failed"); return 2; }
  hipMemcpy(c, dc, sizeof c, hipMemcpyDeviceToHost);
  for (int i = 0; i < n; ++i) if (c[i] != 3 * i) { std::printf("wrong at %d\n", i); return 1; }
  std::printf("ok: %d elements\n", n);
  return 0;
}
