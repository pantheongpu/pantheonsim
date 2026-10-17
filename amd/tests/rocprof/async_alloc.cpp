// Stream-ordered allocation: hipMallocAsync, a copy and a kernel that use the
// block, and hipFreeAsync.
#include "common.h"

int main() {
  const int n = 1024;
  const size_t bytes = n * sizeof(float);
  float *ha = new float[n], *hc = new float[n];
  for (int i = 0; i < n; i++) ha[i] = float(i);
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  float *da, *dc;
  CHECK(hipMallocAsync(reinterpret_cast<void**>(&da), bytes, s));
  CHECK(hipMallocAsync(reinterpret_cast<void**>(&dc), bytes, s));
  CHECK(hipMemcpyAsync(da, ha, bytes, hipMemcpyHostToDevice, s));
  hipLaunchKernelGGL(vadd, dim3(n / 256), dim3(256), 0, s, da, da, dc, n);
  CHECK(hipMemcpyAsync(hc, dc, bytes, hipMemcpyDeviceToHost, s));
  CHECK(hipFreeAsync(da, s));
  CHECK(hipFreeAsync(dc, s));
  CHECK(hipStreamSynchronize(s));
  int wrong = 0;
  for (int i = 0; i < n; i++) wrong += hc[i] != float(2 * i);
  CHECK(hipStreamDestroy(s));
  std::printf("async_alloc wrong %d of %d\n", wrong, n);
  return wrong != 0;
}
