// The smallest complete program: allocate, copy in, launch, synchronize, copy
// out, free.
#include "common.h"

int main() {
  const int n = 1024;
  const size_t bytes = n * sizeof(float);
  float *ha = new float[n], *hb = new float[n], *hc = new float[n];
  for (int i = 0; i < n; i++) { ha[i] = float(i); hb[i] = float(2 * i); }
  float *da, *db, *dc;
  CHECK(hipMalloc(&da, bytes));
  CHECK(hipMalloc(&db, bytes));
  CHECK(hipMalloc(&dc, bytes));
  CHECK(hipMemcpy(da, ha, bytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(db, hb, bytes, hipMemcpyHostToDevice));
  hipLaunchKernelGGL(vadd, dim3(n / 256), dim3(256), 0, 0, da, db, dc, n);
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(hc, dc, bytes, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; i++) wrong += hc[i] != float(3 * i);
  CHECK(hipFree(da));
  CHECK(hipFree(db));
  CHECK(hipFree(dc));
  std::printf("vecadd wrong %d of %d\n", wrong, n);
  return wrong != 0;
}
