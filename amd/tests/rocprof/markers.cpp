// ROCTx markers around HIP calls: a nested pair of ranges, a mark, a range
// started and stopped by id, and names given to the thread and the device.
#include <rocprofiler-sdk-roctx/roctx.h>

#include "common.h"

int main() {
  const int n = 256;
  const size_t bytes = n * sizeof(float);
  float *h = new float[n];
  for (int i = 0; i < n; i++) h[i] = float(i);
  float *d, *o;
  roctxNameOsThread("main");
  roctxNameHipDevice("the card", 0);
  roctxMarkA("start");
  roctxRangePushA("setup");
  CHECK(hipMalloc(&d, bytes));
  CHECK(hipMalloc(&o, bytes));
  CHECK(hipMemcpy(d, h, bytes, hipMemcpyHostToDevice));
  roctxRangePop();
  roctx_range_id_t id = roctxRangeStartA("compute");
  roctxRangePushA("launch");
  hipLaunchKernelGGL(vadd, dim3(n / 64), dim3(64), 0, 0, d, d, o, n);
  roctxRangePop();
  CHECK(hipDeviceSynchronize());
  roctxRangeStop(id);
  CHECK(hipMemcpy(h, o, bytes, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (int i = 0; i < n; i++) wrong += h[i] != float(2 * i);
  CHECK(hipFree(d));
  CHECK(hipFree(o));
  roctxMarkA("end");
  std::printf("markers wrong %d of %d\n", wrong, n);
  return wrong != 0;
}
