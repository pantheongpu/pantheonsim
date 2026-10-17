// A stream captured into a graph (a copy, a kernel and a copy back), then
// instantiated and launched twice.
#include "common.h"

int main() {
  const int n = 256;
  const size_t bytes = n * sizeof(float);
  float *ha = new float[n], *hc = new float[n];
  for (int i = 0; i < n; i++) ha[i] = float(i);
  float *da, *db, *dc;
  CHECK(hipMalloc(&da, bytes));
  CHECK(hipMalloc(&db, bytes));
  CHECK(hipMalloc(&dc, bytes));
  CHECK(hipMemcpy(db, ha, bytes, hipMemcpyHostToDevice));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  hipGraph_t g;
  hipGraphExec_t x;
  CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  CHECK(hipMemcpyAsync(da, ha, bytes, hipMemcpyHostToDevice, s));
  hipLaunchKernelGGL(vadd, dim3(n / 64), dim3(64), 0, s, da, db, dc, n);
  CHECK(hipMemcpyAsync(hc, dc, bytes, hipMemcpyDeviceToHost, s));
  CHECK(hipStreamEndCapture(s, &g));
  CHECK(hipGraphInstantiate(&x, g, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(x, s));
  CHECK(hipGraphLaunch(x, s));
  CHECK(hipStreamSynchronize(s));
  int wrong = 0;
  for (int i = 0; i < n; i++) wrong += hc[i] != float(2 * i);
  CHECK(hipGraphExecDestroy(x));
  CHECK(hipGraphDestroy(g));
  CHECK(hipStreamDestroy(s));
  CHECK(hipFree(da));
  CHECK(hipFree(db));
  CHECK(hipFree(dc));
  std::printf("graph wrong %d of %d\n", wrong, n);
  return wrong != 0;
}
