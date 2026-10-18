// What a graph instantiation allocates for itself, by the graph's size: one, two and five
// kernel nodes, a copy-only graph, the same graph instantiated twice, and the executable
// graph destroyed before and after the graph.
#include "common.h"

static hipGraph_t capture(hipStream_t s, int kernels, bool copy, const float* d, float* h) {
  hipGraph_t g;
  CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
  if (copy) CHECK(hipMemcpyAsync(const_cast<float*>(d), h, 1024, hipMemcpyHostToDevice, s));
  for (int i = 0; i < kernels; i++) hipLaunchKernelGGL(vadd, dim3(4), dim3(64), 0, s, d, d, const_cast<float*>(d), 256);
  CHECK(hipStreamEndCapture(s, &g));
  return g;
}

int main() {
  float *d, *h = new float[256]();
  CHECK(hipMalloc(&d, 1024));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  hipGraphExec_t x[5];
  hipGraph_t g[5];
  const int kernels[5] = {1, 2, 5, 0, 1};
  for (int i = 0; i < 5; i++) {
    g[i] = capture(s, kernels[i], i >= 2, d, h);
    CHECK(hipGraphInstantiate(&x[i], g[i], nullptr, nullptr, 0));
  }
  CHECK(hipGraphLaunch(x[0], s));
  CHECK(hipStreamSynchronize(s));
  CHECK(hipGraphExecDestroy(x[0]));
  CHECK(hipGraphDestroy(g[1]));
  CHECK(hipGraphExecDestroy(x[1]));
  CHECK(hipGraphDestroy(g[0]));
  for (int i = 2; i < 5; i++) {
    CHECK(hipGraphExecDestroy(x[i]));
    CHECK(hipGraphDestroy(g[i]));
  }
  CHECK(hipStreamDestroy(s));
  CHECK(hipFree(d));
  std::printf("graph_sizes done\n");
  return 0;
}
