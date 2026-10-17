// Two streams with a kernel each, ordered by an event: the second stream waits
// for the first stream's kernel before it runs its own.
#include "common.h"

__global__ void bump(int* p) { atomicAdd(p, 1); }

int main() {
  int* d;
  CHECK(hipMalloc(&d, sizeof(int)));
  CHECK(hipMemset(d, 0, sizeof(int)));
  hipStream_t s1, s2;
  CHECK(hipStreamCreate(&s1));
  CHECK(hipStreamCreate(&s2));
  hipEvent_t e;
  CHECK(hipEventCreate(&e));
  hipLaunchKernelGGL(bump, dim3(1), dim3(64), 0, s1, d);
  CHECK(hipEventRecord(e, s1));
  CHECK(hipStreamWaitEvent(s2, e, 0));
  hipLaunchKernelGGL(bump, dim3(2), dim3(64), 0, s2, d);
  CHECK(hipStreamSynchronize(s2));
  CHECK(hipEventSynchronize(e));
  CHECK(hipStreamSynchronize(s1));
  int h = 0;
  CHECK(hipMemcpy(&h, d, sizeof h, hipMemcpyDeviceToHost));
  CHECK(hipEventDestroy(e));
  CHECK(hipStreamDestroy(s1));
  CHECK(hipStreamDestroy(s2));
  CHECK(hipFree(d));
  std::printf("streams_events count %d (want 192)\n", h);
  return h != 192;
}
