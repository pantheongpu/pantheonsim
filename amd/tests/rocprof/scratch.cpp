// Private (scratch) segments of different sizes, launched in turn and on two streams: the
// runtime grows the queue's scratch pool when a kernel needs more than it has, and the
// profiler records each allocation.
#include "common.h"

template <int N>
__global__ void spill(int* out, int k) {
  int a[N];                           // indexed at run time: lives in scratch
  for (int i = 0; i < N; i++) a[i] = i * k;
  int s = 0;
  for (int i = 0; i < 16; i++) s += a[(threadIdx.x * 7 + i * k) % N];
  out[blockIdx.x * blockDim.x + threadIdx.x] = s;
}

int main() {
  int* d;
  CHECK(hipMalloc(&d, 4096 * sizeof(int)));
  hipStream_t s1, s2;
  CHECK(hipStreamCreate(&s1));
  CHECK(hipStreamCreate(&s2));
  hipLaunchKernelGGL(spill<64>, dim3(2), dim3(64), 0, s1, d, 3);
  CHECK(hipStreamSynchronize(s1));
  hipLaunchKernelGGL(spill<64>, dim3(2), dim3(64), 0, s1, d, 3);
  CHECK(hipStreamSynchronize(s1));
  hipLaunchKernelGGL(spill<512>, dim3(2), dim3(64), 0, s1, d, 3);
  CHECK(hipStreamSynchronize(s1));
  hipLaunchKernelGGL(spill<64>, dim3(2), dim3(64), 0, s1, d, 3);
  CHECK(hipStreamSynchronize(s1));
  hipLaunchKernelGGL(spill<1024>, dim3(2), dim3(64), 0, s1, d, 3);
  CHECK(hipStreamSynchronize(s1));
  hipLaunchKernelGGL(spill<64>, dim3(2), dim3(64), 0, s2, d, 3);
  CHECK(hipStreamSynchronize(s2));
  hipLaunchKernelGGL(spill<512>, dim3(2), dim3(64), 0, s2, d, 3);
  CHECK(hipStreamSynchronize(s2));
  hipLaunchKernelGGL(spill<64>, dim3(2), dim3(64), 0, 0, d, 3);
  CHECK(hipDeviceSynchronize());
  hipLaunchKernelGGL(spill<2048>, dim3(2), dim3(64), 0, 0, d, 3);
  CHECK(hipDeviceSynchronize());
  CHECK(hipStreamDestroy(s1));
  CHECK(hipStreamDestroy(s2));
  CHECK(hipFree(d));
  std::printf("scratch done\n");
  return 0;
}
