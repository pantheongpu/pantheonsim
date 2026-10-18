// Which calls make the runtime allocate for itself: device and event setup first, then a
// launch on the default stream, then allocation variants, streams made with flags and
// priorities, and an event wait. The profiler's allocation records say where each of the
// runtime's own blocks (staging buffers, queue state) appears.
#include "common.h"

int main() {
  int n = 0;
  CHECK(hipGetDeviceCount(&n));
  CHECK(hipSetDevice(0));
  hipDeviceProp_t prop;
  CHECK(hipGetDeviceProperties(&prop, 0));
  hipEvent_t e;
  CHECK(hipEventCreate(&e));
  CHECK(hipDeviceSynchronize());
  hipLaunchKernelGGL(vadd, dim3(1), dim3(64), 0, 0, nullptr, nullptr, nullptr, 0);
  CHECK(hipDeviceSynchronize());
  void *a, *b, *c;
  CHECK(hipMalloc(&a, 1 << 16));
  size_t pitch;
  CHECK(hipMallocPitch(&b, &pitch, 1000, 10));
  CHECK(hipMallocManaged(&c, 1 << 16));
  void* d;
  CHECK(hipExtMallocWithFlags(&d, 1 << 16, hipDeviceMallocFinegrained));
  CHECK(hipEventRecord(e, 0));
  CHECK(hipEventSynchronize(e));
  hipStream_t s1, s2;
  CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
  CHECK(hipStreamCreateWithPriority(&s2, hipStreamDefault, -1));
  CHECK(hipMemcpyAsync(a, d, 1 << 16, hipMemcpyDeviceToDevice, 0));
  CHECK(hipMemsetAsync(a, 1, 1 << 16, s1));
  CHECK(hipMemsetAsync(a, 1, 1 << 16, s2));
  CHECK(hipDeviceSynchronize());
  CHECK(hipStreamDestroy(s1));
  CHECK(hipStreamDestroy(s2));
  CHECK(hipEventDestroy(e));
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipFree(c));
  CHECK(hipFree(d));
  std::printf("first_calls done\n");
  return 0;
}
