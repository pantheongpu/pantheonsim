// Fills and device-to-device copies, blocking and on a stream.
#include "common.h"

int main() {
  const size_t bytes = 4096;
  unsigned char* h = new unsigned char[bytes];
  unsigned char *a, *b, *c;
  CHECK(hipMalloc(&a, bytes));
  CHECK(hipMalloc(&b, bytes));
  CHECK(hipMalloc(&c, bytes));
  CHECK(hipMemset(a, 0x5a, bytes));
  CHECK(hipMemcpy(b, a, bytes, hipMemcpyDeviceToDevice));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  CHECK(hipMemsetAsync(c, 0x11, bytes, s));
  CHECK(hipMemcpyAsync(a, c, bytes, hipMemcpyDeviceToDevice, s));
  CHECK(hipMemsetD32Async(reinterpret_cast<hipDeviceptr_t>(c), 0x01020304, bytes / 4, s));
  CHECK(hipStreamSynchronize(s));
  CHECK(hipMemcpy(h, b, bytes, hipMemcpyDeviceToHost));
  int wrong = 0;
  for (size_t i = 0; i < bytes; i++) wrong += h[i] != 0x5a;
  CHECK(hipMemcpy(h, a, bytes, hipMemcpyDeviceToHost));
  for (size_t i = 0; i < bytes; i++) wrong += h[i] != 0x11;
  CHECK(hipStreamDestroy(s));
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipFree(c));
  std::printf("memset_d2d wrong %d\n", wrong);
  return wrong != 0;
}
