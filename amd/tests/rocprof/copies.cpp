// Copies between pinned host memory and the device, on a stream and blocking:
// the sizes where ROCm's runtime hands a copy to a DMA engine, which is a
// memory-copy record, instead of to a kernel.
#include "common.h"

int main() {
  const size_t bytes = 1 << 20;
  char* pinned;
  CHECK(hipHostMalloc(reinterpret_cast<void**>(&pinned), bytes, hipHostMallocDefault));
  char* pageable = new char[bytes];
  for (size_t i = 0; i < bytes; i++) pinned[i] = char(i * 7);
  char *a, *b;
  CHECK(hipMalloc(&a, bytes));
  CHECK(hipMalloc(&b, bytes));
  hipStream_t s;
  CHECK(hipStreamCreate(&s));
  CHECK(hipMemcpyAsync(a, pinned, bytes, hipMemcpyHostToDevice, s));
  CHECK(hipMemcpyAsync(b, a, bytes, hipMemcpyDeviceToDevice, s));
  CHECK(hipMemcpyAsync(pinned, b, bytes / 2, hipMemcpyDeviceToHost, s));
  CHECK(hipStreamSynchronize(s));
  CHECK(hipMemcpy(a, pageable, bytes, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(pageable, a, bytes, hipMemcpyDeviceToHost));
  CHECK(hipStreamDestroy(s));
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipHostFree(pinned));
  std::printf("copies done\n");
  return 0;
}
