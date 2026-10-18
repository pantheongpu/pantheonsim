// The kernels ROCm's runtime runs for the program: how it copies within a device, copies
// small host transfers and fills buffers. Each call below is one dispatch of
// __amd_rocclr_copyBuffer, ...copyBufferAligned or ...fillBufferAligned, whose grid depends
// on the size, the alignment and the number of compute units.
#include "common.h"

int main() {
  const size_t cap = 64u << 20;
  char *a, *b;
  CHECK(hipMalloc(&a, cap));
  CHECK(hipMalloc(&b, cap));
  char* h;
  CHECK(hipHostMalloc(reinterpret_cast<void**>(&h), cap, hipHostMallocDefault));
  const size_t sizes[] = {1, 3, 4, 7, 64, 100, 1000, 4096, 16384, 20000, 65536, 100000, 1u << 20, 4u << 20, 16u << 20, 64u << 20};
  for (size_t n : sizes) CHECK(hipMemcpy(b, a, n, hipMemcpyDeviceToDevice));
  // unaligned source or destination
  CHECK(hipMemcpy(b + 1, a, 4096, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b, a + 2, 4096, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b + 4, a + 4, 4096, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b + 8, a + 8, 1 << 20, hipMemcpyDeviceToDevice));
  for (size_t n : {size_t(1), size_t(3), size_t(100), size_t(1000), size_t(20000), size_t(100000)})
    CHECK(hipMemcpy(b + 1, a + 1, n, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b + 4, a, 20000, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b, a + 8, 20000, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b + 16, a + 32, 20000, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(b + 3, a + 1, 1 << 20, hipMemcpyDeviceToDevice));
  CHECK(hipMemcpy(a + 1, h, 1000, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(a, h + 4, 20000, hipMemcpyHostToDevice));
  // host transfers a kernel does (up to 16 KiB)
  for (size_t n : {size_t(1), size_t(100), size_t(4096), size_t(16384)}) {
    CHECK(hipMemcpy(a, h, n, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(h, a, n, hipMemcpyDeviceToHost));
  }
  CHECK(hipMemcpy(a + 1, h, 4096, hipMemcpyHostToDevice));
  CHECK(hipMemcpy(h + 1, a, 4096, hipMemcpyDeviceToHost));
  for (size_t n : sizes) CHECK(hipMemset(b, 7, n));
  CHECK(hipMemset(b + 1, 7, 4096));
  CHECK(hipMemset(b + 4, 7, 4099));
  CHECK(hipMemset(b + 1, 7, 1 << 20));
  CHECK(hipMemset(b + 3, 7, 100000));
  CHECK(hipMemset(b + 16, 7, 100));
  CHECK(hipMemset(b + 8, 7, 100000));
  CHECK(hipMemset(b + 1, 7, 3));
  CHECK(hipMemsetD8(reinterpret_cast<hipDeviceptr_t>(b), 7, 1000));
  CHECK(hipMemsetD16(reinterpret_cast<hipDeviceptr_t>(b), 7, 1000));
  CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(b), 7, 1000));
  CHECK(hipMemsetD32(reinterpret_cast<hipDeviceptr_t>(b), 7, 1u << 20));
  CHECK(hipDeviceSynchronize());
  CHECK(hipFree(a));
  CHECK(hipFree(b));
  CHECK(hipHostFree(h));
  std::printf("runtime_kernels done\n");
  return 0;
}
