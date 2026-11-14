// The callee of host_gpu_rdc.cu, in a translation unit of its own (built with -rdc=true): a device function the kernel cannot inline.
#include <cstdint>

__device__ __noinline__ uint32_t mix(uint32_t x, int rounds) {
#pragma unroll 1
  for (int r = 0; r < rounds; ++r) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    x += uint32_t(r) * 0x9e3779b9u;
  }
  return x;
}
