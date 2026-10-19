// Stream-ordered allocation by size and pattern: blocks of 1 KiB, 40 MiB and 100 MiB, a
// free that is synchronized and one that is not, a second stream, and hipMallocFromPoolAsync
// on the default pool.
#include "common.h"

int main() {
  hipStream_t s, t;
  CHECK(hipStreamCreate(&s));
  CHECK(hipStreamCreate(&t));
  void *a, *b, *c, *d, *e;
  CHECK(hipMallocAsync(&a, 1024, s));
  CHECK(hipMallocAsync(&b, 40u << 20, s));
  CHECK(hipFreeAsync(a, s));
  CHECK(hipStreamSynchronize(s));
  CHECK(hipMallocAsync(&c, 100u << 20, t));
  CHECK(hipFreeAsync(b, s));
  CHECK(hipFreeAsync(c, t));
  CHECK(hipDeviceSynchronize());
  hipMemPool_t pool;
  CHECK(hipDeviceGetDefaultMemPool(&pool, 0));
  CHECK(hipMallocFromPoolAsync(&d, 4096, pool, s));
  CHECK(hipMallocAsync(&e, 4096, s));
  CHECK(hipFreeAsync(d, s));
  CHECK(hipFreeAsync(e, s));
  CHECK(hipStreamSynchronize(s));
  CHECK(hipStreamDestroy(s));
  CHECK(hipStreamDestroy(t));
  std::printf("async_sizes done\n");
  return 0;
}
