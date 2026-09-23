// Stream-ordered memory pools, the part of the API that is worth having: a pool
// holds what was freed to it and hands it back, so an allocator asks the driver
// once and reuses after that. PyTorch's async allocator and RAPIDS' RMM both
// work this way, and both read the pool's statistics to decide what to do.
//
// The default release threshold is 0, which the documentation says means freed
// memory goes straight back to the device -- so the caching only starts once a
// threshold is set. This checks both halves of that, and the statistics.
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)

static unsigned long long attr(cudaMemPool_t pool, cudaMemPoolAttr a) {
  unsigned long long v = 0;
  if (cudaMemPoolGetAttribute(pool, a, &v) != cudaSuccess) return ~0ull;
  return v;
}

__global__ void fill(int* p, int n, int v) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] = v;
}

int main() {
  const size_t kBytes = 1 << 20;   // 1 MiB
  cudaMemPool_t pool = nullptr;
  CK(cudaDeviceGetDefaultMemPool(&pool, 0));

  // Nothing allocated yet.
  if (attr(pool, cudaMemPoolAttrUsedMemCurrent) != 0 ||
      attr(pool, cudaMemPoolAttrReservedMemCurrent) != 0) {
    printf("FAIL a fresh pool reports used %llu reserved %llu\n",
           attr(pool, cudaMemPoolAttrUsedMemCurrent), attr(pool, cudaMemPoolAttrReservedMemCurrent));
    return 1;
  }

  // An allocation, used by a kernel, then freed. With the default threshold of
  // 0 the memory goes back to the device rather than into the pool.
  int* a = nullptr;
  CK(cudaMallocAsync(reinterpret_cast<void**>(&a), kBytes, 0));
  if (attr(pool, cudaMemPoolAttrUsedMemCurrent) != kBytes) {
    printf("FAIL after one allocation, used is %llu\n", attr(pool, cudaMemPoolAttrUsedMemCurrent));
    return 1;
  }
  const int n = 256;
  fill<<<1, n>>>(a, n, 7);
  CK(cudaDeviceSynchronize());
  int back[n] = {0};
  CK(cudaMemcpy(back, a, sizeof back, cudaMemcpyDeviceToHost));
  if (back[0] != 7 || back[n - 1] != 7) {
    printf("FAIL pool memory did not take the kernel's writes: %d %d\n", back[0], back[n - 1]);
    return 1;
  }
  CK(cudaFreeAsync(a, 0));
  if (attr(pool, cudaMemPoolAttrUsedMemCurrent) != 0 ||
      attr(pool, cudaMemPoolAttrReservedMemCurrent) != 0) {
    printf("FAIL with a threshold of 0, a free kept used %llu reserved %llu\n",
           attr(pool, cudaMemPoolAttrUsedMemCurrent), attr(pool, cudaMemPoolAttrReservedMemCurrent));
    return 1;
  }
  // The high-water marks remember what the current totals no longer do.
  if (attr(pool, cudaMemPoolAttrUsedMemHigh) != kBytes) {
    printf("FAIL the used high-water mark is %llu\n", attr(pool, cudaMemPoolAttrUsedMemHigh));
    return 1;
  }

  // Now a threshold: the pool keeps freed memory and the next allocation of the
  // same size reuses it, which is the point of the API.
  unsigned long long threshold = 4 * kBytes;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold));
  if (attr(pool, cudaMemPoolAttrReleaseThreshold) != threshold) {
    printf("FAIL the threshold reads back as %llu\n", attr(pool, cudaMemPoolAttrReleaseThreshold));
    return 1;
  }
  int* b = nullptr;
  CK(cudaMallocAsync(reinterpret_cast<void**>(&b), kBytes, 0));
  CK(cudaFreeAsync(b, 0));
  if (attr(pool, cudaMemPoolAttrReservedMemCurrent) != kBytes ||
      attr(pool, cudaMemPoolAttrUsedMemCurrent) != 0) {
    printf("FAIL after a free under the threshold: reserved %llu used %llu\n",
           attr(pool, cudaMemPoolAttrReservedMemCurrent),
           attr(pool, cudaMemPoolAttrUsedMemCurrent));
    return 1;
  }
  int* c = nullptr;
  CK(cudaMallocAsync(reinterpret_cast<void**>(&c), kBytes, 0));
  if (c != b) {
    printf("FAIL the pool did not reuse the cached block (%p then %p)\n",
           static_cast<void*>(b), static_cast<void*>(c));
    return 1;
  }
  // Reuse, not a second request to the device.
  if (attr(pool, cudaMemPoolAttrReservedMemCurrent) != kBytes) {
    printf("FAIL reuse asked the device for more: reserved %llu\n",
           attr(pool, cudaMemPoolAttrReservedMemCurrent));
    return 1;
  }
  CK(cudaFreeAsync(c, 0));

  // Trimming gives the cache back on demand.
  CK(cudaMemPoolTrimTo(pool, 0));
  if (attr(pool, cudaMemPoolAttrReservedMemCurrent) != 0) {
    printf("FAIL after trimming, reserved is %llu\n",
           attr(pool, cudaMemPoolAttrReservedMemCurrent));
    return 1;
  }
  // And a high-water mark resets by writing zero to it, as documented.
  unsigned long long zero = 0;
  CK(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &zero));
  if (attr(pool, cudaMemPoolAttrUsedMemHigh) != 0) {
    printf("FAIL the high-water mark did not reset\n");
    return 1;
  }

  // A pool of one's own, with allocations taken from it explicitly.
  cudaMemPoolProps props{};
  props.allocType = cudaMemAllocationTypePinned;
  props.handleTypes = cudaMemHandleTypeNone;
  props.location.type = cudaMemLocationTypeDevice;
  props.location.id = 0;
  cudaMemPool_t mine = nullptr;
  CK(cudaMemPoolCreate(&mine, &props));
  int* d = nullptr;
  CK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&d), kBytes, mine, 0));
  if (attr(mine, cudaMemPoolAttrUsedMemCurrent) != kBytes ||
      attr(pool, cudaMemPoolAttrUsedMemCurrent) != 0) {
    printf("FAIL an explicit pool's allocation was counted against the wrong pool\n");
    return 1;
  }
  fill<<<1, n>>>(d, n, 9);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost));
  if (back[0] != 9) {
    printf("FAIL memory from an explicit pool did not work: %d\n", back[0]);
    return 1;
  }
  // A pool with allocations outstanding cannot be destroyed.
  WANT(cudaMemPoolDestroy(mine), cudaErrorInvalidValue);
  CK(cudaFreeAsync(d, 0));
  CK(cudaMemPoolDestroy(mine));
  // Nor can the device's own pool be destroyed.
  WANT(cudaMemPoolDestroy(pool), cudaErrorInvalidValue);

  // The device's current pool can be pointed at one of ours, and
  // cudaMallocAsync then takes from that one.
  cudaMemPool_t current = nullptr;
  CK(cudaMemPoolCreate(&mine, &props));
  CK(cudaDeviceSetMemPool(0, mine));
  CK(cudaDeviceGetMemPool(&current, 0));
  if (current != mine) {
    printf("FAIL the device's pool is not the one that was set\n");
    return 1;
  }
  int* e = nullptr;
  CK(cudaMallocAsync(reinterpret_cast<void**>(&e), kBytes, 0));
  if (attr(mine, cudaMemPoolAttrUsedMemCurrent) != kBytes) {
    printf("FAIL cudaMallocAsync did not take from the device's current pool\n");
    return 1;
  }
  CK(cudaFreeAsync(e, 0));
  CK(cudaDeviceSetMemPool(0, pool));
  CK(cudaMemPoolDestroy(mine));

  // What the API refuses.
  WANT(cudaMemPoolCreate(&mine, nullptr), cudaErrorInvalidValue);
  props.location.id = 99;
  WANT(cudaMemPoolCreate(&mine, &props), cudaErrorInvalidDevice);
  props.location.id = 0;
  cudaMemPool_t bogus = reinterpret_cast<cudaMemPool_t>(0x1234);
  WANT(cudaMemPoolTrimTo(bogus, 0), cudaErrorInvalidValue);
  unsigned long long v = 1;
  WANT(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &v), cudaErrorInvalidValue);
  WANT(cudaMemPoolSetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &v), cudaErrorInvalidValue);

  // ---- managed memory hints -------------------------------------------------
  //
  // Advice moves nothing here, but it is state: what was set comes back, a
  // range that disagrees answers "no single value", and both calls refuse
  // memory the API does not take hints about.
  int* m = nullptr;
  const size_t kInts = 4096;
  CK(cudaMallocManaged(reinterpret_cast<void**>(&m), kInts * sizeof(int)));
  int hint = -99;
  CK(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributeReadMostly, m,
                              kInts * sizeof(int)));
  if (hint != 0) { printf("FAIL fresh managed memory reports read-mostly %d\n", hint); return 1; }

#if CUDART_VERSION >= 13000
  cudaMemLocation dev0{};
  dev0.type = cudaMemLocationTypeDevice;
  dev0.id = 0;
  CK(cudaMemAdvise(m, kInts * sizeof(int), cudaMemAdviseSetReadMostly, dev0));
  CK(cudaMemAdvise(m, kInts * sizeof(int), cudaMemAdviseSetPreferredLocation, dev0));
  CK(cudaMemAdvise(m, kInts * sizeof(int), cudaMemAdviseSetAccessedBy, dev0));
  CK(cudaMemPrefetchAsync(m, kInts * sizeof(int), dev0, 0, 0));
#else
  CK(cudaMemAdvise(m, kInts * sizeof(int), cudaMemAdviseSetReadMostly, 0));
  CK(cudaMemAdvise(m, kInts * sizeof(int), cudaMemAdviseSetPreferredLocation, 0));
  CK(cudaMemAdvise(m, kInts * sizeof(int), cudaMemAdviseSetAccessedBy, 0));
  CK(cudaMemPrefetchAsync(m, kInts * sizeof(int), 0, 0));
#endif
  CK(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributeReadMostly, m,
                              kInts * sizeof(int)));
  if (hint != 1) { printf("FAIL read-mostly was set but reads %d\n", hint); return 1; }
  CK(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributePreferredLocation, m,
                              kInts * sizeof(int)));
  if (hint != 0) { printf("FAIL preferred location reads %d, expected device 0\n", hint); return 1; }
  CK(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributeLastPrefetchLocation, m,
                              kInts * sizeof(int)));
  if (hint != 0) { printf("FAIL last prefetch location reads %d\n", hint); return 1; }
  int accessed[4] = {-9, -9, -9, -9};
  CK(cudaMemRangeGetAttribute(accessed, sizeof accessed, cudaMemRangeAttributeAccessedBy, m,
                              kInts * sizeof(int)));
  if (accessed[0] != 0 || accessed[1] != cudaInvalidDeviceId) {
    printf("FAIL accessed-by reads %d %d\n", accessed[0], accessed[1]);
    return 1;
  }

  // Unset half of it: the whole range no longer agrees, so read-mostly is 0
  // over all of it and still 1 over the half that kept it.
#if CUDART_VERSION >= 13000
  CK(cudaMemAdvise(m, (kInts / 2) * sizeof(int), cudaMemAdviseUnsetReadMostly, dev0));
#else
  CK(cudaMemAdvise(m, (kInts / 2) * sizeof(int), cudaMemAdviseUnsetReadMostly, 0));
#endif
  CK(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributeReadMostly, m,
                              kInts * sizeof(int)));
  if (hint != 0) { printf("FAIL a half-unset range reports read-mostly %d\n", hint); return 1; }
  CK(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributeReadMostly,
                              m + kInts / 2, (kInts / 2) * sizeof(int)));
  if (hint != 1) { printf("FAIL the half that kept read-mostly reports %d\n", hint); return 1; }

  // Memory that is not managed takes no hints, and neither does a device that
  // does not exist.
  int* plain = nullptr;
  CK(cudaMalloc(reinterpret_cast<void**>(&plain), 1024));
#if CUDART_VERSION >= 13000
  WANT(cudaMemAdvise(plain, 1024, cudaMemAdviseSetReadMostly, dev0), cudaErrorInvalidValue);
  WANT(cudaMemPrefetchAsync(plain, 1024, dev0, 0, 0), cudaErrorInvalidValue);
  cudaMemLocation nowhere{};
  nowhere.type = cudaMemLocationTypeDevice;
  nowhere.id = 99;
  WANT(cudaMemAdvise(m, 1024, cudaMemAdviseSetAccessedBy, nowhere), cudaErrorInvalidDevice);
#else
  WANT(cudaMemAdvise(plain, 1024, cudaMemAdviseSetReadMostly, 0), cudaErrorInvalidValue);
  WANT(cudaMemPrefetchAsync(plain, 1024, 0, 0), cudaErrorInvalidValue);
  WANT(cudaMemAdvise(m, 1024, cudaMemAdviseSetAccessedBy, 99), cudaErrorInvalidDevice);
#endif
  WANT(cudaMemRangeGetAttribute(&hint, sizeof hint, cudaMemRangeAttributeReadMostly, plain, 1024),
       cudaErrorInvalidValue);
  CK(cudaFree(plain));
  CK(cudaFree(m));

  printf("PASS\n");
  return 0;
}
