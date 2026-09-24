// cudaDeviceSetLimit and cudaDeviceGetLimit, and the heap limit being real.
//
// The heap is the one limit that changes what a kernel can do here: malloc() in
// a kernel draws from a heap of cudaLimitMallocHeapSize bytes, so a program
// that raises the limit gets the room, and an allocation past it returns null
// the way it does on hardware. The test allocates up to the limit and one past
// it to show both.
#include <cuda_runtime.h>
#include <cstdio>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)
#define WANT(x, want) do { cudaError_t e_ = (x); if (e_ != (want)) { \
  printf("FAIL %s -> %s, expected %s\n", #x, cudaGetErrorString(e_), #want); return 1; } } while (0)
#define CHECK(c) do { if (!(c)) { printf("FAIL %s (line %d)\n", #c, __LINE__); return 1; } } while (0)

static const size_t kMiB = 1u << 20;

// Thread i asks for one MiB, touches it, and reports whether it got it. Every
// allocation is held until all threads have tried, so the total is what counts.
__global__ void grab(int* got, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  char* p = nullptr;
  if (i < n) {
    p = static_cast<char*>(malloc(kMiB));
    got[i] = p != nullptr;
    if (p) p[kMiB - 1] = 1;
  }
  __syncthreads();
  free(p);
}

__global__ void say() { printf("device printf ran\n"); }

static size_t limit(cudaLimit which) {
  size_t v = 0;
  return cudaDeviceGetLimit(&v, which) == cudaSuccess ? v : ~size_t{0};
}

int main() {
  // ---- the defaults ---------------------------------------------------------
  CHECK(limit(cudaLimitStackSize) == 1024);
  CHECK(limit(cudaLimitPrintfFifoSize) == kMiB);
  CHECK(limit(cudaLimitMallocHeapSize) == 8 * kMiB);
  CHECK(limit(cudaLimitDevRuntimePendingLaunchCount) == 2048);
  CHECK(limit(cudaLimitDevRuntimeSyncDepth) == 2);   // an A10 is below 9.0
  CHECK(limit(cudaLimitPersistingL2CacheSize) == 0);

  // ---- what the driver may round or clamp ----------------------------------
  CK(cudaDeviceSetLimit(cudaLimitStackSize, 1000));
  CHECK(limit(cudaLimitStackSize) == 1008);           // rounded up to 16 bytes
  CK(cudaDeviceSetLimit(cudaLimitMaxL2FetchGranularity, 4096));
  CHECK(limit(cudaLimitMaxL2FetchGranularity) == 128);
  CK(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 100));
  CHECK(limit(cudaLimitDevRuntimeSyncDepth) == 24);
  CK(cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 4 * kMiB));
  CHECK(limit(cudaLimitPersistingL2CacheSize) == 0);   // the device sets none aside
  // 7 is not a limit, and still a value the enum can hold; casting a number
  // outside its range would itself be undefined behaviour.
  WANT(cudaDeviceSetLimit(static_cast<cudaLimit>(7), 1), cudaErrorUnsupportedLimit);
  size_t unused = 0;
  WANT(cudaDeviceGetLimit(&unused, static_cast<cudaLimit>(7)), cudaErrorUnsupportedLimit);

  // ---- the heap -------------------------------------------------------------
  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 32 * kMiB));
  CHECK(limit(cudaLimitMallocHeapSize) == 32 * kMiB);

  const int threads = 40;
  int* got = nullptr;
  CK(cudaMalloc(&got, threads * sizeof(int)));
  CK(cudaMemset(got, 0, threads * sizeof(int)));
  grab<<<1, threads>>>(got, threads);
  CK(cudaDeviceSynchronize());
  int host[threads];
  CK(cudaMemcpy(host, got, sizeof host, cudaMemcpyDeviceToHost));
  int succeeded = 0;
  for (int i = 0; i < threads; ++i) succeeded += host[i];
  // Thirty-two MiB of heap is thirty-two one-MiB allocations -- four times what
  // the default would have given -- and the thirty-third gets null.
  if (succeeded != 32) {
    printf("FAIL %d of %d one-MiB allocations succeeded under a 32 MiB heap\n", succeeded, threads);
    return 1;
  }

  // A kernel that uses the heap has launched, so the heap is fixed now.
  WANT(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 64 * kMiB), cudaErrorInvalidValue);
  CHECK(limit(cudaLimitMallocHeapSize) == 32 * kMiB);

  // ---- the printf buffer, the same rule -------------------------------------
  CK(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 4 * kMiB));
  CHECK(limit(cudaLimitPrintfFifoSize) == 4 * kMiB);
  say<<<1, 1>>>();
  CK(cudaDeviceSynchronize());
  WANT(cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 8 * kMiB), cudaErrorInvalidValue);

  // ---- a reset starts over ----------------------------------------------------
  CK(cudaFree(got));
  CK(cudaDeviceReset());
  CHECK(limit(cudaLimitMallocHeapSize) == 8 * kMiB);
  CHECK(limit(cudaLimitStackSize) == 1024);
  CK(cudaDeviceSetLimit(cudaLimitMallocHeapSize, 16 * kMiB));   // settable again
  printf("PASS\n");
  return 0;
}
