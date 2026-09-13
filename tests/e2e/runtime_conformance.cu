// The CUDA runtime API's documented behaviour at its edges, on a simulated
// 2 x Tesla T4. Each check was written from the documentation, and the ones
// marked with a note used to disagree with it.
#include <cstdio>
#include <cstdlib>
#include <cstring>
static int fails = 0;
#define CHECK(name, cond, ...)                                   \
  do {                                                           \
    bool ok_ = (cond);                                           \
    std::printf("%s %s", ok_ ? "PASS" : "FAIL", name);           \
    if (!ok_) { std::printf("  -> "); std::printf(__VA_ARGS__); ++fails; } \
    std::printf("\n");                                           \
  } while (0)

__global__ void fill(int* d, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) d[i] = i;
}
__global__ void oob(int* d) { d[threadIdx.x + (1 << 20)] = 1; }

int main() {
  int n = -1;
  cudaGetDeviceCount(&n);
  CHECK("device count", n == 2, "got %d", n);
  cudaError_t e = cudaSetDevice(2);
  CHECK("cudaSetDevice out of range", e == cudaErrorInvalidDevice, "got %d", e);
  cudaGetLastError();
  CHECK("cudaFree(nullptr)", cudaFree(nullptr) == cudaSuccess, "");

  int* big = nullptr;
  e = cudaMalloc(&big, (size_t)1 << 50);
  CHECK("cudaMalloc beyond the device", e == cudaErrorMemoryAllocation, "got %d", e);
  CHECK("peek does not reset", cudaPeekAtLastError() == cudaErrorMemoryAllocation, "");
  CHECK("get returns it", cudaGetLastError() == cudaErrorMemoryAllocation, "");
  CHECK("and resets", cudaGetLastError() == cudaSuccess, "");

  cudaDeviceProp p;
  std::memset(&p, 0, sizeof p);
  cudaGetDeviceProperties(&p, 0);
  CHECK("properties", !std::strcmp(p.name, "Tesla T4") && p.major == 7 && p.minor == 5 &&
                          p.maxThreadsPerBlock == 1024 && p.maxThreadsDim[2] == 64,
        "%s %d.%d %d %d", p.name, p.major, p.minor, p.maxThreadsPerBlock, p.maxThreadsDim[2]);
  size_t free0 = 0, total = 0;
  cudaMemGetInfo(&free0, &total);
  CHECK("memGetInfo total", total == p.totalGlobalMem && free0 <= total, "");

  // Where a pointer lives. Pinned and managed memory used to report
  // "unregistered".
  int* d = nullptr;
  cudaMalloc(&d, 64 * sizeof(int));
  cudaPointerAttributes a;
  std::memset(&a, 0xff, sizeof a);
  CHECK("attributes: device", cudaPointerGetAttributes(&a, d) == cudaSuccess &&
                                  a.type == cudaMemoryTypeDevice && a.device == 0, "type %d", (int)a.type);
  int* host = (int*)std::malloc(64);
  std::memset(&a, 0xff, sizeof a);
  CHECK("attributes: malloc is unregistered", cudaPointerGetAttributes(&a, host) == cudaSuccess &&
                                                 a.type == cudaMemoryTypeUnregistered, "type %d", (int)a.type);
  int* pinned = nullptr;
  cudaHostAlloc(&pinned, 64 * sizeof(int), 0);
  std::memset(&a, 0xff, sizeof a);
  cudaPointerGetAttributes(&a, pinned + 3);   // an interior pointer, too
  CHECK("attributes: pinned is host", a.type == cudaMemoryTypeHost, "type %d", (int)a.type);
  int* m = nullptr;
  cudaMallocManaged(&m, 64 * sizeof(int));
  std::memset(&a, 0xff, sizeof a);
  cudaPointerGetAttributes(&a, m);
  CHECK("attributes: managed", a.type == cudaMemoryTypeManaged, "type %d", (int)a.type);

  // Launch configurations the device cannot run. These used to be
  // cudaErrorInvalidValue.
  fill<<<1, 2000>>>(d, 64);
  e = cudaGetLastError();
  CHECK("2000 threads per block", e == cudaErrorInvalidConfiguration, "got %d %s", e, cudaGetErrorName(e));
  fill<<<dim3(1, 1, 1), dim3(1, 1, 65)>>>(d, 64);
  e = cudaGetLastError();
  CHECK("blockDim.z 65", e == cudaErrorInvalidConfiguration, "got %d", e);
  fill<<<0, 16>>>(d, 64);
  e = cudaGetLastError();
  CHECK("an empty grid", e == cudaErrorInvalidConfiguration, "got %d", e);
  fill<<<1, 64>>>(d, 64);
  CHECK("a valid launch still works after them", cudaDeviceSynchronize() == cudaSuccess, "");

  cudaStream_t s;
  cudaStreamCreate(&s);
  for (int i = 0; i < 64; ++i) pinned[i] = 1000 + i;
  cudaMemcpyAsync(d, pinned, 64 * sizeof(int), cudaMemcpyHostToDevice, s);
  std::memset(pinned, 0, 64 * sizeof(int));
  cudaMemcpyAsync(pinned, d, 64 * sizeof(int), cudaMemcpyDeviceToHost, s);
  cudaStreamSynchronize(s);
  CHECK("async round trip", pinned[63] == 1063, "got %d", pinned[63]);

  // Last: a kernel's illegal address corrupts the context, and it stays
  // corrupted until a reset. The next call used to succeed.
  oob<<<1, 1>>>(d);
  e = cudaDeviceSynchronize();
  CHECK("out-of-bounds write -> illegal address", e == cudaErrorIllegalAddress, "got %d", e);
  int* after = nullptr;
  CHECK("sticky: the next malloc fails", cudaMalloc(&after, 16) == cudaErrorIllegalAddress, "");
  CHECK("sticky: reading it does not clear it", cudaGetLastError() == cudaErrorIllegalAddress &&
                                                   cudaGetLastError() == cudaErrorIllegalAddress, "");
  CHECK("reset clears it", cudaDeviceReset() == cudaSuccess && cudaMalloc(&after, 16) == cudaSuccess, "");

  std::printf("%d failed\n", fails);
  return fails != 0;
}
