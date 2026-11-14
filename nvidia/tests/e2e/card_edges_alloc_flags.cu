// The flags of cudaEventCreateWithFlags, cudaHostAlloc and cudaMallocManaged,
// as an RTX 3060 (CUDA 13.0) checks them. The simulator took any value.
//
//   - Events: Default, BlockingSync, DisableTiming and Interprocess, and an
//     interprocess event must not time (Interprocess without DisableTiming is
//     refused). Any other bit is cudaErrorInvalidValue and makes no event.
//   - cudaHostAlloc: Portable, Mapped and WriteCombined; any other bit is
//     cudaErrorInvalidValue and allocates nothing. A zero-byte request still
//     succeeds first.
//   - cudaMallocManaged: exactly one of cudaMemAttachGlobal and
//     cudaMemAttachHost. 0, both, and any other bit are cudaErrorInvalidValue.
//     A zero-byte request still succeeds first.
#include <cuda_runtime.h>

#include <cstdio>

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL %s -> %d, expected %d\n", #x, got_, static_cast<int>(want)); ++fails; } } while (0)

int main() {
  WANT(cudaFree(nullptr), cudaSuccess);
  const unsigned flags[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0x10, 0x80, 0x100, 0x8000};
  for (unsigned f : flags) {
    cudaEvent_t e = nullptr;
    const bool ok = f <= 7 && f != 4 && f != 5;
    const cudaError_t r = cudaEventCreateWithFlags(&e, f);
    if (r != (ok ? cudaSuccess : cudaErrorInvalidValue)) { std::printf("FAIL event flags %#x -> %d\n", f, (int)r); ++fails; }
    WANT(cudaGetLastError(), ok ? cudaSuccess : cudaErrorInvalidValue);
    if (r == cudaSuccess) WANT(cudaEventDestroy(e), cudaSuccess);
  }
  for (unsigned f : flags) {
    void* p = nullptr;
    const bool ok = f <= 7;
    const cudaError_t r = cudaHostAlloc(&p, 64, f);
    if (r != (ok ? cudaSuccess : cudaErrorInvalidValue)) { std::printf("FAIL cudaHostAlloc flags %#x -> %d\n", f, (int)r); ++fails; }
    WANT(cudaGetLastError(), ok ? cudaSuccess : cudaErrorInvalidValue);
    if (r == cudaSuccess) WANT(cudaFreeHost(p), cudaSuccess);
  }
  for (unsigned f : flags) {
    void* p = nullptr;
    const bool ok = f == cudaMemAttachGlobal || f == cudaMemAttachHost;
    const cudaError_t r = cudaMallocManaged(&p, 64, f);
    if (r != (ok ? cudaSuccess : cudaErrorInvalidValue)) { std::printf("FAIL cudaMallocManaged flags %#x -> %d\n", f, (int)r); ++fails; }
    WANT(cudaGetLastError(), ok ? cudaSuccess : cudaErrorInvalidValue);
    if (r == cudaSuccess) WANT(cudaFree(p), cudaSuccess);
  }
  void* p = reinterpret_cast<void*>(1);
  WANT(cudaMallocManaged(&p, 0, 0), cudaSuccess);
  WANT(p == nullptr, true);
  WANT(cudaHostAlloc(&p, 0, 8), cudaSuccess);
  WANT(cudaMallocManaged(nullptr, 16, 0), cudaErrorInvalidValue);
  WANT(cudaHostAlloc(nullptr, 16, 8), cudaErrorInvalidValue);
  WANT(cudaEventCreateWithFlags(nullptr, 8), cudaErrorInvalidValue);
  cudaGetLastError();
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
