// A kernel on one device that reads, writes and atomically adds to another device's allocations (issue #598).
//
//   peer_kernel direct     device 1 enables peer access to device 0, then a kernel on device 1 sums a buffer of device 0's, stores into
//                          it and adds to it atomically; a kernel on device 0 then reads what device 1's kernel wrote. Prints the results.
//   peer_kernel without    the same read without enabling peer access first: the kernel faults (cudaErrorIllegalAddress), on a card
//                          too (measured on two RTX 3060s, which have no peer path at all)
//   peer_kernel disabled   enabled, then disabled again: the read faults as if it had never been enabled
//
// Each mode runs in a process of its own, because a fault ends the context.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

__global__ void sum_peer(const int* peer, int n, long long* out) {
  long long s = 0;
  for (int i = 0; i < n; i++) s += peer[i];
  *out = s;
}

__global__ void write_peer(int* peer, int n, unsigned* counter) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) peer[i] = 3 * i + 1;
  atomicAdd(counter, 1u);
}

__global__ void check_own(const int* own, int n, int* bad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n && own[i] != 3 * i + 1) atomicAdd(bad, 1);
}

#define CHECK(x)                                                              \
  do {                                                                        \
    cudaError_t e_ = (x);                                                     \
    if (e_ != cudaSuccess) {                                                  \
      std::printf("%s: %s\n", #x, cudaGetErrorName(e_));                      \
      return 1;                                                               \
    }                                                                         \
  } while (0)

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "direct";
  int count = 0, can = 0;
  CHECK(cudaGetDeviceCount(&count));
  if (count < 2) {
    std::printf("SKIP: needs two devices\n");
    return 0;
  }
  CHECK(cudaDeviceCanAccessPeer(&can, 1, 0));
  std::printf("cudaDeviceCanAccessPeer(1, 0) = %d\n", can);

  const int n = 1000;
  int host[n];
  for (int i = 0; i < n; i++) host[i] = i;
  int* d0 = nullptr;
  unsigned* c0 = nullptr;
  int* bad0 = nullptr;
  CHECK(cudaSetDevice(0));
  CHECK(cudaMalloc(&d0, n * sizeof(int)));
  CHECK(cudaMalloc(&c0, sizeof(unsigned)));
  CHECK(cudaMalloc(&bad0, sizeof(int)));
  CHECK(cudaMemcpy(d0, host, n * sizeof(int), cudaMemcpyHostToDevice));
  CHECK(cudaMemset(c0, 0, sizeof(unsigned)));
  CHECK(cudaMemset(bad0, 0, sizeof(int)));

  long long* s1 = nullptr;
  long long sum = -1;
  CHECK(cudaSetDevice(1));
  CHECK(cudaMalloc(&s1, sizeof(long long)));
  if (!std::strcmp(mode, "direct") || !std::strcmp(mode, "disabled")) {
    if (!can) {
      std::printf("SKIP: no peer path\n");
      return 0;
    }
    CHECK(cudaDeviceEnablePeerAccess(0, 0));
    if (!std::strcmp(mode, "disabled")) CHECK(cudaDeviceDisablePeerAccess(0));
  }
  sum_peer<<<1, 1>>>(d0, n, s1);
  CHECK(cudaGetLastError());
  const cudaError_t e = cudaDeviceSynchronize();
  std::printf("the read of device 0's buffer from device 1: %s\n", cudaGetErrorName(e));
  if (e != cudaSuccess) return 0;
  CHECK(cudaMemcpy(&sum, s1, sizeof sum, cudaMemcpyDeviceToHost));
  std::printf("sum = %lld\n", sum);

  write_peer<<<(n + 127) / 128, 128>>>(d0, n, c0);
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  unsigned c = 0;
  CHECK(cudaMemcpy(&c, c0, sizeof c, cudaMemcpyDeviceToHost));
  std::printf("atomic adds from device 1 into device 0's counter = %u\n", c);

  CHECK(cudaSetDevice(0));
  check_own<<<(n + 127) / 128, 128>>>(d0, n, bad0);
  CHECK(cudaDeviceSynchronize());
  int bad = -1;
  CHECK(cudaMemcpy(&bad, bad0, sizeof bad, cudaMemcpyDeviceToHost));
  std::printf("words device 0 reads back wrong = %d\n", bad);

  CHECK(cudaFree(d0));
  CHECK(cudaFree(c0));
  CHECK(cudaFree(bad0));
  CHECK(cudaSetDevice(1));
  CHECK(cudaFree(s1));
  return 0;
}
