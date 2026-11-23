// Version-matrix check for HIP: shared memory, a block barrier, atomics, a stream and an event, with answers known
// without a GPU. Built with the matrix leg's hipcc for the simulated GPU's gfx name and run with `vgpu run`.
#include <hip/hip_runtime.h>

#include <cstdio>
#include <vector>

#define CK(call)                                                                                   \
  do {                                                                                             \
    if ((call) != hipSuccess) { std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #call); return 1; } \
  } while (0)

__global__ void block_sum(const int* in, int* total, int n) {
  __shared__ int part[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  part[threadIdx.x] = i < n ? in[i] : 0;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(total, part[0]);
}

int main() {
  int rt = 0;
  CK(hipRuntimeGetVersion(&rt));
  std::printf("hip runtime %d\n", rt);
  const int n = 10000;
  std::vector<int> h(n);
  long want = 0;
  for (int i = 0; i < n; ++i) { h[i] = i % 17; want += h[i]; }
  int *d, *t;
  CK(hipMalloc(&d, n * sizeof(int)));
  CK(hipMalloc(&t, sizeof(int)));
  hipStream_t s;
  CK(hipStreamCreate(&s));
  hipEvent_t e;
  CK(hipEventCreate(&e));
  CK(hipMemcpyAsync(d, h.data(), n * sizeof(int), hipMemcpyHostToDevice, s));
  CK(hipMemsetAsync(t, 0, sizeof(int), s));
  block_sum<<<(n + 255) / 256, 256, 0, s>>>(d, t, n);
  CK(hipEventRecord(e, s));
  CK(hipEventSynchronize(e));
  int got = 0;
  CK(hipMemcpy(&got, t, sizeof got, hipMemcpyDeviceToHost));
  if (got != want) { std::printf("FAIL sum %d want %ld\n", got, want); return 1; }
  std::printf("PASS\n");
  return 0;
}
