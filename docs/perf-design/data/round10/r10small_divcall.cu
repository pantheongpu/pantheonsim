// Round 10 "small": a divergent warp that alternates between the kernel's section and a device function's (two-entry code hint).
// The functions are __noinline__ so they are separate code sections; lanes take different ones per iteration.
#include <cstdio>
#include <cstdint>
#include <vector>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("FAIL %s\n", cudaGetErrorString(e)); return 1; } } while (0)
__device__ __noinline__ uint32_t fa(uint32_t v) { return v * 2654435761u + 12345u; }
__global__ void k(uint32_t* out, int iters) {
  uint32_t i = blockIdx.x * blockDim.x + threadIdx.x, v = i;
  for (int k = 0; k < iters; ++k) {
    if ((v >> 3) & 1u) v = fa(v);    // some lanes are in the device function while the others run the kernel's own code
    else v = (v ^ (v << 5)) + 1u;
    v += k;
  }
  out[i] = v;
}
static uint32_t ha(uint32_t v) { return v * 2654435761u + 12345u; }
int main() {
  const int blocks = 8, threads = 128, n = blocks * threads, iters = 20000;
  uint32_t* d;
  CK(cudaMalloc(&d, n * 4));
  k<<<blocks, threads>>>(d, iters);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<uint32_t> o(n);
  CK(cudaMemcpy(o.data(), d, n * 4, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n; ++i) {
    uint32_t v = i;
    for (int k = 0; k < iters; ++k) { v = ((v >> 3) & 1u) ? ha(v) : (v ^ (v << 5)) + 1u; v += k; }
    if (v != o[i]) { std::printf("FAIL %d\n", i); return 1; }
  }
  std::printf("PASS\n");
  return 0;
}
