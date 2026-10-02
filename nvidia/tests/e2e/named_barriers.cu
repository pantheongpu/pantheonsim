// Named barriers with a thread count, the way CUTLASS's warp-specialised
// kernels use them: two producer warps synchronise among themselves
// (bar.sync 1, 64), then hand their data to two consumer warps without
// waiting (bar.arrive 2, 128), which wait for it (bar.sync 2, 128) and read it.
// A counted barrier completes when that many threads have arrived, not when
// every thread of the block has -- held to the whole block, the producers and
// the consumers each wait at their own barrier for the others, forever.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void handoff(int* out, int rounds) {
  __shared__ int data[64];
  const int warp = threadIdx.x / 32;
  for (int r = 0; r < rounds; ++r) {
    if (warp < 2) {
      data[threadIdx.x] = (threadIdx.x + 1) * (r + 3);
      asm volatile("bar.sync 1, 64;" ::: "memory");     // the producers only
      asm volatile("bar.arrive 2, 128;" ::: "memory");  // to the consumers, without waiting
      asm volatile("bar.sync 3, 128;" ::: "memory");    // until the consumers have read it
    } else {
      asm volatile("bar.sync 2, 128;" ::: "memory");    // the producers' data is in
      const int i = threadIdx.x - 64;
      out[r * 64 + i] = data[63 - i] + i;
      asm volatile("bar.arrive 3, 128;" ::: "memory");  // done with it
    }
  }
}

int main() {
  const int rounds = 4;
  int* out = nullptr;
  cudaMalloc(&out, rounds * 64 * sizeof(int));
  handoff<<<2, 128>>>(out, rounds);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  int host[rounds * 64];
  cudaMemcpy(host, out, sizeof host, cudaMemcpyDeviceToHost);
  int bad = 0;
  long long sum = 0;
  for (int r = 0; r < rounds; ++r)
    for (int i = 0; i < 64; ++i) {
      const int want = (63 - i + 1) * (r + 3) + i;
      bad += host[r * 64 + i] != want;
      sum += host[r * 64 + i];
    }
  std::printf("%s: %d of %d values wrong, sum %lld\n", bad ? "FAIL" : "ok", bad, rounds * 64, sum);
  cudaFree(out);
  return bad ? 1 : 0;
}
