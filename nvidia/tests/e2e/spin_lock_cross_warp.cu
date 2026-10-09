// A block-wide shared-memory spin lock taken lane by lane inside a divergent region, with a search loop
// that leaves early inside the critical section: the shape of RXMesh's slice_patches (RXMeshDynamic
// tests). The warps of the block spin on one lock, and the lane that holds it stands at the BSYNC that
// closes its own spin loop, waiting for the lanes that spin because it holds the lock. An RTX 3060 runs
// it: NVIDIA's scheduler guarantees forward progress to threads in a YIELD loop.
//
// The simulator lets the lanes parked at a BSYNC go on once every warp is spinning or parked. The first
// version of that decided per warp and released every parked lane of a warp whose lanes were spinning,
// even when the lock was held by a lane of another warp. The warp's lanes parked at the outer BSYNC then
// ran on, set up the same barrier registers again for the next round, and the barriers each waited for
// lanes parked at the other: this program ran for ever. Now the release is decided over the whole block
// and goes to the lanes parked at the most deeply nested barrier, where a lock holder stands. Every count
// is computed on the host and the program passes on the card.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void k(int* out, int rounds) {
  __shared__ int lock, counter;
  __shared__ int tab[8];
  if (threadIdx.x == 0) { lock = 0; counter = 0; }
  if (threadIdx.x < 8) tab[threadIdx.x] = threadIdx.x * 3;
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  int mine = 0;
  for (int r = 0; r < rounds; ++r) {
    if ((lane + warp + r) % 3 != 0) {           // a divergent region: two lanes in three take the lock
      while (atomicCAS(&lock, 0, 1) != 0) {}
      int found = 0;
      for (int i = 0; i < 8; ++i) {             // the search leaves early, inside the critical section
        if (*(volatile int*)&tab[i] == ((lane + r) % 8) * 3) { found = i + 1; break; }
      }
      int c = *(volatile int*)&counter;
      *(volatile int*)&counter = c + (found ? 1 : 0);
      __threadfence_block();
      atomicExch(&lock, 0);
      mine += found ? 1 : 0;
    }
  }
  __syncthreads();
  atomicAdd(&out[blockIdx.x], mine);
  if (threadIdx.x == 0) out[2 + blockIdx.x] = counter;
}
int main() {
  int bad = 0;
  for (int threads : {64, 96, 128, 256}) {
    int* d; cudaMalloc(&d, 4 * sizeof(int)); cudaMemset(d, 0, 4 * sizeof(int));
    k<<<2, threads>>>(d, 4);
    cudaError_t e = cudaDeviceSynchronize();
    int h[4]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int want = 0;
    for (int t = 0; t < threads; ++t)
      for (int r = 0; r < 4; ++r) want += ((t & 31) + (t >> 5) + r) % 3 != 0;
    const bool ok = e == cudaSuccess && h[0] == want && h[1] == want && h[2] == want && h[3] == want;
    printf("threads=%d err=%d sums=%d,%d counters=%d,%d want %d %s\n", threads, (int)e, h[0], h[1], h[2], h[3], want, ok ? "ok" : "BAD");
    bad += !ok;
    cudaFree(d);
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
