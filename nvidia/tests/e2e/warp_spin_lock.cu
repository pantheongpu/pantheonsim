// Lanes of one warp contending for a spin lock.
//
// Since Volta every thread makes progress on its own, so the classic lock --
// spin on atomicCAS, do the work, release -- works even when the lanes that
// contend for it share a warp: the lane holding the lock gets to run while its
// warp-mates spin. The simulator ran a warp's divergent paths lowest pc first,
// and the spinning lanes' path (back at the top of the loop) always came
// first, so the holder never reached its release: stdgpu's hash-map inserts
// hung this way.
#include <cstdio>

__device__ int lock_word = 0;
__device__ unsigned counter = 0;

__global__ void locked_increment(int rounds) {
  for (int r = 0; r < rounds; ++r) {
    while (atomicCAS(&lock_word, 0, 1) != 0) {
    }
    __threadfence();
    const unsigned c = *(volatile unsigned*)&counter;
    *(volatile unsigned*)&counter = c + 1;   // a plain read-modify-write: the lock is what makes it safe
    __threadfence();
    atomicExch(&lock_word, 0);
  }
}

int main() {
  // Two warps per block and two blocks contend within a warp and across both,
  // which is the whole property; every hand-over inside a warp waits out the
  // scheduler's starvation limit, so more rounds only cost time (four blocks
  // of three rounds took 230 s of a 300 s limit under AddressSanitizer).
  const int blocks = 2, threads = 64, rounds = 1;
  locked_increment<<<blocks, threads>>>(rounds);
  unsigned h = 0;
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpyFromSymbol(&h, counter, sizeof h) != cudaSuccess) {
    std::printf("FAIL %s\n", cudaGetErrorString(cudaGetLastError()));
    return 1;
  }
  const unsigned want = blocks * threads * rounds;
  if (h != want) std::printf("FAIL counter %u, want %u\n", h, want);
  std::printf(h == want ? "PASS\n" : "FAILED\n");
  return h == want ? 0 : 1;
}
