// A spin lock taken by every thread of a warp, with the first CAS won by lane 0.
// ptxas compiles the loop to
//   BSSY B0 ; YIELD ; ATOMS.CAS ; @P1 BRA loop ; BSYNC B0 ; (critical section) ; ATOMS.EXCH
// so the lane that wins stands at the BSYNC closing the loop and waits for the
// lanes that are spinning, which spin because it holds the lock. An RTX 3080 Ti
// runs it: NVIDIA's scheduler guarantees forward progress to threads in a YIELD
// loop. The simulator ran the spinners (the group at the lowest address) for
// ever, in two ways: it never gave a turn to the winner when the winner was the
// lowest-numbered lane, and once the winner did reach the BSYNC it waited there.
// This is the shape of RXMesh's patch locks (RXMeshDynamic tests), which ran
// for a billion instructions and then stopped. Every count is computed on the
// host and the program passes on the card.
#include <cstdio>
#include <cuda_runtime.h>
// A shared-memory spin lock taken by every thread of a block.
__global__ void k(int* out, int rounds) {
  __shared__ int lock;
  __shared__ int counter;
  if (threadIdx.x == 0) { lock = 0; counter = 0; }
  __syncthreads();
  for (int r = 0; r < rounds; ++r) {
    while (atomicCAS(&lock, 0, 1) != 0) {}      // lane 0 of a warp wins the first try
    int c = *(volatile int*)&counter;           // the critical section
    *(volatile int*)&counter = c + 1;
    __threadfence_block();
    atomicExch(&lock, 0);
  }
  __syncthreads();
  if (threadIdx.x == 0) out[blockIdx.x] = counter;
}
int main() {
  int bad = 0;
  for (int threads : {32, 33, 96, 128, 256}) {
    int* d; cudaMalloc(&d, 8 * sizeof(int)); cudaMemset(d, 0xff, 8 * sizeof(int));
    k<<<2, threads>>>(d, 3);
    cudaError_t e = cudaDeviceSynchronize();
    int h[2]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    const bool ok = e == cudaSuccess && h[0] == threads * 3 && h[1] == threads * 3;
    printf("threads=%d err=%d counters=%d,%d want %d %s\n", threads, (int)e, h[0], h[1], threads * 3, ok ? "ok" : "BAD");
    bad += !ok;
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
