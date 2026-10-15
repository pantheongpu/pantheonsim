// Spin locks taken at several call sites by lanes of the same warps: the shape of RXMesh's
// cavity-graph mutexes (RandomCollapse and PatchSlicing in its RXMeshDynamic tests). Each site is its
// own inlined copy of `while (atomicCAS(...) != 0) {}`, so a warp holds lane groups spinning at three
// different code addresses, lanes of one warp contend for the same lock, and the lane that holds a
// lock can stand at the BSYNC that closes its spin loop, waiting for a lane that spins because it
// holds the lock. An RTX 3060 runs it: NVIDIA's scheduler guarantees forward progress to threads in
// a YIELD loop.
//
// Two things in the simulator's scheduler made this run for ever:
//  - When a YIELD loop gave way it let in the next group above the LOWEST one. A group let in that
//    yielded in turn was followed by itself again, so with two groups spinning, a third that held
//    the lock they wait for never got a turn. Now the next group is the one after the group that
//    yielded, wrapping round.
//  - The release of lanes parked at a BSYNC (see spin_lock_cross_warp.cu) waits until every warp is
//    spinning, and a warp with several groups taking turns never counted as spinning: its YIELDs
//    always had another group to let in. Now a warp that has yielded 64 times without writing
//    memory counts as spinning too.
// Every count is computed on the host and the program passes on the card.
#include <cstdio>
#include <cuda_runtime.h>
constexpr int kLocks = 3;
__global__ void k(int* out, int rounds) {
  __shared__ int lock[kLocks];
  __shared__ int count[3][kLocks];
  if (threadIdx.x < kLocks) {
    lock[threadIdx.x] = 0;
    count[0][threadIdx.x] = count[1][threadIdx.x] = count[2][threadIdx.x] = 0;
  }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int r = 0; r < rounds; ++r) {
    const int site = (lane * 5 + warp + r) % 3, which = (lane + 2 * warp + 3 * r) % kLocks;
    int* m = &lock[which];
    if (site == 0) {
      while (atomicCAS(m, 0, 1) != 0) {}
      *(volatile int*)&count[0][which] += 1;
      __threadfence_block();
      atomicExch(m, 0);
    } else if (site == 1) {
      while (atomicCAS(m, 0, 1) != 0) {}
      *(volatile int*)&count[1][which] += 2;
      __threadfence_block();
      atomicExch(m, 0);
    } else {
      while (atomicCAS(m, 0, 1) != 0) {}
      *(volatile int*)&count[2][which] += 3;
      __threadfence_block();
      atomicExch(m, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0)
    for (int s = 0; s < 3; ++s)
      for (int i = 0; i < kLocks; ++i) out[blockIdx.x * 9 + s * 3 + i] = count[s][i];
}

// One lock for the whole block, three copies of the spin loop: a warp has lanes spinning at several
// addresses while the lane that holds the lock stands at the BSYNC closing its own loop.
__global__ void k2(int* out, int rounds) {
  __shared__ int lock, count[3];
  if (threadIdx.x == 0) { lock = 0; count[0] = count[1] = count[2] = 0; }
  __syncthreads();
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int r = 0; r < rounds; ++r) {
    const int site = (lane * 7 + warp + r) % 3;
    if (site == 0) {
      while (atomicCAS(&lock, 0, 1) != 0) {}
      *(volatile int*)&count[0] += 1;
      __threadfence_block();
      atomicExch(&lock, 0);
    } else if (site == 1) {
      while (atomicCAS(&lock, 0, 1) != 0) {}
      *(volatile int*)&count[1] += 2;
      __threadfence_block();
      atomicExch(&lock, 0);
    } else {
      while (atomicCAS(&lock, 0, 1) != 0) {}
      *(volatile int*)&count[2] += 3;
      __threadfence_block();
      atomicExch(&lock, 0);
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) for (int s = 0; s < 3; ++s) out[blockIdx.x * 3 + s] = count[s];
}
int main() {
  int bad = 0;
  for (int threads : {64, 96, 128, 256}) {
    const int rounds = 6, blocks = 2;
    int* d; cudaMalloc(&d, blocks * 9 * sizeof(int)); cudaMemset(d, 0, blocks * 9 * sizeof(int));
    k<<<blocks, threads>>>(d, rounds);
    cudaError_t e = cudaDeviceSynchronize();
    int h[18]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int want[9] = {};
    for (int t = 0; t < threads; ++t)
      for (int r = 0; r < rounds; ++r) {
        const int lane = t & 31, warp = t >> 5;
        const int site = (lane * 5 + warp + r) % 3, which = (lane + 2 * warp + 3 * r) % kLocks;
        want[site * 3 + which] += site + 1;
      }
    bool ok = e == cudaSuccess;
    for (int b = 0; b < blocks; ++b)
      for (int i = 0; i < 9; ++i) ok = ok && h[b * 9 + i] == want[i];
    printf("threads=%d err=%d counts=%d,%d,%d/%d,%d,%d/%d,%d,%d want %d,%d,%d/%d,%d,%d/%d,%d,%d %s\n", threads, (int)e, h[0], h[1], h[2],
           h[3], h[4], h[5], h[6], h[7], h[8], want[0], want[1], want[2], want[3], want[4], want[5], want[6], want[7], want[8], ok ? "ok" : "BAD");
    bad += !ok;
    cudaFree(d);
  }
  for (int threads : {64, 96, 128, 256}) {
    const int rounds = 6, blocks = 2;
    int* d; cudaMalloc(&d, blocks * 3 * sizeof(int)); cudaMemset(d, 0, blocks * 3 * sizeof(int));
    k2<<<blocks, threads>>>(d, rounds);
    cudaError_t e = cudaDeviceSynchronize();
    int h[6]; cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
    int want[3] = {};
    for (int t = 0; t < threads; ++t)
      for (int r = 0; r < rounds; ++r) want[(((t & 31) * 7 + (t >> 5) + r) % 3)] += ((((t & 31) * 7 + (t >> 5) + r) % 3) + 1);
    bool ok = e == cudaSuccess;
    for (int b = 0; b < blocks; ++b)
      for (int i = 0; i < 3; ++i) ok = ok && h[b * 3 + i] == want[i];
    printf("one lock, threads=%d err=%d counts=%d,%d,%d/%d,%d,%d want %d,%d,%d %s\n", threads, (int)e, h[0], h[1], h[2], h[3], h[4], h[5], want[0], want[1], want[2],
           ok ? "ok" : "BAD");
    bad += !ok;
    cudaFree(d);
  }
  printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
