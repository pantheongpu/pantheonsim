// A CTA pair's Tensor Memory allocation (tcgen05.alloc/dealloc.cta_group::2),
// as Blackwell's 2-SM GEMMs make it: both CTAs of a cluster of two allocate
// together, store to the columns they were given and read them back, then
// free them and give up the permit -- once, as a GEMM kernel does. ptxas
// turns the paired allocation into a handshake in the driver's reserved
// shared memory -- the leader claims the columns and signals its peer, the
// peer acknowledges -- whose barriers must start as the driver leaves them;
// started wrong, either CTA waits for ever. Each CTA prints the column it got
// and what came back, so the SASS run and the PTX run must print the same.
#include <cstdio>
#include <cuda_runtime.h>

#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM101_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
#define HAVE_TCGEN05 1
#endif

constexpr int kRounds = 1;

__global__ void __cluster_dims__(2, 1, 1) pair_alloc(unsigned* out) {
#ifdef HAVE_TCGEN05
  __shared__ __align__(16) unsigned taddr;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  const unsigned slot = static_cast<unsigned>(__cvta_generic_to_shared(&taddr));
  for (int r = 0; r < kRounds; ++r) {
    if (warp == 0)
      asm volatile("tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [%0], %1;" ::"r"(slot), "r"(64u) : "memory");
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    asm volatile("tcgen05.fence::after_thread_sync;" ::: "memory");
    const unsigned base = taddr;
    if (warp == 0) {
      // Lane i's row of column 0: a value that names the round, the CTA and the lane.
      const unsigned v = 1000u * (r + 1) + 100u * rank + lane;
      asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0], {%1};" ::"r"(base), "r"(v) : "memory");
      asm volatile("tcgen05.wait::st.sync.aligned;" ::: "memory");
      unsigned back;
      asm volatile("tcgen05.ld.sync.aligned.32x32b.x1.b32 {%0}, [%1];" : "=r"(back) : "r"(base) : "memory");
      asm volatile("tcgen05.wait::ld.sync.aligned;" ::: "memory");
      out[(rank * kRounds + r) * 33 + lane] = back;
      if (lane == 0) out[(rank * kRounds + r) * 33 + 32] = base & 0xffff;   // the column
    }
    asm volatile("tcgen05.fence::before_thread_sync;" ::: "memory");
    __syncthreads();
    if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::2.sync.aligned.b32 %0, %1;" ::"r"(base), "r"(64u) : "memory");
  }
  if (warp == 0) asm volatile("tcgen05.relinquish_alloc_permit.cta_group::2.sync.aligned;" ::: "memory");
#else
  (void)out;
#endif
}

int main() {
  unsigned* out = nullptr;
  const int n = 2 * kRounds * 33;
  cudaMalloc(&out, n * sizeof(unsigned));
  cudaMemset(out, 0, n * sizeof(unsigned));
  pair_alloc<<<2, 128>>>(out);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); return 1; }
  unsigned h[n];
  cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (unsigned rank = 0; rank < 2; ++rank)
    for (int r = 0; r < kRounds; ++r) {
      const unsigned* row = h + (rank * kRounds + r) * 33;
      for (unsigned lane = 0; lane < 32; ++lane) bad += row[lane] != 1000u * (r + 1) + 100u * rank + lane;
      std::printf("CTA %u round %d: column %u, lane 31 read %u\n", rank, r, row[32], row[31]);
    }
  std::printf("%s: %d values wrong\n", bad ? "FAIL" : "ok", bad);
  return bad ? 1 : 0;
}
