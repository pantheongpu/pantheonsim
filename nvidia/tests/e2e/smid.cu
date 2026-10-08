// %smid, which SASS reads as SR_VIRTUALSMID: distinct among the blocks resident
// at once, as on the hardware, because kernels keep per-SM state indexed by
// it -- CUTLASS's grouped GEMMs a tensor map per SM, which two resident CTAs
// sharing an SM number overwrite for each other. A grid that fits the device
// takes one SM a block; every block claims its SM's slot with an atomic and
// reports a clash. Read as 0, every block after the first clashed.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void claim(unsigned* owner, unsigned* clashes, unsigned* range) {
  if (threadIdx.x != 0) return;
  unsigned sm, nsm;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(sm));
  asm volatile("mov.u32 %0, %%nsmid;" : "=r"(nsm));
  if (sm >= nsm) atomicAdd(range, 1u);
  else if (atomicCAS(&owner[sm], 0xffffffffu, blockIdx.x) != 0xffffffffu) atomicAdd(clashes, 1u);
}

int main() {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  unsigned *owner, *counts;
  cudaMalloc(&owner, sms * sizeof(unsigned));
  cudaMalloc(&counts, 2 * sizeof(unsigned));
  cudaMemset(owner, 0xff, sms * sizeof(unsigned));
  cudaMemset(counts, 0, 2 * sizeof(unsigned));
  claim<<<sms, 32>>>(owner, counts, counts + 1);   // one block an SM: all resident at once
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL: launch\n"); return 1; }
  unsigned c[2];
  cudaMemcpy(c, counts, sizeof c, cudaMemcpyDeviceToHost);
  std::printf("%s: %d blocks, %u shared an SM number, %u outside %%nsmid\n", c[0] || c[1] ? "FAIL" : "ok", sms, c[0], c[1]);
  return c[0] || c[1] ? 1 : 0;
}
