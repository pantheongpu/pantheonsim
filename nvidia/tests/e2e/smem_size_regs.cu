// The shared-memory size registers: %dynamic_smem_size, %total_smem_size and,
// from sm_90, %aggr_smem_size, for a block with 400 static bytes and 1024
// dynamic ones. run_sass_archs.sh runs it on each generation's SASS and on its
// PTX, which must agree: ptxas reads the first from bank 0, the second as
// SR_SMEMSZ less the reserved size, the third as SR_SMEMSZ clamped.
//
// The expected values are the RTX 3060's for the first two: the dynamic
// bytes, and the block's own shared memory in allocation units (128 bytes
// from compute capability 8.0, 256 before). The third adds the driver's
// reserved shared memory (cudaDevAttrReservedSharedMemoryPerBlock), as the
// PTX ISA defines it; no sm_90 card was at hand to measure it.
#include <cstdio>

__global__ void sizes(unsigned* out) {
  __shared__ unsigned tile[100];
  extern __shared__ unsigned dyn[];
  unsigned d, t, a = 0, have_aggr = 0;
  asm volatile("mov.u32 %0, %%dynamic_smem_size;" : "=r"(d));
  asm volatile("mov.u32 %0, %%total_smem_size;" : "=r"(t));
  // PTX ISA 8.1 (CUDA 12.1) and sm_90.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900 && \
    (__CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 1))
  asm volatile("mov.u32 %0, %%aggr_smem_size;" : "=r"(a));
  have_aggr = 1;
#endif
  tile[threadIdx.x] = d;
  dyn[threadIdx.x] = t;
  __syncthreads();
  if (threadIdx.x == 0) {
    out[0] = d;
    out[1] = t;
    out[2] = a;
    out[3] = have_aggr;
    out[4] = tile[1] + dyn[1];
  }
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  unsigned* d_out;
  cudaMalloc(&d_out, 5 * sizeof(unsigned));
  sizes<<<1, 32, 1024>>>(d_out);
  unsigned out[5] = {};
  cudaMemcpy(out, d_out, sizeof out, cudaMemcpyDeviceToHost);
  const unsigned unit = prop.major >= 8 ? 128 : 256;
  const unsigned total = (400 + 1024 + unit - 1) / unit * unit;
  const unsigned reserved = static_cast<unsigned>(prop.reservedSharedMemPerBlock);
  int bad = cudaGetLastError() != cudaSuccess;
  std::printf("dynamic %u total %u", out[0], out[1]);
  if (out[3]) std::printf(" aggr %u", out[2]);
  std::printf("\n");
  bad += out[0] != 1024;
  bad += out[1] != total;
  bad += out[3] && out[2] != total + reserved;
  bad += out[4] != 1024 + total;
  std::printf(bad ? "FAIL\n" : "PASS\n");
  return bad != 0;
}
