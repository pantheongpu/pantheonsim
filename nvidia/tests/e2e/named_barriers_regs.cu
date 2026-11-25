// named_barriers.cu with the barrier numbers and thread counts read at run time, so that ptxas writes BAR with registers: `bar.sync %a, %b`
// (one register holding the barrier in its low four bits and the count from bit 16), `bar.sync %a, 64` and `bar.sync 1, %b`, and the same for
// bar.arrive. cub's DeviceScan from CUDA 13.5 does this on sm_120 (#758: "unknown opcode 0x31d" in the SASS decoder). Two producer warps
// synchronise among themselves, then hand their data to two consumer warps without waiting, which wait for it and read it; a count or a barrier
// number read wrongly leaves warps waiting for ever, or lets the consumers read before the producers wrote.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void handoff(int* out, int rounds, unsigned prod_bar, unsigned prod_cnt, unsigned data_bar, unsigned cons_cnt, unsigned done_bar) {
  __shared__ int data[64];
  const int warp = threadIdx.x / 32;
  for (int r = 0; r < rounds; ++r) {
    if (warp < 2) {
      data[threadIdx.x] = (threadIdx.x + 1) * (r + 3);
      if (r & 1) asm volatile("bar.sync %0, %1;" :: "r"(prod_bar), "r"(prod_cnt) : "memory");   // both in registers
      else asm volatile("bar.sync 1, %0;" :: "r"(prod_cnt) : "memory");                          // the count in a register
      asm volatile("bar.arrive %0, %1;" :: "r"(data_bar), "r"(cons_cnt) : "memory");            // to the consumers, without waiting
      asm volatile("bar.sync %0, 128;" :: "r"(done_bar) : "memory");                            // until the consumers have read it
    } else {
      asm volatile("bar.sync %0, %1;" :: "r"(data_bar), "r"(cons_cnt) : "memory");              // the producers' data is in
      const int i = threadIdx.x - 64;
      out[r * 64 + i] = data[63 - i] + i;
      asm volatile("bar.arrive 3, %0;" :: "r"(cons_cnt) : "memory");                            // done with it
    }
  }
}

int main() {
  const int rounds = 4;
  int* out = nullptr;
  cudaMalloc(&out, rounds * 64 * sizeof(int));
  handoff<<<2, 128>>>(out, rounds, 1, 64, 2, 128, 3);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) { std::printf("FAIL: %s\n", cudaGetErrorString(e)); cudaFree(out); return 1; }
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
