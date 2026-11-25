// Lanes that leave a data-dependent loop early and then read __activemask(): on a card the warp reconverges at the end of the loop (ptxas puts a
// convergence barrier there), so every lane reads the whole warp's mask, and a warp reduction that takes its mask from __activemask() counts all
// 32 lanes. The simulator ran the lanes that had left first once they had waited some thousand instructions behind the lanes still looping (its
// guard against a lane spinning on another), so a loop of a few thousand instructions per lane handed them a partial mask and the reduction
// merged only part of the warp (#759: QuantCUDA's barrier-option kernel). Each lane's loop runs from 100 to over 3000 iterations; the total is
// checked against the host's, and no lane may read a partial mask.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void loop_then_reduce(const unsigned* iterations, unsigned* sums, unsigned* partial_masks) {
  const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned n = iterations[tid];
  unsigned x = tid + 1, acc = 0;
  for (unsigned j = 0;; ++j) {
    x = x * 1664525u + 1013904223u;
    acc ^= x >> 7;
    if (j >= n) break;   // each lane leaves at its own iteration
  }
  const unsigned mask = __activemask();   // the whole warp, as on a card
  if (mask != 0xffffffffu) atomicAdd(partial_masks, 1u);
  const int lane = threadIdx.x & 31;
  const int active = __popc(mask);
  unsigned v = acc & 0xff;
  for (int off = 16; off > 0; off >>= 1) {
    const unsigned o = __shfl_down_sync(mask, v, off);
    if (lane + off < active) v += o;
  }
  if (lane == 0) sums[tid >> 5] = v;
}

int main() {
  const unsigned blocks = 2, threads = 128, n = blocks * threads;
  static unsigned iterations[n];
  for (unsigned i = 0; i < n; ++i) iterations[i] = 100 + 97 * (i % 32) + 13 * (i / 32);
  unsigned *d_it, *d_sums, *d_partial;
  cudaMalloc(&d_it, sizeof iterations);
  cudaMalloc(&d_sums, (n / 32) * sizeof(unsigned));
  cudaMalloc(&d_partial, sizeof(unsigned));
  cudaMemcpy(d_it, iterations, sizeof iterations, cudaMemcpyHostToDevice);
  cudaMemset(d_sums, 0, (n / 32) * sizeof(unsigned));
  cudaMemset(d_partial, 0, sizeof(unsigned));
  loop_then_reduce<<<blocks, threads>>>(d_it, d_sums, d_partial);
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("FAIL: %s\n", cudaGetErrorString(e));
    return 1;
  }
  unsigned sums[n / 32], partial = 0;
  cudaMemcpy(sums, d_sums, sizeof sums, cudaMemcpyDeviceToHost);
  cudaMemcpy(&partial, d_partial, sizeof partial, cudaMemcpyDeviceToHost);
  unsigned bad = partial;
  unsigned long long total = 0;
  for (unsigned w = 0; w < n / 32; ++w) {
    unsigned want = 0;
    for (unsigned l = 0; l < 32; ++l) {
      const unsigned tid = w * 32 + l;
      unsigned x = tid + 1, acc = 0;
      for (unsigned j = 0;; ++j) {
        x = x * 1664525u + 1013904223u;
        acc ^= x >> 7;
        if (j >= iterations[tid]) break;
      }
      want += acc & 0xff;
    }
    bad += sums[w] != want;
    total += sums[w];
  }
  std::printf("%s: %u wrong (%u lanes read a partial mask), total %llu\n", bad ? "FAIL" : "ok", bad, partial, total);
  cudaFree(d_it);
  cudaFree(d_sums);
  cudaFree(d_partial);
  return bad ? 1 : 0;
}
