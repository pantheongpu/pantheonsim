// Barriers inside device functions the compiler does not inline: every -G
// (debug) build, and any function marked __noinline__. A warp stops in the
// middle of the callee at the barrier and the other warps run until they get
// there too -- the shape of a CUB-style block reduction, a __syncthreads_count
// vote, a recursive function that synchronises at each level, and a named
// barrier only some warps take.
//
// Built twice by run_device_function_barriers.sh: with __noinline__ at -O3,
// and with -G, where nothing is inlined at all.
#include <cstdio>
#include <cuda_runtime.h>

constexpr int kThreads = 256;

// A tree reduction in shared memory, synchronising between levels.
__device__ __noinline__ int block_sum(int v, int* scratch) {
  const int t = threadIdx.x;
  scratch[t] = v;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (t < stride) scratch[t] += scratch[t + stride];
    __syncthreads();
  }
  const int total = scratch[0];
  __syncthreads();   // scratch is reused by the caller's next call
  return total;
}

// A block-wide vote, one more call level down.
__device__ __noinline__ int count_true(bool p) { return __syncthreads_count(p); }
__device__ __noinline__ int count_multiples(int k) { return count_true(threadIdx.x % k == 0); }

// Recursion that synchronises at every level: level n writes, syncs, and
// reads what a thread in another warp wrote.
__device__ __noinline__ int ladder(int n, int* scratch) {
  if (n == 0) return 0;
  const int t = threadIdx.x;
  scratch[t] = n * 1000 + t;
  __syncthreads();
  const int partner = scratch[(t + 32 * n) % blockDim.x];
  __syncthreads();
  return partner + ladder(n - 1, scratch);
}

// A named barrier for two warps only, inside a helper only they call: the
// other warps never arrive, and must not be needed.
__device__ __noinline__ int pair_exchange(int* scratch) {
  const int t = threadIdx.x;
  scratch[t] = t + 7;
  asm volatile("bar.sync 1, 64;" ::: "memory");
  return scratch[63 - t];
}

// A barrier two calls down, reached through a function that does nothing else.
__device__ __noinline__ void inner_sync() { __syncthreads(); }
__device__ __noinline__ void outer_sync() { inner_sync(); }

__global__ void kernel(int* out) {
  __shared__ int scratch[kThreads];
  const int t = threadIdx.x;
  int* o = out + t * 6;

  o[0] = block_sum(t, scratch);
  o[1] = block_sum(t % 7, scratch);
  o[2] = count_multiples(3);
  o[3] = ladder(3, scratch);

  // A value written by another warp before the helper's barrier is visible
  // after it.
  scratch[t] = t * 3;
  outer_sync();
  o[4] = scratch[kThreads - 1 - t];
  __syncthreads();

  o[5] = t < 64 ? pair_exchange(scratch) : -1;
}

int main() {
  int* d;
  cudaMalloc(&d, kThreads * 6 * sizeof(int));
  kernel<<<2, kThreads>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    std::printf("kernel failed: %s\n", cudaGetErrorString(e));
    return 1;
  }
  int h[kThreads * 6];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  int sum7 = 0;
  for (int t = 0; t < kThreads; ++t) sum7 += t % 7;
  int bad = 0;
  auto check = [&](int t, int slot, int want, const char* what) {
    const int got = h[t * 6 + slot];
    if (got != want && bad++ < 20) std::printf("thread %d, %s: got %d, want %d\n", t, what, got, want);
  };
  for (int t = 0; t < kThreads; ++t) {
    check(t, 0, kThreads * (kThreads - 1) / 2, "block_sum(t)");
    check(t, 1, sum7, "block_sum(t % 7)");
    check(t, 2, (kThreads + 2) / 3, "__syncthreads_count two calls deep");
    int want = 0;
    for (int n = 3; n > 0; --n) want += n * 1000 + (t + 32 * n) % kThreads;
    check(t, 3, want, "recursive ladder");
    check(t, 4, (kThreads - 1 - t) * 3, "a barrier two calls down");
    check(t, 5, t < 64 ? 63 - t + 7 : -1, "named barrier for two warps in a helper");
  }
  if (bad) {
    std::printf("%d mismatches\nFAIL\n", bad);
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
