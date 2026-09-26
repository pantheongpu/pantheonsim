// Blocks that wait on each other through a semaphore in global memory, the
// way CUTLASS's serial split-K does: slice z of a tile waits until slice z-1
// has added its part and released the semaphore. The wait is CUTLASS's own
// shape -- thread 0 re-reads the semaphore with an acquire load while the
// block polls it with __syncthreads_and -- which nvcc compiles to a bar.red
// inside a brace block with its own %p1/%p2, and a loop in which thread 0's
// fetch sits at a higher pc than the barrier its warp-mates return to.
// Run with the grid spread over several host threads, so the waiting blocks
// really do spin while the ones they wait for run. Prints PASS on the last line.
#include <cstdio>
#include <vector>

constexpr int kTiles = 8, kSlices = 4, kThreads = 128;

__device__ int fetch(const int* p) {
  int v;
  asm volatile("ld.global.acquire.gpu.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__global__ void split_k(int* sem, int* out) {
  const int tile = blockIdx.x, slice = blockIdx.z;
  // This slice's part: a value every thread can check afterwards.
  const int part = (slice + 1) * 1000 + tile * 10 + (threadIdx.x % 7);
  int state = -1;
  if (threadIdx.x == 0) state = fetch(&sem[tile]);
  while (__syncthreads_and(state != slice)) {
    if (threadIdx.x == 0) state = fetch(&sem[tile]);
  }
  out[tile * kThreads + threadIdx.x] += part;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int next = slice + 1 == kSlices ? 0 : slice + 1;   // the last one resets it
    asm volatile("st.global.release.gpu.b32 [%0], %1;" :: "l"(&sem[tile]), "r"(next) : "memory");
  }
}

int main() {
  int *sem, *out;
  cudaMalloc(&sem, kTiles * sizeof(int));
  cudaMalloc(&out, kTiles * kThreads * sizeof(int));
  cudaMemset(sem, 0, kTiles * sizeof(int));
  cudaMemset(out, 0, kTiles * kThreads * sizeof(int));
  split_k<<<dim3(kTiles, 1, kSlices), kThreads>>>(sem, out);
  const cudaError_t e = cudaDeviceSynchronize();
  std::vector<int> h(kTiles * kThreads), s(kTiles);
  cudaMemcpy(h.data(), out, h.size() * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(s.data(), sem, s.size() * sizeof(int), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int t = 0; t < kTiles; ++t) {
    bad += s[t] != 0;
    for (int i = 0; i < kThreads; ++i) {
      int want = 0;
      for (int z = 0; z < kSlices; ++z) want += (z + 1) * 1000 + t * 10 + (i % 7);
      bad += h[t * kThreads + i] != want;
    }
  }
  std::printf("%s, %d wrong\n", cudaGetErrorString(e), bad);
  std::printf("%s\n", e == cudaSuccess && bad == 0 ? "PASS" : "FAIL");
  return e == cudaSuccess && bad == 0 ? 0 : 1;
}
