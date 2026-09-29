// atomicCAS: memory is replaced by the new value only where it holds the
// compare value, and the old value comes back either way.
//
// The simulator had the two operands of atom.cas the other way round. A CAS
// loop then took every failure for success and never stored anything, so
// stdgpu's atomic sum came out 0 and its hash-map inserts spun forever.
#include <cstdio>

__global__ void one_cas(unsigned* v, unsigned* got) {
  got[0] = atomicCAS(&v[0], 5u, 7u);   // holds 5: swapped to 7, returns 5
  got[1] = atomicCAS(&v[1], 5u, 7u);   // holds 6: untouched, returns 6
}
template <class T>
__global__ void cas_sum(T* v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  T old = *(volatile T*)v;
  for (;;) {
    const T got = atomicCAS(v, old, old + T(i + 1));
    if (got == old) break;
    old = got;
  }
}
__global__ void cas_shared(unsigned* out) {
  __shared__ unsigned s;
  if (threadIdx.x == 0) s = 0;
  __syncthreads();
  unsigned old = s;
  for (;;) {
    const unsigned got = atomicCAS(&s, old, old + 1);
    if (got == old) break;
    old = got;
  }
  __syncthreads();
  if (threadIdx.x == 0) *out = s;
}

int main() {
  int bad = 0;
  unsigned *v, *got;
  cudaMalloc(&v, 2 * sizeof(unsigned));
  cudaMalloc(&got, 2 * sizeof(unsigned));
  const unsigned init[2] = {5, 6};
  cudaMemcpy(v, init, sizeof init, cudaMemcpyHostToDevice);
  one_cas<<<1, 1>>>(v, got);
  unsigned hv[2], hg[2];
  cudaMemcpy(hv, v, sizeof hv, cudaMemcpyDeviceToHost);
  cudaMemcpy(hg, got, sizeof hg, cudaMemcpyDeviceToHost);
  if (hv[0] != 7 || hg[0] != 5) { std::printf("FAIL matching CAS: memory %u returned %u, want 7 and 5\n", hv[0], hg[0]); ++bad; }
  if (hv[1] != 6 || hg[1] != 6) { std::printf("FAIL mismatching CAS: memory %u returned %u, want 6 and 6\n", hv[1], hg[1]); ++bad; }

  const int n = 4096;
  unsigned* s32;
  unsigned long long* s64;
  cudaMalloc(&s32, sizeof *s32);
  cudaMalloc(&s64, sizeof *s64);
  cudaMemset(s32, 0, sizeof *s32);
  cudaMemset(s64, 0, sizeof *s64);
  cas_sum<<<n / 128, 128>>>(s32, n);
  cas_sum<<<n / 128, 128>>>(s64, n);
  unsigned h32 = 0;
  unsigned long long h64 = 0;
  cudaMemcpy(&h32, s32, sizeof h32, cudaMemcpyDeviceToHost);
  cudaMemcpy(&h64, s64, sizeof h64, cudaMemcpyDeviceToHost);
  const unsigned long long want = (unsigned long long)n * (n + 1) / 2;
  if (h32 != want) { std::printf("FAIL 32-bit CAS sum over blocks: %u, want %llu\n", h32, want); ++bad; }
  if (h64 != want) { std::printf("FAIL 64-bit CAS sum over blocks: %llu, want %llu\n", h64, want); ++bad; }

  unsigned hs = 0;
  cas_shared<<<1, 256>>>(got);
  cudaMemcpy(&hs, got, sizeof hs, cudaMemcpyDeviceToHost);
  if (hs != 256) { std::printf("FAIL shared-memory CAS count: %u, want 256\n", hs); ++bad; }
  if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL %s\n", cudaGetErrorString(cudaGetLastError())); ++bad; }
  std::printf(bad ? "FAILED\n" : "PASS\n");
  return bad ? 1 : 0;
}
