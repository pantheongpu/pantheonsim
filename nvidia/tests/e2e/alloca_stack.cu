// alloca() in device code -- PTX's alloca, and stacksave/stackrestore around
// it -- checked against an RTX 3060 (sm_86): per-thread allocations of
// different sizes, and a device function called 50 times whose allocas must
// be freed when it returns, or the 1 KiB stack overflows. The outputs are
// hashed and compared with the card's; prints PASS on the last line, and runs
// the same on a GPU.
#include <cstdio>
#include <alloca.h>
__device__ __noinline__ unsigned inner(int n, unsigned seed) {
  unsigned* a = static_cast<unsigned*>(alloca(n * sizeof(unsigned)));
  for (int i = 0; i < n; ++i) a[i] = seed * 31u + i;
  unsigned s = 0;
  for (int i = n - 1; i >= 0; --i) s = s * 7u + a[i];
  return s;
}
__global__ void k(unsigned* out) {
  const int t = threadIdx.x;
  unsigned* b = static_cast<unsigned*>(alloca((t % 5 + 1) * 8));
  for (int i = 0; i < (t % 5 + 1) * 2; ++i) b[i] = t + i;
  unsigned acc = 0;
  for (int r = 0; r < 50; ++r) acc += inner(20 + t % 7, acc + r);   // 50 calls: the callee's allocas must be freed
  for (int i = 0; i < (t % 5 + 1) * 2; ++i) acc ^= b[i] << i;
  out[t] = acc;
}
int main() {
  unsigned* d;
  cudaMalloc(&d, 64 * 4);
  k<<<1, 64>>>(d);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) { printf("error: %s\n", cudaGetErrorString(e)); return 1; }
  unsigned h[64];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  unsigned long long hash = 1469598103934665603ull;
  for (unsigned v : h) { hash ^= v; hash *= 1099511628211ull; }
  const unsigned long long want = 0xf568a22f650d3aa7ull;   // the card's
  printf("hash %016llx (the card's %016llx)\n", hash, want);
  puts(hash == want ? "PASS" : "FAIL");
  return hash == want ? 0 : 1;
}
