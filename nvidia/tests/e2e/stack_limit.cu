// cudaLimitStackSize reaches the launch: a program that raises it gets the
// bigger per-thread stack. Each thread allocas 3000 bytes, past the 1 KiB
// default, after the limit is raised to 4 KiB, and the values come back as
// the host computes them. The simulator used to keep every launch at 1 KiB
// whatever the limit said, so this ran off the top of the stack. Runs the
// same on a GPU (checked on an RTX 3060). The size is a kernel argument: a
// constant one is compiled to a fixed local frame, not to alloca.
#include <cstdio>
#include <alloca.h>
constexpr int kWords = 750;   // 3000 bytes
__global__ void k(unsigned* out, int words) {
  const unsigned t = threadIdx.x;
  unsigned* a = static_cast<unsigned*>(alloca(words * sizeof(unsigned)));
  for (int i = 0; i < words; ++i) a[i] = t * 2654435761u + i;
  unsigned s = 0;
  for (int i = words - 1; i >= 0; --i) s = s * 33u + a[i];
  out[t] = s;
}
int main() {
  size_t stack = 0;
  cudaDeviceGetLimit(&stack, cudaLimitStackSize);
  printf("default stack %zu\n", stack);
  if (cudaDeviceSetLimit(cudaLimitStackSize, 4096)) { puts("FAIL: could not raise the stack limit"); return 1; }
  cudaDeviceGetLimit(&stack, cudaLimitStackSize);
  printf("raised to %zu\n", stack);
  unsigned* d;
  cudaMalloc(&d, 64 * 4);
  k<<<1, 64>>>(d, kWords);
  cudaError_t e = cudaDeviceSynchronize();
  if (e) { printf("error: %s\nFAIL\n", cudaGetErrorString(e)); return 1; }
  unsigned h[64];
  cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (unsigned t = 0; t < 64; ++t) {
    unsigned s = 0;
    for (int i = kWords - 1; i >= 0; --i) s = s * 33u + (t * 2654435761u + i);
    bad += h[t] != s;
  }
  printf("%d of 64 threads wrong\n", bad);
  puts(bad == 0 && stack >= 3000 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
