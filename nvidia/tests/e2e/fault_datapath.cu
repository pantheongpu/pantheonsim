// A kernel that copies a buffer, and a host check of what came back: the
// program `vgpu fault arm` is tested against. It knows nothing of VirtualGPU.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void copy(const unsigned* in, unsigned* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

int main() {
  const int n = 4096;
  const size_t bytes = n * sizeof(unsigned);
  unsigned* want = static_cast<unsigned*>(malloc(bytes));
  unsigned* got = static_cast<unsigned*>(malloc(bytes));
  for (int i = 0; i < n; ++i) want[i] = 0x5A5A0000u + static_cast<unsigned>(i);
  unsigned *in = nullptr, *out = nullptr;
  cudaMalloc(&in, bytes);
  cudaMalloc(&out, bytes);
  cudaMemcpy(in, want, bytes, cudaMemcpyHostToDevice);
  cudaMemset(out, 0, bytes);
  copy<<<(n + 255) / 256, 256>>>(in, out, n);
  // Every path frees what it allocated: the e2e run is under LeakSanitizer, and
  // the uncorrectable-error path is the one that returns early.
  int rc = 0;
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("launch failed: %d %s\n", static_cast<int>(e), cudaGetErrorName(e));
    rc = 3;
  } else {
    cudaMemcpy(got, out, bytes, cudaMemcpyDeviceToHost);
    int bad = 0;
    for (int i = 0; i < n; ++i) bad += got[i] != want[i];
    printf("mismatches: %d\n", bad);
    rc = bad ? 1 : 0;
  }
  cudaFree(in);
  cudaFree(out);
  free(want);
  free(got);
  return rc;
}
