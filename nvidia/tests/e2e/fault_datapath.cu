// A kernel that copies a buffer, and a host check of what came back: the
// program `vgpu fault arm` is tested against. It knows nothing of VirtualGPU.
// With the argument "shared", the copy is staged through shared memory; with
// "wild" it reads far past its buffer, and with "misaligned" it reads a word
// at an odd address -- faults of the program's own making. With "float" each
// value goes through a floating-point multiply by one, which leaves it as it
// was unless the multiply's result is corrupted.
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

__global__ void copy(const unsigned* in, unsigned* out, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i];
}

__global__ void staged(const unsigned* in, unsigned* out, int n) {
  __shared__ unsigned tile[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  tile[threadIdx.x] = i < n ? in[i] : 0u;
  __syncthreads();
  // Each thread writes out a neighbour's element, so the value really is read
  // back from shared memory.
  const unsigned j = (threadIdx.x + 1) % blockDim.x;
  const int k = blockIdx.x * blockDim.x + static_cast<int>(j);
  if (k < n) out[k] = tile[j];
}

__global__ void offset_copy(const unsigned* in, unsigned* out, int n, long long skew_bytes) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = *reinterpret_cast<const unsigned*>(reinterpret_cast<const char*>(in + i) + skew_bytes);
}

__global__ void scaled(const unsigned* in, unsigned* out, int n, float one) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = __float_as_uint(__uint_as_float(in[i]) * one);
}

int main(int argc, char** argv) {
  const char mode = argc > 1 ? argv[1][0] : 'c';
  const bool through_shared = mode == 's';
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
  if (mode == 'f')
    scaled<<<(n + 255) / 256, 256>>>(in, out, n, 1.0f);
  else if (mode == 'w')
    offset_copy<<<(n + 255) / 256, 256>>>(in, out, n, 1LL << 32);
  else if (mode == 'm')
    offset_copy<<<(n + 255) / 256, 256>>>(in, out, n, 1);
  else if (through_shared)
    staged<<<(n + 255) / 256, 256>>>(in, out, n);
  else
    copy<<<(n + 255) / 256, 256>>>(in, out, n);
  // Every path frees what it allocated: the e2e run is under LeakSanitizer, and
  // the uncorrectable-error path is the one that returns early.
  int rc = 0;
  const cudaError_t e = cudaDeviceSynchronize();
  if (e != cudaSuccess) {
    printf("launch failed: %d %s\n", static_cast<int>(e), cudaGetErrorName(e));
    rc = 3;
  } else if (const cudaError_t c = cudaMemcpy(got, out, bytes, cudaMemcpyDeviceToHost);
             c != cudaSuccess) {
    printf("copy failed: %d %s\n", static_cast<int>(c), cudaGetErrorName(c));
    rc = 4;
  } else {
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
