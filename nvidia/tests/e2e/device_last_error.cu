// The device runtime's error, per thread: cudaPeekAtLastError reads it,
// cudaGetLastError reads and clears it, and a thread sees only the errors of
// the device-runtime calls it made itself. The expected values are an RTX
// 3060's: the odd threads' child launch has a zero-sized block (error 9,
// cudaErrorInvalidConfiguration), the even threads' launch succeeds.
#include <cstdio>
#include <cuda_runtime.h>

__global__ void child(int* p) { p[threadIdx.x] += 1; }

__global__ void parent(int* out, int* data) {
  const int t = threadIdx.x;
  out[t * 6 + 0] = (int)cudaPeekAtLastError();   // nothing has failed yet
  if (t & 1) child<<<1, 0>>>(data);
  else child<<<1, 1>>>(data);
  out[t * 6 + 1] = (int)cudaPeekAtLastError();   // peeking does not clear ...
  out[t * 6 + 2] = (int)cudaPeekAtLastError();
  out[t * 6 + 3] = (int)cudaGetLastError();      // ... getting does
  out[t * 6 + 4] = (int)cudaGetLastError();
  out[t * 6 + 5] = (int)cudaPeekAtLastError();
}

int main() {
  int *out, *data;
  cudaMallocManaged(&out, 4 * 6 * sizeof(int));
  cudaMallocManaged(&data, 64 * sizeof(int));
  for (int i = 0; i < 64; ++i) data[i] = 0;
  parent<<<1, 4>>>(out, data);
  const cudaError_t e = cudaDeviceSynchronize();
  int bad = e != cudaSuccess;
  for (int t = 0; t < 4; ++t) {
    const int err = (t & 1) ? 9 : 0;
    const int want[6] = {0, err, err, err, 0, 0};
    for (int j = 0; j < 6; ++j)
      if (out[t * 6 + j] != want[j]) {
        std::printf("FAIL: thread %d, value %d: %d, the card gives %d\n", t, j, out[t * 6 + j], want[j]);
        ++bad;
      }
  }
  // The even threads' children ran: one thread each, adding 1 to data[0].
  if (data[0] != 2) {
    std::printf("FAIL: the successful child launches added %d, want 2\n", data[0]);
    ++bad;
  }
  cudaFree(out);
  cudaFree(data);
  std::printf("%s\n", bad ? "FAIL" : "PASS");
  return bad ? 1 : 0;
}
