// A program built for the per-thread default stream, as nvcc
// --default-stream per-thread builds one (COLMAP does): the CUDA headers map
// cudaMemcpy to cudaMemcpy_ptds, kernel launches to cudaLaunchKernel_ptsz and
// so on, and stream 0 means the calling thread's own default stream. Each
// call here goes through those names, and the results are checked. Built by
// run_per_thread_stream.sh with --default-stream per-thread.
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define OK(call) check((call) == cudaSuccess, #call)

__global__ void scale(float* v, int n, float by) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) v[i] *= by;
}

__device__ int counter;

int main() {
  const int n = 256;
  float h[n], back[n];
  for (int i = 0; i < n; ++i) h[i] = static_cast<float>(i);
  float* d = nullptr;
  OK(cudaMalloc(&d, n * sizeof(float)));
  OK(cudaMemcpy(d, h, sizeof h, cudaMemcpyHostToDevice));
  scale<<<(n + 127) / 128, 128>>>(d, n, 2.0f);
  OK(cudaGetLastError());
  OK(cudaStreamSynchronize(0));
  OK(cudaMemcpy(back, d, sizeof back, cudaMemcpyDeviceToHost));
  check(back[0] == 0.0f && back[255] == 510.0f, "a kernel on the per-thread default stream");

  // Pitched copies and fills, synchronous and on stream 0.
  float* p = nullptr;
  size_t pitch = 0;
  OK(cudaMallocPitch(&p, &pitch, 16 * sizeof(float), 4));
  OK(cudaMemset2D(p, pitch, 0, 16 * sizeof(float), 4));
  OK(cudaMemcpy2D(p, pitch, h, 16 * sizeof(float), 16 * sizeof(float), 4, cudaMemcpyHostToDevice));
  std::memset(back, 0, sizeof back);
  OK(cudaMemcpy2DAsync(back, 16 * sizeof(float), p, pitch, 16 * sizeof(float), 4, cudaMemcpyDeviceToHost, 0));
  OK(cudaStreamSynchronize(0));
  check(back[17] == h[17] && back[63] == h[63], "cudaMemcpy2D there and back");
  OK(cudaMemsetAsync(d, 0, sizeof h, 0));
  OK(cudaMemcpyAsync(back, d, sizeof back, cudaMemcpyDeviceToHost, 0));
  OK(cudaStreamSynchronize(0));
  check(back[100] == 0.0f, "cudaMemsetAsync on stream 0");

  // A __device__ variable, and an event on the default stream.
  const int seven = 7;
  OK(cudaMemcpyToSymbol(counter, &seven, sizeof seven));
  int got = 0;
  OK(cudaMemcpyFromSymbol(&got, counter, sizeof got));
  check(got == 7, "a __device__ variable written and read");
  cudaEvent_t ev;
  OK(cudaEventCreate(&ev));
  OK(cudaEventRecord(ev, 0));
  OK(cudaEventSynchronize(ev));
  OK(cudaStreamQuery(0));
  OK(cudaEventDestroy(ev));

  OK(cudaFree(p));
  OK(cudaFree(d));
  std::printf(failures ? "FAIL: %d per-thread default stream checks\n" : "PASS: every per-thread default stream check\n", failures);
  return failures ? 1 : 0;
}
