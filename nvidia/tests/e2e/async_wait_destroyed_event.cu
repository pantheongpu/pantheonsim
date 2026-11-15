// cudaEventDestroy right after cudaStreamWaitEvent (the reduced repro of Ta-Cuda/tacuda#5). The wait is legal CUDA: it has taken the
// event's state at the call, so destroying the event must not release the waiting stream early. Stream B runs a long kernel and a
// kernel that writes 42s, the event is recorded behind them, stream A waits for it and is then given a copy of what B wrote; the event
// is destroyed straight after the wait. A real RTX 3060 copies the 42s in every round; the simulator's queued wait used to look the
// event up by handle when its stream reached it, found nothing, took the event for complete and let the copy run first.
#include <cuda_runtime.h>

#include <cstdio>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  std::fflush(stdout);
  if (!ok) ++failures;
}

__global__ void spin(float* sink, int iters) {
  float v = threadIdx.x;
  for (int i = 0; i < iters; ++i) v = v * 0.999f + 1.0f;
  if (v < 0) sink[0] = v;   // never true; keeps the loop
}
__global__ void write42(float* out, int n) {
  for (int i = threadIdx.x; i < n; i += blockDim.x) out[i] = 42.0f;
}

int main() {
  const int n = 256, rounds = 30;
  cudaStream_t a, b;
  check(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking) == cudaSuccess, "stream A");
  check(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking) == cudaSuccess, "stream B");
  float *dout = nullptr, *dsink = nullptr, *hout = nullptr;
  check(cudaMalloc(&dout, n * sizeof(float)) == cudaSuccess, "device buffer");
  check(cudaMalloc(&dsink, sizeof(float)) == cudaSuccess, "sink");
  check(cudaHostAlloc(&hout, n * sizeof(float), cudaHostAllocDefault) == cudaSuccess, "pinned buffer");
  if (failures) return 1;

  int stale = 0, bad_calls = 0;
  for (int r = 0; r < rounds; ++r) {
    cudaMemset(dout, 0, n * sizeof(float));
    for (int i = 0; i < n; ++i) hout[i] = -1.0f;
    cudaDeviceSynchronize();
    cudaEvent_t ev;
    bad_calls += cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess;
    spin<<<1, 32, 0, b>>>(dsink, 100000);
    write42<<<1, 128, 0, b>>>(dout, n);
    bad_calls += cudaEventRecord(ev, b) != cudaSuccess;
    bad_calls += cudaStreamWaitEvent(a, ev, 0) != cudaSuccess;
    bad_calls += cudaEventDestroy(ev) != cudaSuccess;   // straight after the wait
    bad_calls += cudaMemcpyAsync(hout, dout, n * sizeof(float), cudaMemcpyDeviceToHost, a) != cudaSuccess;
    bad_calls += cudaStreamSynchronize(a) != cudaSuccess;
    bad_calls += cudaStreamSynchronize(b) != cudaSuccess;
    int wrong = 0;
    for (int i = 0; i < n; ++i) wrong += hout[i] != 42.0f;
    stale += wrong != 0;
  }
  check(bad_calls == 0, "every call in the rounds succeeds");
  std::printf("%d of %d rounds copied stale data\n", stale, rounds);
  check(stale == 0, "a stream that waited on an event destroyed right after the wait still waits for it");
  return failures ? 1 : 0;
}
