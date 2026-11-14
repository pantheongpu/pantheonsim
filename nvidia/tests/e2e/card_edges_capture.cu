// What a stream capture answers once something has invalidated it, as an RTX
// 3060 (CUDA 13.0) does in every capture mode.
//
//   - Every launch, cudaMemcpyAsync, cudaMemsetAsync, cudaEventRecord and
//     cudaStreamWaitEvent on a stream of the invalidated capture fails with
//     cudaErrorStreamCaptureInvalidated (901), until cudaStreamEndCapture ends
//     it (901, no graph). The simulator recorded them and answered success.
//   - cudaEventSynchronize and cudaEventQuery of an event recorded in the
//     capture answer cudaErrorCapturedEvent and invalidate the capture in
//     Relaxed mode too (the simulator kept capturing and built a graph); once
//     the capture is invalidated they answer cudaErrorInvalidValue.
//     cudaEventElapsedTime answers cudaErrorCapturedEvent and does not
//     invalidate it.
//   - cudaStreamEndCapture on a stream that joined the capture (not the one
//     that began it) is cudaErrorStreamCaptureUnmatched and invalidates the
//     capture: the simulator let the capture go on.
#include <cuda_runtime.h>

#include <cstdio>

static int fails = 0;
static int mode_ = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL mode %d line %d: %s -> %d, expected %d\n", mode_, __LINE__, #x, got_, static_cast<int>(want)); ++fails; } } while (0)

__global__ void bump(int* p) { if (p) *p += 1; }

static cudaStream_t c, s2;
static cudaEvent_t e, e0;
static int* d;

static cudaStreamCaptureStatus status(cudaStream_t s) {
  cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
  cudaStreamIsCapturing(s, &st);
  return st;
}
// Ends the capture on c, which must have been invalidated: 901 and no graph.
static void end_invalidated() {
  cudaGraph_t g = nullptr;
  WANT(cudaStreamEndCapture(c, &g), cudaErrorStreamCaptureInvalidated);
  WANT(g == nullptr, true);
  if (g) cudaGraphDestroy(g);
  WANT(status(c), cudaStreamCaptureStatusNone);
  WANT(status(s2), cudaStreamCaptureStatusNone);
  cudaGetLastError();
}

int main() {
  WANT(cudaMalloc(&d, 64), cudaSuccess);
  WANT(cudaStreamCreateWithFlags(&c, cudaStreamNonBlocking), cudaSuccess);
  WANT(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), cudaSuccess);
  WANT(cudaEventCreate(&e), cudaSuccess);
  WANT(cudaEventCreate(&e0), cudaSuccess);
  WANT(cudaEventRecord(e0, 0), cudaSuccess);
  WANT(cudaEventSynchronize(e0), cudaSuccess);
  float ms = 0;
  for (mode_ = 0; mode_ < 3; ++mode_) {
    const cudaStreamCaptureMode mode = static_cast<cudaStreamCaptureMode>(mode_);

    // A query of a stream taking part invalidates the capture (all modes); then everything on its streams is 901.
    WANT(cudaStreamBeginCapture(c, mode), cudaSuccess);
    WANT(cudaEventRecord(e, c), cudaSuccess);
    WANT(cudaStreamWaitEvent(s2, e, 0), cudaSuccess);
    WANT(cudaStreamQuery(s2), cudaErrorStreamCaptureUnsupported);
    WANT(status(c), cudaStreamCaptureStatusInvalidated);
    bump<<<1, 1, 0, c>>>(d);
    WANT(cudaGetLastError(), cudaErrorStreamCaptureInvalidated);
    bump<<<1, 1, 0, s2>>>(d);
    WANT(cudaGetLastError(), cudaErrorStreamCaptureInvalidated);
    WANT(cudaMemcpyAsync(d + 1, d, 4, cudaMemcpyDeviceToDevice, c), cudaErrorStreamCaptureInvalidated);
    WANT(cudaMemsetAsync(d, 0, 4, s2), cudaErrorStreamCaptureInvalidated);
    WANT(cudaEventRecord(e, s2), cudaErrorStreamCaptureInvalidated);
    WANT(cudaStreamWaitEvent(c, e, 0), cudaErrorStreamCaptureInvalidated);
    WANT(cudaGetLastError(), cudaErrorStreamCaptureInvalidated);
    end_invalidated();

    // Waiting for or querying a captured event: 907 and the capture is invalidated; afterwards 1.
    WANT(cudaStreamBeginCapture(c, mode), cudaSuccess);
    bump<<<1, 1, 0, c>>>(d);
    WANT(cudaEventRecord(e, c), cudaSuccess);
    WANT(cudaEventSynchronize(e), cudaErrorCapturedEvent);
    WANT(cudaGetLastError(), cudaErrorCapturedEvent);
    WANT(status(c), cudaStreamCaptureStatusInvalidated);
    WANT(cudaEventQuery(e), cudaErrorInvalidValue);
    WANT(cudaEventSynchronize(e), cudaErrorInvalidValue);
    end_invalidated();
    WANT(cudaStreamBeginCapture(c, mode), cudaSuccess);
    WANT(cudaEventRecord(e, c), cudaSuccess);
    WANT(cudaEventQuery(e), cudaErrorCapturedEvent);
    WANT(status(c), cudaStreamCaptureStatusInvalidated);
    end_invalidated();
    // A time does not invalidate it.
    WANT(cudaStreamBeginCapture(c, mode), cudaSuccess);
    WANT(cudaEventRecord(e, c), cudaSuccess);
    WANT(cudaEventElapsedTime(&ms, e0, e), cudaErrorCapturedEvent);
    WANT(status(c), cudaStreamCaptureStatusActive);
    {
      cudaGraph_t g = nullptr;
      WANT(cudaStreamEndCapture(c, &g), cudaSuccess);
      WANT(g != nullptr, true);
      if (g) WANT(cudaGraphDestroy(g), cudaSuccess);
    }
    cudaGetLastError();

    // Ending the capture from a stream that joined it.
    WANT(cudaStreamBeginCapture(c, mode), cudaSuccess);
    WANT(cudaEventRecord(e, c), cudaSuccess);
    WANT(cudaStreamWaitEvent(s2, e, 0), cudaSuccess);
    {
      cudaGraph_t g = nullptr;
      WANT(cudaStreamEndCapture(s2, &g), cudaErrorStreamCaptureUnmatched);
      WANT(g == nullptr, true);
    }
    WANT(status(c), cudaStreamCaptureStatusInvalidated);
    WANT(status(s2), cudaStreamCaptureStatusInvalidated);
    end_invalidated();
  }
  WANT(cudaDeviceSynchronize(), cudaSuccess);
  WANT(cudaEventDestroy(e), cudaSuccess);
  WANT(cudaEventDestroy(e0), cudaSuccess);
  WANT(cudaStreamDestroy(c), cudaSuccess);
  WANT(cudaStreamDestroy(s2), cudaSuccess);
  WANT(cudaFree(d), cudaSuccess);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
