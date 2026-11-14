// Every runtime call that fails makes its error the thread's last error, which
// cudaPeekAtLastError then reports and cudaGetLastError reports and clears. On
// an RTX 3060 (CUDA 13.0) that holds for each call here; the simulator only did
// it for the calls that went through its guard, so these left the last error
// clear: cudaEventElapsedTime on an event never recorded, cudaStreamWaitEvent
// with an unknown flag, cudaStreamEndCapture on a stream not capturing,
// cudaStreamBeginCapture on the legacy stream, cudaMemcpy2D with a pitch
// narrower than a row, cudaMemcpy3D and cudaMemset3D with bad arguments, and
// cudaEventQuery of an event recorded in a capture. A query's not-ready is not
// an error and sets nothing.
#include <cuda_runtime.h>

#include <cstdio>

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL %s -> %d, expected %d\n", #x, got_, static_cast<int>(want)); ++fails; } } while (0)
// The call fails with `code`, and the last error is then `code` once.
#define FAILS_AND_SETS(call, code) do { WANT(call, code); WANT(cudaPeekAtLastError(), code); \
  WANT(cudaGetLastError(), code); WANT(cudaGetLastError(), cudaSuccess); } while (0)

int main() {
  char *d = nullptr, *d2 = nullptr;
  WANT(cudaMalloc(&d, 256), cudaSuccess);
  WANT(cudaMalloc(&d2, 256), cudaSuccess);
  cudaStream_t s;
  WANT(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), cudaSuccess);
  cudaEvent_t a, b;
  WANT(cudaEventCreate(&a), cudaSuccess);
  WANT(cudaEventCreate(&b), cudaSuccess);
  float ms = 0;

  FAILS_AND_SETS(cudaEventElapsedTime(&ms, a, b), cudaErrorInvalidResourceHandle);
  FAILS_AND_SETS(cudaEventElapsedTime(nullptr, a, b), cudaErrorInvalidValue);
  FAILS_AND_SETS(cudaStreamWaitEvent(s, a, 5), cudaErrorInvalidValue);
  cudaGraph_t g = nullptr;
  FAILS_AND_SETS(cudaStreamEndCapture(s, &g), cudaErrorIllegalState);
  FAILS_AND_SETS(cudaStreamBeginCapture(0, cudaStreamCaptureModeGlobal), cudaErrorStreamCaptureUnsupported);
  FAILS_AND_SETS(cudaStreamBeginCapture(s, static_cast<cudaStreamCaptureMode>(9)), cudaErrorInvalidValue);
  FAILS_AND_SETS(cudaMemcpy2D(d2, 4, d, 8, 8, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidPitchValue);
  FAILS_AND_SETS(cudaMemcpy2D(d2, 8, d, 8, 300, 2, cudaMemcpyDeviceToDevice), cudaErrorInvalidPitchValue);
  FAILS_AND_SETS(cudaMemcpy3D(nullptr), cudaErrorInvalidValue);
  cudaPitchedPtr p3 = make_cudaPitchedPtr(d, 16, 16, 2);
  FAILS_AND_SETS(cudaMemset3D(p3, 5, make_cudaExtent(17, 2, 2)), cudaErrorInvalidValue);

  // A failure inside a capture: the capture's own codes are last errors too.
  WANT(cudaEventRecord(b, s), cudaSuccess);
  WANT(cudaEventSynchronize(b), cudaSuccess);
  WANT(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), cudaSuccess);
  WANT(cudaEventRecord(a, s), cudaSuccess);
  FAILS_AND_SETS(cudaEventElapsedTime(&ms, b, a), cudaErrorCapturedEvent);
  FAILS_AND_SETS(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), cudaErrorIllegalState);
  WANT(cudaStreamEndCapture(s, &g), cudaSuccess);
  if (g) WANT(cudaGraphDestroy(g), cudaSuccess);

  // A later failure replaces an earlier one; a success leaves it alone.
  WANT(cudaStreamWaitEvent(s, a, 5), cudaErrorInvalidValue);
  WANT(cudaEventElapsedTime(&ms, a, b), cudaErrorInvalidValue);   // a was recorded in a capture that has ended
  WANT(cudaMemset(d, 0, 16), cudaSuccess);
  WANT(cudaGetLastError(), cudaErrorInvalidValue);
  WANT(cudaGetLastError(), cudaSuccess);
  // cudaGetLastError and cudaPeekAtLastError do not make what they return a failure of their own.
  WANT(cudaPeekAtLastError(), cudaSuccess);

  WANT(cudaEventDestroy(a), cudaSuccess);
  WANT(cudaEventDestroy(b), cudaSuccess);
  WANT(cudaStreamDestroy(s), cudaSuccess);
  WANT(cudaFree(d), cudaSuccess);
  WANT(cudaFree(d2), cudaSuccess);
  WANT(cudaGetLastError(), cudaSuccess);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
