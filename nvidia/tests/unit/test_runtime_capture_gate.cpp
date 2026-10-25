// The gate in front of every allocation, copy and launch (capture_unsafe_gate in runtime_api.cpp): while a stream
// capture is going on, a call that is not safe to make during one is refused and invalidates the capture; with none
// going on the gate costs the call nothing, and it must notice a capture that begins and one that ends, from any
// thread.
#include <cuda_runtime_api.h>

#include <thread>

#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

VTEST(an_unsafe_call_is_refused_during_a_global_capture_and_allowed_after_it) {
  cudaStream_t s = nullptr;
  VCHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
  void* p = nullptr;
  VCHECK_EQ(cudaMalloc(&p, 256), cudaSuccess);   // (no capture)
  VCHECK_EQ(cudaFree(p), cudaSuccess);

  VCHECK_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal), cudaSuccess);
  void* q = nullptr;
  VCHECK_EQ(cudaMalloc(&q, 256), cudaErrorStreamCaptureUnsupported);
  // a global capture is every thread's
  cudaError_t other = cudaSuccess;
  std::thread([&] {
    void* r = nullptr;
    other = cudaMalloc(&r, 256);
  }).join();
  VCHECK_EQ(other, cudaErrorStreamCaptureUnsupported);
  cudaGraph_t g = nullptr;
  VCHECK_EQ(cudaStreamEndCapture(s, &g), cudaErrorStreamCaptureInvalidated);   // the refused call spoiled it
  (void)cudaGetLastError();

  VCHECK_EQ(cudaMalloc(&q, 256), cudaSuccess);   // the capture is over
  VCHECK_EQ(cudaFree(q), cudaSuccess);
  std::thread([&] {
    void* r = nullptr;
    other = cudaMalloc(&r, 256);
    if (other == cudaSuccess) cudaFree(r);
  }).join();
  VCHECK_EQ(other, cudaSuccess);
  VCHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
}

// A relaxed capture lets the call through.
VTEST(a_relaxed_capture_does_not_refuse) {
  cudaStream_t s = nullptr;
  VCHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
  VCHECK_EQ(cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed), cudaSuccess);
  void* p = nullptr;
  VCHECK_EQ(cudaMalloc(&p, 256), cudaSuccess);
  VCHECK_EQ(cudaFree(p), cudaSuccess);
  cudaGraph_t g = nullptr;
  VCHECK_EQ(cudaStreamEndCapture(s, &g), cudaSuccess);
  if (g) cudaGraphDestroy(g);
  VCHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
}

VTEST_MAIN
