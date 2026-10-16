// Shared by the rocprofv3 comparison cases (amd/tests/e2e/run_rocprof_case.sh).
// Each case is a few HIP calls that take milliseconds on a card; the profiler's
// trace of them is what is compared, so a case keeps its calls fixed and its
// sizes small.
#pragma once
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

#define CHECK(call)                                                              \
  do {                                                                           \
    hipError_t e_ = (call);                                                      \
    if (e_ != hipSuccess) {                                                      \
      std::fprintf(stderr, "%s:%d: %s: %s\n", __FILE__, __LINE__, #call, hipGetErrorString(e_)); \
      std::exit(1);                                                              \
    }                                                                            \
  } while (0)

__global__ void vadd(const float* a, const float* b, float* c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
