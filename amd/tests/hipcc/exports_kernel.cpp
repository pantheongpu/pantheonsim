// The code object amd/tests/hipcc/exports.cpp loads as a module, a library,
// a fat binary and a link's input: one kernel that adds to every element.
#include <hip/hip_runtime.h>

extern "C" __global__ void add_to(int* p, int v, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] += v;
}
