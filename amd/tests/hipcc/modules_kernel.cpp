// The code object amd/tests/hipcc/modules.cpp loads, built by build.sh for
// gfx9-4-generic only: code a gfx942 device runs because it is a member of
// that family. One kernel, and one global to find.
#include <hip/hip_runtime.h>

__device__ int int_var = 7;

extern "C" __global__ void scale(int* p, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) p[i] *= int_var;
}
