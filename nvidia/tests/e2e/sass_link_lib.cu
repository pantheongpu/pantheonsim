// The library half of nvjitlink_sass (see sass_link_main.cu).
#include "sass_link_shared.h"

__device__ int lib_var = 17;
__device__ int lib_counter;
__constant__ int lib_const[8] = {1, 2, 3, 4, 5, 6, 7, 8};
__device__ int lib_array[3] = {5, 6, 7};
__device__ int* lib_ptr = &lib_array[1];

static __device__ __noinline__ int file_scope(int x) { return x + 2000; }   // the main module has one too
__device__ int lib_file_scope_of(int x) { return file_scope(x); }

__device__ int lib_add(int x) { return x + lib_var + main_var; }
__device__ int lib_negate(int x) { return -x; }
__device__ int lib_template(int x) { return twice_plus<3>(x); }

__device__ __noinline__ int lib_scratch2(int t) {
  __shared__ char bytes[64];
  bytes[t] = static_cast<char>(t + 9);
  __syncthreads();
  const int v = bytes[(t + 3) & 63];
  __syncthreads();
  return v;
}
// 2 when both its own values and lib_scratch2's are right.
__device__ __noinline__ int lib_scratch(int t) {
  __shared__ int scratch[64];
  scratch[t] = t * 5;
  __syncthreads();
  int ok = lib_scratch2(t) == ((t + 3) & 63) + 9;
  ok += scratch[(t + 2) & 63] == ((t + 2) & 63) * 5;
  __syncthreads();
  return ok;
}

// A kernel of the library's own, which reads the main module's variable.
__global__ void lib_kernel(int* out) {
  out[threadIdx.x] = lib_add(threadIdx.x) + main_const[threadIdx.x & 7] + main_counter * 0;
}
