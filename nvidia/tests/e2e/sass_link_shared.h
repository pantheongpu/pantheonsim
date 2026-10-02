// What sass_link_main.cu and sass_link_lib.cu share: declarations of each
// other's definitions, and a template both instantiate.
#pragma once

extern __device__ int main_var;
extern __device__ int main_counter;
extern __constant__ int main_const[8];
extern __device__ int lib_var;
extern __device__ int lib_counter;
extern __constant__ int lib_const[8];
extern __device__ int* lib_ptr;

__device__ int lib_add(int x);
__device__ int lib_negate(int x);
__device__ int lib_file_scope_of(int x);
__device__ int lib_template(int x);
__device__ int lib_scratch(int t);

template <int N>
__device__ __noinline__ int twice_plus(int x) { return 2 * x + N; }
