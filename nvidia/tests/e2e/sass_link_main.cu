// The kernel half of nvjitlink_sass: built with nvcc -rdc for SASS only and
// linked with sass_link_lib.cu by nvJitLink (run_jit_link_sass.sh). Every
// kind of reference that crosses the two modules, or that only the link can
// resolve, is here:
//   - calls into the other module, and through a pointer to a function there;
//   - its variables, initialised and not, and its __constant__ data, read
//     directly and indexed (constant-bank offsets the link fixes);
//   - a file-scope function of the same name in each module, each of which
//     must stay its module's;
//   - a template both modules instantiate (a weak definition in each);
//   - __shared__ variables in device functions of both modules and in the
//     kernel, and extern __shared__ memory behind them (offsets the link lays
//     out), checked for overlap by every thread writing its own pattern;
//   - a pointer initialised to another variable's address, printf, malloc.
#include <cstdio>

#include "sass_link_shared.h"

__device__ int main_var = 3;           // the library reads it
__device__ int main_counter;           // and adds to it
__constant__ int main_const[8] = {10, 20, 30, 40, 50, 60, 70, 80};
__device__ int main_array[4] = {100, 200, 300, 400};
__device__ int* main_ptr = &main_array[2];   // a relocation in initialised data

static __device__ __noinline__ int file_scope(int x) { return x + 1000; }   // the library has one too

// A device function's shared memory, live across a call into the library's
// (which keeps its own live across a call to another): no two may overlap.
// 3 when every value read back is the one written.
__device__ __noinline__ int main_scratch(int t) {
  __shared__ int scratch[64];
  scratch[t] = t * 7;
  __syncthreads();
  int ok = lib_scratch(t);
  ok += scratch[(t + 1) & 63] == ((t + 1) & 63) * 7;
  __syncthreads();
  return ok;
}

typedef int (*op_t)(int);
__device__ int main_twice(int x) { return 2 * x; }
__device__ op_t ops[2] = {main_twice, lib_negate};   // one function from each module

extern __shared__ int dynamic[];

__global__ void run_all(int* out, int sel) {
  __shared__ int mine[64];             // the kernel's own shared memory
  const int t = threadIdx.x;           // launched with 64 threads
  mine[t] = t + 5;
  dynamic[t] = t * 3;
  __syncthreads();
  int r = 0;
  if (t == 0) {
    out[0] = lib_add(5);                          // a call into the library
    out[1] = lib_var;                             // its variable
    out[2] = lib_const[3] + lib_const[sel & 7];   // its constants, direct and indexed
    out[3] = main_const[2] + main_const[sel & 7];
    out[4] = file_scope(1) + lib_file_scope_of(1);
    out[5] = ops[0](21) + ops[1](4) + ops[sel & 1](1);
    out[6] = twice_plus<3>(4) + lib_template(4);  // the weak template, both copies
    out[7] = *main_ptr + *lib_ptr;
    int* heap = static_cast<int*>(malloc(4 * sizeof(int)));
    heap[0] = 11;
    out[8] = heap[0];
    free(heap);
    printf("sass_link: printf from linked SASS, %d\n", 42);
  }
  // Shared memory of the kernel, of a function here, of two functions in the
  // library, and the dynamic array, all live at once: 5 per thread.
  r += main_scratch(t);
  r += mine[t] == t + 5;
  r += dynamic[(t + 1) & 63] == ((t + 1) & 63) * 3;
  atomicAdd(&out[9], r);
  atomicAdd(&main_counter, 1);
  atomicAdd(&lib_counter, 1);
}
