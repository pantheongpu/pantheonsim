// Turbo mode and a kernel that can assert or trap (docs/perf-design/12-host-gpu-turbo.md section 20). One mode a process:
//   assertok / assertfire   a kernel with assert(), whose condition holds / fails on thread 3
//   trapok / trapfire       a kernel with __trap() that is / is not reached by thread 3
//   plain                   a kernel with neither, in a module that holds a __device__ variable only the HOST names (cudaMemcpyToSymbol): no instruction does
// Built as SASS (the card path with a cubin) and as PTX only (the card compiles it). With -DREFERENCE_VAR the kernel of `plain` reads the variable; with -DNO_ASSERT
// the module has no assert (nvcc's PTX for one has a call the text scan cannot place, and message strings the kernel reads: a PTX module with an assert stays in the simulator).
// The program prints the error codes it sees, so that a card that fails the launch (the stand-in driver: STUB_LAUNCH_FAIL=710 or 719) and the simulator, which
// fires the assert or the trap itself, print the same lines.
#include <cassert>
#include <cstdio>
#include <cstring>
#include <cuda_runtime.h>

static __device__ int host_only[4];

#ifndef NO_ASSERT
__global__ void k_assert(int* p, int fire) {
  p[threadIdx.x] = 1;
  assert(!(fire && threadIdx.x == 3));
  p[threadIdx.x + 32] = 2;
}
#endif
__global__ void k_trap(int* p, int fire) {
  p[threadIdx.x] = 1;
  if (fire && threadIdx.x == 3) __trap();
  p[threadIdx.x + 32] = 2;
}
__global__ void k_plain(int* p) {
  p[threadIdx.x] = 1;
#ifdef REFERENCE_VAR
  p[threadIdx.x + 32] = host_only[threadIdx.x & 3];
#else
  p[threadIdx.x + 32] = 2;
#endif
}

int main(int argc, char** argv) {
  const char* mode = argc > 1 ? argv[1] : "plain";
  int* d = nullptr;
  cudaMalloc(&d, 1024);
  cudaMemset(d, 0, 1024);
  const int init[4] = {5, 6, 7, 8};
  cudaMemcpyToSymbol(host_only, init, sizeof init);
#ifndef NO_ASSERT
  if (!strcmp(mode, "assertok")) k_assert<<<1, 32>>>(d, 0);
  else if (!strcmp(mode, "assertfire")) k_assert<<<1, 32>>>(d, 1);
  else
#endif
  if (!strcmp(mode, "trapok")) k_trap<<<1, 32>>>(d, 0);
  else if (!strcmp(mode, "trapfire")) k_trap<<<1, 32>>>(d, 1);
  else k_plain<<<1, 32>>>(d);
  const cudaError_t sync = cudaDeviceSynchronize();
  int h[64] = {};
  const cudaError_t copy = cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  const cudaError_t next = cudaMalloc((void**)&d, 16);
  std::printf("mode=%s sync=%d copy=%d later_call=%d\n", mode, static_cast<int>(sync), static_cast<int>(copy), static_cast<int>(next));
  return 0;
}
