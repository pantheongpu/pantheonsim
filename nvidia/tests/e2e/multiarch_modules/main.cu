// Which module a stub registered by two translation units uses, and what a kernel that only a unit built for a
// newer GPU holds does. Argument: the compute capability of the simulated GPU (80, 89, 90).
//
// - tier_kernel<7> is in both units: on an sm_80 or sm_89 GPU the sm_80 unit's code runs it (the sm_90-only unit,
//   registered last or first, is not the one used), on sm_90 either does. cudaGetFuncBySymbol and a launch agree.
// - tier_kernel<9> is in the sm_90-only unit alone: an sm_80 GPU has no code for it, which is
//   cudaErrorNoKernelImageForDevice from cudaGetFuncBySymbol and from the launch -- not an invalid-PTX error.
#include <cstdio>
#include <cstdlib>
#include <cuda.h>
#include <cuda_runtime.h>
#include "kernel.cuh"

static int fails = 0;
#define CHECK(cond, ...) do { if (!(cond)) { std::printf("FAIL: "); std::printf(__VA_ARGS__); std::printf("\n"); ++fails; } } while (0)

static cudaError_t launch(const void* stub, int* out) {
  void* args[] = {&out};
  cudaGetLastError();
  cudaError_t e = cudaLaunchKernel(stub, dim3(1), dim3(32), args, 0, nullptr);
  if (e == cudaSuccess) e = cudaDeviceSynchronize();
  return e;
}

int main(int argc, char** argv) {
  const int cc = argc > 1 ? std::atoi(argv[1]) : 90;
  cuInit(0);   // cudaGetFuncBySymbol hands back a driver CUfunction: the driver library has to be loaded
  int* out = nullptr;
  cudaMalloc(&out, 32 * sizeof(int));
  const void* shared_a = base_stub_shared();
  const void* shared_b = hopper_stub_shared();
  const void* only = hopper_stub_only();
  CHECK(shared_a == shared_b, "the two units' stubs for tier_kernel<7> are different symbols (%p, %p)", shared_a, shared_b);

  for (const void* stub : {shared_a, shared_b}) {
    cudaError_t e = launch(stub, out);
    CHECK(e == cudaSuccess, "launching tier_kernel<7>: %s", cudaGetErrorString(e));
    int h[32] = {};
    cudaMemcpy(h, out, sizeof h, cudaMemcpyDeviceToHost);
    bool ok = true;
    for (int i = 0; i < 32; ++i) ok = ok && h[i] == 700 + i;
    CHECK(ok, "tier_kernel<7> gave a wrong answer");
#if CUDART_VERSION >= 12000
    cudaFunction_t f = nullptr;
    e = cudaGetFuncBySymbol(&f, stub);
    CHECK(e == cudaSuccess && f, "cudaGetFuncBySymbol(tier_kernel<7>): %s", cudaGetErrorString(e));
#endif
  }

  cudaMemset(out, 0, 32 * sizeof(int));
  const cudaError_t e = launch(only, out);
  if (cc >= 90) {
    CHECK(e == cudaSuccess, "launching tier_kernel<9> on sm_%d: %s", cc, cudaGetErrorString(e));
  } else {
    CHECK(e == cudaErrorNoKernelImageForDevice, "launching tier_kernel<9> on sm_%d gave %s (%d), not cudaErrorNoKernelImageForDevice",
          cc, cudaGetErrorString(e), static_cast<int>(e));
  }
#if CUDART_VERSION >= 12000
  cudaFunction_t f = nullptr;
  const cudaError_t g = cudaGetFuncBySymbol(&f, only);
  if (cc >= 90) {
    CHECK(g == cudaSuccess, "cudaGetFuncBySymbol(tier_kernel<9>) on sm_%d: %s", cc, cudaGetErrorString(g));
  } else {
    CHECK(g == cudaErrorNoKernelImageForDevice, "cudaGetFuncBySymbol(tier_kernel<9>) on sm_%d gave %s (%d), not cudaErrorNoKernelImageForDevice",
          cc, cudaGetErrorString(g), static_cast<int>(g));
  }
#endif
  std::printf("%s\n", fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
