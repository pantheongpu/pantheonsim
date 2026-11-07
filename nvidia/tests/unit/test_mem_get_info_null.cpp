// cuMemGetInfo and cudaMemGetInfo with a null free or total pointer. Either may be null on a real GPU and the
// call fills in the other (cuda-api-wrappers asks for the total alone, cuMemGetInfo(nullptr, &total)); the
// simulator answered INVALID_VALUE.
#include <cuda.h>
#include <cuda_runtime.h>

#include <cstddef>

#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

namespace {
struct Env {
  CUcontext ctx = nullptr;
  CUdevice dev = 0;
  Env() {
    cuInit(0);
    cuDeviceGet(&dev, 0);
    cuDevicePrimaryCtxRetain(&ctx, dev);
    cuCtxSetCurrent(ctx);
  }
  ~Env() { cuDevicePrimaryCtxRelease(dev); }
};
}  // namespace

VTEST(driver_mem_get_info_takes_either_pointer_null) {
  Env e;
  size_t free_b = 0, total_b = 0;
  VCHECK_EQ(cuMemGetInfo(&free_b, &total_b), CUDA_SUCCESS);
  VCHECK(total_b > 0);
  VCHECK(free_b > 0 && free_b <= total_b);

  size_t total_only = 0;
  VCHECK_EQ(cuMemGetInfo(nullptr, &total_only), CUDA_SUCCESS);
  VCHECK_EQ(total_only, total_b);

  size_t free_only = 0;
  VCHECK_EQ(cuMemGetInfo(&free_only, nullptr), CUDA_SUCCESS);
  VCHECK(free_only > 0 && free_only <= total_b);

  // The _v2 entry point is the same call.
  size_t t2 = 0;
  VCHECK_EQ(cuMemGetInfo_v2(nullptr, &t2), CUDA_SUCCESS);
  VCHECK_EQ(t2, total_b);
}

VTEST(runtime_mem_get_info_takes_either_pointer_null) {
  Env e;
  size_t free_b = 0, total_b = 0;
  VCHECK_EQ(cudaMemGetInfo(&free_b, &total_b), cudaSuccess);
  size_t total_only = 0;
  VCHECK_EQ(cudaMemGetInfo(nullptr, &total_only), cudaSuccess);
  VCHECK_EQ(total_only, total_b);
  size_t free_only = 0;
  VCHECK_EQ(cudaMemGetInfo(&free_only, nullptr), cudaSuccess);
  VCHECK(free_only > 0 && free_only <= total_b);
}

VTEST_MAIN
