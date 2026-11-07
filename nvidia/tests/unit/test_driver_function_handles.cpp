// cuModuleGetFunction gives the same CUfunction for the same module and name, and cuModuleEnumerateFunctions
// gives those handles. On a real GPU the handle is the function's, so programs compare them (cuda-api-wrappers
// compares its kernel wrappers by handle); the simulator made a fresh handle on every call.
#include <cuda.h>

#include <cstring>
#include <string>

#include "vtest.hpp"

namespace {

const char* const kPtx = R"(
.version 7.0
.target sm_70
.address_size 64
.visible .entry k1() { ret; }
.visible .entry k2() { ret; }
)";

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

VTEST(the_same_module_and_name_give_the_same_function) {
  Env e;
  CUmodule mod = nullptr;
  VCHECK_EQ(cuModuleLoadData(&mod, kPtx), CUDA_SUCCESS);
  CUfunction f1 = nullptr, f1b = nullptr, f2 = nullptr;
  VCHECK_EQ(cuModuleGetFunction(&f1, mod, "k1"), CUDA_SUCCESS);
  VCHECK_EQ(cuModuleGetFunction(&f2, mod, "k2"), CUDA_SUCCESS);
  VCHECK_EQ(cuModuleGetFunction(&f1b, mod, "k1"), CUDA_SUCCESS);
  VCHECK(f1 == f1b);
  VCHECK(f1 != f2);

  // A second module with the same kernels has functions of its own.
  CUmodule mod2 = nullptr;
  VCHECK_EQ(cuModuleLoadData(&mod2, kPtx), CUDA_SUCCESS);
  CUfunction g1 = nullptr;
  VCHECK_EQ(cuModuleGetFunction(&g1, mod2, "k1"), CUDA_SUCCESS);
  VCHECK(g1 != f1);

  // A name the module does not have is still an error, and makes no handle.
  CUfunction none = nullptr;
  VCHECK_EQ(cuModuleGetFunction(&none, mod, "nope"), CUDA_ERROR_NOT_FOUND);

  VCHECK_EQ(cuModuleUnload(mod2), CUDA_SUCCESS);
  VCHECK_EQ(cuModuleUnload(mod), CUDA_SUCCESS);
}

VTEST(enumerated_functions_are_the_ones_looked_up_by_name) {
  Env e;
  CUmodule mod = nullptr;
  VCHECK_EQ(cuModuleLoadData(&mod, kPtx), CUDA_SUCCESS);
  unsigned n = 0;
  VCHECK_EQ(cuModuleGetFunctionCount(&n, mod), CUDA_SUCCESS);
  VCHECK_EQ(n, 2u);
  // Enumerate before and after the lookups: either order gives one handle per function.
  CUfunction before[2] = {};
  VCHECK_EQ(cuModuleEnumerateFunctions(before, 2, mod), CUDA_SUCCESS);
  CUfunction f1 = nullptr, f2 = nullptr;
  VCHECK_EQ(cuModuleGetFunction(&f1, mod, "k1"), CUDA_SUCCESS);
  VCHECK_EQ(cuModuleGetFunction(&f2, mod, "k2"), CUDA_SUCCESS);
  CUfunction after[2] = {};
  VCHECK_EQ(cuModuleEnumerateFunctions(after, 2, mod), CUDA_SUCCESS);
  for (int i = 0; i < 2; ++i) {
    VCHECK(before[i] == after[i]);
    const char* name = "";
    VCHECK_EQ(cuFuncGetName(&name, after[i]), CUDA_SUCCESS);
    VCHECK(after[i] == (std::strcmp(name, "k1") == 0 ? f1 : f2));
  }
  VCHECK(after[0] != after[1]);
  VCHECK_EQ(cuModuleUnload(mod), CUDA_SUCCESS);
}

VTEST_MAIN
