// The C ABI takes any integer for an enum parameter, and a program may pass one the enum does not declare on purpose
// (an error-path check). The shim must not load such a value as its enum type before the call's body has judged it:
// under UBSan's enum check that is a runtime error. Each call here is answered with an error code, and nothing else.
#include <cuda.h>
#include <cuda_runtime_api.h>

#include "vtest.hpp"

extern "C" const char* __asan_default_options() { return "detect_leaks=1:detect_odr_violation=0"; }

// Each entry point is called as a C caller with an int in the enum's place would (the parameter is an int in the C ABI):
// the program never holds the enum value, and the sanitizer sees no load of one on this side.
VTEST(an_undeclared_enum_value_is_answered_with_an_error_not_loaded_as_the_enum) {
  VCHECK_EQ(cudaSetDevice(0), cudaSuccess);
  cudaStream_t s = nullptr;
  VCHECK_EQ(cudaStreamCreate(&s), cudaSuccess);
  using BeginCapture = cudaError_t (*)(cudaStream_t, int);
  VCHECK(reinterpret_cast<BeginCapture>(&cudaStreamBeginCapture)(s, -1) != cudaSuccess);
  (void)cudaGetLastError();

  cudaMemPool_t pool = nullptr;
  VCHECK_EQ(cudaDeviceGetDefaultMemPool(&pool, 0), cudaSuccess);
  int handle = 0;
  using Export = cudaError_t (*)(void*, cudaMemPool_t, int, unsigned);
  VCHECK(reinterpret_cast<Export>(&cudaMemPoolExportToShareableHandle)(&handle, pool, 8, 0) != cudaSuccess);
  (void)cudaGetLastError();

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = 0;
  size_t granularity = 0;
  using Granularity = CUresult (*)(size_t*, const CUmemAllocationProp*, int);
  VCHECK(reinterpret_cast<Granularity>(&cuMemGetAllocationGranularity)(&granularity, &prop, 2) != CUDA_SUCCESS);

  const char* name = nullptr;
  using ErrorName = CUresult (*)(int, const char**);
  VCHECK(reinterpret_cast<ErrorName>(&cuGetErrorName)(12345, &name) != CUDA_SUCCESS);
  const char* text = nullptr;
  VCHECK(reinterpret_cast<ErrorName>(&cuGetErrorString)(12345, &text) != CUDA_SUCCESS);
  VCHECK_EQ(cudaStreamDestroy(s), cudaSuccess);
}

VTEST_MAIN
