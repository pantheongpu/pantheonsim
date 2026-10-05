// The legacy cuBLAS API's helpers (cublas.h): cublasInit and cublasShutdown,
// cublasGetError, cublasAlloc and cublasFree, cublasSetKernelStream and the
// legacy cublasGetVersion. Its BLAS routines are generated
// (generated/cublas_legacy.cpp) onto the _v2 ones, run on the handle these
// keep. NVIDIA's libcublas.so.13 still exports the whole legacy API.
//
// What an RTX 3060's cuBLAS 13.0 does, measured, and done the same here:
// see the comment on each.
#include <cublas.h>

#include <atomic>
#include <mutex>

#include "cublas_legacy.hpp"

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

namespace vgpu_legacy {
namespace {
std::mutex g_mu;
cublasHandle_t g_handle = nullptr;
std::atomic<int> g_error{CUBLAS_STATUS_SUCCESS};
}  // namespace

cublasHandle_t handle() {
  std::lock_guard<std::mutex> lock(g_mu);
  if (!g_handle && cublasCreate_v2(&g_handle) != CUBLAS_STATUS_SUCCESS) g_handle = nullptr;
  return g_handle;
}
void record(cublasStatus_t st) { g_error = st; }
}  // namespace vgpu_legacy

using vgpu_legacy::g_handle;

// cublasInit and cublasShutdown always succeed on the card, a second call of
// either included, and neither is needed for the routines to run.
VGPU_EXPORT cublasStatus_t cublasInit(void) {
  std::lock_guard<std::mutex> lock(vgpu_legacy::g_mu);
  if (g_handle) return CUBLAS_STATUS_SUCCESS;
  return cublasCreate_v2(&g_handle);
}
VGPU_EXPORT cublasStatus_t cublasShutdown(void) {
  std::lock_guard<std::mutex> lock(vgpu_legacy::g_mu);
  if (g_handle) cublasDestroy_v2(g_handle);
  g_handle = nullptr;
  return CUBLAS_STATUS_SUCCESS;
}
// The status of the latest legacy call, which this then clears.
VGPU_EXPORT cublasStatus_t cublasGetError(void) {
  return (cublasStatus_t)vgpu_legacy::g_error.exchange(CUBLAS_STATUS_SUCCESS);
}
// With or without cublasInit; a NULL pointer is INVALID_VALUE.
VGPU_EXPORT cublasStatus_t cublasGetVersion(int* version) {
  if (!version) return CUBLAS_STATUS_INVALID_VALUE;
  *version = CUBLAS_VERSION;
  return CUBLAS_STATUS_SUCCESS;
}
// A size or element size that is not positive is INVALID_VALUE.
VGPU_EXPORT cublasStatus_t cublasAlloc(int n, int elemSize, void** devicePtr) {
  if (n <= 0 || elemSize <= 0 || !devicePtr) return CUBLAS_STATUS_INVALID_VALUE;
  return cudaMalloc(devicePtr, (size_t)n * (size_t)elemSize) == cudaSuccess ? CUBLAS_STATUS_SUCCESS
                                                                           : CUBLAS_STATUS_ALLOC_FAILED;
}
// NULL is fine; a pointer cudaFree refuses is INTERNAL_ERROR.
VGPU_EXPORT cublasStatus_t cublasFree(void* devicePtr) {
  return cudaFree(devicePtr) == cudaSuccess ? CUBLAS_STATUS_SUCCESS : CUBLAS_STATUS_INTERNAL_ERROR;
}
VGPU_EXPORT cublasStatus_t cublasSetKernelStream(cudaStream_t stream) {
  cublasHandle_t h = vgpu_legacy::handle();
  if (!h) return CUBLAS_STATUS_NOT_INITIALIZED;
  return cublasSetStream_v2(h, stream);
}
