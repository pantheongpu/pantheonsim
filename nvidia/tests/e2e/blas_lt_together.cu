// A program that uses cuBLAS and cuBLASLt both, linked in that order (as
// PyTorch's CUDA library is): every cuBLASLt call must reach cuBLASLt. A
// lookup by name in the global scope (dlsym(RTLD_DEFAULT), which is how a
// framework resolving entry points at run time finds them) takes the first
// library in load order that exports the name, so a cuBLASLt name exported by
// libcublas -- which NVIDIA's libcublas does not do -- would be found there
// instead. Passes on the card.
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <dlfcn.h>

#include <cstdio>

static int failures = 0;
#define IS(call, want)                                                              \
  do {                                                                              \
    const bool ok_ = (int)(call) == (int)(want);                                    \
    std::printf("%-4s %s -> %s\n", ok_ ? "ok" : "FAIL", #call, #want);              \
    if (!ok_) ++failures;                                                           \
  } while (0)

int main() {
  cublasHandle_t h = nullptr;
  cublasLtHandle_t lt = nullptr;
  IS(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);
  IS(cublasLtCreate(&lt), CUBLAS_STATUS_SUCCESS);
  cublasLtMatmulDesc_t desc = nullptr;
  cublasLtMatrixLayout_t layout = nullptr;
  cublasLtMatmulPreference_t pref = nullptr;
  IS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  const cublasOperation_t t = CUBLAS_OP_T;
  IS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t), CUBLAS_STATUS_SUCCESS);
  IS(cublasLtMatrixLayoutCreate(&layout, CUDA_R_32F, 4, 4, 4), CUBLAS_STATUS_SUCCESS);
  const int32_t batch = 2;
  IS(cublasLtMatrixLayoutSetAttribute(layout, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof batch),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasLtMatmulPreferenceCreate(&pref), CUBLAS_STATUS_SUCCESS);
  const size_t ws = 0;
  IS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasLtMatmulPreferenceDestroy(pref), CUBLAS_STATUS_SUCCESS);
  IS(cublasLtMatrixLayoutDestroy(layout), CUBLAS_STATUS_SUCCESS);
  IS(cublasLtMatmulDescDestroy(desc), CUBLAS_STATUS_SUCCESS);
  // Each name, looked up globally, is cuBLASLt's own.
  Dl_info info{};
  void* self = dlsym(RTLD_DEFAULT, "cublasLtCreate");
  void* lib = self && dladdr(self, &info) ? dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD) : nullptr;
  for (const char* name : {"cublasLtMatmul", "cublasLtMatmulAlgoGetHeuristic", "cublasLtMatmulDescCreate",
                           "cublasLtMatmulDescDestroy", "cublasLtMatmulDescSetAttribute",
                           "cublasLtMatmulPreferenceCreate", "cublasLtMatmulPreferenceDestroy",
                           "cublasLtMatmulPreferenceSetAttribute", "cublasLtMatrixLayoutCreate",
                           "cublasLtMatrixLayoutDestroy", "cublasLtMatrixLayoutSetAttribute"}) {
    const bool ok = lib && dlsym(RTLD_DEFAULT, name) == dlsym(lib, name);
    std::printf("%-4s %s resolves to libcublasLt\n", ok ? "ok" : "FAIL", name);
    if (!ok) ++failures;
  }
  IS(cublasLtDestroy(lt), CUBLAS_STATUS_SUCCESS);
  IS(cublasDestroy(h), CUBLAS_STATUS_SUCCESS);
  std::printf(failures ? "FAIL: %d checks\n" : "PASS: cuBLASLt calls reach cuBLASLt beside cuBLAS\n", failures);
  return failures ? 1 : 0;
}
