// cuBLAS settings and statuses, against an RTX 3060's cuBLAS 13.0.
//
//   - cublasSetPointerMode takes HOST and DEVICE only; any other value is
//     CUBLAS_STATUS_INVALID_VALUE and the mode stays. The simulator stored it,
//     and the calls after it read their scalars as device pointers.
//   - cublasSetMathMode takes 0..31 (the modes and the reduced-precision flag)
//     and refuses anything above, keeping the mode it had.
//   - cublasGetStatusName and cublasGetStatusString: every status's own name
//     and sentence, "unrecognized status" for a value that is none. The
//     simulator called five of them CUBLAS_STATUS_INTERNAL_ERROR and returned
//     the name for the string.
//   - An empty product may pass null matrices: cublasSgemm with m = n = k = 0
//     and null A, B and C succeeds (the simulator refused a null C).
//   - cublasSetMatrix with a leading dimension below the row count is
//     CUBLAS_STATUS_MAPPING_ERROR and leaves the runtime's last error clear.
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

static int fails = 0;
#define WANT(x, want) do { const int got_ = static_cast<int>(x); if (got_ != static_cast<int>(want)) { \
  std::printf("FAIL %s -> %d, expected %d\n", #x, got_, static_cast<int>(want)); ++fails; } } while (0)
#define WANT_STR(x, want) do { const char* got_ = (x); if (std::strcmp(got_, want) != 0) { \
  std::printf("FAIL %s -> \"%s\", expected \"%s\"\n", #x, got_, want); ++fails; } } while (0)

int main() {
  cublasHandle_t h;
  WANT(cublasCreate(&h), CUBLAS_STATUS_SUCCESS);

  // Pointer mode.
  cublasPointerMode_t pm;
  WANT(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE), CUBLAS_STATUS_SUCCESS);
  WANT(cublasSetPointerMode(h, static_cast<cublasPointerMode_t>(2)), CUBLAS_STATUS_INVALID_VALUE);
  WANT(cublasGetPointerMode(h, &pm), CUBLAS_STATUS_SUCCESS);
  WANT(pm, CUBLAS_POINTER_MODE_DEVICE);
  WANT(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST), CUBLAS_STATUS_SUCCESS);
  WANT(cublasSetPointerMode(h, static_cast<cublasPointerMode_t>(5)), CUBLAS_STATUS_INVALID_VALUE);
  WANT(cublasGetPointerMode(h, &pm), CUBLAS_STATUS_SUCCESS);
  WANT(pm, CUBLAS_POINTER_MODE_HOST);
  // ... and the scalars are still read from the host.
  float* x = nullptr;
  WANT(cudaMalloc(&x, 2 * sizeof(float)), cudaSuccess);
  const float xs[2] = {10, 20};
  WANT(cudaMemcpy(x, xs, sizeof xs, cudaMemcpyHostToDevice), cudaSuccess);
  const float three = 3;
  WANT(cublasSscal(h, 2, &three, x, 1), CUBLAS_STATUS_SUCCESS);
  float back[2] = {};
  WANT(cudaMemcpy(back, x, sizeof back, cudaMemcpyDeviceToHost), cudaSuccess);
  WANT(back[0] == 30 && back[1] == 60, true);

  // Math mode.
  cublasMath_t mm;
  for (int m = 0; m <= 31; ++m) {
    WANT(cublasSetMathMode(h, static_cast<cublasMath_t>(m)), CUBLAS_STATUS_SUCCESS);
    WANT(cublasGetMathMode(h, &mm), CUBLAS_STATUS_SUCCESS);
    WANT(mm, m);
  }
  for (int m : {32, 33, 64, 0x100, 0x7777, -1}) {
    WANT(cublasSetMathMode(h, static_cast<cublasMath_t>(m)), CUBLAS_STATUS_INVALID_VALUE);
    WANT(cublasGetMathMode(h, &mm), CUBLAS_STATUS_SUCCESS);
    WANT(mm, 31);
  }
  WANT(cublasSetMathMode(h, CUBLAS_DEFAULT_MATH), CUBLAS_STATUS_SUCCESS);

  // Names and sentences.
  const struct { int s; const char* name; const char* text; } st[] = {
      {0, "CUBLAS_STATUS_SUCCESS", "success"},
      {1, "CUBLAS_STATUS_NOT_INITIALIZED", "the library was not initialized"},
      {2, "unrecognized status", "unrecognized status"},
      {3, "CUBLAS_STATUS_ALLOC_FAILED", "the resource allocation failed"},
      {7, "CUBLAS_STATUS_INVALID_VALUE", "an unsupported value or parameter was passed to the function"},
      {8, "CUBLAS_STATUS_ARCH_MISMATCH", "the function requires an architectural feature absent from the device"},
      {11, "CUBLAS_STATUS_MAPPING_ERROR", "an access to GPU memory space failed"},
      {13, "CUBLAS_STATUS_EXECUTION_FAILED", "the function failed to launch on the GPU"},
      {14, "CUBLAS_STATUS_INTERNAL_ERROR", "an internal operation failed"},
      {15, "CUBLAS_STATUS_NOT_SUPPORTED", "the requested functionality is not supported"},
      {16, "CUBLAS_STATUS_LICENSE_ERROR", "the license check failed"},
      {17, "unrecognized status", "unrecognized status"},
      {99, "unrecognized status", "unrecognized status"},
  };
  for (const auto& e : st) {
    WANT_STR(cublasGetStatusName(static_cast<cublasStatus_t>(e.s)), e.name);
    WANT_STR(cublasGetStatusString(static_cast<cublasStatus_t>(e.s)), e.text);
  }

  // An empty product with null matrices.
  const float one = 1, zero = 0;
  WANT(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 0, 0, 0, &one, nullptr, 1, nullptr, 1, &zero, nullptr, 1),
       CUBLAS_STATUS_SUCCESS);
  WANT(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, 2, 2, 0, &one, nullptr, 2, nullptr, 1, &zero, nullptr, 2),
       CUBLAS_STATUS_INVALID_VALUE);

  // A leading dimension below the rows: a mapping error, and nothing for cudaGetLastError.
  float host[16] = {};
  float* dm = nullptr;
  WANT(cudaMalloc(&dm, 64 * sizeof(float)), cudaSuccess);
  WANT(cudaGetLastError(), cudaSuccess);
  WANT(cublasSetMatrix(3, 2, 4, host, 2, dm, 3), CUBLAS_STATUS_MAPPING_ERROR);
  WANT(cublasSetMatrix(2, 2, 4, host, 2, dm, 1), CUBLAS_STATUS_MAPPING_ERROR);
  WANT(cublasGetMatrix(3, 2, 4, dm, 3, host, 2), CUBLAS_STATUS_MAPPING_ERROR);
  WANT(cudaGetLastError(), cudaSuccess);

  WANT(cudaFree(x), cudaSuccess);
  WANT(cudaFree(dm), cudaSuccess);
  WANT(cublasDestroy(h), CUBLAS_STATUS_SUCCESS);
  std::printf("%s\n", fails ? "FAILED" : "PASS");
  return fails ? 1 : 0;
}
