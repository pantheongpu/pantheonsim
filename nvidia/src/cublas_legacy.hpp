// The legacy cuBLAS API's state, shared by the hand-written helpers
// (cublas_legacy_core.cpp) and the generated BLAS routines
// (generated/cublas_legacy.cpp): one library-wide handle, which cublasInit
// creates and cublasShutdown destroys, and the last error for cublasGetError.
#pragma once

#include <cublas_api.h>

#include <type_traits>

namespace vgpu_legacy {

// The legacy handle, created on first use: an RTX 3060's cuBLAS runs a legacy
// routine without cublasInit, and after cublasShutdown (measured). Null only
// when one cannot be created.
cublasHandle_t handle();
// Keeps the status of the latest call for cublasGetError: a success after a
// failure clears it, as on the card.
void record(cublasStatus_t st);

// Records `st` and returns what a legacy routine returns when it fails: 0,
// or nothing.
template <class R>
R fail(cublasStatus_t st) {
  record(st);
  if constexpr (!std::is_void_v<R>) return R{};
}

// The char options, either case; anything else is refused -- except by gemv,
// which on the card reads any trans but N (or C, for the complex forms) as T.
inline bool op(char c, cublasOperation_t* out) {
  switch (c) {
    case 'N': case 'n': *out = CUBLAS_OP_N; return true;
    case 'T': case 't': *out = CUBLAS_OP_T; return true;
    case 'C': case 'c': *out = CUBLAS_OP_C; return true;
    default: return false;
  }
}
inline bool op_gemv(char c, cublasOperation_t* out) {
  *out = c == 'N' || c == 'n' ? CUBLAS_OP_N : c == 'C' || c == 'c' ? CUBLAS_OP_C : CUBLAS_OP_T;
  return true;
}
inline bool fill(char c, cublasFillMode_t* out) {
  switch (c) {
    case 'U': case 'u': *out = CUBLAS_FILL_MODE_UPPER; return true;
    case 'L': case 'l': *out = CUBLAS_FILL_MODE_LOWER; return true;
    default: return false;
  }
}
inline bool side(char c, cublasSideMode_t* out) {
  switch (c) {
    case 'L': case 'l': *out = CUBLAS_SIDE_LEFT; return true;
    case 'R': case 'r': *out = CUBLAS_SIDE_RIGHT; return true;
    default: return false;
  }
}
inline bool diag(char c, cublasDiagType_t* out) {
  switch (c) {
    case 'U': case 'u': *out = CUBLAS_DIAG_UNIT; return true;
    case 'N': case 'n': *out = CUBLAS_DIAG_NON_UNIT; return true;
    default: return false;
  }
}

}  // namespace vgpu_legacy
