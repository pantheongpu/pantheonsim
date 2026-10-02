// The complex (C and Z) half of cuSOLVER's dense API: included at the end of
// cusolver_api.cpp, whose helpers (known, set_info, lwork_hint, x_workspace,
// JacobiInfo, pointers) it uses. The arithmetic is complex_linalg.hpp's, in
// std::complex<double>, following LAPACK's conventions (geqrf/ungqr/unmqr must
// agree: PyTorch passes one's output to the others).
//
// What torch.linalg reaches on complex tensors: LU (getrf/getrs), Cholesky
// (potrf/potrs and batched), QR (geqrf, ungqr, unmqr), Hermitian eigen (heevd,
// heevj and batched), SVD (gesvd, gesvdj and batched, gesvdaStridedBatched),
// and complex types through the 64-bit X API (Xpotrf, Xpotrs, Xgeqrf,
// Xsyevd, Xgeev). The symmetric indefinite routines (sytrf, Xsytrs, sytri)
// are in cusolver_api.cpp, one template for all four types.

#include "complex_linalg.hpp"

namespace {

using vgpu_la::cd;
using vgpu_la::CMat;

inline cd sol_to_cd(const cuComplex& v) { return {v.x, v.y}; }
inline cd sol_to_cd(const cuDoubleComplex& v) { return {v.x, v.y}; }
template <class T> T sol_from_cd(cd v);
template <> inline cuComplex sol_from_cd<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> inline cuDoubleComplex sol_from_cd<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
template <class T> using SolReal = std::conditional_t<std::is_same_v<T, cuComplex>, float, double>;

// A column-major complex matrix from device memory, `cols` columns of `ld`.
template <class T> bool cload(const void* dev, int ld, int cols, CMat* m) {
  m->ld = ld;
  const size_t n = (size_t)std::max(ld, 0) * std::max(cols, 0);
  m->v.assign(n, 0);
  if (!n) return true;
  std::vector<T> raw(n);
  if (cudaMemcpy(raw.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess) return false;
  for (size_t i = 0; i < n; ++i) m->v[i] = sol_to_cd(raw[i]);
  return true;
}
template <class T> bool csave(void* dev, const std::vector<cd>& v) {
  if (v.empty()) return true;
  std::vector<T> raw(v.size());
  for (size_t i = 0; i < v.size(); ++i) raw[i] = sol_from_cd<T>(v[i]);
  return cudaMemcpy(dev, raw.data(), raw.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}
// A matrix into `ld`-strided device memory, rows x cols of it.
template <class T> bool csave_ld(void* dev, const CMat& m, int rows, int cols, int ld) {
  CMat out;
  if (!cload<T>(dev, ld, cols, &out)) return false;  // what lies between the columns stays
  for (int c = 0; c < cols; ++c)
    for (int r = 0; r < rows; ++r) out(r, c) = m(r, c);
  return csave<T>(dev, out.v);
}
template <class R> bool rsave(void* dev, const std::vector<double>& v) {
  if (v.empty()) return true;
  std::vector<R> raw(v.begin(), v.end());
  return cudaMemcpy(dev, raw.data(), raw.size() * sizeof(R), cudaMemcpyHostToDevice) == cudaSuccess;
}

vgpu_la::Op sol_op(cublasOperation_t t) {
  return t == CUBLAS_OP_N ? vgpu_la::Op::N : t == CUBLAS_OP_T ? vgpu_la::Op::T : vgpu_la::Op::C;
}

template <class T>
cusolverStatus_t c_potrf(const void* A, int n, int lda, bool lower, int* info) {
  CMat a;
  if (!cload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const int rc = vgpu_la::cholesky(a, n, lower);
  set_info(info, rc);
  if (rc == 0 && !csave<T>(const_cast<void*>(A), a.v)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  return CUSOLVER_STATUS_SUCCESS;
}
template <class T>
cusolverStatus_t c_potrs(const void* A, int n, int lda, bool lower, void* B, int nrhs, int ldb, int* info) {
  CMat a, b;
  if (!cload<T>(A, lda, n, &a) || !cload<T>(B, ldb, nrhs, &b)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  vgpu_la::cholesky_solve(a, n, lower, b, nrhs);
  set_info(info, 0);
  return csave<T>(B, b.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}
template <class T>
cusolverStatus_t c_geqrf(void* A, int m, int n, int lda, void* tau, int* info) {
  CMat a;
  if (!cload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<cd> t;
  vgpu_la::geqrf(a, m, n, &t);
  set_info(info, 0);
  return csave<T>(A, a.v) && csave<T>(tau, t) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}
// Eigenvalues ascending into W (real), vectors over A when asked.
template <class T>
cusolverStatus_t c_heev(void* A, int n, int lda, void* W, cusolverEigMode_t jobz, cublasFillMode_t uplo, int* info) {
  CMat a;
  if (!cload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<double> w;
  CMat v;
  vgpu_la::heev(a, n, uplo == CUBLAS_FILL_MODE_LOWER, &w, &v);
  set_info(info, 0);
  if (!rsave<SolReal<T>>(W, w)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  if (jobz == CUSOLVER_EIG_MODE_VECTOR && !csave_ld<T>(A, v, n, n, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  return CUSOLVER_STATUS_SUCCESS;
}
// S (real, descending), and with vectors U (m x ucols) and V (n x vcols) --
// V itself, as gesvdj returns it, not V^H.
template <class T>
cusolverStatus_t c_svd(const void* A, int m, int n, int lda, bool full, cusolverEigMode_t jobz, void* S, void* U,
                       int ldu, void* V, int ldv, int* info) {
  CMat a;
  if (!cload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  std::vector<double> s;
  CMat u, v;
  vgpu_la::svd(a, m, n, full, &s, &u, &v);
  set_info(info, 0);
  if (!rsave<SolReal<T>>(S, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  if (jobz == CUSOLVER_EIG_MODE_VECTOR) {
    const int k = std::min(m, n);
    if (U && !csave_ld<T>(U, u, m, full ? m : k, ldu)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    if (V && !csave_ld<T>(V, v, n, full ? n : k, ldv)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  }
  return CUSOLVER_STATUS_SUCCESS;
}

bool is_complex(cudaDataType t) { return t == CUDA_C_32F || t == CUDA_C_64F; }
// Runs body<T> for a complex element type named by a cudaDataType.
template <class F> cusolverStatus_t by_ctype(cudaDataType t, F&& body) {
  if (t == CUDA_C_32F) return body(cuComplex{});
  if (t == CUDA_C_64F) return body(cuDoubleComplex{});
  return CUSOLVER_STATUS_NOT_SUPPORTED;
}

}  // namespace

#define VGPU_COMPLEX_SOLVER(P, T, R)                                                                              \
  /* LU */                                                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##getrf_bufferSize(cusolverDnHandle_t h, int m, int n, T*, int,      \
                                                               int* lwork) {                                     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##getrf(cusolverDnHandle_t h, int m, int n, T* A, int lda, T*,       \
                                                    int* devIpiv, int* info) {                                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < 0 || n < 0 || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;                             \
    CMat a;                                                                                                       \
    if (!cload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;                                        \
    std::vector<int> ipiv;                                                                                        \
    set_info(info, vgpu_la::lu(a, m, n, devIpiv ? &ipiv : nullptr));                                              \
    if (devIpiv && !ipiv.empty() &&                                                                               \
        cudaMemcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess)       \
      return CUSOLVER_STATUS_EXECUTION_FAILED;                                                                    \
    return csave<T>(A, a.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;                         \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##getrs(cusolverDnHandle_t h, cublasOperation_t t, int n, int nrhs,  \
                                                    const T* A, int lda, const int* devIpiv, T* B, int ldb,      \
                                                    int* info) {                                                 \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    CMat a, b;                                                                                                    \
    if (!cload<T>(A, lda, n, &a) || !cload<T>(B, ldb, nrhs, &b)) return CUSOLVER_STATUS_EXECUTION_FAILED;         \
    std::vector<int> ipiv;                                                                                        \
    if (devIpiv) {                                                                                                \
      ipiv.resize(n);                                                                                             \
      cudaMemcpy(ipiv.data(), devIpiv, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost);                          \
    }                                                                                                             \
    vgpu_la::lu_solve(a, n, devIpiv ? ipiv.data() : nullptr, sol_op(t), b, nrhs);                                \
    set_info(info, 0);                                                                                            \
    return csave<T>(B, b.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;                         \
  }                                                                                                               \
  /* Cholesky */                                                                                                  \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrf_bufferSize(cusolverDnHandle_t h, cublasFillMode_t, int n,    \
                                                               T*, int, int* lwork) {                            \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(n);                                                                            \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrf(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, T* A,    \
                                                    int lda, T*, int, int* info) {                               \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (n < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                                      \
    return c_potrf<T>(A, n, lda, uplo == CUBLAS_FILL_MODE_LOWER, info);                                           \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrs(cusolverDnHandle_t h, cublasFillMode_t uplo, int n, int nrhs, \
                                                    const T* A, int lda, T* B, int ldb, int* info) {             \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    return c_potrs<T>(A, n, lda, uplo == CUBLAS_FILL_MODE_LOWER, B, nrhs, ldb, info);                             \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrfBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, int n,   \
                                                           T* Aarray[], int lda, int* infos, int batch) {        \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (n < 0 || batch < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                         \
    const auto ptrs = pointers<T>(Aarray, batch);                                                                 \
    for (int b = 0; b < batch; ++b) {                                                                             \
      const cusolverStatus_t st = c_potrf<T>(ptrs[b], n, lda, uplo == CUBLAS_FILL_MODE_LOWER, infos + b);         \
      if (st != CUSOLVER_STATUS_SUCCESS) return st;                                                               \
    }                                                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrsBatched(cusolverDnHandle_t h, cublasFillMode_t uplo, int n,   \
                                                           int nrhs, T* Aarray[], int lda, T* Barray[], int ldb, \
                                                           int* info, int batch) {                               \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (nrhs != 1) return CUSOLVER_STATUS_INVALID_VALUE; /* cuSOLVER supports one right-hand side here */         \
    const auto pa = pointers<T>(Aarray, batch), pb = pointers<T>(Barray, batch);                                  \
    for (int b = 0; b < batch; ++b) {                                                                             \
      const cusolverStatus_t st = c_potrs<T>(pa[b], n, lda, uplo == CUBLAS_FILL_MODE_LOWER, pb[b], nrhs, ldb,     \
                                             nullptr);                                                            \
      if (st != CUSOLVER_STATUS_SUCCESS) return st;                                                               \
    }                                                                                                             \
    set_info(info, 0);                                                                                            \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  /* QR */                                                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##geqrf_bufferSize(cusolverDnHandle_t h, int m, int n, T*, int,      \
                                                               int* lwork) {                                     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##geqrf(cusolverDnHandle_t h, int m, int n, T* A, int lda, T* tau,   \
                                                    T*, int, int* info) {                                        \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < 0 || n < 0 || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;                             \
    return c_geqrf<T>(A, m, n, lda, tau, info);                                                                   \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##ungqr_bufferSize(cusolverDnHandle_t h, int m, int n, int,          \
                                                               const T*, int, const T*, int* lwork) {            \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##ungqr(cusolverDnHandle_t h, int m, int n, int k, T* A, int lda,    \
                                                    const T* tau, T*, int, int* info) {                          \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < 0 || n < 0 || n > m || k < 0 || k > n || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;  \
    CMat a, t;                                                                                                    \
    if (!cload<T>(A, lda, n, &a) || !cload<T>(tau, std::max(k, 1), 1, &t)) return CUSOLVER_STATUS_EXECUTION_FAILED; \
    vgpu_la::ungqr(a, m, n, k, t.v.data());                                                                       \
    set_info(info, 0);                                                                                            \
    return csave<T>(A, a.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;                         \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##unmqr_bufferSize(cusolverDnHandle_t h, cublasSideMode_t,           \
                                                               cublasOperation_t, int m, int n, int, const T*,   \
                                                               int, const T*, const T*, int, int* lwork) {       \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##unmqr(cusolverDnHandle_t h, cublasSideMode_t side,                 \
                                                    cublasOperation_t trans, int m, int n, int k, const T* A,    \
                                                    int lda, const T* tau, T* C, int ldc, T*, int, int* info) {  \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (trans == CUBLAS_OP_T) return CUSOLVER_STATUS_INVALID_VALUE; /* complex Q takes N or C */                  \
    const bool left = side == CUBLAS_SIDE_LEFT;                                                                   \
    CMat a, t, c;                                                                                                 \
    if (!cload<T>(A, lda, k, &a) || !cload<T>(tau, std::max(k, 1), 1, &t) || !cload<T>(C, ldc, n, &c))           \
      return CUSOLVER_STATUS_EXECUTION_FAILED;                                                                    \
    vgpu_la::unmqr(left, sol_op(trans), m, n, k, a, t.v.data(), c);                                               \
    set_info(info, 0);                                                                                            \
    return csave<T>(C, c.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;                         \
  }                                                                                                               \
  /* Hermitian eigen */                                                                                           \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##heevd_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t,          \
                                                               cublasFillMode_t, int n, const T*, int, const R*, \
                                                               int* lwork) {                                     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(n);                                                                            \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##heevd(cusolverDnHandle_t h, cusolverEigMode_t jobz,                \
                                                    cublasFillMode_t uplo, int n, T* A, int lda, R* W, T*, int,  \
                                                    int* info) {                                                 \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (n < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                                      \
    return c_heev<T>(A, n, lda, W, jobz, uplo, info);                                                             \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##heevj_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t,          \
                                                               cublasFillMode_t, int n, const T*, int, const R*, \
                                                               int* lwork, syevjInfo_t) {                        \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(n);                                                                            \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##heevj(cusolverDnHandle_t h, cusolverEigMode_t jobz,                \
                                                    cublasFillMode_t uplo, int n, T* A, int lda, R* W, T*, int,  \
                                                    int* info, syevjInfo_t) {                                    \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (n < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                                      \
    return c_heev<T>(A, n, lda, W, jobz, uplo, info);                                                             \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##heevjBatched_bufferSize(                                           \
      cusolverDnHandle_t h, cusolverEigMode_t, cublasFillMode_t, int n, const T*, int, const R*, int* lwork,      \
      syevjInfo_t, int) {                                                                                         \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(n);                                                                            \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##heevjBatched(cusolverDnHandle_t h, cusolverEigMode_t jobz,         \
                                                           cublasFillMode_t uplo, int n, T* A, int lda, R* W,    \
                                                           T*, int, int* info, syevjInfo_t, int batch) {         \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (n < 0 || batch < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                         \
    for (int b = 0; b < batch; ++b) {                                                                             \
      const cusolverStatus_t st = c_heev<T>(A + (size_t)b * lda * n, n, lda, W + (size_t)b * n, jobz, uplo,       \
                                            info ? info + b : nullptr);                                           \
      if (st != CUSOLVER_STATUS_SUCCESS) return st;                                                               \
    }                                                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  /* SVD */                                                                                                       \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvd_bufferSize(cusolverDnHandle_t h, int m, int n, int* lwork) { \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  /* U and VT = V^H; hardware's gesvd needs m >= n, as the real one here does. */                                 \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvd(cusolverDnHandle_t h, signed char jobu, signed char jobvt,   \
                                                    int m, int n, T* A, int lda, R* S, T* U, int ldu, T* VT,     \
                                                    int ldvt, T*, int, R*, int* info) {                          \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < n) return CUSOLVER_STATUS_NOT_SUPPORTED;                                                              \
    if (jobu == 'O' || jobvt == 'O') return CUSOLVER_STATUS_NOT_SUPPORTED;                                        \
    CMat a;                                                                                                       \
    if (!cload<T>(A, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;                                        \
    std::vector<double> s;                                                                                        \
    CMat u, v;                                                                                                    \
    vgpu_la::svd(a, m, n, jobu == 'A', &s, &u, &v);                                                               \
    set_info(info, 0);                                                                                            \
    if (!rsave<R>(S, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;                                                 \
    if (U && jobu != 'N' && !csave_ld<T>(U, u, m, jobu == 'A' ? m : n, ldu)) return CUSOLVER_STATUS_EXECUTION_FAILED; \
    if (VT && jobvt != 'N') {                                                                                     \
      CMat vh{std::vector<cd>((size_t)n * n), n};                                                                 \
      for (int c = 0; c < n; ++c)                                                                                 \
        for (int r = 0; r < n; ++r) vh(r, c) = std::conj(v(c, r));                                                \
      if (!csave_ld<T>(VT, vh, n, n, ldvt)) return CUSOLVER_STATUS_EXECUTION_FAILED;                              \
    }                                                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvdj_bufferSize(cusolverDnHandle_t h, cusolverEigMode_t, int,    \
                                                                int m, int n, const T*, int, const R*, const T*, \
                                                                int, const T*, int, int* lwork, gesvdjInfo_t) {  \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvdj(cusolverDnHandle_t h, cusolverEigMode_t jobz, int econ,     \
                                                     int m, int n, T* A, int lda, R* S, T* U, int ldu, T* V,     \
                                                     int ldv, T*, int, int* info, gesvdjInfo_t) {                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < 0 || n < 0 || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;                             \
    return c_svd<T>(A, m, n, lda, !econ, jobz, S, U, ldu, V, ldv, info);                                          \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvdjBatched_bufferSize(                                          \
      cusolverDnHandle_t h, cusolverEigMode_t, int m, int n, const T*, int, const R*, const T*, int, const T*,   \
      int, int* lwork, gesvdjInfo_t, int) {                                                                       \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  /* Each matrix at A + b * lda * n; full U (m x m) and V (n x n), as the batched form computes. */             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvdjBatched(cusolverDnHandle_t h, cusolverEigMode_t jobz, int m, \
                                                            int n, T* A, int lda, R* S, T* U, int ldu, T* V,     \
                                                            int ldv, T*, int, int* info, gesvdjInfo_t,           \
                                                            int batch) {                                         \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < 0 || n < 0 || batch < 0 || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;                \
    const int k = std::min(m, n);                                                                                 \
    for (int b = 0; b < batch; ++b) {                                                                             \
      const cusolverStatus_t st = c_svd<T>(A + (size_t)b * lda * n, m, n, lda, true, jobz, S + (size_t)b * k,     \
                                           U ? U + (size_t)b * ldu * m : nullptr, ldu,                            \
                                           V ? V + (size_t)b * ldv * n : nullptr, ldv, info ? info + b : nullptr);\
      if (st != CUSOLVER_STATUS_SUCCESS) return st;                                                               \
    }                                                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvdaStridedBatched_bufferSize(                                   \
      cusolverDnHandle_t h, cusolverEigMode_t, int, int m, int n, const T*, int, long long, const R*, long long,  \
      const T*, int, long long, const T*, int, long long, int* lwork, int) {                                      \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                                               \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }                                                                                                               \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvdaStridedBatched(                                              \
      cusolverDnHandle_t h, cusolverEigMode_t jobz, int rank, int m, int n, const T* A, int lda, long long sA,    \
      R* S, long long sS, T* U, int ldu, long long sU, T* V, int ldv, long long sV, T*, int, int* info,           \
      double* nrm, int batch) {                                                                                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                                        \
    if (m < n || rank < 1 || rank > n) return CUSOLVER_STATUS_INVALID_VALUE;                                      \
    for (int b = 0; b < batch; ++b) {                                                                             \
      CMat a;                                                                                                     \
      if (!cload<T>(A + b * sA, lda, n, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;                             \
      std::vector<double> s;                                                                                      \
      CMat u, v;                                                                                                  \
      vgpu_la::svd(a, m, n, false, &s, &u, &v);                                                                   \
      s.resize(rank);                                                                                             \
      if (!rsave<R>(S + b * sS, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;                                      \
      if (jobz == CUSOLVER_EIG_MODE_VECTOR &&                                                                     \
          (!csave_ld<T>(U + b * sU, u, m, rank, ldu) || !csave_ld<T>(V + b * sV, v, n, rank, ldv)))               \
        return CUSOLVER_STATUS_EXECUTION_FAILED;                                                                  \
      if (info) set_info(info + b, 0);                                                                            \
      if (nrm) nrm[b] = 0.0;                                                                                      \
    }                                                                                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                                               \
  }
VGPU_COMPLEX_SOLVER(C, cuComplex, float)
VGPU_COMPLEX_SOLVER(Z, cuDoubleComplex, double)
#undef VGPU_COMPLEX_SOLVER
