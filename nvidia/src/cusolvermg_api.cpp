// libvgpucusolvermg -- VirtualGPU's cusolverMg, presented as libcusolverMg.so.
//
// The multi-GPU dense solvers: getrf/getrs, potrf/potrs/potri and syevd, on a
// matrix spread over the grid's devices. Each call gathers the matrix to one
// device, runs this simulator's cusolverDn on it (cusolver_api.cpp, through
// the public API: this library links libcusolver), and scatters the result
// back, so any number of devices behaves as one larger one.
//
// The layout is NVIDIA's, as an RTX 3060 pair showed it with CUDA 13.0
// (nvidia/tests/e2e/solver_mg_paths.cu): a 1 x G grid, columns in blocks of
// the descriptor's column block size dealt round-robin over the devices,
// block b on device b % G as its local block b / G; every device holds its
// columns contiguously with leading dimension = the matrix's row count. IPIV
// is spread the same way, as a 1 x N row. info and syevd's W are host memory.
//
// A submatrix (IA, JA, base 1, ScaLAPACK's convention) is gathered, worked
// on, and scattered back; nothing outside it changes. getrf's IPIV goes to
// columns JA.. of the IPIV row, its pivots relative to the submatrix, as
// NVIDIA's writes them. What that card answered and this follows: a grid
// with more than one row of devices is refused when it is created
// (INVALID_VALUE: NVIDIA's supports 1-D column block cyclic only, as it
// documents); syevd on a submatrix other than IA = JA = 1 is INVALID_VALUE;
// potrf of a matrix that is not positive definite returns INTERNAL_ERROR with
// info set; getrs with op other than N, and the upper triangle anywhere, are
// INVALID_VALUE (potrf's and potri's _bufferSize already refuse upper;
// potrs's and syevd's only the call itself). NVIDIA's getrf on a submatrix
// that starts below its diagonal block (IA > JA) returns neither the LU of the
// submatrix nor anything recognisable; this returns the LU.
//
// potri leaves the upper triangle as it was (NVIDIA's writes scratch there);
// the inverse is in the lower one, as the API documents.
#include <cusolverDn.h>
#include <cusolverMg.h>

#include <algorithm>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <set>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace {

struct Handle {
  std::vector<int> devices;
  cusolverDnHandle_t dn = nullptr;
};
struct Grid {
  int rows = 1, cols = 1;
  std::vector<int> ids;
};
struct Desc {
  int64_t m = 0, n = 0, mb = 0, nb = 0;
  cudaDataType type = CUDA_R_64F;
  const Grid* grid = nullptr;
};

std::mutex& g_mu = *new std::mutex;
std::set<const void*>& g_live = *new std::set<const void*>;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(p); }
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

size_t elem_bytes(cudaDataType t) {
  switch (t) {
    case CUDA_R_32F: return 4;
    case CUDA_R_64F: case CUDA_C_32F: return 8;
    case CUDA_C_64F: return 16;
    default: return 0;
  }
}

// Where global column j lives: device slot and the column within it.
struct Place { int slot; int64_t col; };
Place place(const Desc& d, int64_t j) {
  const int g = (int)d.grid->ids.size();
  const int64_t block = j / d.nb;
  return {(int)(block % g), (block / g) * d.nb + j % d.nb};
}

// The rows x cols submatrix starting at global (i0, j0) (0-based), as one
// column-major host array with ld = rows.
bool gather(const Desc& d, void* const* parts, int64_t i0, int64_t j0, int64_t rows, int64_t cols,
            std::vector<uint8_t>* out) {
  const size_t eb = elem_bytes(d.type);
  out->assign((size_t)rows * cols * eb, 0);
  for (int64_t j = 0; j < cols; ++j) {
    const Place p = place(d, j0 + j);
    const uint8_t* src = static_cast<const uint8_t*>(parts[p.slot]) + (size_t)(p.col * d.m + i0) * eb;
    if (rows && cudaMemcpy(out->data() + (size_t)j * rows * eb, src, (size_t)rows * eb, cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
  }
  return true;
}
bool scatter(const Desc& d, void* const* parts, int64_t i0, int64_t j0, int64_t rows, int64_t cols,
             const std::vector<uint8_t>& in) {
  const size_t eb = elem_bytes(d.type);
  for (int64_t j = 0; j < cols; ++j) {
    const Place p = place(d, j0 + j);
    uint8_t* dst = static_cast<uint8_t*>(parts[p.slot]) + (size_t)(p.col * d.m + i0) * eb;
    if (rows && cudaMemcpy(dst, in.data() + (size_t)j * rows * eb, (size_t)rows * eb, cudaMemcpyHostToDevice) != cudaSuccess)
      return false;
  }
  return true;
}
// IPIV, a 1 x N row in the same column blocks: n entries from column j0.
bool gather_ipiv(const Desc& d, int* const* parts, int64_t j0, int64_t n, std::vector<int>* out) {
  out->assign((size_t)n, 0);
  for (int64_t j = 0; j < n; ++j) {
    const Place p = place(d, j0 + j);
    if (cudaMemcpy(&(*out)[(size_t)j], parts[p.slot] + p.col, sizeof(int), cudaMemcpyDeviceToHost) != cudaSuccess)
      return false;
  }
  return true;
}
bool scatter_ipiv(const Desc& d, int* const* parts, int64_t j0, const std::vector<int>& in) {
  for (size_t j = 0; j < in.size(); ++j) {
    const Place p = place(d, j0 + (int64_t)j);
    if (cudaMemcpy(parts[p.slot] + p.col, &in[j], sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) return false;
  }
  return true;
}

// A device copy of host bytes, freed on scope exit.
struct DevBuf {
  void* p = nullptr;
  explicit DevBuf(size_t bytes) { cudaMalloc(&p, std::max<size_t>(bytes, 16)); }
  ~DevBuf() { cudaFree(p); }
  bool put(const void* h, size_t bytes) { return !bytes || cudaMemcpy(p, h, bytes, cudaMemcpyHostToDevice) == cudaSuccess; }
  bool get(void* h, size_t bytes) const { return !bytes || cudaMemcpy(h, p, bytes, cudaMemcpyDeviceToHost) == cudaSuccess; }
};

cusolverStatus_t check(cusolverMgHandle_t h, const void* descr, int ia, int ja, cudaDataType compute) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!known(descr)) return CUSOLVER_STATUS_INVALID_VALUE;
  const Desc& d = *static_cast<const Desc*>(descr);
  if (ia < 1 || ja < 1 || ia > std::max<int64_t>(d.m, 1) || ja > std::max<int64_t>(d.n, 1))
    return CUSOLVER_STATUS_INVALID_VALUE;
  if (compute != d.type) return CUSOLVER_STATUS_INVALID_VALUE;
  return CUSOLVER_STATUS_SUCCESS;
}
// The rows x cols submatrix at (ia, ja), base 1, fits in the matrix.
bool fits(const Desc& d, int ia, int ja, int64_t rows, int64_t cols) {
  return rows >= 0 && cols >= 0 && ia - 1 + rows <= d.m && ja - 1 + cols <= d.n;
}
const Desc& desc(const void* d) { return *static_cast<const Desc*>(d); }
int dev_info(const DevBuf& b) {
  int v = 0;
  b.get(&v, sizeof v);
  return v;
}

// cusolverDn by element type.
cusolverStatus_t dn_getrf(cusolverDnHandle_t h, cudaDataType t, int m, int n, void* A, int lda, int* ipiv, int* info) {
  int lw = 0;
  switch (t) {
    case CUDA_R_32F: {
      cusolverDnSgetrf_bufferSize(h, m, n, static_cast<float*>(A), lda, &lw);
      DevBuf w((size_t)lw * 4);
      return cusolverDnSgetrf(h, m, n, static_cast<float*>(A), lda, static_cast<float*>(w.p), ipiv, info);
    }
    case CUDA_R_64F: {
      cusolverDnDgetrf_bufferSize(h, m, n, static_cast<double*>(A), lda, &lw);
      DevBuf w((size_t)lw * 8);
      return cusolverDnDgetrf(h, m, n, static_cast<double*>(A), lda, static_cast<double*>(w.p), ipiv, info);
    }
    case CUDA_C_32F: {
      cusolverDnCgetrf_bufferSize(h, m, n, static_cast<cuComplex*>(A), lda, &lw);
      DevBuf w((size_t)lw * 8);
      return cusolverDnCgetrf(h, m, n, static_cast<cuComplex*>(A), lda, static_cast<cuComplex*>(w.p), ipiv, info);
    }
    case CUDA_C_64F: {
      cusolverDnZgetrf_bufferSize(h, m, n, static_cast<cuDoubleComplex*>(A), lda, &lw);
      DevBuf w((size_t)lw * 16);
      return cusolverDnZgetrf(h, m, n, static_cast<cuDoubleComplex*>(A), lda, static_cast<cuDoubleComplex*>(w.p),
                              ipiv, info);
    }
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
}
cusolverStatus_t dn_getrs(cusolverDnHandle_t h, cudaDataType t, cublasOperation_t op, int n, int nrhs, void* A, int lda,
                          const int* ipiv, void* B, int ldb, int* info) {
  switch (t) {
    case CUDA_R_32F:
      return cusolverDnSgetrs(h, op, n, nrhs, static_cast<float*>(A), lda, ipiv, static_cast<float*>(B), ldb, info);
    case CUDA_R_64F:
      return cusolverDnDgetrs(h, op, n, nrhs, static_cast<double*>(A), lda, ipiv, static_cast<double*>(B), ldb, info);
    case CUDA_C_32F:
      return cusolverDnCgetrs(h, op, n, nrhs, static_cast<cuComplex*>(A), lda, ipiv, static_cast<cuComplex*>(B), ldb,
                              info);
    case CUDA_C_64F:
      return cusolverDnZgetrs(h, op, n, nrhs, static_cast<cuDoubleComplex*>(A), lda, ipiv,
                              static_cast<cuDoubleComplex*>(B), ldb, info);
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
}
cusolverStatus_t dn_potrf(cusolverDnHandle_t h, cudaDataType t, cublasFillMode_t uplo, int n, void* A, int lda,
                          int* info) {
  int lw = 0;
  switch (t) {
    case CUDA_R_32F: {
      cusolverDnSpotrf_bufferSize(h, uplo, n, static_cast<float*>(A), lda, &lw);
      DevBuf w((size_t)lw * 4);
      return cusolverDnSpotrf(h, uplo, n, static_cast<float*>(A), lda, static_cast<float*>(w.p), lw, info);
    }
    case CUDA_R_64F: {
      cusolverDnDpotrf_bufferSize(h, uplo, n, static_cast<double*>(A), lda, &lw);
      DevBuf w((size_t)lw * 8);
      return cusolverDnDpotrf(h, uplo, n, static_cast<double*>(A), lda, static_cast<double*>(w.p), lw, info);
    }
    case CUDA_C_32F: {
      cusolverDnCpotrf_bufferSize(h, uplo, n, static_cast<cuComplex*>(A), lda, &lw);
      DevBuf w((size_t)lw * 8);
      return cusolverDnCpotrf(h, uplo, n, static_cast<cuComplex*>(A), lda, static_cast<cuComplex*>(w.p), lw, info);
    }
    case CUDA_C_64F: {
      cusolverDnZpotrf_bufferSize(h, uplo, n, static_cast<cuDoubleComplex*>(A), lda, &lw);
      DevBuf w((size_t)lw * 16);
      return cusolverDnZpotrf(h, uplo, n, static_cast<cuDoubleComplex*>(A), lda, static_cast<cuDoubleComplex*>(w.p), lw,
                              info);
    }
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
}
cusolverStatus_t dn_potrs(cusolverDnHandle_t h, cudaDataType t, cublasFillMode_t uplo, int n, int nrhs, void* A,
                          int lda, void* B, int ldb, int* info) {
  switch (t) {
    case CUDA_R_32F:
      return cusolverDnSpotrs(h, uplo, n, nrhs, static_cast<float*>(A), lda, static_cast<float*>(B), ldb, info);
    case CUDA_R_64F:
      return cusolverDnDpotrs(h, uplo, n, nrhs, static_cast<double*>(A), lda, static_cast<double*>(B), ldb, info);
    case CUDA_C_32F:
      return cusolverDnCpotrs(h, uplo, n, nrhs, static_cast<cuComplex*>(A), lda, static_cast<cuComplex*>(B), ldb, info);
    case CUDA_C_64F:
      return cusolverDnZpotrs(h, uplo, n, nrhs, static_cast<cuDoubleComplex*>(A), lda,
                              static_cast<cuDoubleComplex*>(B), ldb, info);
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
}
cusolverStatus_t dn_syevd(cusolverDnHandle_t h, cudaDataType t, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n,
                          void* A, int lda, void* W, int* info) {
  int lw = 0;
  switch (t) {
    case CUDA_R_32F: {
      cusolverDnSsyevd_bufferSize(h, jobz, uplo, n, static_cast<float*>(A), lda, static_cast<float*>(W), &lw);
      DevBuf w((size_t)lw * 4);
      return cusolverDnSsyevd(h, jobz, uplo, n, static_cast<float*>(A), lda, static_cast<float*>(W),
                              static_cast<float*>(w.p), lw, info);
    }
    case CUDA_R_64F: {
      cusolverDnDsyevd_bufferSize(h, jobz, uplo, n, static_cast<double*>(A), lda, static_cast<double*>(W), &lw);
      DevBuf w((size_t)lw * 8);
      return cusolverDnDsyevd(h, jobz, uplo, n, static_cast<double*>(A), lda, static_cast<double*>(W),
                              static_cast<double*>(w.p), lw, info);
    }
    case CUDA_C_32F: {
      cusolverDnCheevd_bufferSize(h, jobz, uplo, n, static_cast<cuComplex*>(A), lda, static_cast<float*>(W), &lw);
      DevBuf w((size_t)lw * 8);
      return cusolverDnCheevd(h, jobz, uplo, n, static_cast<cuComplex*>(A), lda, static_cast<float*>(W),
                              static_cast<cuComplex*>(w.p), lw, info);
    }
    case CUDA_C_64F: {
      cusolverDnZheevd_bufferSize(h, jobz, uplo, n, static_cast<cuDoubleComplex*>(A), lda, static_cast<double*>(W),
                                  &lw);
      DevBuf w((size_t)lw * 16);
      return cusolverDnZheevd(h, jobz, uplo, n, static_cast<cuDoubleComplex*>(A), lda, static_cast<double*>(W),
                              static_cast<cuDoubleComplex*>(w.p), lw, info);
    }
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
}

// The inverse from a Cholesky factor, on the host: the triangle uplo names
// of (L L^H)^{-1} (or (U^H U)^{-1}); the other is left as it was.
template <class T> T from_c(std::complex<double> v) {
  if constexpr (std::is_floating_point_v<T>) return (T)v.real();
  else return T(v);
}
template <class T> void potri_host(bool lower, int n, T* a) {
  using C = std::complex<double>;
  auto at = [&](int i, int j) -> T& { return a[(size_t)j * n + i]; };
  // The factor as L (lower): for upper, L = U^H.
  std::vector<C> L((size_t)n * n, 0.0), Li((size_t)n * n, 0.0);
  for (int j = 0; j < n; ++j)
    for (int i = j; i < n; ++i) L[(size_t)j * n + i] = lower ? C(at(i, j)) : std::conj(C(at(j, i)));
  for (int j = 0; j < n; ++j) {  // L^{-1}, column by column
    Li[(size_t)j * n + j] = 1.0 / L[(size_t)j * n + j];
    for (int i = j + 1; i < n; ++i) {
      C s = 0;
      for (int k = j; k < i; ++k) s += L[(size_t)k * n + i] * Li[(size_t)j * n + k];
      Li[(size_t)j * n + i] = -s / L[(size_t)i * n + i];
    }
  }
  for (int j = 0; j < n; ++j)  // (L L^H)^{-1} = L^{-H} L^{-1}
    for (int i = j; i < n; ++i) {
      C s = 0;
      for (int k = i; k < n; ++k) s += std::conj(Li[(size_t)i * n + k]) * Li[(size_t)j * n + k];
      if (lower) at(i, j) = from_c<T>(s);
      else at(j, i) = from_c<T>(std::conj(s));
    }
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT cusolverStatus_t cusolverMgCreate(cusolverMgHandle_t* h) {
  if (!h) return CUSOLVER_STATUS_INVALID_VALUE;
  auto* x = new Handle();
  if (cusolverDnCreate(&x->dn) != CUSOLVER_STATUS_SUCCESS) {
    delete x;
    return CUSOLVER_STATUS_INTERNAL_ERROR;
  }
  *h = reinterpret_cast<cusolverMgHandle_t>(track(x));
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverMgDestroy(cusolverMgHandle_t h) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  untrack(h);
  auto* x = reinterpret_cast<Handle*>(h);
  cusolverDnDestroy(x->dn);
  delete x;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverMgDeviceSelect(cusolverMgHandle_t h, int count, int ids[]) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  int have = 0;
  cudaGetDeviceCount(&have);
  if (count < 1 || !ids) return CUSOLVER_STATUS_INVALID_VALUE;
  for (int i = 0; i < count; ++i)
    if (ids[i] < 0 || ids[i] >= have) return CUSOLVER_STATUS_INVALID_VALUE;
  reinterpret_cast<Handle*>(h)->devices.assign(ids, ids + count);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverMgCreateDeviceGrid(cudaLibMgGrid_t* grid, int32_t rows, int32_t cols,
                                                        const int32_t ids[], cusolverMgGridMapping_t) {
  // One row of devices only (1-D column block cyclic), as NVIDIA's refuses others.
  if (!grid || rows != 1 || cols < 1 || !ids) return CUSOLVER_STATUS_INVALID_VALUE;
  auto* g = new Grid();
  g->rows = rows;
  g->cols = cols;
  g->ids.assign(ids, ids + (size_t)rows * cols);
  *grid = track(g);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverMgDestroyGrid(cudaLibMgGrid_t grid) {
  if (!known(grid)) return CUSOLVER_STATUS_INVALID_VALUE;
  untrack(grid);
  delete static_cast<Grid*>(grid);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverMgCreateMatrixDesc(cudaLibMgMatrixDesc_t* d, int64_t m, int64_t n, int64_t mb,
                                                        int64_t nb, cudaDataType t, const cudaLibMgGrid_t grid) {
  if (!d || !known(grid) || m < 0 || n < 0 || mb < 1 || nb < 1) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!elem_bytes(t)) return CUSOLVER_STATUS_NOT_SUPPORTED;
  *d = track(new Desc{m, n, mb, nb, t, static_cast<const Grid*>(grid)});
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverMgDestroyMatrixDesc(cudaLibMgMatrixDesc_t d) {
  if (!known(d)) return CUSOLVER_STATUS_INVALID_VALUE;
  untrack(d);
  delete static_cast<Desc*>(d);
  return CUSOLVER_STATUS_SUCCESS;
}

// Workspace sizes: the work happens on one device, in buffers of this
// library's own, so callers' buffers are not touched. getrf's is 0, as
// NVIDIA's answers.
static cusolverStatus_t token(cusolverMgHandle_t h, const void* d, int ia, int ja, cudaDataType c, int64_t n,
                              int64_t* lwork, int64_t value) {
  if (const cusolverStatus_t st = check(h, d, ia, ja, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  if (!lwork) return CUSOLVER_STATUS_INVALID_VALUE;
  *lwork = value < 0 ? std::max<int64_t>(1, n) * 64 : value;
  return CUSOLVER_STATUS_SUCCESS;
}

VGPU_EXPORT cusolverStatus_t cusolverMgGetrf_bufferSize(cusolverMgHandle_t h, int, int N, void*[], int IA, int JA,
                                                        cudaLibMgMatrixDesc_t dA, int*[], cudaDataType c,
                                                        int64_t* lwork) {
  return token(h, dA, IA, JA, c, N, lwork, 0);
}
VGPU_EXPORT cusolverStatus_t cusolverMgGetrf(cusolverMgHandle_t h, int M, int N, void* A[], int IA, int JA,
                                             cudaLibMgMatrixDesc_t dA, int* ipiv[], cudaDataType c, void*[], int64_t,
                                             int* info) {
  if (const cusolverStatus_t st = check(h, dA, IA, JA, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  const Desc& d = desc(dA);
  if (!fits(d, IA, JA, M, N) || !A) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<uint8_t> a;
  if (!gather(d, A, IA - 1, JA - 1, M, N, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  DevBuf da(a.size()), dp((size_t)std::max(1, std::min(M, N)) * sizeof(int)), di(sizeof(int));
  if (!da.put(a.data(), a.size())) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const cusolverStatus_t st = dn_getrf(reinterpret_cast<Handle*>(h)->dn, d.type, M, N, da.p, std::max(1, M),
                                       ipiv ? static_cast<int*>(dp.p) : nullptr, static_cast<int*>(di.p));
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  if (info) *info = dev_info(di);
  // The pivots, relative to the submatrix, at columns JA.. of the IPIV row.
  std::vector<int> piv((size_t)std::min(M, N));
  if (!da.get(a.data(), a.size()) || !dp.get(piv.data(), piv.size() * sizeof(int)) ||
      !scatter(d, A, IA - 1, JA - 1, M, N, a) || (ipiv && !scatter_ipiv(d, ipiv, JA - 1, piv)))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  return CUSOLVER_STATUS_SUCCESS;
}

VGPU_EXPORT cusolverStatus_t cusolverMgGetrs_bufferSize(cusolverMgHandle_t h, cublasOperation_t, int N, int, void*[],
                                                        int IA, int JA, cudaLibMgMatrixDesc_t dA, int*[], void*[],
                                                        int IB, int JB, cudaLibMgMatrixDesc_t dB, cudaDataType c,
                                                        int64_t* lwork) {
  if (const cusolverStatus_t st = check(h, dB, IB, JB, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  return token(h, dA, IA, JA, c, N, lwork, -1);
}
VGPU_EXPORT cusolverStatus_t cusolverMgGetrs(cusolverMgHandle_t h, cublasOperation_t op, int N, int NRHS, void* A[],
                                             int IA, int JA, cudaLibMgMatrixDesc_t dA, int* ipiv[], void* B[], int IB,
                                             int JB, cudaLibMgMatrixDesc_t dB, cudaDataType c, void*[], int64_t,
                                             int* info) {
  if (const cusolverStatus_t st = check(h, dA, IA, JA, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  if (const cusolverStatus_t st = check(h, dB, IB, JB, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  const Desc &a_d = desc(dA), &b_d = desc(dB);
  // A itself only, as NVIDIA's (its _bufferSize takes a transpose; getrs does not).
  if (op != CUBLAS_OP_N) return CUSOLVER_STATUS_INVALID_VALUE;
  if (!fits(a_d, IA, JA, N, N) || !fits(b_d, IB, JB, N, NRHS) || !A || !B || !ipiv)
    return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<uint8_t> a, b;
  std::vector<int> piv;
  if (!gather(a_d, A, IA - 1, JA - 1, N, N, &a) || !gather(b_d, B, IB - 1, JB - 1, N, NRHS, &b) ||
      !gather_ipiv(a_d, ipiv, JA - 1, N, &piv))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  DevBuf da(a.size()), db(b.size()), dp(piv.size() * sizeof(int)), di(sizeof(int));
  if (!da.put(a.data(), a.size()) || !db.put(b.data(), b.size()) || !dp.put(piv.data(), piv.size() * sizeof(int)))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  const cusolverStatus_t st = dn_getrs(reinterpret_cast<Handle*>(h)->dn, a_d.type, op, N, NRHS, da.p, std::max(1, N),
                                       static_cast<int*>(dp.p), db.p, std::max(1, N), static_cast<int*>(di.p));
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  if (info) *info = dev_info(di);
  return db.get(b.data(), b.size()) && scatter(b_d, B, IB - 1, JB - 1, N, NRHS, b) ? CUSOLVER_STATUS_SUCCESS
                                                                                   : CUSOLVER_STATUS_EXECUTION_FAILED;
}

VGPU_EXPORT cusolverStatus_t cusolverMgPotrf_bufferSize(cusolverMgHandle_t h, cublasFillMode_t uplo, int N, void*[],
                                                        int IA, int JA, cudaLibMgMatrixDesc_t dA, cudaDataType c,
                                                        int64_t* lwork) {
  if (uplo != CUBLAS_FILL_MODE_LOWER && known(h)) return CUSOLVER_STATUS_INVALID_VALUE;
  return token(h, dA, IA, JA, c, N, lwork, -1);
}
VGPU_EXPORT cusolverStatus_t cusolverMgPotrf(cusolverMgHandle_t h, cublasFillMode_t uplo, int N, void* A[], int IA,
                                             int JA, cudaLibMgMatrixDesc_t dA, cudaDataType c, void*[], int64_t,
                                             int* info) {
  if (const cusolverStatus_t st = check(h, dA, IA, JA, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  const Desc& d = desc(dA);
  if (!fits(d, IA, JA, N, N) || !A || uplo != CUBLAS_FILL_MODE_LOWER) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<uint8_t> a;
  if (!gather(d, A, IA - 1, JA - 1, N, N, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  DevBuf da(a.size()), di(sizeof(int));
  if (!da.put(a.data(), a.size())) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const cusolverStatus_t st =
      dn_potrf(reinterpret_cast<Handle*>(h)->dn, d.type, uplo, N, da.p, std::max(1, N), static_cast<int*>(di.p));
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  const int bad = dev_info(di);
  if (info) *info = bad;
  // Not positive definite: INTERNAL_ERROR with info set, as NVIDIA's answers it.
  if (bad > 0) return CUSOLVER_STATUS_INTERNAL_ERROR;
  return da.get(a.data(), a.size()) && scatter(d, A, IA - 1, JA - 1, N, N, a) ? CUSOLVER_STATUS_SUCCESS
                                                                              : CUSOLVER_STATUS_EXECUTION_FAILED;
}

VGPU_EXPORT cusolverStatus_t cusolverMgPotrs_bufferSize(cusolverMgHandle_t h, cublasFillMode_t, int n, int, void*[],
                                                        int IA, int JA, cudaLibMgMatrixDesc_t dA, void*[], int IB,
                                                        int JB, cudaLibMgMatrixDesc_t dB, cudaDataType c,
                                                        int64_t* lwork) {
  if (const cusolverStatus_t st = check(h, dB, IB, JB, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  return token(h, dA, IA, JA, c, n, lwork, -1);
}
VGPU_EXPORT cusolverStatus_t cusolverMgPotrs(cusolverMgHandle_t h, cublasFillMode_t uplo, int n, int nrhs, void* A[],
                                             int IA, int JA, cudaLibMgMatrixDesc_t dA, void* B[], int IB, int JB,
                                             cudaLibMgMatrixDesc_t dB, cudaDataType c, void*[], int64_t, int* info) {
  if (const cusolverStatus_t st = check(h, dA, IA, JA, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  if (const cusolverStatus_t st = check(h, dB, IB, JB, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  const Desc &a_d = desc(dA), &b_d = desc(dB);
  if (!fits(a_d, IA, JA, n, n) || !fits(b_d, IB, JB, n, nrhs) || !A || !B || uplo != CUBLAS_FILL_MODE_LOWER)
    return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<uint8_t> a, b;
  if (!gather(a_d, A, IA - 1, JA - 1, n, n, &a) || !gather(b_d, B, IB - 1, JB - 1, n, nrhs, &b))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  DevBuf da(a.size()), db(b.size()), di(sizeof(int));
  if (!da.put(a.data(), a.size()) || !db.put(b.data(), b.size())) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const cusolverStatus_t st = dn_potrs(reinterpret_cast<Handle*>(h)->dn, a_d.type, uplo, n, nrhs, da.p, std::max(1, n),
                                       db.p, std::max(1, n), static_cast<int*>(di.p));
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  if (info) *info = dev_info(di);
  return db.get(b.data(), b.size()) && scatter(b_d, B, IB - 1, JB - 1, n, nrhs, b) ? CUSOLVER_STATUS_SUCCESS
                                                                                   : CUSOLVER_STATUS_EXECUTION_FAILED;
}

VGPU_EXPORT cusolverStatus_t cusolverMgPotri_bufferSize(cusolverMgHandle_t h, cublasFillMode_t uplo, int N, void*[],
                                                        int IA, int JA, cudaLibMgMatrixDesc_t dA, cudaDataType c,
                                                        int64_t* lwork) {
  if (uplo != CUBLAS_FILL_MODE_LOWER && known(h)) return CUSOLVER_STATUS_INVALID_VALUE;
  return token(h, dA, IA, JA, c, N, lwork, -1);
}
VGPU_EXPORT cusolverStatus_t cusolverMgPotri(cusolverMgHandle_t h, cublasFillMode_t uplo, int N, void* A[], int IA,
                                             int JA, cudaLibMgMatrixDesc_t dA, cudaDataType c, void*[], int64_t,
                                             int* info) {
  if (const cusolverStatus_t st = check(h, dA, IA, JA, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  const Desc& d = desc(dA);
  if (!fits(d, IA, JA, N, N) || !A) return CUSOLVER_STATUS_INVALID_VALUE;
  if (uplo != CUBLAS_FILL_MODE_LOWER) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<uint8_t> sq;  // the N x N submatrix, contiguous
  if (!gather(d, A, IA - 1, JA - 1, N, N, &sq)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const bool lower = uplo == CUBLAS_FILL_MODE_LOWER;
  int bad = 0;  // a zero on the factor's diagonal: no inverse
  auto diag_zero = [&](auto tag) {
    using T = decltype(tag);
    const T* p = reinterpret_cast<const T*>(sq.data());
    for (int i = 0; i < N && !bad; ++i)
      if (std::abs(std::complex<double>(p[(size_t)i * N + i])) == 0.0) bad = i + 1;
  };
  switch (d.type) {
    case CUDA_R_32F: diag_zero(float{}); if (!bad) potri_host(lower, N, reinterpret_cast<float*>(sq.data())); break;
    case CUDA_R_64F: diag_zero(double{}); if (!bad) potri_host(lower, N, reinterpret_cast<double*>(sq.data())); break;
    case CUDA_C_32F:
      diag_zero(std::complex<float>{});
      if (!bad) potri_host(lower, N, reinterpret_cast<std::complex<float>*>(sq.data()));
      break;
    case CUDA_C_64F:
      diag_zero(std::complex<double>{});
      if (!bad) potri_host(lower, N, reinterpret_cast<std::complex<double>*>(sq.data()));
      break;
    default: return CUSOLVER_STATUS_NOT_SUPPORTED;
  }
  if (info) *info = bad;
  if (bad) return CUSOLVER_STATUS_SUCCESS;
  return scatter(d, A, IA - 1, JA - 1, N, N, sq) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;
}

VGPU_EXPORT cusolverStatus_t cusolverMgSyevd_bufferSize(cusolverMgHandle_t h, cusolverEigMode_t, cublasFillMode_t,
                                                        int N, void*[], int IA, int JA, cudaLibMgMatrixDesc_t dA,
                                                        void*, cudaDataType, cudaDataType c, int64_t* lwork) {
  return token(h, dA, IA, JA, c, N, lwork, -1);
}
VGPU_EXPORT cusolverStatus_t cusolverMgSyevd(cusolverMgHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo,
                                             int N, void* A[], int IA, int JA, cudaLibMgMatrixDesc_t dA, void* W,
                                             cudaDataType tw, cudaDataType c, void*[], int64_t, int* info) {
  if (const cusolverStatus_t st = check(h, dA, IA, JA, c); st != CUSOLVER_STATUS_SUCCESS) return st;
  const Desc& d = desc(dA);
  // The whole matrix only: a submatrix is INVALID_VALUE with info 0, as NVIDIA's answers it.
  if (IA != 1 || JA != 1) {
    if (info) *info = 0;
    return CUSOLVER_STATUS_INVALID_VALUE;
  }
  if (!fits(d, IA, JA, N, N) || !A || !W) return CUSOLVER_STATUS_INVALID_VALUE;
  const cudaDataType real = d.type == CUDA_R_32F || d.type == CUDA_C_32F ? CUDA_R_32F : CUDA_R_64F;
  if (tw != real || uplo != CUBLAS_FILL_MODE_LOWER) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<uint8_t> a;
  if (!gather(d, A, 0, 0, d.m, N, &a)) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const size_t wb = (size_t)N * elem_bytes(real);
  DevBuf da(a.size()), dw(wb), di(sizeof(int));
  if (!da.put(a.data(), a.size())) return CUSOLVER_STATUS_EXECUTION_FAILED;
  const cusolverStatus_t st =
      dn_syevd(reinterpret_cast<Handle*>(h)->dn, d.type, jobz, uplo, N, da.p, (int)d.m, dw.p, static_cast<int*>(di.p));
  if (st != CUSOLVER_STATUS_SUCCESS) return st;
  if (info) *info = dev_info(di);
  if (!dw.get(W, wb)) return CUSOLVER_STATUS_EXECUTION_FAILED;  // W is host memory
  if (jobz != CUSOLVER_EIG_MODE_VECTOR) return CUSOLVER_STATUS_SUCCESS;
  return da.get(a.data(), a.size()) && scatter(d, A, 0, 0, d.m, N, a) ? CUSOLVER_STATUS_SUCCESS
                                                                       : CUSOLVER_STATUS_EXECUTION_FAILED;
}
