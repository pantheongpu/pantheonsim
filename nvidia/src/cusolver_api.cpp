// libvgpucusolver -- VirtualGPU's cuSOLVER, presented as libcusolver.so.12.
//
// The dense (cusolverDn) factorizations: Cholesky, LU with partial pivoting,
// Householder QR, symmetric eigendecomposition and the SVD, in single and
// double precision. Everything runs on the host in double and is rounded to the
// caller's type, so the single-precision results agree with hardware to
// single-precision rounding rather than bit-for-bit.
//
// Conventions follow LAPACK exactly, because that is what callers unpack:
// column-major storage with a leading dimension, 1-based pivots, reflectors
// stored below the diagonal with their tau vector, eigenvalues ascending,
// singular values descending.
//
// Not implemented: the sparse (cusolverSp) and multi-GPU (cusolverMg) modules,
// the 64-bit generic API, and the Jacobi/randomized variants. Those return
// CUSOLVER_STATUS_NOT_SUPPORTED.
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include <cuda_runtime.h>

namespace {

bool quiet() { const char* q = std::getenv("VGPU_QUIET"); return q && q[0] == '1'; }

struct Handle { cudaStream_t stream = nullptr; };

std::mutex g_mu;
std::set<const void*> g_live;
template <class T> T* track(T* p) { std::lock_guard<std::mutex> l(g_mu); g_live.insert(p); return p; }
bool known(const void* p) { std::lock_guard<std::mutex> l(g_mu); return p && g_live.count(p); }
void untrack(const void* p) { std::lock_guard<std::mutex> l(g_mu); g_live.erase(p); }

// Column-major access into a host copy that keeps the caller's leading dimension.
struct Mat {
  std::vector<double> v;
  int ld = 0;
  double& operator()(int r, int c) { return v[(size_t)c * ld + r]; }
  double operator()(int r, int c) const { return v[(size_t)c * ld + r]; }
};

template <class T> bool load(const void* dev, size_t n, Mat* m, int ld) {
  m->ld = ld;
  m->v.assign(n, 0.0);
  if (!n) return true;
  std::vector<T> tmp(n);
  if (cudaMemcpy(tmp.data(), dev, n * sizeof(T), cudaMemcpyDeviceToHost) != cudaSuccess)
    return false;
  for (size_t i = 0; i < n; ++i) m->v[i] = (double)tmp[i];
  return true;
}
template <class T> bool save(void* dev, const std::vector<double>& v) {
  if (v.empty()) return true;
  std::vector<T> tmp(v.size());
  for (size_t i = 0; i < v.size(); ++i) tmp[i] = (T)v[i];
  return cudaMemcpy(dev, tmp.data(), tmp.size() * sizeof(T), cudaMemcpyHostToDevice) == cudaSuccess;
}
void set_info(int* dev_info, int value) {
  if (dev_info) cudaMemcpy(dev_info, &value, sizeof(int), cudaMemcpyHostToDevice);
}

/* ---- factorizations, all column-major and all in double ---- */

// Cholesky. Returns 0, or the 1-based index of the leading minor that failed.
int potrf(Mat& A, int n, bool lower) {
  for (int j = 0; j < n; ++j) {
    double d = A(j, j);
    for (int k = 0; k < j; ++k) {
      const double v = lower ? A(j, k) : A(k, j);
      d -= v * v;
    }
    if (!(d > 0.0)) return j + 1;
    d = std::sqrt(d);
    (lower ? A(j, j) : A(j, j)) = d;
    for (int i = j + 1; i < n; ++i) {
      double s = lower ? A(i, j) : A(j, i);
      for (int k = 0; k < j; ++k)
        s -= (lower ? A(i, k) : A(k, i)) * (lower ? A(j, k) : A(k, j));
      (lower ? A(i, j) : A(j, i)) = s / d;
    }
  }
  return 0;
}

// LU with partial pivoting. ipiv is LAPACK's: 1-based row that row i was
// swapped with. A null ipiv means the caller asked for no pivoting.
int getrf(Mat& A, int m, int n, std::vector<int>* ipiv) {
  const int k = std::min(m, n);
  if (ipiv) ipiv->assign(k, 0);
  int info = 0;
  for (int j = 0; j < k; ++j) {
    int p = j;
    if (ipiv) {
      double best = std::fabs(A(j, j));
      for (int i = j + 1; i < m; ++i)
        if (std::fabs(A(i, j)) > best) { best = std::fabs(A(i, j)); p = i; }
      (*ipiv)[j] = p + 1;
      if (p != j)
        for (int c = 0; c < n; ++c) std::swap(A(j, c), A(p, c));
    }
    const double piv = A(j, j);
    if (piv == 0.0) { if (!info) info = j + 1; continue; }
    for (int i = j + 1; i < m; ++i) {
      A(i, j) /= piv;
      const double f = A(i, j);
      for (int c = j + 1; c < n; ++c) A(i, c) -= f * A(j, c);
    }
  }
  return info;
}

void apply_pivots(Mat& B, int n, int nrhs, const std::vector<int>& ipiv, bool forward) {
  const int k = (int)ipiv.size();
  for (int s = 0; s < k; ++s) {
    const int i = forward ? s : k - 1 - s;
    const int p = ipiv[i] - 1;
    if (p == i || p < 0 || p >= n) continue;
    for (int c = 0; c < nrhs; ++c) std::swap(B(i, c), B(p, c));
  }
}

// Triangular solves. `unit` means an implicit unit diagonal (LU's L).
void trsm_lower(const Mat& A, Mat& B, int n, int nrhs, bool unit, bool trans) {
  for (int c = 0; c < nrhs; ++c) {
    if (!trans) {
      for (int i = 0; i < n; ++i) {
        double s = B(i, c);
        for (int k = 0; k < i; ++k) s -= A(i, k) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    } else {
      for (int i = n - 1; i >= 0; --i) {
        double s = B(i, c);
        for (int k = i + 1; k < n; ++k) s -= A(k, i) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    }
  }
}

void trsm_upper(const Mat& A, Mat& B, int n, int nrhs, bool unit, bool trans) {
  for (int c = 0; c < nrhs; ++c) {
    if (!trans) {
      for (int i = n - 1; i >= 0; --i) {
        double s = B(i, c);
        for (int k = i + 1; k < n; ++k) s -= A(i, k) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    } else {
      for (int i = 0; i < n; ++i) {
        double s = B(i, c);
        for (int k = 0; k < i; ++k) s -= A(k, i) * B(k, c);
        B(i, c) = unit ? s : s / A(i, i);
      }
    }
  }
}

// Householder QR, LAPACK's dgeqrf/dlarfg convention: beta carries the opposite
// sign to the leading element, v[0] is an implicit 1, and the rest of v
// overwrites the column below the diagonal.
void geqrf(Mat& A, int m, int n, std::vector<double>* tau) {
  const int k = std::min(m, n);
  tau->assign(k, 0.0);
  std::vector<double> v((size_t)m);
  for (int j = 0; j < k; ++j) {
    double norm = 0.0;
    for (int i = j; i < m; ++i) norm += A(i, j) * A(i, j);
    norm = std::sqrt(norm);
    if (norm == 0.0) { (*tau)[j] = 0.0; continue; }
    const double alpha = A(j, j);
    const double beta = alpha >= 0.0 ? -norm : norm;
    const double t = (beta - alpha) / beta;
    (*tau)[j] = t;
    const double scale = alpha - beta;
    v[j] = 1.0;
    for (int i = j + 1; i < m; ++i) { A(i, j) /= scale; v[i] = A(i, j); }
    A(j, j) = beta;
    for (int c = j + 1; c < n; ++c) {
      double dot = 0.0;
      for (int i = j; i < m; ++i) dot += v[i] * A(i, c);
      dot *= t;
      for (int i = j; i < m; ++i) A(i, c) -= dot * v[i];
    }
  }
}

// Overwrite the first n columns of A with the corresponding columns of Q.
void orgqr(Mat& A, int m, int n, int k, const std::vector<double>& tau) {
  std::vector<double> q((size_t)m * n, 0.0);
  auto Q = [&](int r, int c) -> double& { return q[(size_t)c * m + r]; };
  for (int c = 0; c < n; ++c) Q(c, c) = 1.0;
  std::vector<double> v((size_t)m);
  for (int j = k - 1; j >= 0; --j) {
    v[j] = 1.0;
    for (int i = j + 1; i < m; ++i) v[i] = A(i, j);
    for (int c = 0; c < n; ++c) {
      double dot = 0.0;
      for (int i = j; i < m; ++i) dot += v[i] * Q(i, c);
      dot *= tau[j];
      for (int i = j; i < m; ++i) Q(i, c) -= dot * v[i];
    }
  }
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r) A(r, c) = Q(r, c);
}

// C := op(Q) C  (left) or C op(Q)  (right), Q given by the reflectors in A.
void ormqr(const Mat& A, Mat& C, int m, int n, int k, const std::vector<double>& tau, bool left,
           bool trans) {
  const int rows = left ? m : n;  // length of each reflector's active span
  std::vector<double> v((size_t)rows);
  // Q = H(0) H(1) ... H(k-1); Q^T applies them in order, Q in reverse.
  for (int s = 0; s < k; ++s) {
    const int j = (left != trans) ? k - 1 - s : s;
    for (int i = 0; i < rows; ++i) v[i] = i < j ? 0.0 : (i == j ? 1.0 : A(i, j));
    if (left) {
      for (int c = 0; c < n; ++c) {
        double dot = 0.0;
        for (int i = j; i < m; ++i) dot += v[i] * C(i, c);
        dot *= tau[j];
        for (int i = j; i < m; ++i) C(i, c) -= dot * v[i];
      }
    } else {
      for (int r = 0; r < m; ++r) {
        double dot = 0.0;
        for (int i = j; i < n; ++i) dot += v[i] * C(r, i);
        dot *= tau[j];
        for (int i = j; i < n; ++i) C(r, i) -= dot * v[i];
      }
    }
  }
}

// Cyclic Jacobi for the symmetric eigenproblem. Eigenvalues come back ascending
// with their vectors, matching LAPACK's syevd.
void syev(std::vector<double>& a, int n, std::vector<double>* w, std::vector<double>* vecs) {
  auto A = [&](int r, int c) -> double& { return a[(size_t)c * n + r]; };
  vecs->assign((size_t)n * n, 0.0);
  auto V = [&](int r, int c) -> double& { return (*vecs)[(size_t)c * n + r]; };
  for (int i = 0; i < n; ++i) V(i, i) = 1.0;
  for (int sweep = 0; sweep < 100; ++sweep) {
    double off = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) off += A(p, q) * A(p, q);
    if (off <= 1e-30) break;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        if (std::fabs(A(p, q)) < 1e-300) continue;
        const double theta = (A(q, q) - A(p, p)) / (2.0 * A(p, q));
        const double t = (theta >= 0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), s = t * c;
        for (int i = 0; i < n; ++i) {
          const double aip = A(i, p), aiq = A(i, q);
          A(i, p) = c * aip - s * aiq;
          A(i, q) = s * aip + c * aiq;
        }
        for (int i = 0; i < n; ++i) {
          const double api = A(p, i), aqi = A(q, i);
          A(p, i) = c * api - s * aqi;
          A(q, i) = s * api + c * aqi;
        }
        for (int i = 0; i < n; ++i) {
          const double vip = V(i, p), viq = V(i, q);
          V(i, p) = c * vip - s * viq;
          V(i, q) = s * vip + c * viq;
        }
      }
  }
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int x, int y) { return A(x, x) < A(y, y); });
  w->resize(n);
  std::vector<double> sorted((size_t)n * n);
  for (int i = 0; i < n; ++i) {
    (*w)[i] = A(order[i], order[i]);
    for (int r = 0; r < n; ++r) sorted[(size_t)i * n + r] = V(r, order[i]);
  }
  *vecs = std::move(sorted);
}

// One-sided Jacobi SVD of an m-by-n matrix with m >= n. Columns of `a` become
// U*S; `vt` accumulates V, and is transposed by the caller.
void svd(std::vector<double>& a, int m, int n, std::vector<double>* s, std::vector<double>* v) {
  auto A = [&](int r, int c) -> double& { return a[(size_t)c * m + r]; };
  v->assign((size_t)n * n, 0.0);
  auto V = [&](int r, int c) -> double& { return (*v)[(size_t)c * n + r]; };
  for (int i = 0; i < n; ++i) V(i, i) = 1.0;
  for (int sweep = 0; sweep < 60; ++sweep) {
    double worst = 0.0;
    for (int p = 0; p < n; ++p)
      for (int q = p + 1; q < n; ++q) {
        double app = 0, aqq = 0, apq = 0;
        for (int i = 0; i < m; ++i) {
          app += A(i, p) * A(i, p);
          aqq += A(i, q) * A(i, q);
          apq += A(i, p) * A(i, q);
        }
        if (app == 0.0 || aqq == 0.0) continue;
        worst = std::max(worst, std::fabs(apq) / std::sqrt(app * aqq));
        if (std::fabs(apq) <= 1e-15 * std::sqrt(app * aqq)) continue;
        const double theta = (aqq - app) / (2.0 * apq);
        const double t = (theta >= 0 ? 1.0 : -1.0) /
                         (std::fabs(theta) + std::sqrt(theta * theta + 1.0));
        const double c = 1.0 / std::sqrt(t * t + 1.0), sn = t * c;
        for (int i = 0; i < m; ++i) {
          const double aip = A(i, p), aiq = A(i, q);
          A(i, p) = c * aip - sn * aiq;
          A(i, q) = sn * aip + c * aiq;
        }
        for (int i = 0; i < n; ++i) {
          const double vip = V(i, p), viq = V(i, q);
          V(i, p) = c * vip - sn * viq;
          V(i, q) = sn * vip + c * viq;
        }
      }
    if (worst < 1e-14) break;
  }
  s->assign(n, 0.0);
  for (int c = 0; c < n; ++c) {
    double norm = 0.0;
    for (int i = 0; i < m; ++i) norm += A(i, c) * A(i, c);
    (*s)[c] = std::sqrt(norm);
  }
  // LAPACK reports singular values in descending order.
  std::vector<int> order(n);
  for (int i = 0; i < n; ++i) order[i] = i;
  std::sort(order.begin(), order.end(), [&](int x, int y) { return (*s)[x] > (*s)[y]; });
  std::vector<double> a2((size_t)m * n), v2((size_t)n * n), s2(n);
  for (int c = 0; c < n; ++c) {
    const int o = order[c];
    s2[c] = (*s)[o];
    // Fix the sign so the first nonzero component of each left vector is
    // positive: the decomposition is otherwise only unique up to a sign, and a
    // convention makes the result reproducible.
    double sign = 1.0;
    for (int i = 0; i < m; ++i)
      if (std::fabs(A(i, o)) > 1e-12) { sign = A(i, o) < 0 ? -1.0 : 1.0; break; }
    for (int i = 0; i < m; ++i) a2[(size_t)c * m + i] = sign * A(i, o);
    for (int i = 0; i < n; ++i) v2[(size_t)c * n + i] = sign * V(i, o);
  }
  a = std::move(a2);
  *v = std::move(v2);
  *s = std::move(s2);
}

int lwork_hint(int n) { return std::max(1, n) * 64; }

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

/* ---- handle ---- */

VGPU_EXPORT cusolverStatus_t cusolverDnCreate(cusolverDnHandle_t* h) {
  if (!h) return CUSOLVER_STATUS_INVALID_VALUE;
  *h = reinterpret_cast<cusolverDnHandle_t>(track(new Handle()));
  if (!quiet())
    std::fprintf(stderr, "[vgpu] cuSOLVER handle created (host-computed dense factorizations)\n");
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnDestroy(cusolverDnHandle_t h) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  untrack(h); delete reinterpret_cast<Handle*>(h);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnSetStream(cusolverDnHandle_t h, cudaStream_t s) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  reinterpret_cast<Handle*>(h)->stream = s;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverDnGetStream(cusolverDnHandle_t h, cudaStream_t* s) {
  if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (s) *s = reinterpret_cast<Handle*>(h)->stream;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverGetProperty(libraryPropertyType type, int* v) {
  if (!v) return CUSOLVER_STATUS_INVALID_VALUE;
  switch (type) {
    case MAJOR_VERSION: *v = CUSOLVER_VER_MAJOR; break;
    case MINOR_VERSION: *v = CUSOLVER_VER_MINOR; break;
    case PATCH_LEVEL: *v = CUSOLVER_VER_PATCH; break;
    default: return CUSOLVER_STATUS_INVALID_VALUE;
  }
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverGetVersion(int* v) {
  if (!v) return CUSOLVER_STATUS_INVALID_VALUE;
  *v = CUSOLVER_VERSION;
  return CUSOLVER_STATUS_SUCCESS;
}

/* ---- Cholesky ---- */

#define VGPU_POTRF(P, T)                                                                       \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrf_bufferSize(                                \
      cusolverDnHandle_t h, cublasFillMode_t, int n, T*, int, int* lwork) {                    \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                     \
    if (lwork) *lwork = lwork_hint(n);                                                         \
    return CUSOLVER_STATUS_SUCCESS;                                                            \
  }                                                                                            \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrf(cusolverDnHandle_t h, cublasFillMode_t uplo,\
                                                    int n, T* A, int lda, T*, int, int* info) { \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                     \
    if (n < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                   \
    Mat a;                                                                                     \
    if (!load<T>(A, (size_t)lda * n, &a, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;        \
    const int rc = potrf(a, n, uplo == CUBLAS_FILL_MODE_LOWER);                                \
    set_info(info, rc);                                                                        \
    if (rc == 0 && !save<T>(A, a.v)) return CUSOLVER_STATUS_EXECUTION_FAILED;                  \
    return CUSOLVER_STATUS_SUCCESS;                                                            \
  }                                                                                            \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##potrs(cusolverDnHandle_t h, cublasFillMode_t uplo,\
                                                    int n, int nrhs, const T* A, int lda, T* B, \
                                                    int ldb, int* info) {                       \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                     \
    Mat a, b;                                                                                  \
    if (!load<T>(A, (size_t)lda * n, &a, lda) || !load<T>(B, (size_t)ldb * nrhs, &b, ldb))     \
      return CUSOLVER_STATUS_EXECUTION_FAILED;                                                 \
    if (uplo == CUBLAS_FILL_MODE_LOWER) {                                                      \
      trsm_lower(a, b, n, nrhs, false, false);                                                 \
      trsm_lower(a, b, n, nrhs, false, true);                                                  \
    } else {                                                                                   \
      trsm_upper(a, b, n, nrhs, false, true);                                                  \
      trsm_upper(a, b, n, nrhs, false, false);                                                 \
    }                                                                                          \
    set_info(info, 0);                                                                         \
    return save<T>(B, b.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;       \
  }
VGPU_POTRF(S, float)
VGPU_POTRF(D, double)

/* ---- LU ---- */

#define VGPU_GETRF(P, T)                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##getrf_bufferSize(cusolverDnHandle_t h, int m,     \
                                                               int n, T*, int, int* lwork) {    \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##getrf(cusolverDnHandle_t h, int m, int n, T* A,   \
                                                    int lda, T*, int* devIpiv, int* info) {     \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (m < 0 || n < 0 || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;           \
    Mat a;                                                                                      \
    if (!load<T>(A, (size_t)lda * n, &a, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;         \
    std::vector<int> ipiv;                                                                      \
    const int rc = getrf(a, m, n, devIpiv ? &ipiv : nullptr);                                   \
    set_info(info, rc);                                                                         \
    if (devIpiv && !ipiv.empty() &&                                                             \
        cudaMemcpy(devIpiv, ipiv.data(), ipiv.size() * sizeof(int), cudaMemcpyHostToDevice) !=  \
            cudaSuccess)                                                                        \
      return CUSOLVER_STATUS_EXECUTION_FAILED;                                                  \
    return save<T>(A, a.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;        \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##getrs(                                            \
      cusolverDnHandle_t h, cublasOperation_t trans, int n, int nrhs, const T* A, int lda,      \
      const int* devIpiv, T* B, int ldb, int* info) {                                           \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    Mat a, b;                                                                                   \
    if (!load<T>(A, (size_t)lda * n, &a, lda) || !load<T>(B, (size_t)ldb * nrhs, &b, ldb))      \
      return CUSOLVER_STATUS_EXECUTION_FAILED;                                                  \
    std::vector<int> ipiv;                                                                      \
    if (devIpiv) {                                                                              \
      ipiv.resize(n);                                                                           \
      if (cudaMemcpy(ipiv.data(), devIpiv, (size_t)n * sizeof(int), cudaMemcpyDeviceToHost) !=  \
          cudaSuccess)                                                                          \
        return CUSOLVER_STATUS_EXECUTION_FAILED;                                                \
    }                                                                                           \
    if (trans == CUBLAS_OP_N) {                                                                 \
      if (!ipiv.empty()) apply_pivots(b, n, nrhs, ipiv, true);                                  \
      trsm_lower(a, b, n, nrhs, true, false);                                                   \
      trsm_upper(a, b, n, nrhs, false, false);                                                  \
    } else {                                                                                    \
      trsm_upper(a, b, n, nrhs, false, true);                                                   \
      trsm_lower(a, b, n, nrhs, true, true);                                                    \
      if (!ipiv.empty()) apply_pivots(b, n, nrhs, ipiv, false);                                 \
    }                                                                                           \
    set_info(info, 0);                                                                          \
    return save<T>(B, b.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;        \
  }
VGPU_GETRF(S, float)
VGPU_GETRF(D, double)

/* ---- QR ---- */

#define VGPU_GEQRF(P, T)                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##geqrf_bufferSize(cusolverDnHandle_t h, int m,     \
                                                               int n, T*, int, int* lwork) {    \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##geqrf(cusolverDnHandle_t h, int m, int n, T* A,   \
                                                    int lda, T* tau, T*, int, int* info) {      \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (m < 0 || n < 0 || lda < std::max(1, m)) return CUSOLVER_STATUS_INVALID_VALUE;           \
    Mat a;                                                                                      \
    if (!load<T>(A, (size_t)lda * n, &a, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;         \
    std::vector<double> t;                                                                      \
    geqrf(a, m, n, &t);                                                                         \
    set_info(info, 0);                                                                          \
    if (!save<T>(A, a.v) || !save<T>(tau, t)) return CUSOLVER_STATUS_EXECUTION_FAILED;          \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##orgqr_bufferSize(                                 \
      cusolverDnHandle_t h, int m, int n, int, const T*, int, const T*, int* lwork) {           \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##orgqr(cusolverDnHandle_t h, int m, int n, int k,  \
                                                    T* A, int lda, const T* tau, T*, int,       \
                                                    int* info) {                                \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    Mat a;                                                                                      \
    if (!load<T>(A, (size_t)lda * n, &a, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;         \
    Mat t;                                                                                      \
    if (!load<T>(tau, (size_t)k, &t, k)) return CUSOLVER_STATUS_EXECUTION_FAILED;               \
    orgqr(a, m, n, k, t.v);                                                                     \
    set_info(info, 0);                                                                          \
    return save<T>(A, a.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;        \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##ormqr_bufferSize(                                 \
      cusolverDnHandle_t h, cublasSideMode_t, cublasOperation_t, int m, int n, int, const T*,   \
      int, const T*, const T*, int, int* lwork) {                                               \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##ormqr(                                            \
      cusolverDnHandle_t h, cublasSideMode_t side, cublasOperation_t trans, int m, int n, int k, \
      const T* A, int lda, const T* tau, T* C, int ldc, T*, int, int* info) {                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    const int arows = side == CUBLAS_SIDE_LEFT ? m : n;                                         \
    Mat a, c, t;                                                                                \
    if (!load<T>(A, (size_t)lda * k, &a, lda) || !load<T>(C, (size_t)ldc * n, &c, ldc) ||       \
        !load<T>(tau, (size_t)k, &t, k))                                                        \
      return CUSOLVER_STATUS_EXECUTION_FAILED;                                                  \
    (void)arows;                                                                                \
    ormqr(a, c, m, n, k, t.v, side == CUBLAS_SIDE_LEFT, trans != CUBLAS_OP_N);                  \
    set_info(info, 0);                                                                          \
    return save<T>(C, c.v) ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_EXECUTION_FAILED;        \
  }
VGPU_GEQRF(S, float)
VGPU_GEQRF(D, double)

/* ---- symmetric eigenproblem ---- */

#define VGPU_SYEVD(P, T)                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##syevd_bufferSize(                                 \
      cusolverDnHandle_t h, cusolverEigMode_t, cublasFillMode_t, int n, const T*, int,          \
      const T*, int* lwork) {                                                                   \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (lwork) *lwork = lwork_hint(n);                                                          \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##syevd(                                            \
      cusolverDnHandle_t h, cusolverEigMode_t jobz, cublasFillMode_t uplo, int n, T* A, int lda, \
      T* W, T*, int, int* info) {                                                               \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (n < 0 || lda < std::max(1, n)) return CUSOLVER_STATUS_INVALID_VALUE;                    \
    Mat a;                                                                                      \
    if (!load<T>(A, (size_t)lda * n, &a, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;         \
    /* Only one triangle is meaningful; mirror it so the solver sees a full     \
       symmetric matrix. */                                                                     \
    std::vector<double> full((size_t)n * n);                                                    \
    for (int c = 0; c < n; ++c)                                                                 \
      for (int r = 0; r < n; ++r) {                                                             \
        const bool take = uplo == CUBLAS_FILL_MODE_LOWER ? r >= c : r <= c;                     \
        full[(size_t)c * n + r] = take ? a(r, c) : a(c, r);                                     \
      }                                                                                         \
    std::vector<double> w, v;                                                                   \
    syev(full, n, &w, &v);                                                                      \
    set_info(info, 0);                                                                          \
    if (!save<T>(W, w)) return CUSOLVER_STATUS_EXECUTION_FAILED;                                \
    if (jobz == CUSOLVER_EIG_MODE_VECTOR) {                                                     \
      for (int c = 0; c < n; ++c)                                                               \
        for (int r = 0; r < n; ++r) a(r, c) = v[(size_t)c * n + r];                             \
      if (!save<T>(A, a.v)) return CUSOLVER_STATUS_EXECUTION_FAILED;                            \
    }                                                                                           \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }
VGPU_SYEVD(S, float)
VGPU_SYEVD(D, double)

/* ---- SVD ---- */

#define VGPU_GESVD(P, T)                                                                        \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvd_bufferSize(cusolverDnHandle_t h, int m,     \
                                                               int n, int* lwork) {             \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    if (lwork) *lwork = lwork_hint(std::max(m, n));                                             \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }                                                                                             \
  VGPU_EXPORT cusolverStatus_t cusolverDn##P##gesvd(                                            \
      cusolverDnHandle_t h, signed char jobu, signed char jobvt, int m, int n, T* A, int lda,   \
      T* S, T* U, int ldu, T* VT, int ldvt, T*, int, T*, int* info) {                           \
    if (!known(h)) return CUSOLVER_STATUS_NOT_INITIALIZED;                                      \
    /* Hardware's gesvd requires m >= n; say so rather than inventing an answer. */             \
    if (m < n) return CUSOLVER_STATUS_NOT_SUPPORTED;                                            \
    if (jobu == 'O' || jobvt == 'O') return CUSOLVER_STATUS_NOT_SUPPORTED;                      \
    Mat a;                                                                                      \
    if (!load<T>(A, (size_t)lda * n, &a, lda)) return CUSOLVER_STATUS_EXECUTION_FAILED;         \
    std::vector<double> packed((size_t)m * n);                                                  \
    for (int c = 0; c < n; ++c)                                                                 \
      for (int r = 0; r < m; ++r) packed[(size_t)c * m + r] = a(r, c);                          \
    std::vector<double> s, v;                                                                   \
    svd(packed, m, n, &s, &v);                                                                  \
    set_info(info, 0);                                                                          \
    if (!save<T>(S, s)) return CUSOLVER_STATUS_EXECUTION_FAILED;                                \
    if (U && jobu != 'N') {                                                                     \
      const int ucols = jobu == 'A' ? m : n;                                                    \
      std::vector<double> u((size_t)ldu * ucols, 0.0);                                          \
      for (int c = 0; c < std::min(n, ucols); ++c) {                                            \
        const double sc = s[c] > 0 ? 1.0 / s[c] : 0.0;                                          \
        for (int r = 0; r < m; ++r) u[(size_t)c * ldu + r] = packed[(size_t)c * m + r] * sc;    \
      }                                                                                         \
      if (!save<T>(U, u)) return CUSOLVER_STATUS_EXECUTION_FAILED;                              \
    }                                                                                           \
    if (VT && jobvt != 'N') {                                                                   \
      const int vrows = jobvt == 'A' ? n : n;                                                   \
      std::vector<double> vt((size_t)ldvt * n, 0.0);                                            \
      for (int c = 0; c < n; ++c)                                                               \
        for (int r = 0; r < vrows; ++r) vt[(size_t)c * ldvt + r] = v[(size_t)r * n + c];        \
      if (!save<T>(VT, vt)) return CUSOLVER_STATUS_EXECUTION_FAILED;                            \
    }                                                                                           \
    return CUSOLVER_STATUS_SUCCESS;                                                             \
  }
VGPU_GESVD(S, float)
VGPU_GESVD(D, double)
