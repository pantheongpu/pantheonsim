// The cuSOLVER and cuBLAS entry points torch.linalg reaches, checked by the
// defining property of each result rather than against a second implementation:
// a solve's residual, a factorization's reconstruction, an eigenpair's
// equation, the orthogonality of the vectors.
//
//   cholesky, cholesky_solve   cusolverDnXpotrf/Xpotrs, potrfBatched/potrsBatched
//   qr                         cusolverDnXgeqrf
//   eigh                       cusolverDnXsyevd, syevj, syevjBatched, XsyevBatched
//   eig                        cusolverDnXgeev
//   svd                        gesvdj, gesvdjBatched, gesvdaStridedBatched
//   inv, solve (batched)       cublasgetrfBatched, cublasgetrsBatched
//   solve_triangular, lstsq    cublastrsm, cublastrsmBatched
//
// Every matrix is column-major, as both libraries take it.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <complex>
#include <cmath>
#include <cstdio>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}

#define CK(x)                                                                    \
  do {                                                                           \
    const int r_ = (int)(x);                                                     \
    if (r_ != 0) {                                                               \
      std::printf("FAIL %s returned %d\n", #x, r_);                             \
      ++failures;                                                                \
      return;                                                                    \
    }                                                                            \
  } while (0)

template <class T> using M = std::vector<T>;

static double val(int i, int j, int seed) {
  return std::sin(0.7 * i + 1.3 * j + 0.1 * seed) + 0.3 * std::cos(2.1 * i * j + seed);
}
// A general m x n matrix.
template <class T> M<T> general(int m, int n, int seed) {
  M<T> a((size_t)m * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) a[i + (size_t)j * m] = (T)val(i, j, seed);
  return a;
}
// A symmetric positive definite n x n matrix: G^T G + n I.
template <class T> M<T> spd(int n, int seed) {
  const M<T> g = general<T>(n, n, seed);
  M<T> a((size_t)n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double s = i == j ? n : 0;
      for (int k = 0; k < n; ++k) s += (double)g[k + (size_t)i * n] * g[k + (size_t)j * n];
      a[i + (size_t)j * n] = (T)s;
    }
  return a;
}
// A symmetric (not definite) n x n matrix.
template <class T> M<T> symmetric(int n, int seed) {
  M<T> a((size_t)n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) a[i + (size_t)j * n] = a[j + (size_t)i * n] = (T)val(i, j, seed);
  return a;
}

template <class T> T* upload(const M<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> M<T> download(const T* d, size_t n) {
  M<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// max |A x - b| over every column, relative to |b|.
template <class T> double residual(const M<T>& a, int n, const M<T>& x, const M<T>& b, int nrhs) {
  double err = 0, scale = 1e-30;
  for (int c = 0; c < nrhs; ++c)
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int k = 0; k < n; ++k) s += (double)a[i + (size_t)k * n] * x[k + (size_t)c * n];
      err = std::fmax(err, std::fabs(s - b[i + (size_t)c * n]));
      scale = std::fmax(scale, std::fabs((double)b[i + (size_t)c * n]));
    }
  return err / scale;
}

// max |A - U diag(S) V^T| relative to max |A|, over the leading k triplets,
// and max |U^T U - I|, |V^T V - I| over those columns.
template <class T>
double svd_error(const T* a, int m, int n, const T* s, const T* u, int ldu, const T* v, int ldv, int k,
                 bool* orthonormal) {
  double err = 0, scale = 1e-30;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double r = 0;
      for (int t = 0; t < k; ++t) r += (double)u[i + (size_t)t * ldu] * s[t] * v[j + (size_t)t * ldv];
      err = std::fmax(err, std::fabs(r - a[i + (size_t)j * m]));
      scale = std::fmax(scale, std::fabs((double)a[i + (size_t)j * m]));
    }
  double orth = 0;
  for (int p = 0; p < k; ++p)
    for (int q = 0; q < k; ++q) {
      double du = 0, dv = 0;
      for (int i = 0; i < m; ++i) du += (double)u[i + (size_t)p * ldu] * u[i + (size_t)q * ldu];
      for (int j = 0; j < n; ++j) dv += (double)v[j + (size_t)p * ldv] * v[j + (size_t)q * ldv];
      orth = std::fmax(orth, std::fmax(std::fabs(du - (p == q)), std::fabs(dv - (p == q))));
    }
  bool sorted = true;
  for (int t = 1; t < k; ++t) sorted = sorted && s[t - 1] >= s[t];
  *orthonormal = orth < (sizeof(T) == 4 ? 1e-4 : 1e-8) && sorted;
  return err / scale;
}

// max |A v - w v| over every eigenpair, relative to max |w|, and whether the
// eigenvalues come back ascending as cuSOLVER returns them.
template <class T> double eig_error(const M<T>& a, int n, const T* v, const T* w, bool* ascending) {
  double err = 0, scale = 1e-30;
  for (int t = 0; t < n; ++t) {
    scale = std::fmax(scale, std::fabs((double)w[t]));
    for (int i = 0; i < n; ++i) {
      double s = 0;
      for (int k = 0; k < n; ++k) s += (double)a[i + (size_t)k * n] * v[k + (size_t)t * n];
      err = std::fmax(err, std::fabs(s - (double)w[t] * v[i + (size_t)t * n]));
    }
  }
  *ascending = true;
  for (int t = 1; t < n; ++t) *ascending = *ascending && w[t - 1] <= w[t];
  return err / scale;
}

static cusolverDnHandle_t sh;
static cusolverDnParams_t params;
static cublasHandle_t bh;

static void cholesky() {
  const int n = 6, nrhs = 2;
  const auto a = spd<float>(n, 1);
  const auto b = general<float>(n, nrhs, 2);
  float* da = upload(a);
  float* db = upload(b);
  int* dinfo = upload(M<int>{7});
  size_t dev = 0, host = 0;
  CK(cusolverDnXpotrf_bufferSize(sh, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, da, n, CUDA_R_32F,
                                 &dev, &host));
  void* work = nullptr;
  cudaMalloc(&work, dev);
  CK(cusolverDnXpotrf(sh, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, da, n, CUDA_R_32F, work, dev,
                      nullptr, host, dinfo));
  CK(cusolverDnXpotrs(sh, params, CUBLAS_FILL_MODE_LOWER, n, nrhs, CUDA_R_32F, da, n, CUDA_R_32F, db, n,
                      dinfo));
  const auto x = download(db, b.size());
  const int info = download(dinfo, 1)[0];
  const double e = residual(a, n, x, b, nrhs);
  check(e < 1e-5 && info == 0, "Xpotrf + Xpotrs solve an SPD system", e);
  cudaFree(work);
  cudaFree(da);
  cudaFree(db);
  cudaFree(dinfo);
}

static void cholesky_batched() {
  const int n = 5, batch = 3;
  M<double*> pa(batch), pb(batch);
  M<M<double>> as, bs;
  for (int k = 0; k < batch; ++k) {
    as.push_back(spd<double>(n, 10 + k));
    bs.push_back(general<double>(n, 1, 20 + k));
    pa[k] = upload(as[k]);
    pb[k] = upload(bs[k]);
  }
  double** dpa = upload(pa);
  double** dpb = upload(pb);
  int* dinfo = upload(M<int>(batch, 7));
  CK(cusolverDnDpotrfBatched(sh, CUBLAS_FILL_MODE_UPPER, n, dpa, n, dinfo, batch));
  const auto infos = download(dinfo, batch);
  CK(cusolverDnDpotrsBatched(sh, CUBLAS_FILL_MODE_UPPER, n, 1, dpa, n, dpb, n, dinfo, batch));
  double e = 0;
  for (int k = 0; k < batch; ++k) e = std::fmax(e, residual(as[k], n, download(pb[k], n), bs[k], 1));
  const bool ok = std::all_of(infos.begin(), infos.end(), [](int i) { return i == 0; });
  check(e < 1e-12 && ok, "potrfBatched + potrsBatched (upper) solve each system", e);
  for (int k = 0; k < batch; ++k) { cudaFree(pa[k]); cudaFree(pb[k]); }
  cudaFree(dpa);
  cudaFree(dpb);
  cudaFree(dinfo);
}

// A = QR, so R^T R = A^T A whatever Q is; R is the upper triangle geqrf leaves.
static void qr() {
  const int m = 7, n = 4;
  const auto a = general<double>(m, n, 3);
  double* da = upload(a);
  double* dtau = upload(M<double>(n));
  int* dinfo = upload(M<int>{7});
  size_t dev = 0, host = 0;
  CK(cusolverDnXgeqrf_bufferSize(sh, params, m, n, CUDA_R_64F, da, m, CUDA_R_64F, dtau, CUDA_R_64F, &dev,
                                 &host));
  void* work = nullptr;
  cudaMalloc(&work, dev);
  CK(cusolverDnXgeqrf(sh, params, m, n, CUDA_R_64F, da, m, CUDA_R_64F, dtau, CUDA_R_64F, work, dev,
                      nullptr, host, dinfo));
  const auto f = download(da, a.size());
  double e = 0, scale = 1e-30;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      double rr = 0, aa = 0;
      for (int k = 0; k <= std::min(i, j); ++k) rr += f[k + (size_t)i * m] * f[k + (size_t)j * m];
      for (int k = 0; k < m; ++k) aa += a[k + (size_t)i * m] * a[k + (size_t)j * m];
      e = std::fmax(e, std::fabs(rr - aa));
      scale = std::fmax(scale, std::fabs(aa));
    }
  check(e / scale < 1e-12 && download(dinfo, 1)[0] == 0, "Xgeqrf leaves R with R^T R = A^T A", e / scale);
  cudaFree(work);
  cudaFree(da);
  cudaFree(dtau);
  cudaFree(dinfo);
}

static void eigh() {
  const int n = 6;
  {
    const auto a = symmetric<float>(n, 4);
    float* da = upload(a);
    float* dw = upload(M<float>(n));
    int* dinfo = upload(M<int>{7});
    size_t dev = 0, host = 0;
    CK(cusolverDnXsyevd_bufferSize(sh, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
                                   CUDA_R_32F, da, n, CUDA_R_32F, dw, CUDA_R_32F, &dev, &host));
    void* work = nullptr;
    cudaMalloc(&work, dev);
    CK(cusolverDnXsyevd(sh, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_32F, da, n,
                        CUDA_R_32F, dw, CUDA_R_32F, work, dev, nullptr, host, dinfo));
    bool asc;
    const auto v = download(da, a.size());
    const auto w = download(dw, n);
    const double e = eig_error(a, n, v.data(), w.data(), &asc);
    check(e < 1e-5 && asc, "Xsyevd eigenpairs satisfy A v = w v, ascending", e);
    cudaFree(work);
    cudaFree(da);
    cudaFree(dw);
    cudaFree(dinfo);
  }
  {
    const auto a = symmetric<double>(n, 5);
    double* da = upload(a);
    double* dw = upload(M<double>(n));
    int* dinfo = upload(M<int>{7});
    syevjInfo_t info;
    CK(cusolverDnCreateSyevjInfo(&info));
    CK(cusolverDnXsyevjSetSortEig(info, 1));
    int lwork = 0;
    CK(cusolverDnDsyevj_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, da, n, dw,
                                   &lwork, info));
    double* work = upload(M<double>(std::max(lwork, 1)));
    CK(cusolverDnDsyevj(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, da, n, dw, work, lwork,
                        dinfo, info));
    bool asc;
    const auto v = download(da, a.size());
    const auto w = download(dw, n);
    const double e = eig_error(a, n, v.data(), w.data(), &asc);
    check(e < 1e-12 && asc, "syevj eigenpairs satisfy A v = w v, ascending", e);
    cusolverDnDestroySyevjInfo(info);
    cudaFree(work);
    cudaFree(da);
    cudaFree(dw);
    cudaFree(dinfo);
  }
  {
    const int batch = 3, m = 4;
    M<float> a;
    for (int k = 0; k < batch; ++k) {
      const auto one = symmetric<float>(m, 30 + k);
      a.insert(a.end(), one.begin(), one.end());
    }
    float* da = upload(a);
    float* dw = upload(M<float>((size_t)m * batch));
    int* dinfo = upload(M<int>(batch, 7));
    syevjInfo_t info;
    CK(cusolverDnCreateSyevjInfo(&info));
    int lwork = 0;
    CK(cusolverDnSsyevjBatched_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m, da, m,
                                          dw, &lwork, info, batch));
    float* work = upload(M<float>(std::max(lwork, 1)));
    CK(cusolverDnSsyevjBatched(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, m, da, m, dw, work,
                               lwork, dinfo, info, batch));
    const auto v = download(da, a.size());
    const auto w = download(dw, (size_t)m * batch);
    double e = 0;
    bool all_asc = true;
    for (int k = 0; k < batch; ++k) {
      bool asc;
      const M<float> ak(a.begin() + (size_t)k * m * m, a.begin() + (size_t)(k + 1) * m * m);
      e = std::fmax(e, eig_error(ak, m, v.data() + (size_t)k * m * m, w.data() + (size_t)k * m, &asc));
      all_asc = all_asc && asc;
    }
    check(e < 1e-5 && all_asc, "syevjBatched eigenpairs satisfy A v = w v for each matrix", e);
    cusolverDnDestroySyevjInfo(info);
    cudaFree(work);
    cudaFree(da);
    cudaFree(dw);
    cudaFree(dinfo);
  }
}

static void svd() {
  gesvdjInfo_t info;
  CK(cusolverDnCreateGesvdjInfo(&info));
  CK(cusolverDnXgesvdjSetTolerance(info, 1e-7));
  CK(cusolverDnXgesvdjSetMaxSweeps(info, 50));
  // Tall, wide, economy and full: torch.linalg.svd takes each of these paths.
  struct Shape { int m, n, econ; const char* what; };
  const Shape shapes[] = {{6, 4, 0, "gesvdj 6x4 full: A = U S V^T, U and V orthonormal"},
                          {6, 4, 1, "gesvdj 6x4 economy: A = U S V^T, U and V orthonormal"},
                          {3, 5, 1, "gesvdj 3x5 (wide) economy: A = U S V^T, U and V orthonormal"}};
  for (const Shape& s : shapes) {
    const int m = s.m, n = s.n, k = std::min(m, n);
    const int ucols = s.econ ? k : m, vcols = s.econ ? k : n;
    const auto a = general<float>(m, n, m * 10 + n);
    float* da = upload(a);
    float* ds = upload(M<float>(k));
    float* du = upload(M<float>((size_t)m * ucols));
    float* dv = upload(M<float>((size_t)n * vcols));
    int* dinfo = upload(M<int>{7});
    int lwork = 0;
    CK(cusolverDnSgesvdj_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, s.econ, m, n, da, m, ds, du, m, dv, n,
                                    &lwork, info));
    float* work = upload(M<float>(std::max(lwork, 1)));
    CK(cusolverDnSgesvdj(sh, CUSOLVER_EIG_MODE_VECTOR, s.econ, m, n, da, m, ds, du, m, dv, n, work, lwork,
                         dinfo, info));
    bool orth;
    const auto hs = download(ds, k), hu = download(du, (size_t)m * ucols), hv = download(dv, (size_t)n * vcols);
    const double e = svd_error(a.data(), m, n, hs.data(), hu.data(), m, hv.data(), n, k, &orth);
    check(e < 1e-5 && orth && download(dinfo, 1)[0] == 0, s.what, e);
    if (!s.econ) {  // the columns past min(m, n) complete U to an orthonormal basis
      double worst = 0;
      for (int p = 0; p < ucols; ++p)
        for (int q = 0; q < ucols; ++q) {
          double d = 0;
          for (int i = 0; i < m; ++i) d += (double)hu[i + (size_t)p * m] * hu[i + (size_t)q * m];
          worst = std::fmax(worst, std::fabs(d - (p == q)));
        }
      check(worst < 1e-5, "gesvdj full: all of U is orthonormal, not just its leading columns", worst);
    }
    cudaFree(work);
    cudaFree(da);
    cudaFree(ds);
    cudaFree(du);
    cudaFree(dv);
    cudaFree(dinfo);
  }
  {  // Batched: small square matrices packed one after another.
    const int m = 4, n = 4, batch = 3;
    M<double> a;
    for (int b = 0; b < batch; ++b) {
      const auto one = general<double>(m, n, 40 + b);
      a.insert(a.end(), one.begin(), one.end());
    }
    double* da = upload(a);
    double* ds = upload(M<double>((size_t)n * batch));
    double* du = upload(M<double>((size_t)m * m * batch));
    double* dv = upload(M<double>((size_t)n * n * batch));
    int* dinfo = upload(M<int>(batch, 7));
    int lwork = 0;
    CK(cusolverDnDgesvdjBatched_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, m, n, da, m, ds, du, m, dv, n,
                                           &lwork, info, batch));
    double* work = upload(M<double>(std::max(lwork, 1)));
    CK(cusolverDnDgesvdjBatched(sh, CUSOLVER_EIG_MODE_VECTOR, m, n, da, m, ds, du, m, dv, n, work, lwork,
                                dinfo, info, batch));
    const auto hs = download(ds, (size_t)n * batch), hu = download(du, (size_t)m * m * batch),
               hv = download(dv, (size_t)n * n * batch);
    double e = 0;
    bool all = true;
    for (int b = 0; b < batch; ++b) {
      bool orth;
      e = std::fmax(e, svd_error(a.data() + (size_t)b * m * n, m, n, hs.data() + (size_t)b * n,
                                 hu.data() + (size_t)b * m * m, m, hv.data() + (size_t)b * n * n, n, n, &orth));
      all = all && orth;
    }
    // Jacobi iterates to a tolerance: NVIDIA's own lands near 1e-10 here.
    check(e < 1e-8 && all, "gesvdjBatched: each A = U S V^T, U and V orthonormal", e);
    cudaFree(work);
    cudaFree(da);
    cudaFree(ds);
    cudaFree(du);
    cudaFree(dv);
    cudaFree(dinfo);
  }
  {  // Strided batched, approximate by name: tall matrices, leading `rank` triplets.
    const int m = 7, n = 3, rank = 3, batch = 2;
    const long long sa = (long long)m * n, ss = rank, su = (long long)m * rank, sv = (long long)n * rank;
    M<float> a;
    for (int b = 0; b < batch; ++b) {
      const auto one = general<float>(m, n, 50 + b);
      a.insert(a.end(), one.begin(), one.end());
    }
    float* da = upload(a);
    float* ds = upload(M<float>(ss * batch));
    float* du = upload(M<float>(su * batch));
    float* dv = upload(M<float>(sv * batch));
    int* dinfo = upload(M<int>(batch, 7));
    int lwork = 0;
    CK(cusolverDnSgesvdaStridedBatched_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, rank, m, n, da, m, sa, ds, ss,
                                                  du, m, su, dv, n, sv, &lwork, batch));
    float* work = upload(M<float>(std::max(lwork, 1)));
    M<double> nrm(batch, -1);
    CK(cusolverDnSgesvdaStridedBatched(sh, CUSOLVER_EIG_MODE_VECTOR, rank, m, n, da, m, sa, ds, ss, du, m, su,
                                       dv, n, sv, work, lwork, dinfo, nrm.data(), batch));
    const auto hs = download(ds, ss * batch), hu = download(du, su * batch), hv = download(dv, sv * batch);
    double e = 0;
    bool all = true;
    for (int b = 0; b < batch; ++b) {
      bool orth;
      e = std::fmax(e, svd_error(a.data() + b * sa, m, n, hs.data() + b * ss, hu.data() + b * su, m,
                                 hv.data() + b * sv, n, rank, &orth));
      all = all && orth;
    }
    check(e < 1e-5 && all, "gesvdaStridedBatched: each A = U S V^T at full rank", e);
    cudaFree(work);
    cudaFree(da);
    cudaFree(ds);
    cudaFree(du);
    cudaFree(dv);
    cudaFree(dinfo);
  }
  cusolverDnDestroyGesvdjInfo(info);
}

// torch.linalg.solve and inv on a batch: LU with pivoting, then the solve
// against the factors, both through arrays of device pointers.
static void lu_batched() {
  const int n = 5, batch = 3;
  M<float*> pa(batch), pb(batch);
  M<M<float>> as, bs;
  for (int k = 0; k < batch; ++k) {
    auto a = general<float>(n, n, 60 + k);
    for (int i = 0; i < n; ++i) a[i + (size_t)i * n] += 0.1f;  // nonsingular, still needs pivoting
    as.push_back(a);
    bs.push_back(general<float>(n, 2, 70 + k));
    pa[k] = upload(as[k]);
    pb[k] = upload(bs[k]);
  }
  // The last matrix is singular: its info must say which pivot was zero.
  M<float> sing((size_t)n * n, 0.0f);
  for (int j = 0; j < n; ++j) sing[0 + (size_t)j * n] = sing[1 + (size_t)j * n] = (float)(j + 1);
  pa.push_back(upload(sing));
  float** dpa = upload(pa);
  float** dpb = upload(pb);
  int* dpiv = upload(M<int>((size_t)n * (batch + 1)));
  int* dinfo = upload(M<int>(batch + 1, 7));
  CK(cublasSgetrfBatched(bh, n, dpa, n, dpiv, dinfo, batch + 1));
  const auto infos = download(dinfo, batch + 1);
  int hinfo = 7;
  CK(cublasSgetrsBatched(bh, CUBLAS_OP_N, n, 2, dpa, n, dpiv, dpb, n, &hinfo, batch));
  double e = 0;
  for (int k = 0; k < batch; ++k) e = std::fmax(e, residual(as[k], n, download(pb[k], (size_t)n * 2), bs[k], 2));
  bool ok = hinfo == 0;
  for (int k = 0; k < batch; ++k) ok = ok && infos[k] == 0;
  check(e < 1e-5 && ok, "getrfBatched + getrsBatched solve each system", e);
  check(infos[batch] > 0, "getrfBatched reports a singular matrix in its info", infos[batch]);
  for (float* p : pa) cudaFree(p);
  for (float* p : pb) cudaFree(p);
  cudaFree(dpa);
  cudaFree(dpb);
  cudaFree(dpiv);
  cudaFree(dinfo);
}

// op(A) X = alpha B or X op(A) = alpha B, A triangular: the residual of each
// form, with the other triangle of A filled with values that must be ignored.
static void triangular() {
  const int m = 5, n = 3;
  struct Case { cublasSideMode_t side; cublasFillMode_t uplo; cublasOperation_t op; cublasDiagType_t diag; const char* what; };
  const Case cases[] = {
      {CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, "trsm left, lower: A X = alpha B"},
      {CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T, CUBLAS_DIAG_UNIT, "trsm left, upper, transposed, unit: A^T X = alpha B"},
      {CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT, "trsm right, upper: X A = alpha B"},
      {CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_T, CUBLAS_DIAG_NON_UNIT, "trsm right, lower, transposed: X A^T = alpha B"},
  };
  const double alpha = 0.5;
  for (const Case& c : cases) {
    const int k = c.side == CUBLAS_SIDE_LEFT ? m : n;
    auto a = general<double>(k, k, 80);
    for (int i = 0; i < k; ++i) a[i + (size_t)i * k] = 3.0 + i;  // well conditioned
    const auto b = general<double>(m, n, 81);
    double* da = upload(a);
    double* db = upload(b);
    CK(cublasDtrsm(bh, c.side, c.uplo, c.op, c.diag, m, n, &alpha, da, k, db, m));
    const auto x = download(db, b.size());
    // The effective op(A), from the named triangle only.
    auto M = [&](int i, int j) {
      const int r = c.op == CUBLAS_OP_N ? i : j, col = c.op == CUBLAS_OP_N ? j : i;
      const bool in = c.uplo == CUBLAS_FILL_MODE_LOWER ? col <= r : col >= r;
      if (r == col && c.diag == CUBLAS_DIAG_UNIT) return 1.0;
      return in ? a[r + (size_t)col * k] : 0.0;
    };
    double err = 0, scale = 1e-30;
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0;
        if (c.side == CUBLAS_SIDE_LEFT) for (int t = 0; t < m; ++t) s += M(i, t) * x[t + (size_t)j * m];
        else for (int t = 0; t < n; ++t) s += x[i + (size_t)t * m] * M(t, j);
        err = std::fmax(err, std::fabs(s - alpha * b[i + (size_t)j * m]));
        scale = std::fmax(scale, std::fabs(alpha * b[i + (size_t)j * m]));
      }
    check(err / scale < 1e-12, c.what, err / scale);
    cudaFree(da);
    cudaFree(db);
  }
  {  // Batched, single precision, alpha from device memory.
    const int batch = 3;
    M<float*> pa(batch), pb(batch);
    M<M<float>> as, bs;
    for (int t = 0; t < batch; ++t) {
      auto a = general<float>(m, m, 90 + t);
      for (int i = 0; i < m; ++i) a[i + (size_t)i * m] = 4.0f;
      as.push_back(a);
      bs.push_back(general<float>(m, n, 95 + t));
      pa[t] = upload(as[t]);
      pb[t] = upload(bs[t]);
    }
    float** dpa = upload(pa);
    float** dpb = upload(pb);
    float* dalpha = upload(M<float>{2.0f});
    CK(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_DEVICE));
    const cublasStatus_t st = cublasStrsmBatched(bh, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,
                                                 CUBLAS_DIAG_NON_UNIT, m, n, dalpha, dpa, m, dpb, m, batch);
    CK(cublasSetPointerMode(bh, CUBLAS_POINTER_MODE_HOST));
    CK(st);
    double err = 0;
    for (int t = 0; t < batch; ++t) {
      const auto x = download(pb[t], (size_t)m * n);
      for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j) {
          double s = 0;
          for (int q = i; q < m; ++q) s += (double)as[t][i + (size_t)q * m] * x[q + (size_t)j * m];
          err = std::fmax(err, std::fabs(s - 2.0 * bs[t][i + (size_t)j * m]));
        }
    }
    check(err < 1e-4, "trsmBatched (device alpha): each A X = alpha B", err);
    for (int t = 0; t < batch; ++t) { cudaFree(pa[t]); cudaFree(pb[t]); }
    cudaFree(dpa);
    cudaFree(dpb);
    cudaFree(dalpha);
  }
}

// XsyevBatched and Xgeev arrived in CUDA 12.6 and 12.8; the headers before
// them do not declare them, so this part is built only where they exist.
#if CUDART_VERSION >= 13000
static void eig_64bit() {
  {  // XsyevBatched: a batch of symmetric matrices, vectors in place.
    const int n = 5, batch = 3;
    M<double> a;
    for (int b = 0; b < batch; ++b) {
      const auto one = symmetric<double>(n, 60 + b);
      a.insert(a.end(), one.begin(), one.end());
    }
    double* da = upload(a);
    double* dw = upload(M<double>((size_t)n * batch));
    int* dinfo = upload(M<int>(batch, 7));
    size_t dev = 0, host = 0;
    CK(cusolverDnXsyevBatched_bufferSize(sh, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n,
                                         CUDA_R_64F, da, n, CUDA_R_64F, dw, CUDA_R_64F, &dev, &host, batch));
    void* work = nullptr;
    cudaMalloc(&work, std::max<size_t>(dev, 1));
    M<char> hwork(std::max<size_t>(host, 1));
    CK(cusolverDnXsyevBatched(sh, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, da, n,
                              CUDA_R_64F, dw, CUDA_R_64F, work, dev, hwork.data(), host, dinfo, batch));
    const auto v = download(da, a.size());
    const auto w = download(dw, (size_t)n * batch);
    double e = 0;
    bool all_asc = true;
    for (int b = 0; b < batch; ++b) {
      bool asc;
      const M<double> ab(a.begin() + (size_t)b * n * n, a.begin() + (size_t)(b + 1) * n * n);
      e = std::fmax(e, eig_error(ab, n, v.data() + (size_t)b * n * n, w.data() + (size_t)b * n, &asc));
      all_asc = all_asc && asc;
    }
    check(e < 1e-10 && all_asc, "XsyevBatched eigenpairs satisfy A v = w v for each matrix", e);
    cudaFree(work);
    cudaFree(da);
    cudaFree(dw);
    cudaFree(dinfo);
  }
  {  // Xgeev: a real matrix with a complex pair, right vectors in LAPACK's packing.
    const int n = 6;
    auto a = general<double>(n, n, 70);
    a[0 + 1 * n] = 3.0;  // a strong rotation between the first two coordinates
    a[1 + 0 * n] = -3.0;
    double* da = upload(a);
    std::complex<double>* dw = upload(M<std::complex<double>>(n));
    double* dvr = upload(M<double>((size_t)n * n));
    int* dinfo = upload(M<int>{7});
    size_t dev = 0, host = 0;
    CK(cusolverDnXgeev_bufferSize(sh, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, n, CUDA_R_64F, da,
                                  n, CUDA_C_64F, dw, CUDA_R_64F, nullptr, n, CUDA_R_64F, dvr, n, CUDA_R_64F, &dev,
                                  &host));
    void* work = nullptr;
    cudaMalloc(&work, std::max<size_t>(dev, 1));
    M<char> hwork(std::max<size_t>(host, 1));
    CK(cusolverDnXgeev(sh, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_VECTOR, n, CUDA_R_64F, da, n,
                       CUDA_C_64F, dw, CUDA_R_64F, nullptr, n, CUDA_R_64F, dvr, n, CUDA_R_64F, work, dev,
                       hwork.data(), host, dinfo));
    const auto w = download(dw, n);
    const auto vr = download(dvr, (size_t)n * n);
    const int info = download(dinfo, 1)[0];
    // Unpack: a pair (w[j], w[j+1] = conj) shares columns j (real) and j+1 (imaginary).
    double err = 0, scale = 1e-30;
    int pairs = 0;
    bool unit = true, packing = true;
    for (int j = 0; j < n; ++j) {
      M<std::complex<double>> x(n);
      if (w[j].imag() == 0) {
        for (int i = 0; i < n; ++i) x[i] = vr[i + (size_t)j * n];
      } else {
        const bool first = w[j].imag() > 0;
        if (first) {
          ++pairs;
          packing = packing && j + 1 < n && w[j + 1] == std::conj(w[j]);
        }
        const int c = first ? j : j - 1;
        for (int i = 0; i < n; ++i)
          x[i] = std::complex<double>(vr[i + (size_t)c * n], (first ? 1.0 : -1.0) * vr[i + (size_t)(c + 1) * n]);
      }
      double nrm = 0;
      for (int i = 0; i < n; ++i) {
        std::complex<double> s = 0;
        for (int k = 0; k < n; ++k) s += a[i + (size_t)k * n] * x[k];
        err = std::fmax(err, std::abs(s - w[j] * x[i]));
        nrm += std::norm(x[i]);
      }
      unit = unit && std::fabs(std::sqrt(nrm) - 1.0) < 1e-10;
      scale = std::fmax(scale, std::abs(w[j]));
    }
    check(info == 0 && err / scale < 1e-10 && pairs >= 1 && packing && unit,
          "Xgeev: A v = w v for every eigenpair, conjugate pairs packed as LAPACK packs them, unit vectors",
          err / scale);
    // W typed real, as PyTorch passes it: the real parts, then the imaginary parts.
    double* dwr = upload(M<double>((size_t)2 * n, -99.0));
    double* da2 = upload(a);
    CK(cusolverDnXgeev(sh, params, CUSOLVER_EIG_MODE_NOVECTOR, CUSOLVER_EIG_MODE_NOVECTOR, n, CUDA_R_64F, da2, n,
                       CUDA_R_64F, dwr, CUDA_R_64F, nullptr, n, CUDA_R_64F, nullptr, n, CUDA_R_64F, work, dev,
                       hwork.data(), host, dinfo));
    const auto wr = download(dwr, (size_t)2 * n);
    double werr = 0;
    for (int j = 0; j < n; ++j) werr = std::fmax(werr, std::abs(std::complex<double>(wr[j], wr[n + j]) - w[j]));
    check(werr / scale < 1e-12, "Xgeev with W typed real returns WR then WI, as LAPACK's geev does", werr / scale);
    cudaFree(dwr);
    cudaFree(da2);
    cudaFree(work);
    cudaFree(da);
    cudaFree(dw);
    cudaFree(dvr);
    cudaFree(dinfo);
  }
}
#else
static void eig_64bit() { std::printf("ok   XsyevBatched and Xgeev: not in this toolkit's headers, skipped\n"); }
#endif

int main() {
  if (cusolverDnCreate(&sh) || cusolverDnCreateParams(&params) || cublasCreate(&bh)) {
    std::printf("FAIL: could not create the library handles\n");
    return 1;
  }
  cholesky();
  cholesky_batched();
  qr();
  eigh();
  svd();
  lu_batched();
  triangular();
  eig_64bit();
  cublasDestroy(bh);
  cusolverDnDestroyParams(params);
  cusolverDnDestroy(sh);
  std::printf(failures ? "FAIL: %d solver checks\n" : "PASS: every solver check\n", failures);
  return failures ? 1 : 0;
}
