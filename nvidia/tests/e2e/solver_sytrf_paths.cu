// cuSOLVER's symmetric indefinite factorization and what uses it, its row
// interchanges, and the eigendecomposition of a general complex matrix --
// checked by each result's defining property, so the same program checks
// NVIDIA's libraries and VirtualGPU's.
//
//   sytrf + Xsytrs    A X = B for S, D, C, Z (complex symmetric, not
//                     Hermitian), both triangles, sizes from 1 to past
//                     cuSOLVER's block size; 2x2 pivots forced by a zero
//                     diagonal; the other triangle untouched; info for an
//                     exactly singular D
//   sytri             A A^-1 = I from the stored triangle. NVIDIA's sytri, on
//                     an RTX 3060 with CUDA 13.0, returns success and leaves A
//                     as it was; that is accepted on hardware and not here
//   laswp             LAPACK's row interchanges, forward and backward
//   Xgeev (complex)   A v = w v, unit vectors with the largest component real,
//                     the eigenvalues summing to the trace; mixed types
//                     refused with INVALID_VALUE. CUDA 12.0's header has no
//                     Xgeev, so it is looked up by name.
//
// Tolerances: residuals relative to ||A|| ||X||, 1e-5 in single precision and
// 1e-12 in double.
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;
// run_lib_check.sh runs this on a simulated GPU it names in VGPU_GPU.
static const bool on_sim = std::getenv("VGPU_GPU") != nullptr;

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

template <class T> T* upload(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

static cusolverDnHandle_t h;

// Element types: the device type, its value as cdouble, and the entry points.
template <class T> struct Ty;
template <> struct Ty<float> {
  static constexpr const char* name = "S";
  static constexpr double tol = 1e-5;
  static float make(cdouble v) { return (float)v.real(); }
  static cdouble get(float v) { return v; }
  static constexpr cudaDataType type = CUDA_R_32F;
  static cusolverStatus_t bs(int n, float* A, int lda, int* l) { return cusolverDnSsytrf_bufferSize(h, n, A, lda, l); }
  static cusolverStatus_t trf(cublasFillMode_t u, int n, float* A, int lda, int* p, float* w, int l, int* i) {
    return cusolverDnSsytrf(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t tri_bs(cublasFillMode_t u, int n, float* A, int lda, const int* p, int* l) {
    return cusolverDnSsytri_bufferSize(h, u, n, A, lda, p, l);
  }
  static cusolverStatus_t tri(cublasFillMode_t u, int n, float* A, int lda, const int* p, float* w, int l, int* i) {
    return cusolverDnSsytri(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t laswp(int n, float* A, int lda, int k1, int k2, const int* p, int inc) {
    return cusolverDnSlaswp(h, n, A, lda, k1, k2, p, inc);
  }
};
template <> struct Ty<double> {
  static constexpr const char* name = "D";
  static constexpr double tol = 1e-12;
  static double make(cdouble v) { return v.real(); }
  static cdouble get(double v) { return v; }
  static constexpr cudaDataType type = CUDA_R_64F;
  static cusolverStatus_t bs(int n, double* A, int lda, int* l) { return cusolverDnDsytrf_bufferSize(h, n, A, lda, l); }
  static cusolverStatus_t trf(cublasFillMode_t u, int n, double* A, int lda, int* p, double* w, int l, int* i) {
    return cusolverDnDsytrf(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t tri_bs(cublasFillMode_t u, int n, double* A, int lda, const int* p, int* l) {
    return cusolverDnDsytri_bufferSize(h, u, n, A, lda, p, l);
  }
  static cusolverStatus_t tri(cublasFillMode_t u, int n, double* A, int lda, const int* p, double* w, int l, int* i) {
    return cusolverDnDsytri(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t laswp(int n, double* A, int lda, int k1, int k2, const int* p, int inc) {
    return cusolverDnDlaswp(h, n, A, lda, k1, k2, p, inc);
  }
};
template <> struct Ty<cuComplex> {
  static constexpr const char* name = "C";
  static constexpr double tol = 1e-5;
  static cuComplex make(cdouble v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
  static cdouble get(cuComplex v) { return {v.x, v.y}; }
  static constexpr cudaDataType type = CUDA_C_32F;
  static cusolverStatus_t bs(int n, cuComplex* A, int lda, int* l) {
    return cusolverDnCsytrf_bufferSize(h, n, A, lda, l);
  }
  static cusolverStatus_t trf(cublasFillMode_t u, int n, cuComplex* A, int lda, int* p, cuComplex* w, int l, int* i) {
    return cusolverDnCsytrf(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t tri_bs(cublasFillMode_t u, int n, cuComplex* A, int lda, const int* p, int* l) {
    return cusolverDnCsytri_bufferSize(h, u, n, A, lda, p, l);
  }
  static cusolverStatus_t tri(cublasFillMode_t u, int n, cuComplex* A, int lda, const int* p, cuComplex* w, int l,
                              int* i) {
    return cusolverDnCsytri(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t laswp(int n, cuComplex* A, int lda, int k1, int k2, const int* p, int inc) {
    return cusolverDnClaswp(h, n, A, lda, k1, k2, p, inc);
  }
};
template <> struct Ty<cuDoubleComplex> {
  static constexpr const char* name = "Z";
  static constexpr double tol = 1e-12;
  static cuDoubleComplex make(cdouble v) { return make_cuDoubleComplex(v.real(), v.imag()); }
  static cdouble get(cuDoubleComplex v) { return {v.x, v.y}; }
  static constexpr cudaDataType type = CUDA_C_64F;
  static cusolverStatus_t bs(int n, cuDoubleComplex* A, int lda, int* l) {
    return cusolverDnZsytrf_bufferSize(h, n, A, lda, l);
  }
  static cusolverStatus_t trf(cublasFillMode_t u, int n, cuDoubleComplex* A, int lda, int* p, cuDoubleComplex* w,
                              int l, int* i) {
    return cusolverDnZsytrf(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t tri_bs(cublasFillMode_t u, int n, cuDoubleComplex* A, int lda, const int* p, int* l) {
    return cusolverDnZsytri_bufferSize(h, u, n, A, lda, p, l);
  }
  static cusolverStatus_t tri(cublasFillMode_t u, int n, cuDoubleComplex* A, int lda, const int* p,
                              cuDoubleComplex* w, int l, int* i) {
    return cusolverDnZsytri(h, u, n, A, lda, p, w, l, i);
  }
  static cusolverStatus_t laswp(int n, cuDoubleComplex* A, int lda, int k1, int k2, const int* p, int inc) {
    return cusolverDnZlaswp(h, n, A, lda, k1, k2, p, inc);
  }
};

// A symmetric (complex symmetric for C, Z) n x n matrix, column-major with
// leading dimension lda, its diagonal scaled by `diag` (0 forces 2x2 pivots).
// Padding rows and the other triangle hold markers that must survive.
template <class T> std::vector<T> symmetric(int n, int lda, int seed, double diag, bool upper, std::vector<cdouble>* full) {
  std::vector<T> a((size_t)lda * n);
  full->assign((size_t)n * n, 0);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i) {
      if (i >= n) {
        a[(size_t)j * lda + i] = Ty<T>::make(cdouble(-77.0, 0.5));
        continue;
      }
      const int r = std::min(i, j), c = std::max(i, j);
      cdouble v(std::sin(1.3 * r + 0.7 * c * c + seed) * 2, std::cos(0.9 * r * c + seed));
      if (i == j) v *= diag;
      v = Ty<T>::get(Ty<T>::make(v));  // as the device holds it
      (*full)[(size_t)j * n + i] = v;
      const bool stored = upper ? i <= j : i >= j;
      a[(size_t)j * lda + i] = stored ? Ty<T>::make(v) : Ty<T>::make(cdouble(55.0, -0.25));
    }
  return a;
}


template <class T> static void sytrf_case(int n, bool upper, double diag) {
  const int lda = n + 2, nrhs = 3, ldb = n + 1;
  const cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  std::vector<cdouble> full;
  const std::vector<T> a0 = symmetric<T>(n, lda, n + (upper ? 7 : 3), diag, upper, &full);
  T* dA = upload(a0);
  int lwork = 0;
  CK(Ty<T>::bs(n, dA, lda, &lwork));
  T* work = upload(std::vector<T>(std::max(lwork, 1)));
  int* dpiv = upload(std::vector<int>(n));
  int* dinfo = upload(std::vector<int>{-1});
  CK(Ty<T>::trf(uplo, n, dA, lda, dpiv, work, lwork, dinfo));
  const int info = download(dinfo, 1)[0];
  const auto f = download(dA, a0.size());
  const auto piv = download(dpiv, n);
  bool other = true;  // the other triangle and the padding are as they were
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i)
      if (i >= n || (upper ? i > j : i < j))
        other = other && Ty<T>::get(f[(size_t)j * lda + i]) == Ty<T>::get(a0[(size_t)j * lda + i]);
  bool two = false, valid = true;
  for (int k = 0; k < n; ++k) {
    two = two || piv[k] < 0;
    valid = valid && piv[k] != 0 && std::abs(piv[k]) <= n;
  }

  // Solve A X = B with the 64-bit API, ipiv widened.
  std::vector<T> b((size_t)ldb * nrhs);
  for (size_t i = 0; i < b.size(); ++i) b[i] = Ty<T>::make(cdouble(std::cos(0.37 * (double)i), 0.1 * std::sin((double)i)));
  T* dB = upload(b);
  std::vector<int64_t> p64(piv.begin(), piv.end());
  int64_t* dp64 = upload(p64);
  size_t dws = 0, hws = 0;
  CK(cusolverDnXsytrs_bufferSize(h, uplo, n, nrhs, Ty<T>::type, dA, lda, dp64, Ty<T>::type, dB, ldb, &dws, &hws));
  void* dwork = nullptr;
  cudaMalloc(&dwork, std::max<size_t>(dws, 16));
  std::vector<char> hwork(std::max<size_t>(hws, 16));
  CK(cusolverDnXsytrs(h, uplo, n, nrhs, Ty<T>::type, dA, lda, dp64, Ty<T>::type, dB, ldb, dwork, dws, hwork.data(), hws,
                      dinfo));
  const auto x = download(dB, b.size());
  double res = 0, anorm = 0, xnorm = 0;
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i) {
      cdouble s = 0;
      for (int k = 0; k < n; ++k) s += full[(size_t)k * n + i] * Ty<T>::get(x[(size_t)j * ldb + k]);
      res = std::max(res, std::abs(s - Ty<T>::get(b[(size_t)j * ldb + i])));
      xnorm = std::max(xnorm, std::abs(Ty<T>::get(x[(size_t)j * ldb + i])));
    }
  for (const cdouble& v : full) anorm = std::max(anorm, std::abs(v));
  const double rel = res / (anorm * xnorm * n);
  char what[200];
  std::snprintf(what, sizeof what, "%ssytrf + Xsytrs, n = %d, %s%s: A X = B", Ty<T>::name, n, upper ? "upper" : "lower",
                diag == 0 ? ", zero diagonal" : "");
  check(info == 0 && valid && other && (diag != 0 || n < 2 || two) && rel < Ty<T>::tol, what, rel);

  // The inverse from the factors, in the stored triangle.
  int lw2 = 0;
  CK(Ty<T>::tri_bs(uplo, n, dA, lda, dpiv, &lw2));
  T* work2 = upload(std::vector<T>(std::max(lw2, 1)));
  CK(Ty<T>::tri(uplo, n, dA, lda, dpiv, work2, lw2, dinfo));
  const auto inv = download(dA, a0.size());
  bool untouched = true, other2 = true;
  for (size_t i = 0; i < inv.size(); ++i) untouched = untouched && Ty<T>::get(inv[i]) == Ty<T>::get(f[i]);
  std::vector<cdouble> ifull((size_t)n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i) {
      const bool stored = i < n && (upper ? i <= j : i >= j);
      if (!stored) {
        other2 = other2 && Ty<T>::get(inv[(size_t)j * lda + i]) == Ty<T>::get(a0[(size_t)j * lda + i]);
        continue;
      }
      ifull[(size_t)j * n + i] = ifull[(size_t)i * n + j] = Ty<T>::get(inv[(size_t)j * lda + i]);
    }
  double ierr = 0, inorm = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      cdouble s = 0;
      for (int k = 0; k < n; ++k) s += full[(size_t)k * n + i] * ifull[(size_t)j * n + k];
      ierr = std::max(ierr, std::abs(s - (i == j ? 1.0 : 0.0)));
      inorm = std::max(inorm, std::abs(ifull[(size_t)j * n + i]));
    }
  const double irel = ierr / (anorm * inorm * n);
  std::snprintf(what, sizeof what, "%ssytri, n = %d, %s: A A^-1 = I", Ty<T>::name, n, upper ? "upper" : "lower");
  if (!on_sim && untouched && n > 0)
    std::printf("ok   %s -- NVIDIA's left A as it was, as it does on CUDA 13.0\n", what);
  else
    check(download(dinfo, 1)[0] == 0 && other2 && irel < Ty<T>::tol, what, irel);
  for (void* p : {(void*)dA, (void*)work, (void*)dpiv, (void*)dinfo, (void*)dB, (void*)dp64, dwork, (void*)work2})
    cudaFree(p);
}

template <class T> static void sytrf_all() {
  for (int n : {1, 5, 40, 130})
    for (int upper = 0; upper < 2; ++upper) sytrf_case<T>(n, upper, n == 40 ? 0.0 : 0.1);
}

// D exactly singular: the last row and column zero. info = n, as LAPACK.
static void singular() {
  const int n = 6;
  for (int upper = 0; upper < 2; ++upper) {
    std::vector<double> a((size_t)n * n);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i)
        a[(size_t)j * n + i] = (i == n - 1 || j == n - 1) ? 0.0 : std::sin(1.0 + std::min(i, j) + 0.5 * std::max(i, j));
    double* dA = upload(a);
    int lwork = 0;
    CK(cusolverDnDsytrf_bufferSize(h, n, dA, n, &lwork));
    double* work = upload(std::vector<double>(lwork));
    int* dpiv = upload(std::vector<int>(n));
    int* dinfo = upload(std::vector<int>{-1});
    CK(cusolverDnDsytrf(h, upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, n, dA, n, dpiv, work, lwork, dinfo));
    const int info = download(dinfo, 1)[0];
    check(info == n, upper ? "Dsytrf, upper, singular D: info = n" : "Dsytrf, lower, singular D: info = n", info);
    for (void* p : {(void*)dA, (void*)work, (void*)dpiv, (void*)dinfo}) cudaFree(p);
  }
}

template <class T> static void laswp() {
  const int m = 6, n = 3, lda = 7;
  std::vector<T> a((size_t)lda * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i) a[(size_t)j * lda + i] = Ty<T>::make(cdouble(10 * i + j, -j));
  const std::vector<int> ipiv = {3, 2, 6, 4, 6};  // rows 1..5 interchanged with these, 1-based
  for (int inc : {1, -1}) {
    T* dA = upload(a);
    int* dp = upload(ipiv);
    CK(Ty<T>::laswp(n, dA, lda, 2, 5, dp, inc));
    std::vector<cdouble> ref((size_t)lda * n);
    for (size_t i = 0; i < ref.size(); ++i) ref[i] = Ty<T>::get(a[i]);
    // LAPACK: rows k1..k2 in order (inc > 0) or reverse, ipiv read at the same index.
    for (int s = 0; s < 4; ++s) {
      const int i = inc > 0 ? 2 + s : 5 - s;
      const int ip = ipiv[(size_t)i - 1];
      for (int c = 0; c < n; ++c) std::swap(ref[(size_t)c * lda + i - 1], ref[(size_t)c * lda + ip - 1]);
    }
    const auto got = download(dA, a.size());
    bool same = true;
    for (size_t i = 0; i < got.size(); ++i) same = same && Ty<T>::get(got[i]) == ref[i];
    char what[120];
    std::snprintf(what, sizeof what, "%slaswp, rows 2..5, incx = %d", Ty<T>::name, inc);
    check(same, what, 0);
    cudaFree(dA);
    cudaFree(dp);
  }
  (void)m;
}

using XgeevBs = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t,
                                     int64_t, cudaDataType, const void*, int64_t, cudaDataType, const void*,
                                     cudaDataType, const void*, int64_t, cudaDataType, const void*, int64_t,
                                     cudaDataType, size_t*, size_t*);
using Xgeev = cusolverStatus_t (*)(cusolverDnHandle_t, cusolverDnParams_t, cusolverEigMode_t, cusolverEigMode_t,
                                   int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, cudaDataType, void*,
                                   int64_t, cudaDataType, void*, int64_t, cudaDataType, void*, size_t, void*, size_t,
                                   int*);

template <class T> static void geev_complex(int n, bool vectors) {
  auto bs = reinterpret_cast<XgeevBs>(dlsym(RTLD_DEFAULT, "cusolverDnXgeev_bufferSize"));
  auto geev = reinterpret_cast<Xgeev>(dlsym(RTLD_DEFAULT, "cusolverDnXgeev"));
  if (!bs || !geev) {
    std::printf("FAIL the cuSOLVER this runs against has no cusolverDnXgeev\n");
    ++failures;
    return;
  }
  const int lda = n + 1, ldv = n + 2;
  std::vector<T> a((size_t)lda * n);
  std::vector<cdouble> full((size_t)n * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i) {
      const cdouble v(std::sin(i + 2.0 * j + 1 + n) + 0.3 * std::cos(0.7 * i * j),
                      std::cos(3.0 * i - j) + 0.2 * std::sin(0.5 * i * i + j));
      a[(size_t)j * lda + i] = Ty<T>::make(v);
      if (i < n) full[(size_t)j * n + i] = Ty<T>::get(a[(size_t)j * lda + i]);
    }
  T* dA = upload(a);
  T* dW = upload(std::vector<T>(n));
  T* dV = upload(std::vector<T>((size_t)ldv * n));
  int* dinfo = upload(std::vector<int>{-1});
  cusolverDnParams_t params;
  CK(cusolverDnCreateParams(&params));
  const cudaDataType t = Ty<T>::type;
  const auto jobvr = vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  size_t dws = 0, hws = 0;
  CK(bs(h, params, CUSOLVER_EIG_MODE_NOVECTOR, jobvr, n, t, dA, lda, t, dW, t, nullptr, 1, t, dV, ldv, t, &dws, &hws));
  void* dwork = nullptr;
  cudaMalloc(&dwork, std::max<size_t>(dws, 16));
  std::vector<char> hwork(std::max<size_t>(hws, 16));
  CK(geev(h, params, CUSOLVER_EIG_MODE_NOVECTOR, jobvr, n, t, dA, lda, t, dW, t, nullptr, 1, t, dV, ldv, t, dwork, dws,
          hwork.data(), hws, dinfo));
  const int info = download(dinfo, 1)[0];
  const auto w = download(dW, n);
  cdouble trace = 0, wsum = 0;
  double scale = 0;
  for (int i = 0; i < n; ++i) {
    trace += full[(size_t)i * n + i];
    wsum += Ty<T>::get(w[i]);
    for (int j = 0; j < n; ++j) scale = std::max(scale, std::abs(full[(size_t)j * n + i]));
  }
  double err = std::abs(trace - wsum) / (scale * n);
  bool shape = true;
  if (vectors) {
    const auto v = download(dV, (size_t)ldv * n);
    for (int j = 0; j < n; ++j) {
      double nrm = 0;
      int big = 0;
      for (int i = 0; i < n; ++i) {
        const cdouble x = Ty<T>::get(v[(size_t)j * ldv + i]);
        nrm += std::norm(x);
        if (std::abs(x) > std::abs(Ty<T>::get(v[(size_t)j * ldv + big]))) big = i;
        cdouble s = 0;
        for (int k = 0; k < n; ++k) s += full[(size_t)k * n + i] * Ty<T>::get(v[(size_t)j * ldv + k]);
        err = std::max(err, std::abs(s - Ty<T>::get(w[j]) * x) / (scale * n));
      }
      const cdouble b = Ty<T>::get(v[(size_t)j * ldv + big]);
      shape = shape && std::fabs(std::sqrt(nrm) - 1) < 10 * Ty<T>::tol && b.imag() == 0 && b.real() > 0;
    }
  }
  char what[200];
  std::snprintf(what, sizeof what, "Xgeev %s, n = %d%s", t == CUDA_C_32F ? "C_32F" : "C_64F", n,
                vectors ? ": A v = w v, unit vectors with the largest component real" : ", eigenvalues: their sum is the trace");
  check(info == 0 && shape && err < Ty<T>::tol * 10, what, err);
  if (n == 4 && vectors) {
    const int st = bs(h, params, CUSOLVER_EIG_MODE_NOVECTOR, jobvr, n, t, dA, lda,
                      t == CUDA_C_32F ? CUDA_R_32F : CUDA_R_64F, dW, t, nullptr, 1, t, dV, ldv, t, &dws, &hws);
    check(st == CUSOLVER_STATUS_INVALID_VALUE, "Xgeev refuses a real W for a complex A (INVALID_VALUE)", st);
  }
  cusolverDnDestroyParams(params);
  for (void* p : {(void*)dA, (void*)dW, (void*)dV, (void*)dinfo, dwork}) cudaFree(p);
}

int main() {
  if (cusolverDnCreate(&h)) {
    std::printf("FAIL: cusolverDnCreate\n");
    return 1;
  }
  sytrf_all<float>();
  sytrf_all<double>();
  sytrf_all<cuComplex>();
  sytrf_all<cuDoubleComplex>();
  singular();
  laswp<float>();
  laswp<double>();
  laswp<cuComplex>();
  laswp<cuDoubleComplex>();
  geev_complex<cuDoubleComplex>(4, true);
  geev_complex<cuDoubleComplex>(30, true);
  geev_complex<cuDoubleComplex>(9, false);
  geev_complex<cuComplex>(4, true);
  geev_complex<cuComplex>(12, true);
  cusolverDnDestroy(h);
  std::printf(failures ? "FAIL: %d checks\n" : "PASS: every symmetric indefinite and complex eigen check\n", failures);
  return failures ? 1 : 0;
}
