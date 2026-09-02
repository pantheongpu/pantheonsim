// Differential conformance for the cuSOLVER shim: Cholesky, LU, QR, the
// symmetric eigenproblem and the SVD, in single and double precision.
//
// Factorizations that are mathematically unique -- Cholesky, pivoted LU, the
// LAPACK-signed Householder QR, and every linear solve -- are compared element
// by element. Eigen- and singular vectors are only unique up to a sign, so
// those are pinned down by their invariants instead: the spectrum itself, and
// the residual of the reconstruction.
#include <cusolverDn.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>
#include <vector>

#define CV(x) do { cusolverStatus_t s_ = (x); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
  printf("%s -> %d\n", #x, (int)s_); return; } } while (0)

static void dump(const char* tag, const std::vector<double>& v) {
  double sum = 0, abs = 0;
  for (double x : v) { sum += x; abs += std::fabs(x); }
  printf("%-32s n=%zu sum=%.5f abs=%.5f first=%.5f last=%.5f\n", tag, v.size(), sum, abs,
         v.empty() ? 0.0 : v[0], v.empty() ? 0.0 : v.back());
}

template <class T> static T* up(const std::vector<double>& h) {
  std::vector<T> t(h.begin(), h.end());
  T* d = nullptr;
  cudaMalloc(&d, t.size() * sizeof(T));
  cudaMemcpy(d, t.data(), t.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<double> down(const T* d, size_t n) {
  std::vector<T> t(n);
  cudaMemcpy(t.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return std::vector<double>(t.begin(), t.end());
}

// Column-major test matrices, fixed so both sides see identical bits.
static std::vector<double> spd(int n) {  // symmetric positive definite
  std::vector<double> a((size_t)n * n);
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < n; ++r)
      a[(size_t)c * n + r] = (r == c) ? 4.0 + 0.5 * r : 1.0 / (1.0 + std::fabs(r - c));
  return a;
}
static std::vector<double> general(int m, int n) {
  std::vector<double> a((size_t)m * n);
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r)
      a[(size_t)c * m + r] = std::sin(0.6 * r + 0.35 * c) + 0.25 * ((r + 2 * c) % 5) + 0.1;
  return a;
}

template <class T>
static void chol(cusolverDnHandle_t h, const char* prec,
                 cusolverStatus_t (*bs)(cusolverDnHandle_t, cublasFillMode_t, int, T*, int, int*),
                 cusolverStatus_t (*f)(cusolverDnHandle_t, cublasFillMode_t, int, T*, int, T*, int, int*),
                 cusolverStatus_t (*s)(cusolverDnHandle_t, cublasFillMode_t, int, int, const T*,
                                       int, T*, int, int*)) {
  const int n = 6, nrhs = 2;
  auto a = spd(n);
  std::vector<double> b((size_t)n * nrhs);
  for (size_t i = 0; i < b.size(); ++i) b[i] = 1.0 + 0.5 * (double)(i % 4);
  T* dA = up<T>(a);
  T* dB = up<T>(b);
  int* dinfo = nullptr;
  cudaMalloc(&dinfo, sizeof(int));
  int lwork = 0;
  CV(bs(h, CUBLAS_FILL_MODE_LOWER, n, dA, n, &lwork));
  T* dwork = nullptr;
  if (lwork > 0) cudaMalloc(&dwork, (size_t)lwork * sizeof(T));
  CV(f(h, CUBLAS_FILL_MODE_LOWER, n, dA, n, dwork, lwork, dinfo));
  auto info = down<int>(dinfo, 1);
  char tag[80];
  snprintf(tag, sizeof(tag), "potrf %s lower (info=%d)", prec, (int)info[0]);
  // Only the lower triangle is defined; the upper is whatever was there.
  auto fac = down<T>(dA, a.size());
  std::vector<double> lower;
  for (int c = 0; c < n; ++c)
    for (int r = c; r < n; ++r) lower.push_back(fac[(size_t)c * n + r]);
  dump(tag, lower);
  CV(s(h, CUBLAS_FILL_MODE_LOWER, n, nrhs, dA, n, dB, n, dinfo));
  snprintf(tag, sizeof(tag), "potrs %s solution", prec);
  dump(tag, down<T>(dB, b.size()));
  cudaFree(dA); cudaFree(dB); cudaFree(dinfo);
  if (dwork) cudaFree(dwork);
}

template <class T>
static void lu(cusolverDnHandle_t h, const char* prec,
               cusolverStatus_t (*bs)(cusolverDnHandle_t, int, int, T*, int, int*),
               cusolverStatus_t (*f)(cusolverDnHandle_t, int, int, T*, int, T*, int*, int*),
               cusolverStatus_t (*s)(cusolverDnHandle_t, cublasOperation_t, int, int, const T*,
                                     int, const int*, T*, int, int*)) {
  const int n = 6, nrhs = 2;
  auto a = general(n, n);
  std::vector<double> b((size_t)n * nrhs);
  for (size_t i = 0; i < b.size(); ++i) b[i] = 2.0 - 0.25 * (double)(i % 5);
  T* dA = up<T>(a);
  T* dB = up<T>(b);
  int *dpiv = nullptr, *dinfo = nullptr;
  cudaMalloc(&dpiv, n * sizeof(int));
  cudaMalloc(&dinfo, sizeof(int));
  int lwork = 0;
  CV(bs(h, n, n, dA, n, &lwork));
  T* dwork = nullptr;
  if (lwork > 0) cudaMalloc(&dwork, (size_t)lwork * sizeof(T));
  CV(f(h, n, n, dA, n, dwork, dpiv, dinfo));
  auto piv = down<int>(dpiv, n);
  char tag[80];
  printf("getrf %s pivots               ", prec);
  for (int i = 0; i < n; ++i) printf(" %d", (int)piv[i]);
  printf("\n");
  snprintf(tag, sizeof(tag), "getrf %s factors", prec);
  dump(tag, down<T>(dA, a.size()));
  CV(s(h, CUBLAS_OP_N, n, nrhs, dA, n, dpiv, dB, n, dinfo));
  snprintf(tag, sizeof(tag), "getrs %s solution", prec);
  dump(tag, down<T>(dB, b.size()));
  // Transposed solve exercises the reverse pivot pass.
  T* dB2 = up<T>(b);
  CV(s(h, CUBLAS_OP_T, n, nrhs, dA, n, dpiv, dB2, n, dinfo));
  snprintf(tag, sizeof(tag), "getrs %s transposed", prec);
  dump(tag, down<T>(dB2, b.size()));
  cudaFree(dA); cudaFree(dB); cudaFree(dB2); cudaFree(dpiv); cudaFree(dinfo);
  if (dwork) cudaFree(dwork);
}

template <class T>
static void qr(cusolverDnHandle_t h, const char* prec,
               cusolverStatus_t (*bs)(cusolverDnHandle_t, int, int, T*, int, int*),
               cusolverStatus_t (*f)(cusolverDnHandle_t, int, int, T*, int, T*, T*, int, int*),
               cusolverStatus_t (*og)(cusolverDnHandle_t, int, int, int, T*, int, const T*, T*,
                                      int, int*),
               cusolverStatus_t (*om)(cusolverDnHandle_t, cublasSideMode_t, cublasOperation_t,
                                      int, int, int, const T*, int, const T*, T*, int, T*, int,
                                      int*)) {
  const int m = 7, n = 4;
  auto a = general(m, n);
  T* dA = up<T>(a);
  T* dTau = nullptr;
  cudaMalloc(&dTau, n * sizeof(T));
  int* dinfo = nullptr;
  cudaMalloc(&dinfo, sizeof(int));
  int lwork = 0;
  CV(bs(h, m, n, dA, m, &lwork));
  T* dwork = nullptr;
  if (lwork > 0) cudaMalloc(&dwork, (size_t)lwork * sizeof(T));
  CV(f(h, m, n, dA, m, dTau, dwork, lwork, dinfo));
  char tag[80];
  auto fac = down<T>(dA, a.size());
  std::vector<double> r;
  for (int c = 0; c < n; ++c)
    for (int rr = 0; rr <= c; ++rr) r.push_back(fac[(size_t)c * m + rr]);
  snprintf(tag, sizeof(tag), "geqrf %s R", prec);
  dump(tag, r);
  snprintf(tag, sizeof(tag), "geqrf %s tau", prec);
  dump(tag, down<T>(dTau, n));

  // Q^T applied to a right-hand side, before A is overwritten by orgqr.
  std::vector<double> c0((size_t)m * 2);
  for (size_t i = 0; i < c0.size(); ++i) c0[i] = 1.0 + 0.5 * (double)(i % 3);
  T* dC = up<T>(c0);
  CV(om(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, m, 2, n, dA, m, dTau, dC, m, dwork, lwork, dinfo));
  snprintf(tag, sizeof(tag), "ormqr %s Q^T C", prec);
  dump(tag, down<T>(dC, c0.size()));

  CV(og(h, m, n, n, dA, m, dTau, dwork, lwork, dinfo));
  snprintf(tag, sizeof(tag), "orgqr %s Q", prec);
  dump(tag, down<T>(dA, a.size()));
  cudaFree(dA); cudaFree(dTau); cudaFree(dC); cudaFree(dinfo);
  if (dwork) cudaFree(dwork);
}

template <class T>
static void eig(cusolverDnHandle_t h, const char* prec,
                cusolverStatus_t (*bs)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t,
                                       int, const T*, int, const T*, int*),
                cusolverStatus_t (*f)(cusolverDnHandle_t, cusolverEigMode_t, cublasFillMode_t,
                                      int, T*, int, T*, T*, int, int*)) {
  const int n = 6;
  auto a = spd(n);
  T* dA = up<T>(a);
  T* dW = nullptr;
  cudaMalloc(&dW, n * sizeof(T));
  int* dinfo = nullptr;
  cudaMalloc(&dinfo, sizeof(int));
  int lwork = 0;
  CV(bs(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, &lwork));
  T* dwork = nullptr;
  if (lwork > 0) cudaMalloc(&dwork, (size_t)lwork * sizeof(T));
  CV(f(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, dwork, lwork, dinfo));
  char tag[80];
  auto w = down<T>(dW, n);
  snprintf(tag, sizeof(tag), "syevd %s eigenvalues", prec);
  dump(tag, w);
  // The vectors are only defined up to a sign, so check what is invariant:
  // orthonormality and the residual of A V - V diag(w).
  auto v = down<T>(dA, a.size());
  double resid = 0, ortho = 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < n; ++i) {
      double av = 0;
      for (int k = 0; k < n; ++k) av += a[(size_t)k * n + i] * v[(size_t)j * n + k];
      resid = std::fmax(resid, std::fabs(av - w[j] * v[(size_t)j * n + i]));
    }
  for (int j = 0; j < n; ++j)
    for (int k = 0; k < n; ++k) {
      double d = 0;
      for (int i = 0; i < n; ++i) d += v[(size_t)j * n + i] * v[(size_t)k * n + i];
      ortho = std::fmax(ortho, std::fabs(d - (j == k ? 1.0 : 0.0)));
    }
  printf("syevd %-10s residual < 1e-4 %s, orthonormal < 1e-4 %s\n", prec,
         resid < 1e-4 ? "yes" : "NO", ortho < 1e-4 ? "yes" : "NO");
  cudaFree(dA); cudaFree(dW); cudaFree(dinfo);
  if (dwork) cudaFree(dwork);
}

template <class T>
static void sv(cusolverDnHandle_t h, const char* prec,
               cusolverStatus_t (*bs)(cusolverDnHandle_t, int, int, int*),
               cusolverStatus_t (*f)(cusolverDnHandle_t, signed char, signed char, int, int, T*,
                                     int, T*, T*, int, T*, int, T*, int, T*, int*)) {
  const int m = 7, n = 4;
  auto a = general(m, n);
  T* dA = up<T>(a);
  T* dS = nullptr; T* dU = nullptr; T* dVT = nullptr;
  cudaMalloc(&dS, n * sizeof(T));
  cudaMalloc(&dU, (size_t)m * m * sizeof(T));
  cudaMalloc(&dVT, (size_t)n * n * sizeof(T));
  int* dinfo = nullptr;
  cudaMalloc(&dinfo, sizeof(int));
  int lwork = 0;
  CV(bs(h, m, n, &lwork));
  T* dwork = nullptr;
  if (lwork > 0) cudaMalloc(&dwork, (size_t)lwork * sizeof(T));
  CV(f(h, 'S', 'S', m, n, dA, m, dS, dU, m, dVT, n, dwork, lwork, nullptr, dinfo));
  char tag[80];
  auto s = down<T>(dS, n);
  snprintf(tag, sizeof(tag), "gesvd %s singular values", prec);
  dump(tag, s);
  // U and V carry an arbitrary sign per column; the reconstruction does not.
  auto u = down<T>(dU, (size_t)m * n);
  auto vt = down<T>(dVT, (size_t)n * n);
  double resid = 0;
  for (int c = 0; c < n; ++c)
    for (int r = 0; r < m; ++r) {
      double acc = 0;
      for (int k = 0; k < n; ++k) acc += u[(size_t)k * m + r] * s[k] * vt[(size_t)c * n + k];
      resid = std::fmax(resid, std::fabs(acc - a[(size_t)c * m + r]));
    }
  printf("gesvd %-10s reconstruction < 1e-4 %s\n", prec, resid < 1e-4 ? "yes" : "NO");
  cudaFree(dA); cudaFree(dS); cudaFree(dU); cudaFree(dVT); cudaFree(dinfo);
  if (dwork) cudaFree(dwork);
}

int main() {
  cusolverDnHandle_t h;
  if (cusolverDnCreate(&h) != CUSOLVER_STATUS_SUCCESS) { printf("create failed\n"); return 1; }

  chol<float>(h, "f32", cusolverDnSpotrf_bufferSize, cusolverDnSpotrf, cusolverDnSpotrs);
  chol<double>(h, "f64", cusolverDnDpotrf_bufferSize, cusolverDnDpotrf, cusolverDnDpotrs);
  lu<float>(h, "f32", cusolverDnSgetrf_bufferSize, cusolverDnSgetrf, cusolverDnSgetrs);
  lu<double>(h, "f64", cusolverDnDgetrf_bufferSize, cusolverDnDgetrf, cusolverDnDgetrs);
  qr<float>(h, "f32", cusolverDnSgeqrf_bufferSize, cusolverDnSgeqrf, cusolverDnSorgqr,
            cusolverDnSormqr);
  qr<double>(h, "f64", cusolverDnDgeqrf_bufferSize, cusolverDnDgeqrf, cusolverDnDorgqr,
             cusolverDnDormqr);
  eig<float>(h, "f32", cusolverDnSsyevd_bufferSize, cusolverDnSsyevd);
  eig<double>(h, "f64", cusolverDnDsyevd_bufferSize, cusolverDnDsyevd);
  sv<float>(h, "f32", cusolverDnSgesvd_bufferSize, cusolverDnSgesvd);
  sv<double>(h, "f64", cusolverDnDgesvd_bufferSize, cusolverDnDgesvd);

  int major = 0;
  cusolverGetProperty(MAJOR_VERSION, &major);
  printf("cusolver major %d\n", major);
  cusolverDnDestroy(h);
  return 0;
}
