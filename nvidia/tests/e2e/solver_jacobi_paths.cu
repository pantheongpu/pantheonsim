// cuSOLVER's symmetric eigen- and singular value solvers (syevd, syevj, gesvd, gesvdj and their Jacobi settings), single and
// double precision, over sizes that put the sweeps' vector tails and groups in different places (rows and columns that are
// and are not multiples of four, tall and square matrices, both triangles).
//
// Every result is checked by its defining property, so the same program checks NVIDIA's library and VirtualGPU's:
//     syevd/syevj   A v = w v for every pair, V orthonormal, w ascending
//     gesvd/gesvdj  A = U S V^T, V orthonormal, s descending and not negative
// On VirtualGPU, where the solvers are the cyclic Jacobi methods of src/host_jacobi.hpp, each result (the numbers, the
// sweeps the Jacobi info reports and its residual) is also compared with the digest the plain loops gave before the
// sweeps were put on vector lanes (the values below were taken from that build): the two must agree to the bit.
// Run with --print to list the digests this build gives.
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int failures = 0;
static bool print_digests = false;
static const bool on_sim = std::getenv("VGPU_GPU") != nullptr;

static void check(bool ok, const std::string& what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what.c_str(), err);
  if (!ok) ++failures;
}

struct Expect {
  const char* name;
  uint64_t digest;
};
// Digests of the plain loops' results (see the top of the file).
static const Expect kExpected[] = {
  {"Dsyevd_n1_z0_lower", 0xbacd8594d72ba12eull},
  {"Dsyevd_n1_z3_upper", 0x9131c8b86c17840aull},
  {"Ssyevd_n1_z0_lower", 0xa60886e50684ac32ull},
  {"Dsyevd_n2_z0_lower", 0x7ec29f87a6746cb7ull},
  {"Dsyevd_n2_z3_upper", 0x3b2bbf4cf2238d5eull},
  {"Ssyevd_n2_z0_lower", 0xdb84fc8ba5b677f2ull},
  {"Dsyevd_n3_z0_lower", 0xb8ee5b283fde4c84ull},
  {"Dsyevd_n3_z3_upper", 0x85e14450323292c1ull},
  {"Ssyevd_n3_z0_lower", 0x9ea07346cab3e350ull},
  {"Dsyevd_n5_z0_lower", 0x54df8d89583443c8ull},
  {"Dsyevd_n5_z3_upper", 0xb4710775cb6c5440ull},
  {"Ssyevd_n5_z0_lower", 0xc4669028cb722f72ull},
  {"Dsyevd_n8_z0_lower", 0xba4839fb990036ecull},
  {"Dsyevd_n8_z3_upper", 0x1bb145c34651ac54ull},
  {"Ssyevd_n8_z0_lower", 0xc9543ab4dd422e57ull},
  {"Dsyevd_n13_z0_lower", 0x661a2744e69b7267ull},
  {"Dsyevd_n13_z3_upper", 0xde84870f163dfddcull},
  {"Ssyevd_n13_z0_lower", 0xadafeaa8c6bb2d64ull},
  {"Dsyevd_n24_z0_lower", 0x2dc6590288aa227dull},
  {"Dsyevd_n24_z3_upper", 0xaa0acb78be6774c2ull},
  {"Ssyevd_n24_z0_lower", 0x2ac47591e29dbf3full},
  {"Dsyevd_n33_z0_lower", 0x6a5e50b219c72ddbull},
  {"Dsyevd_n33_z3_upper", 0x590d635994fc3274ull},
  {"Ssyevd_n33_z0_lower", 0x962af5b98eab3b56ull},
  {"Dsyevd_n13_z0_lower_novec", 0x809fca19c63a5a69ull},
  {"Dsyevd_n70_z0_lower", 0x130743715cac20d0ull},
  {"Dsyevj_n8_z0", 0x8205219cdef69336ull},
  {"Dsyevj_n24_z0", 0xb28b4e1e0437f4f4ull},
  {"Dsyevj_n33_z4", 0xe4cf5b0d5d52e9f3ull},
  {"Dgesvd_1x1_z0_AA", 0xec21b3bd0b7ab7cdull},
  {"Sgesvd_1x1_z0_SS", 0xb7c15f58edc4daa3ull},
  {"Dgesvd_3x3_z0_AA", 0x919408da9523ee7aull},
  {"Sgesvd_3x3_z0_SS", 0x4374417fdb86dcf9ull},
  {"Dgesvd_5x5_z0_AA", 0x0cb4920702e8a169ull},
  {"Sgesvd_5x5_z0_SS", 0x1e03993ca69d19c9ull},
  {"Dgesvd_8x8_z0_AA", 0x394e07f6b0b11b53ull},
  {"Sgesvd_8x8_z0_SS", 0xb5168d67f5334db9ull},
  {"Dgesvd_13x13_z0_AA", 0x94f0033482f9f242ull},
  {"Sgesvd_13x13_z0_SS", 0x4304b8431d67ee3cull},
  {"Dgesvd_24x24_z0_AA", 0x7e967f388cff8c44ull},
  {"Sgesvd_24x24_z0_SS", 0x231ad13c847d120bull},
  {"Dgesvd_13x8_z0_AA", 0xdc4d7f8caced3380ull},
  {"Dgesvd_33x20_z0_SS", 0x62716848b3e8b83full},
  {"Dgesvd_24x24_z3_AA", 0x1b81f2baf8764431ull},
  {"Dgesvd_20x11_z0_NN", 0xa5eddc073b06516full},
  {"Dgesvd_70x50_z0_AA", 0x5b0394e315a2312aull},
  {"Dgesvdj_8x8_e0_z0", 0x38d452daa3a8b877ull},
  {"Dgesvdj_13x8_e1_z0", 0xef13f69dda29dc38ull},
  {"Dgesvdj_24x24_e1_z3", 0x4fec293b04258b46ull},
  {"Dgesvdj_8x13_e1_z0", 0x4e83722c323afe35ull},
  {"Dgesvdj_33x33_e0_z0", 0xd17c09e0fda94d16ull},
};

static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const unsigned char* b = static_cast<const unsigned char*>(p);
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 1099511628211ull;
  }
  return h;
}

static void digest(const std::string& name, uint64_t d) {
  if (!on_sim) return;
  if (print_digests) {
    std::printf("  {\"%s\", 0x%016llxull},\n", name.c_str(), static_cast<unsigned long long>(d));
    return;
  }
  for (const Expect& e : kExpected)
    if (name == e.name) {
      check(e.digest == d, "digest " + name, 0);
      return;
    }
  check(false, "digest " + name + " (no expected value)", 0);
}

struct Rng {
  uint64_t s = 0x2545F4914F6CDD1Dull;
  double u() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<double>(s >> 11) * (2.0 / 9007199254740992.0) - 1.0;
  }
};

template <class T> static T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  if (n) cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

static cusolverDnHandle_t h;

template <class T> static T cast(double x) { return static_cast<T>(x); }
template <class T> static double tol() { return sizeof(T) == 4 ? 2e-4 : 1e-10; }

// A symmetric matrix (both triangles equal) with one entry in `zero_every` kept (the others zero) when zero_every > 0.
static std::vector<double> sym(int n, Rng& r, int zero_every) {
  std::vector<double> a(static_cast<size_t>(n) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i <= j; ++i) {
      double x = r.u();
      if (zero_every > 0 && i != j && (i * 7 + j * 3) % zero_every != 0) x = 0.0;
      a[static_cast<size_t>(j) * n + i] = a[static_cast<size_t>(i) * n + j] = x;
    }
  return a;
}

template <class T> static void eig_checks(const std::string& name, const std::vector<double>& a, int n, const std::vector<T>& w,
                                          const std::vector<T>& v) {
  double resid = 0, ortho = 0, scale = 1;
  for (double x : a) scale = std::max(scale, std::fabs(x));
  bool ascending = true;
  for (int k = 0; k < n; ++k) {
    if (k && !(w[k] >= w[k - 1])) ascending = false;
    for (int i = 0; i < n; ++i) {
      double av = 0;
      for (int j = 0; j < n; ++j) av += a[static_cast<size_t>(j) * n + i] * static_cast<double>(v[static_cast<size_t>(k) * n + j]);
      resid = std::max(resid, std::fabs(av - static_cast<double>(w[k]) * static_cast<double>(v[static_cast<size_t>(k) * n + i])));
    }
    for (int l = 0; l < n; ++l) {
      double d = 0;
      for (int i = 0; i < n; ++i) d += static_cast<double>(v[static_cast<size_t>(k) * n + i]) * static_cast<double>(v[static_cast<size_t>(l) * n + i]);
      ortho = std::max(ortho, std::fabs(d - (k == l ? 1.0 : 0.0)));
    }
  }
  check(resid <= tol<T>() * scale * std::max(n, 1) && ortho <= tol<T>() * std::max(n, 1) && ascending, name, std::max(resid, ortho));
}

template <class T> static void svd_checks(const std::string& name, const std::vector<double>& a, int m, int n, const std::vector<T>& s,
                                          const std::vector<T>& u, int ldu, const std::vector<T>& vv, int ldv, bool vt) {
  double resid = 0, ortho = 0, scale = 1;
  for (double x : a) scale = std::max(scale, std::fabs(x));
  bool descending = true;
  for (int k = 0; k < n; ++k)
    if (s[k] < 0 || (k && !(s[k] <= s[k - 1]))) descending = false;
  // V's columns, or VT's rows
  auto V = [&](int i, int k) { return static_cast<double>(vt ? vv[static_cast<size_t>(i) * ldv + k] : vv[static_cast<size_t>(k) * ldv + i]); };
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double x = 0;
      for (int k = 0; k < n; ++k) x += static_cast<double>(u[static_cast<size_t>(k) * ldu + i]) * static_cast<double>(s[k]) * V(j, k);
      resid = std::max(resid, std::fabs(x - a[static_cast<size_t>(j) * m + i]));
    }
  for (int k = 0; k < n; ++k)
    for (int l = 0; l < n; ++l) {
      double d = 0;
      for (int i = 0; i < n; ++i) d += V(i, k) * V(i, l);
      ortho = std::max(ortho, std::fabs(d - (k == l ? 1.0 : 0.0)));
    }
  check(resid <= tol<T>() * scale * std::max(std::max(m, n), 1) && ortho <= tol<T>() * std::max(n, 1) && descending, name,
        std::max(resid, ortho));
}

template <class T> static void run_syevd(int n, int zero_every, bool upper, bool vectors) {
  Rng r;
  r.s += static_cast<uint64_t>(n) * 977 + zero_every;
  const std::vector<double> a = sym(n, r, zero_every);
  std::vector<T> ha(a.size());
  for (size_t i = 0; i < a.size(); ++i) ha[i] = cast<T>(a[i]);
  T* dA = upload(ha);
  T* dW = upload(std::vector<T>(n));
  T* dWork = upload(std::vector<T>(static_cast<size_t>(n) * 64 + 64));
  int* dInfo = upload(std::vector<int>(1));
  const std::string name = std::string(sizeof(T) == 4 ? "Ssyevd" : "Dsyevd") + "_n" + std::to_string(n) + "_z" + std::to_string(zero_every) +
                           (upper ? "_upper" : "_lower") + (vectors ? "" : "_novec");
  int lwork = 0;
  if constexpr (sizeof(T) == 4) {
    cusolverDnSsyevd_bufferSize(h, vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR, upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, n, (float*)dA, n, (float*)dW, &lwork);
    cusolverDnSsyevd(h, vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR, upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, n, (float*)dA, n, (float*)dW, (float*)dWork, std::max(lwork, n * 64), dInfo);
  } else {
    cusolverDnDsyevd_bufferSize(h, vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR, upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, n, (double*)dA, n, (double*)dW, &lwork);
    cusolverDnDsyevd(h, vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR, upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER, n, (double*)dA, n, (double*)dW, (double*)dWork, std::max(lwork, n * 64), dInfo);
  }
  cudaDeviceSynchronize();
  const auto w = download(dW, n);
  const auto v = download(dA, static_cast<size_t>(n) * n);
  const auto info = download(dInfo, 1);
  if (info[0] != 0) check(false, name + " info", info[0]);
  // The values a Jacobi solver reads are one triangle of the matrix.
  std::vector<double> sa = a;
  if (vectors) eig_checks<T>(name, sa, n, w, v);
  else {
    double sum = 0;  // trace = sum of eigenvalues
    for (int i = 0; i < n; ++i) sum += static_cast<double>(w[i]);
    double tr = 0;
    for (int i = 0; i < n; ++i) tr += a[static_cast<size_t>(i) * n + i];
    check(std::fabs(sum - tr) <= tol<T>() * 10 * std::max(n, 1), name, std::fabs(sum - tr));
  }
  digest(name, fnv(v.data(), v.size() * sizeof(T), fnv(w.data(), w.size() * sizeof(T))));
  cudaFree(dA); cudaFree(dW); cudaFree(dWork); cudaFree(dInfo);
}

static void run_syevj(int n, int zero_every) {
  Rng r;
  r.s += static_cast<uint64_t>(n) * 31 + zero_every;
  const std::vector<double> a = sym(n, r, zero_every);
  double* dA = upload(a);
  double* dW = upload(std::vector<double>(n));
  double* dWork = upload(std::vector<double>(static_cast<size_t>(n) * 64 + 64));
  int* dInfo = upload(std::vector<int>(1));
  syevjInfo_t params;
  cusolverDnCreateSyevjInfo(&params);
  const std::string name = "Dsyevj_n" + std::to_string(n) + "_z" + std::to_string(zero_every);
  cusolverDnDsyevj(h, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, dA, n, dW, dWork, n * 64 + 64, dInfo, params);
  cudaDeviceSynchronize();
  int sweeps = 0;
  double residual = 0;
  cusolverDnXsyevjGetSweeps(h, params, &sweeps);
  cusolverDnXsyevjGetResidual(h, params, &residual);
  const auto w = download(dW, n);
  const auto v = download(dA, static_cast<size_t>(n) * n);
  eig_checks<double>(name, a, n, w, v);
  uint64_t d = fnv(v.data(), v.size() * 8, fnv(w.data(), w.size() * 8));
  d = fnv(&residual, 8, fnv(&sweeps, 4, d));
  digest(name, d);
  cusolverDnDestroySyevjInfo(params);
  cudaFree(dA); cudaFree(dW); cudaFree(dWork); cudaFree(dInfo);
}

template <class T> static void run_gesvd(int m, int n, int zero_every, char jobu, char jobvt) {
  Rng r;
  r.s += static_cast<uint64_t>(m) * 131 + static_cast<uint64_t>(n) * 7 + zero_every;
  std::vector<double> a(static_cast<size_t>(m) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double x = r.u();
      if (zero_every > 0 && (i * 5 + j * 3) % zero_every != 0) x = 0.0;
      a[static_cast<size_t>(j) * m + i] = x;
    }
  if (zero_every > 0) a[0] = 1.5;  // keep the matrix from being all zero
  std::vector<T> ha(a.size());
  for (size_t i = 0; i < a.size(); ++i) ha[i] = cast<T>(a[i]);
  // the matrix the solver sees is the rounded one
  for (size_t i = 0; i < a.size(); ++i) a[i] = static_cast<double>(ha[i]);
  const int ldu = m, ldvt = n;
  T* dA = upload(ha);
  T* dS = upload(std::vector<T>(n));
  T* dU = upload(std::vector<T>(static_cast<size_t>(ldu) * m));
  T* dVT = upload(std::vector<T>(static_cast<size_t>(ldvt) * n));
  T* dWork = upload(std::vector<T>(static_cast<size_t>(std::max(m, n)) * 64 + 64));
  int* dInfo = upload(std::vector<int>(1));
  const std::string name = std::string(sizeof(T) == 4 ? "Sgesvd" : "Dgesvd") + "_" + std::to_string(m) + "x" + std::to_string(n) + "_z" +
                           std::to_string(zero_every) + "_" + jobu + jobvt;
  int lwork = 0;
  if constexpr (sizeof(T) == 4) {
    cusolverDnSgesvd_bufferSize(h, m, n, &lwork);
    cusolverDnSgesvd(h, jobu, jobvt, m, n, (float*)dA, m, (float*)dS, (float*)dU, ldu, (float*)dVT, ldvt, (float*)dWork, std::max(lwork, 64), nullptr, dInfo);
  } else {
    cusolverDnDgesvd_bufferSize(h, m, n, &lwork);
    cusolverDnDgesvd(h, jobu, jobvt, m, n, (double*)dA, m, (double*)dS, (double*)dU, ldu, (double*)dVT, ldvt, (double*)dWork, std::max(lwork, 64), nullptr, dInfo);
  }
  cudaDeviceSynchronize();
  const auto s = download(dS, n);
  const auto u = download(dU, static_cast<size_t>(ldu) * m);
  const auto vt = download(dVT, static_cast<size_t>(ldvt) * n);
  if (jobu != 'N' && jobvt != 'N') svd_checks<T>(name, a, m, n, s, u, ldu, vt, ldvt, true);
  else {
    bool descending = true;
    for (int k = 0; k < n; ++k)
      if (s[k] < 0 || (k && !(s[k] <= s[k - 1]))) descending = false;
    check(descending, name, 0);
  }
  uint64_t d = fnv(s.data(), s.size() * sizeof(T));
  if (jobu != 'N') d = fnv(u.data(), u.size() * sizeof(T), d);
  if (jobvt != 'N') d = fnv(vt.data(), vt.size() * sizeof(T), d);
  digest(name, d);
  cudaFree(dA); cudaFree(dS); cudaFree(dU); cudaFree(dVT); cudaFree(dWork); cudaFree(dInfo);
}

static void run_gesvdj(int m, int n, int econ, int zero_every) {
  Rng r;
  r.s += static_cast<uint64_t>(m) * 17 + static_cast<uint64_t>(n) * 5 + zero_every;
  std::vector<double> a(static_cast<size_t>(m) * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double x = r.u();
      if (zero_every > 0 && (i * 5 + j * 3) % zero_every != 0) x = 0.0;
      a[static_cast<size_t>(j) * m + i] = x;
    }
  if (zero_every > 0) a[0] = 1.5;
  const int k = std::min(m, n);
  const int ucols = econ ? k : m, vcols = econ ? k : n;
  double* dA = upload(a);
  double* dS = upload(std::vector<double>(k));
  double* dU = upload(std::vector<double>(static_cast<size_t>(m) * ucols));
  double* dV = upload(std::vector<double>(static_cast<size_t>(n) * vcols));
  double* dWork = upload(std::vector<double>(static_cast<size_t>(std::max(m, n)) * 64 + 64));
  int* dInfo = upload(std::vector<int>(1));
  gesvdjInfo_t params;
  cusolverDnCreateGesvdjInfo(&params);
  const std::string name = "Dgesvdj_" + std::to_string(m) + "x" + std::to_string(n) + "_e" + std::to_string(econ) + "_z" + std::to_string(zero_every);
  cusolverDnDgesvdj(h, CUSOLVER_EIG_MODE_VECTOR, econ, m, n, dA, m, dS, dU, m, dV, n, dWork, static_cast<int>(std::max(m, n)) * 64 + 64, dInfo, params);
  cudaDeviceSynchronize();
  int sweeps = 0;
  double residual = 0;
  cusolverDnXgesvdjGetSweeps(h, params, &sweeps);
  cusolverDnXgesvdjGetResidual(h, params, &residual);
  const auto s = download(dS, k);
  const auto u = download(dU, static_cast<size_t>(m) * ucols);
  const auto v = download(dV, static_cast<size_t>(n) * vcols);
  if (m >= n) {
    std::vector<double> sv(s.begin(), s.end());
    svd_checks<double>(name, a, m, n, sv, u, m, v, n, false);
  } else {
    // the factorization of a wide matrix: A = U S V^T over the first m columns of V
    double resid = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        double x = 0;
        for (int q = 0; q < k; ++q) x += u[static_cast<size_t>(q) * m + i] * s[q] * v[static_cast<size_t>(q) * n + j];
        resid = std::max(resid, std::fabs(x - a[static_cast<size_t>(j) * m + i]));
      }
    check(resid <= 1e-10 * std::max(m, n), name, resid);
  }
  uint64_t d = fnv(v.data(), v.size() * 8, fnv(u.data(), u.size() * 8, fnv(s.data(), s.size() * 8)));
  d = fnv(&residual, 8, fnv(&sweeps, 4, d));
  digest(name, d);
  cusolverDnDestroyGesvdjInfo(params);
  cudaFree(dA); cudaFree(dS); cudaFree(dU); cudaFree(dV); cudaFree(dWork); cudaFree(dInfo);
}

int main(int argc, char** argv) {
  print_digests = argc > 1 && std::strcmp(argv[1], "--print") == 0;
  cusolverDnCreate(&h);
  const int sizes[] = {1, 2, 3, 5, 8, 13, 24, 33};
  for (const int n : sizes) {
    run_syevd<double>(n, 0, false, true);
    run_syevd<double>(n, 3, true, true);
    run_syevd<float>(n, 0, false, true);
  }
  run_syevd<double>(13, 0, false, false);
  run_syevd<double>(70, 0, false, true);
  run_syevj(8, 0);
  run_syevj(24, 0);
  run_syevj(33, 4);
  for (const int n : {1, 3, 5, 8, 13, 24}) {
    run_gesvd<double>(n, n, 0, 'A', 'A');
    run_gesvd<float>(n, n, 0, 'S', 'S');
  }
  run_gesvd<double>(13, 8, 0, 'A', 'A');
  run_gesvd<double>(33, 20, 0, 'S', 'S');
  run_gesvd<double>(24, 24, 3, 'A', 'A');
  run_gesvd<double>(20, 11, 0, 'N', 'N');
  run_gesvd<double>(70, 50, 0, 'A', 'A');
  run_gesvdj(8, 8, 0, 0);
  run_gesvdj(13, 8, 1, 0);
  run_gesvdj(24, 24, 1, 3);
  run_gesvdj(8, 13, 1, 0);
  run_gesvdj(33, 33, 0, 0);
  std::printf("%s\n", failures ? "FAIL" : "PASS");
  return failures ? 1 : 0;
}
