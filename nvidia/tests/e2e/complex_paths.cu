// The complex (C and Z) entry points of cuBLAS and cuSOLVER that PyTorch's
// complex tensors reach, each checked by what defines its result: a product
// against a host reference, a solve's residual, a factorization's
// reconstruction, orthonormal factors, the eigen equation. Column-major, as
// both libraries take matrices.
#include <cublas_v2.h>
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cd = std::complex<double>;
static int failures = 0;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}
#define CK(x)                                                     \
  do {                                                            \
    const int r_ = (int)(x);                                      \
    if (r_ != 0) {                                                \
      std::printf("FAIL %s returned %d\n", #x, r_);              \
      ++failures;                                                 \
      return;                                                     \
    }                                                             \
  } while (0)

// Host matrices are std::complex<double>, column-major, packed. rnd() is
// smooth rather than random, so its rank is small (at most 4): a solve needs
// its diagonal lifted first.
struct HM {
  int m, n;
  std::vector<cd> v;
  cd& operator()(int i, int j) { return v[(size_t)j * m + i]; }
  cd operator()(int i, int j) const { return v[(size_t)j * m + i]; }
};
static HM rnd(int m, int n, int seed) {
  HM a{m, n, std::vector<cd>((size_t)m * n)};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) a(i, j) = cd(std::sin(0.7 * i + 1.3 * j + seed), std::cos(0.9 * i - 0.4 * j + 2 * seed));
  return a;
}
static HM eye(int n) {
  HM a{n, n, std::vector<cd>((size_t)n * n, 0)};
  for (int i = 0; i < n; ++i) a(i, i) = 1;
  return a;
}
enum Op { N, T, C };
static cd opel(const HM& a, Op op, int i, int j) {
  return op == N ? a(i, j) : op == T ? a(j, i) : std::conj(a(j, i));
}
static HM mul(const HM& a, Op oa, const HM& b, Op ob) {
  const int m = oa == N ? a.m : a.n, k = oa == N ? a.n : a.m, n = ob == N ? b.n : b.m;
  HM c{m, n, std::vector<cd>((size_t)m * n, 0)};
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i)
      for (int p = 0; p < k; ++p) c(i, j) += opel(a, oa, i, p) * opel(b, ob, p, j);
  return c;
}
static double diff(const HM& a, const HM& b) {
  double e = 0, s = 1e-30;
  for (size_t i = 0; i < a.v.size(); ++i) {
    e = std::max(e, std::abs(a.v[i] - b.v[i]));
    s = std::max(s, std::abs(b.v[i]));
  }
  return e / s;
}
static HM hpd(int n, int seed) {  // Hermitian positive definite
  HM g = rnd(n, n, seed), a = mul(g, N, g, C);
  for (int i = 0; i < n; ++i) a(i, i) += (double)n;
  return a;
}

// Device copies in either precision.
template <class T> T to_dev(cd v);
template <> cuComplex to_dev<cuComplex>(cd v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
template <> cuDoubleComplex to_dev<cuDoubleComplex>(cd v) { return make_cuDoubleComplex(v.real(), v.imag()); }
static cd from_dev(cuComplex v) { return {v.x, v.y}; }
static cd from_dev(cuDoubleComplex v) { return {v.x, v.y}; }
template <class T> T* up(const HM& a) {
  std::vector<T> h(a.v.size());
  for (size_t i = 0; i < h.size(); ++i) h[i] = to_dev<T>(a.v[i]);
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> HM down(const T* d, int m, int n) {
  std::vector<T> h((size_t)m * n);
  cudaMemcpy(h.data(), d, h.size() * sizeof(T), cudaMemcpyDeviceToHost);
  HM a{m, n, std::vector<cd>(h.size())};
  for (size_t i = 0; i < h.size(); ++i) a.v[i] = from_dev(h[i]);
  return a;
}
template <class R> std::vector<double> down_real(const R* d, int n) {
  std::vector<R> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(R), cudaMemcpyDeviceToHost);
  return std::vector<double>(h.begin(), h.end());
}
template <class T> T** up_ptrs(const std::vector<T*>& p) {
  T** d = nullptr;
  cudaMalloc(&d, p.size() * sizeof(T*));
  cudaMemcpy(d, p.data(), p.size() * sizeof(T*), cudaMemcpyHostToDevice);
  return d;
}

static cublasHandle_t bh;
static cusolverDnHandle_t sh;

static void blas() {
  {  // Cgemm, conjugate-transposed A: C = alpha A^H B + beta C
    const HM a = rnd(5, 4, 1), b = rnd(5, 3, 2), c0 = rnd(4, 3, 3);
    auto *da = up<cuComplex>(a), *db = up<cuComplex>(b), *dc = up<cuComplex>(c0);
    const cuComplex al = make_cuComplex(0.5f, -1.0f), be = make_cuComplex(2.0f, 0.25f);
    CK(cublasCgemm(bh, CUBLAS_OP_C, CUBLAS_OP_N, 4, 3, 5, &al, da, 5, db, 5, &be, dc, 4));
    HM ref = mul(a, C, b, N);
    for (size_t i = 0; i < ref.v.size(); ++i) ref.v[i] = cd(0.5, -1) * ref.v[i] + cd(2, 0.25) * c0.v[i];
    const double e = diff(down(dc, 4, 3), ref);
    check(e < 1e-5, "Cgemm: alpha A^H B + beta C", e);
    cudaFree(da); cudaFree(db); cudaFree(dc);
  }
  {  // ZgemmStridedBatched and CgemmBatched, A B^T
    const int m = 3, n = 4, k = 5, batch = 3;
    HM a{m, k * batch, {}}, b{n, k * batch, {}};
    a.v.clear(); b.v.clear();
    std::vector<HM> as, bs;
    for (int i = 0; i < batch; ++i) {
      as.push_back(rnd(m, k, 10 + i)); bs.push_back(rnd(n, k, 20 + i));
      a.v.insert(a.v.end(), as[i].v.begin(), as[i].v.end());
      b.v.insert(b.v.end(), bs[i].v.begin(), bs[i].v.end());
    }
    auto *da = up<cuDoubleComplex>(a), *db = up<cuDoubleComplex>(b);
    cuDoubleComplex* dc = up<cuDoubleComplex>(HM{m, n * batch, std::vector<cd>((size_t)m * n * batch, 0)});
    const cuDoubleComplex one = make_cuDoubleComplex(1, 0), zero = make_cuDoubleComplex(0, 0);
    CK(cublasZgemmStridedBatched(bh, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &one, da, m, (long long)m * k, db, n,
                                 (long long)n * k, &zero, dc, m, (long long)m * n, batch));
    double e = 0;
    const HM got = down(dc, m, n * batch);
    for (int i = 0; i < batch; ++i) {
      HM part{m, n, std::vector<cd>(got.v.begin() + (size_t)i * m * n, got.v.begin() + (size_t)(i + 1) * m * n)};
      e = std::max(e, diff(part, mul(as[i], N, bs[i], T)));
    }
    check(e < 1e-12, "ZgemmStridedBatched: A B^T per member", e);
    std::vector<cuComplex*> pa, pb, pc;
    for (int i = 0; i < batch; ++i) {
      pa.push_back(up<cuComplex>(as[i])); pb.push_back(up<cuComplex>(bs[i]));
      pc.push_back(up<cuComplex>(HM{m, n, std::vector<cd>((size_t)m * n, 0)}));
    }
    cuComplex **dpa = up_ptrs(pa), **dpb = up_ptrs(pb), **dpc = up_ptrs(pc);
    const cuComplex onef = make_cuComplex(1, 0), zerof = make_cuComplex(0, 0);
    CK(cublasCgemmBatched(bh, CUBLAS_OP_N, CUBLAS_OP_T, m, n, k, &onef, dpa, m, dpb, n, &zerof, dpc, m, batch));
    e = 0;
    for (int i = 0; i < batch; ++i) e = std::max(e, diff(down(pc[i], m, n), mul(as[i], N, bs[i], T)));
    check(e < 1e-5, "CgemmBatched: A B^T per member", e);
    for (int i = 0; i < batch; ++i) { cudaFree(pa[i]); cudaFree(pb[i]); cudaFree(pc[i]); }
    cudaFree(dpa); cudaFree(dpb); cudaFree(dpc); cudaFree(da); cudaFree(db); cudaFree(dc);
  }
  {  // Cgemv with A^H, Cdotc, Caxpy, Csscal
    const HM a = rnd(5, 4, 4), x = rnd(5, 1, 5), y0 = rnd(4, 1, 6);
    auto *da = up<cuComplex>(a), *dx = up<cuComplex>(x), *dy = up<cuComplex>(y0);
    const cuComplex al = make_cuComplex(1, 1), be = make_cuComplex(0, 0);
    CK(cublasCgemv(bh, CUBLAS_OP_C, 5, 4, &al, da, 5, dx, 1, &be, dy, 1));
    HM ref = mul(a, C, x, N);
    for (auto& v : ref.v) v *= cd(1, 1);
    double e = diff(down(dy, 4, 1), ref);
    check(e < 1e-5, "Cgemv: alpha A^H x", e);
    cuComplex dot;
    CK(cublasCdotc(bh, 4, dy, 1, dy, 1, &dot));
    double nrm = 0;
    for (const cd& v : ref.v) nrm += std::norm(v);
    check(std::abs(from_dev(dot) - cd(nrm, 0)) < 1e-4 * nrm, "Cdotc: y^H y is |y|^2", std::abs(from_dev(dot) - cd(nrm)));
    const float two = 2.0f;
    CK(cublasCsscal(bh, 4, &two, dy, 1));
    const cuComplex mhalf = make_cuComplex(-0.5f, 0);
    CK(cublasCaxpy(bh, 4, &mhalf, dy, 1, dy, 1));  // y + (-1/2) y = y/2: back to ref
    e = diff(down(dy, 4, 1), ref);
    check(e < 1e-5, "Csscal then Caxpy: 2y - y = y", e);
    cudaFree(da); cudaFree(dx); cudaFree(dy);
  }
  {  // Ztrsm right, lower, conjugate-transposed: X A^H = alpha B
    const int m = 4, n = 5;
    HM a = rnd(n, n, 7), b = rnd(m, n, 8);
    for (int i = 0; i < n; ++i) a(i, i) += 4.0;
    auto *da = up<cuDoubleComplex>(a), *db = up<cuDoubleComplex>(b);
    const cuDoubleComplex al = make_cuDoubleComplex(0.5, 1);
    CK(cublasZtrsm(bh, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, CUBLAS_DIAG_NON_UNIT, m, n, &al, da, n,
                   db, m));
    HM tri = a;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < j; ++i) tri(i, j) = 0;
    HM ref = b;
    for (auto& v : ref.v) v *= cd(0.5, 1);
    const double e = diff(mul(down(db, m, n), N, tri, C), ref);
    check(e < 1e-12, "Ztrsm right, lower, A^H: X A^H = alpha B", e);
    cudaFree(da); cudaFree(db);
  }
  {  // CgetrfBatched + CgetrsBatched with op C
    const int n = 5, batch = 2;
    std::vector<HM> as, bs;
    std::vector<cuComplex *> pa, pb;
    for (int i = 0; i < batch; ++i) {
      HM a = rnd(n, n, 30 + i);
      for (int d = 0; d < n; ++d) a(d, d) += 0.2;
      as.push_back(a); bs.push_back(rnd(n, 2, 40 + i));
      pa.push_back(up<cuComplex>(as[i])); pb.push_back(up<cuComplex>(bs[i]));
    }
    cuComplex **dpa = up_ptrs(pa), **dpb = up_ptrs(pb);
    int *dpiv = nullptr, *dinfo = nullptr, hinfo = 7;
    cudaMalloc(&dpiv, sizeof(int) * n * batch);
    cudaMalloc(&dinfo, sizeof(int) * batch);
    CK(cublasCgetrfBatched(bh, n, dpa, n, dpiv, dinfo, batch));
    CK(cublasCgetrsBatched(bh, CUBLAS_OP_C, n, 2, (const cuComplex* const*)dpa, n, dpiv, dpb, n, &hinfo, batch));
    double e = 0;
    for (int i = 0; i < batch; ++i) e = std::max(e, diff(mul(as[i], C, down(pb[i], n, 2), N), bs[i]));
    check(e < 1e-5 && hinfo == 0, "CgetrfBatched + CgetrsBatched: A^H X = B", e);
    for (auto p : pa) cudaFree(p);
    for (auto p : pb) cudaFree(p);
    cudaFree(dpa); cudaFree(dpb); cudaFree(dpiv); cudaFree(dinfo);
  }
  {  // Zherk (lower, A^H A) and Zhemv against the matrix it made
    const int n = 4, k = 6;
    const HM a = rnd(k, n, 9);
    auto* da = up<cuDoubleComplex>(a);
    auto* dc = up<cuDoubleComplex>(HM{n, n, std::vector<cd>((size_t)n * n, cd(99, 99))});
    const double one = 1, zero = 0;
    CK(cublasZherk(bh, CUBLAS_FILL_MODE_LOWER, CUBLAS_OP_C, n, k, &one, da, k, &zero, dc, n));
    const HM ref = mul(a, C, a, N), got = down(dc, n, n);
    double e = 0;
    bool upper_kept = true;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) {
        if (i >= j) e = std::max(e, std::abs(got(i, j) - ref(i, j)));
        else upper_kept = upper_kept && got(i, j) == cd(99, 99);
      }
    check(e < 1e-12 && upper_kept, "Zherk: lower triangle of A^H A, upper untouched", e);
    const HM x = rnd(n, 1, 11);
    auto *dx = up<cuDoubleComplex>(x), *dy = up<cuDoubleComplex>(HM{n, 1, std::vector<cd>(n, 0)});
    const cuDoubleComplex al = make_cuDoubleComplex(1, 0), be = make_cuDoubleComplex(0, 0);
    CK(cublasZhemv(bh, CUBLAS_FILL_MODE_LOWER, n, &al, dc, n, dx, 1, &be, dy, 1));
    const double e2 = diff(down(dy, n, 1), mul(ref, N, x, N));
    check(e2 < 1e-12, "Zhemv: y = A x from A's lower triangle", e2);
    cudaFree(da); cudaFree(dc); cudaFree(dx); cudaFree(dy);
  }
}

static void solver() {
  {  // Cgetrf + Cgetrs (A^H X = B)
    const int n = 6;
    HM a = rnd(n, n, 50);
    for (int i = 0; i < n; ++i) a(i, i) += 3.0;  // rnd() alone has rank 4 at most
    const HM b = rnd(n, 2, 51);
    auto *da = up<cuComplex>(a), *db = up<cuComplex>(b);
    int lwork = 0, *dpiv, *dinfo;
    cudaMalloc(&dpiv, n * sizeof(int));
    cudaMalloc(&dinfo, sizeof(int));
    CK(cusolverDnCgetrf_bufferSize(sh, n, n, da, n, &lwork));
    cuComplex* work = up<cuComplex>(HM{std::max(lwork, 1), 1, std::vector<cd>(std::max(lwork, 1))});
    CK(cusolverDnCgetrf(sh, n, n, da, n, work, dpiv, dinfo));
    CK(cusolverDnCgetrs(sh, CUBLAS_OP_C, n, 2, da, n, dpiv, db, n, dinfo));
    const double e = diff(mul(a, C, down(db, n, 2), N), b);
    check(e < 1e-5, "Cgetrf + Cgetrs: A^H X = B", e);
    cudaFree(da); cudaFree(db); cudaFree(dpiv); cudaFree(dinfo); cudaFree(work);
  }
  {  // Zpotrf + Zpotrs (upper), and Xpotrf/Xpotrs on CUDA_C_32F (lower)
    const int n = 5;
    const HM a = hpd(n, 52), b = rnd(n, 2, 53);
    auto *da = up<cuDoubleComplex>(a), *db = up<cuDoubleComplex>(b);
    int lwork = 0, *dinfo;
    cudaMalloc(&dinfo, sizeof(int));
    CK(cusolverDnZpotrf_bufferSize(sh, CUBLAS_FILL_MODE_UPPER, n, da, n, &lwork));
    cuDoubleComplex* work = up<cuDoubleComplex>(HM{std::max(lwork, 1), 1, std::vector<cd>(std::max(lwork, 1))});
    CK(cusolverDnZpotrf(sh, CUBLAS_FILL_MODE_UPPER, n, da, n, work, lwork, dinfo));
    CK(cusolverDnZpotrs(sh, CUBLAS_FILL_MODE_UPPER, n, 2, da, n, db, n, dinfo));
    double e = diff(mul(a, N, down(db, n, 2), N), b);
    check(e < 1e-12, "Zpotrf + Zpotrs (upper): A X = B", e);
    cusolverDnParams_t params;
    CK(cusolverDnCreateParams(&params));
    auto *fa = up<cuComplex>(a), *fb = up<cuComplex>(b);
    size_t ds = 0, hs = 0;
    CK(cusolverDnXpotrf_bufferSize(sh, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_C_32F, fa, n, CUDA_C_32F, &ds, &hs));
    void* xw = nullptr;
    cudaMalloc(&xw, std::max<size_t>(ds, 1));
    std::vector<char> hw(std::max<size_t>(hs, 1));
    CK(cusolverDnXpotrf(sh, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_C_32F, fa, n, CUDA_C_32F, xw, ds, hw.data(), hs, dinfo));
    CK(cusolverDnXpotrs(sh, params, CUBLAS_FILL_MODE_LOWER, n, 2, CUDA_C_32F, fa, n, CUDA_C_32F, fb, n, dinfo));
    e = diff(mul(a, N, down(fb, n, 2), N), b);
    check(e < 1e-5, "Xpotrf + Xpotrs on CUDA_C_32F (lower): A X = B", e);
    cusolverDnDestroyParams(params);
    cudaFree(da); cudaFree(db); cudaFree(fa); cudaFree(fb); cudaFree(dinfo); cudaFree(work); cudaFree(xw);
  }
  {  // Zgeqrf, Zungqr (Q R = A, Q^H Q = I), Zunmqr (Q^H A = R)
    const int m = 7, n = 4;
    const HM a = rnd(m, n, 54);
    auto *da = up<cuDoubleComplex>(a);
    auto* dtau = up<cuDoubleComplex>(HM{n, 1, std::vector<cd>(n)});
    int lwork = 0, *dinfo;
    cudaMalloc(&dinfo, sizeof(int));
    CK(cusolverDnZgeqrf_bufferSize(sh, m, n, da, m, &lwork));
    cuDoubleComplex* work = up<cuDoubleComplex>(HM{std::max(lwork, 64), 1, std::vector<cd>(std::max(lwork, 64))});
    CK(cusolverDnZgeqrf(sh, m, n, da, m, dtau, work, std::max(lwork, 64), dinfo));
    const HM f = down(da, m, n);
    HM r{n, n, std::vector<cd>((size_t)n * n, 0)};
    for (int j = 0; j < n; ++j)
      for (int i = 0; i <= j; ++i) r(i, j) = f(i, j);
    auto* dc = up<cuDoubleComplex>(a);
    CK(cusolverDnZunmqr_bufferSize(sh, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, m, n, n, da, m, dtau, dc, m, &lwork));
    CK(cusolverDnZunmqr(sh, CUBLAS_SIDE_LEFT, CUBLAS_OP_C, m, n, n, da, m, dtau, dc, m, work, std::max(lwork, 64), dinfo));
    const HM qha = down(dc, m, n);
    HM top{n, n, std::vector<cd>((size_t)n * n)};
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < n; ++i) top(i, j) = qha(i, j);
    double below = 0;
    for (int j = 0; j < n; ++j)
      for (int i = n; i < m; ++i) below = std::max(below, std::abs(qha(i, j)));
    double e = diff(top, r);
    check(e < 1e-12 && below < 1e-12, "Zunmqr (left, Q^H): Q^H A is R, zero below", std::max(e, below));
    CK(cusolverDnZungqr_bufferSize(sh, m, n, n, da, m, dtau, &lwork));
    CK(cusolverDnZungqr(sh, m, n, n, da, m, dtau, work, std::max(lwork, 64), dinfo));
    const HM q = down(da, m, n);
    e = std::max(diff(mul(q, N, r, N), a), diff(mul(q, C, q, N), eye(n)));
    check(e < 1e-12, "Zgeqrf + Zungqr: Q R = A, Q^H Q = I", e);
    cudaFree(da); cudaFree(dtau); cudaFree(dinfo); cudaFree(work); cudaFree(dc);
  }
  {  // Cheevd, and Xsyevd on CUDA_C_64F with real eigenvalues
    const int n = 5;
    const HM a = hpd(n, 55);
    auto* da = up<cuComplex>(a);
    float* dw = nullptr;
    cudaMalloc(&dw, n * sizeof(float));
    int lwork = 0, *dinfo;
    cudaMalloc(&dinfo, sizeof(int));
    CK(cusolverDnCheevd_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, da, n, dw, &lwork));
    cuComplex* work = up<cuComplex>(HM{std::max(lwork, 1), 1, std::vector<cd>(std::max(lwork, 1))});
    CK(cusolverDnCheevd(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, da, n, dw, work, lwork, dinfo));
    auto eig_err = [&](const HM& v, const std::vector<double>& w) {
      HM av = mul(a, N, v, N);
      double e = 0;
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) e = std::max(e, std::abs(av(i, j) - w[j] * v(i, j)));
      bool asc = true;
      for (int j = 1; j < n; ++j) asc = asc && w[j - 1] <= w[j];
      return asc ? e / w.back() : 1.0;
    };
    double e = eig_err(down(da, n, n), down_real(dw, n));
    check(e < 1e-5, "Cheevd: A v = w v, w ascending", e);
    cusolverDnParams_t params;
    CK(cusolverDnCreateParams(&params));
    auto* za = up<cuDoubleComplex>(a);
    double* zw = nullptr;
    cudaMalloc(&zw, n * sizeof(double));
    size_t ds = 0, hs = 0;
    CK(cusolverDnXsyevd_bufferSize(sh, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, CUDA_C_64F, za, n,
                                   CUDA_R_64F, zw, CUDA_C_64F, &ds, &hs));
    void* xw = nullptr;
    cudaMalloc(&xw, std::max<size_t>(ds, 1));
    std::vector<char> hw(std::max<size_t>(hs, 1));
    CK(cusolverDnXsyevd(sh, params, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_UPPER, n, CUDA_C_64F, za, n, CUDA_R_64F,
                        zw, CUDA_C_64F, xw, ds, hw.data(), hs, dinfo));
    e = eig_err(down(za, n, n), down_real(zw, n));
    check(e < 1e-12, "Xsyevd on CUDA_C_64F: A v = w v, w real and ascending", e);
    cusolverDnDestroyParams(params);
    cudaFree(da); cudaFree(dw); cudaFree(dinfo); cudaFree(work); cudaFree(za); cudaFree(zw); cudaFree(xw);
  }
  {  // Cgesvdj (tall economy, wide full) and Zgesvd (U, VT = V^H)
    gesvdjInfo_t info;
    CK(cusolverDnCreateGesvdjInfo(&info));
    int* dinfo;
    cudaMalloc(&dinfo, sizeof(int));
    struct Shape { int m, n, econ; const char* what; };
    for (const Shape s : {Shape{6, 4, 1, "Cgesvdj 6x4 economy: A = U S V^H, U and V orthonormal"},
                          Shape{3, 5, 0, "Cgesvdj 3x5 full: A = U S V^H, U and V unitary"}}) {
      const int m = s.m, n = s.n, k = std::min(m, n), uc = s.econ ? k : m, vc = s.econ ? k : n;
      const HM a = rnd(m, n, 60 + m);
      auto* da = up<cuComplex>(a);
      auto* du = up<cuComplex>(HM{m, uc, std::vector<cd>((size_t)m * uc)});
      auto* dv = up<cuComplex>(HM{n, vc, std::vector<cd>((size_t)n * vc)});
      float* ds = nullptr;
      cudaMalloc(&ds, k * sizeof(float));
      int lwork = 0;
      CK(cusolverDnCgesvdj_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, s.econ, m, n, da, m, ds, du, m, dv, n, &lwork, info));
      cuComplex* work = up<cuComplex>(HM{std::max(lwork, 1), 1, std::vector<cd>(std::max(lwork, 1))});
      CK(cusolverDnCgesvdj(sh, CUSOLVER_EIG_MODE_VECTOR, s.econ, m, n, da, m, ds, du, m, dv, n, work, lwork, dinfo, info));
      const HM u = down(du, m, uc), v = down(dv, n, vc);
      const auto sv = down_real(ds, k);
      HM us{m, k, std::vector<cd>((size_t)m * k)};
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i) us(i, j) = u(i, j) * sv[j];
      HM vk{n, k, std::vector<cd>(v.v.begin(), v.v.begin() + (size_t)n * k)};
      const double e = std::max({diff(mul(us, N, vk, C), a), diff(mul(u, C, u, N), eye(uc)), diff(mul(v, C, v, N), eye(vc))});
      check(e < 1e-5, s.what, e);
      cudaFree(da); cudaFree(du); cudaFree(dv); cudaFree(ds); cudaFree(work);
    }
    {
      const int m = 5, n = 3;
      const HM a = rnd(m, n, 70);
      auto* da = up<cuDoubleComplex>(a);
      auto* du = up<cuDoubleComplex>(HM{m, m, std::vector<cd>((size_t)m * m)});
      auto* dvt = up<cuDoubleComplex>(HM{n, n, std::vector<cd>((size_t)n * n)});
      double *ds = nullptr, *rwork = nullptr;
      cudaMalloc(&ds, n * sizeof(double));
      cudaMalloc(&rwork, 5 * n * sizeof(double));
      int lwork = 0;
      CK(cusolverDnZgesvd_bufferSize(sh, m, n, &lwork));
      cuDoubleComplex* work = up<cuDoubleComplex>(HM{std::max(lwork, 1), 1, std::vector<cd>(std::max(lwork, 1))});
      CK(cusolverDnZgesvd(sh, 'A', 'A', m, n, da, m, ds, du, m, dvt, n, work, lwork, rwork, dinfo));
      const HM u = down(du, m, m), vt = down(dvt, n, n);
      const auto sv = down_real(ds, n);
      HM us{m, n, std::vector<cd>((size_t)m * n)};
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) us(i, j) = u(i, j) * sv[j];
      const double e = std::max({diff(mul(us, N, vt, N), a), diff(mul(u, C, u, N), eye(m)), diff(mul(vt, N, vt, C), eye(n))});
      check(e < 1e-12, "Zgesvd 5x3 (A, A): A = U S VT, U and VT unitary", e);
      cudaFree(da); cudaFree(du); cudaFree(dvt); cudaFree(ds); cudaFree(rwork); cudaFree(work);
    }
    cusolverDnDestroyGesvdjInfo(info);
    cudaFree(dinfo);
  }
  {  // CheevjBatched and CpotrfBatched
    const int n = 4, batch = 3;
    HM all{n, n * batch, {}};
    std::vector<HM> as;
    for (int b = 0; b < batch; ++b) {
      as.push_back(hpd(n, 80 + b));
      all.v.insert(all.v.end(), as[b].v.begin(), as[b].v.end());
    }
    auto* da = up<cuComplex>(all);
    float* dw = nullptr;
    cudaMalloc(&dw, n * batch * sizeof(float));
    int* dinfo = nullptr;
    cudaMalloc(&dinfo, batch * sizeof(int));
    syevjInfo_t params;
    CK(cusolverDnCreateSyevjInfo(&params));
    int lwork = 0;
    CK(cusolverDnCheevjBatched_bufferSize(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, da, n, dw, &lwork,
                                          params, batch));
    cuComplex* work = up<cuComplex>(HM{std::max(lwork, 1), 1, std::vector<cd>(std::max(lwork, 1))});
    CK(cusolverDnCheevjBatched(sh, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, n, da, n, dw, work, lwork, dinfo,
                               params, batch));
    const HM vs = down(da, n, n * batch);
    const auto ws = down_real(dw, n * batch);
    double e = 0;
    for (int b = 0; b < batch; ++b) {
      HM v{n, n, std::vector<cd>(vs.v.begin() + (size_t)b * n * n, vs.v.begin() + (size_t)(b + 1) * n * n)};
      HM av = mul(as[b], N, v, N);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i) e = std::max(e, std::abs(av(i, j) - ws[b * n + j] * v(i, j)) / ws[b * n + n - 1]);
    }
    check(e < 1e-5, "CheevjBatched: A v = w v for each matrix", e);
    std::vector<cuComplex*> pa;
    for (int b = 0; b < batch; ++b) pa.push_back(up<cuComplex>(as[b]));
    cuComplex** dpa = up_ptrs(pa);
    CK(cusolverDnCpotrfBatched(sh, CUBLAS_FILL_MODE_LOWER, n, dpa, n, dinfo, batch));
    e = 0;
    for (int b = 0; b < batch; ++b) {
      HM l = down(pa[b], n, n);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < j; ++i) l(i, j) = 0;
      e = std::max(e, diff(mul(l, N, l, C), as[b]));
    }
    check(e < 1e-5, "CpotrfBatched: L L^H = A for each matrix", e);
    cusolverDnDestroySyevjInfo(params);
    for (auto p : pa) cudaFree(p);
    cudaFree(dpa); cudaFree(da); cudaFree(dw); cudaFree(dinfo); cudaFree(work);
  }
}

int main() {
  if (cublasCreate(&bh) || cusolverDnCreate(&sh)) {
    std::printf("FAIL: could not create the library handles\n");
    return 1;
  }
  blas();
  solver();
  cusolverDnDestroy(sh);
  cublasDestroy(bh);
  std::printf(failures ? "FAIL: %d complex checks\n" : "PASS: every complex check\n", failures);
  return failures ? 1 : 0;
}
