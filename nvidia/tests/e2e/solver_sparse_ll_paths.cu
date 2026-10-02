// cusolverSp's low-level preview API (cusolverSp_LOWLEVEL_PREVIEW.h): LU on
// the host, QR and Cholesky on the host and the device, step by step. Checked
// by the solves and factors, and against what an RTX 3060 (CUDA 13.0) gave on
// these very matrices:
//
//   LU    the row order P for thresholds 1, 0.5 and 0 (no column reordering,
//         Q the identity; threshold 0 keeps every diagonal), P A = L U with
//         L's unit diagonal not stored; a zero pivot's position; a symmetric
//         descriptor, and Factor before Analysis or before BufferInfo, all
//         INVALID_VALUE
//   QR    the least-squares x of (A - mu I) x = b; b overwritten by Q^H b,
//         every column reflected (a square matrix's last entry flips too);
//         ZeroPivot's first |R(j,j)| <= tol
//   Chol  A(P, P) = L L^T with P the elimination tree's postorder: Diag in
//         that order; a pivot that is not positive named 0-based by the host
//         form and one higher by the device form
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusolverSp.h>
#include <cusolverSp_LOWLEVEL_PREVIEW.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <vector>

using cd = std::complex<double>;
using V = std::vector<double>;
using I = std::vector<int>;
static int failures = 0;
static cusolverSpHandle_t h;
static cusparseMatDescr_t descr;

static void check(bool ok, const char* what, double err) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}
#define CK(x)                                                \
  do {                                                       \
    const int r_ = (int)(x);                                 \
    if (r_ != 0) {                                           \
      std::printf("FAIL %s returned %d\n", #x, r_);         \
      ++failures;                                            \
      return;                                                \
    }                                                        \
  } while (0)

template <class T> T* upload(const std::vector<T>& v) {
  T* d = nullptr;
  cudaMalloc(&d, std::max<size_t>(1, v.size()) * sizeof(T));
  cudaMemcpy(d, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> std::vector<T> download(const T* d, size_t n) {
  std::vector<T> v(n);
  cudaMemcpy(v.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

struct Csr {
  int m, n;
  I rp, ci;
  V v;
  std::vector<V> dense;
};
// The probe's matrices: entries where (5i + 3j + seed) % 4 == 0, plus a diagonal shift.
static Csr make(int m, int n, int seed, double diag, bool sym = false) {
  Csr c{m, n, {0}, {}, {}, std::vector<V>(m, V(n, 0.0))};
  for (int i = 0; i < m; ++i)
    for (int j = 0; j < n; ++j) {
      double v = (i * 5 + j * 3 + seed) % 4 == 0 ? std::sin(i + 2.0 * j + seed) + 0.3 : 0.0;
      if (i == j) v += diag;
      c.dense[i][j] = v;
    }
  if (sym)
    for (int i = 0; i < m; ++i)
      for (int j = 0; j < i; ++j) c.dense[j][i] = c.dense[i][j];
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      if (c.dense[i][j] != 0) {
        c.ci.push_back(j);
        c.v.push_back(c.dense[i][j]);
      }
    c.rp.push_back((int)c.ci.size());
  }
  return c;
}
static double residual(const std::vector<V>& a, const V& x, const V& b) {
  double r = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    double s = -b[i];
    for (size_t j = 0; j < x.size(); ++j) s += a[i][j] * x[j];
    r = std::max(r, std::fabs(s));
  }
  return r;
}

static void lu() {
  const int n = 6;
  const Csr a = make(n, n, 1, 0.2);
  const int nnz = (int)a.v.size();
  struct Case { double thr; I p; } cases[] = {{1.0, {3, 4, 5, 2, 0, 1}}, {0.5, {3, 4, 2, 5, 0, 1}}, {0.0, {0, 1, 2, 3, 4, 5}}};
  for (const auto& c : cases) {
    csrluInfoHost_t info;
    CK(cusolverSpCreateCsrluInfoHost(&info));
    CK(cusolverSpXcsrluAnalysisHost(h, n, nnz, descr, a.rp.data(), a.ci.data(), info));
    size_t in = 0, ws = 0;
    CK(cusolverSpDcsrluBufferInfoHost(h, n, nnz, descr, a.v.data(), a.rp.data(), a.ci.data(), info, &in, &ws));
    std::vector<char> buf(ws + 16);
    CK(cusolverSpDcsrluFactorHost(h, n, nnz, descr, a.v.data(), a.rp.data(), a.ci.data(), info, c.thr, buf.data()));
    int pos = -7;
    CK(cusolverSpDcsrluZeroPivotHost(h, info, 1e-12, &pos));
    V b = {1, 2, 3, 4, 5, 6}, x(n);
    CK(cusolverSpDcsrluSolveHost(h, n, b.data(), x.data(), info, buf.data()));
    int nl = -1, nu = -1;
    CK(cusolverSpXcsrluNnzHost(h, &nl, &nu, info));
    I P(n), Q(n), lrp(n + 1), lci(nl + 1), urp(n + 1), uci(nu + 1);
    V lv(nl + 1), uv(nu + 1);
    cusparseMatDescr_t dl, du;
    cusparseCreateMatDescr(&dl);
    cusparseCreateMatDescr(&du);
    CK(cusolverSpDcsrluExtractHost(h, P.data(), Q.data(), dl, lv.data(), lrp.data(), lci.data(), du, uv.data(), urp.data(),
                                   uci.data(), info, buf.data()));
    std::vector<V> L(n, V(n, 0.0)), U(n, V(n, 0.0));
    bool strict = true;
    for (int i = 0; i < n; ++i) {
      L[i][i] = 1;
      for (int k = lrp[i]; k < lrp[i + 1]; ++k) {
        strict = strict && lci[k] < i;
        L[i][lci[k]] = lv[k];
      }
      for (int k = urp[i]; k < urp[i + 1]; ++k) U[i][uci[k]] = uv[k];
    }
    double ferr = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += L[i][k] * U[k][j];
        ferr = std::max(ferr, std::fabs(s - a.dense[P[i]][j]));
      }
    bool q_id = true;
    for (int i = 0; i < n; ++i) q_id = q_id && Q[i] == i;
    const double res = residual(a.dense, x, b);
    char what[200];
    std::snprintf(what, sizeof what,
                  "LU threshold %.1f: P as NVIDIA's, Q the identity, P A = L U (L's unit diagonal not stored), A x = b",
                  c.thr);
    check(P == c.p && q_id && strict && pos == -1 && ferr < 1e-12 && res < 1e-12, what, std::max(ferr, res));
    cusolverSpDestroyCsrluInfoHost(info);
    cusparseDestroyMatDescr(dl);
    cusparseDestroyMatDescr(du);
  }
  // A singular matrix: the zero pivot's position; and the refusals.
  Csr s = a;
  for (double& v : s.v) v = 0;
  s.v[0] = 1;
  csrluInfoHost_t info;
  CK(cusolverSpCreateCsrluInfoHost(&info));
  std::vector<char> buf(1 << 16);
  const int before = cusolverSpDcsrluFactorHost(h, n, nnz, descr, s.v.data(), s.rp.data(), s.ci.data(), info, 1.0, buf.data());
  CK(cusolverSpXcsrluAnalysisHost(h, n, nnz, descr, s.rp.data(), s.ci.data(), info));
  const int unbuffered = cusolverSpDcsrluFactorHost(h, n, nnz, descr, s.v.data(), s.rp.data(), s.ci.data(), info, 1.0, buf.data());
  size_t in = 0, ws = 0;
  CK(cusolverSpDcsrluBufferInfoHost(h, n, nnz, descr, s.v.data(), s.rp.data(), s.ci.data(), info, &in, &ws));
  CK(cusolverSpDcsrluFactorHost(h, n, nnz, descr, s.v.data(), s.rp.data(), s.ci.data(), info, 1.0, buf.data()));
  int pos = -7;
  CK(cusolverSpDcsrluZeroPivotHost(h, info, 1e-12, &pos));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
  const int sym = cusolverSpXcsrluAnalysisHost(h, n, nnz, descr, s.rp.data(), s.ci.data(), info);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  check(pos == 1 && before == CUSOLVER_STATUS_INVALID_VALUE && unbuffered == CUSOLVER_STATUS_INVALID_VALUE &&
            sym == CUSOLVER_STATUS_INVALID_VALUE,
        "LU: zero pivot at 1; Factor before Analysis or BufferInfo, a symmetric descriptor INVALID_VALUE", pos);
  cusolverSpDestroyCsrluInfoHost(info);
  // Complex: A x = b.
  {
    std::vector<cuDoubleComplex> zv(a.v.size());
    for (size_t k = 0; k < zv.size(); ++k) zv[k] = make_cuDoubleComplex(a.v[k], 0.25 * std::cos((double)k));
    csrluInfoHost_t zi;
    CK(cusolverSpCreateCsrluInfoHost(&zi));
    CK(cusolverSpXcsrluAnalysisHost(h, n, nnz, descr, a.rp.data(), a.ci.data(), zi));
    size_t in, ws;
    CK(cusolverSpZcsrluBufferInfoHost(h, n, nnz, descr, zv.data(), a.rp.data(), a.ci.data(), zi, &in, &ws));
    std::vector<char> zb(ws + 16);
    CK(cusolverSpZcsrluFactorHost(h, n, nnz, descr, zv.data(), a.rp.data(), a.ci.data(), zi, 1.0, zb.data()));
    std::vector<cuDoubleComplex> b(n), x(n);
    for (int i = 0; i < n; ++i) b[i] = make_cuDoubleComplex(1.0 + i, -0.5 * i);
    CK(cusolverSpZcsrluSolveHost(h, n, b.data(), x.data(), zi, zb.data()));
    double r = 0;
    for (int i = 0; i < n; ++i) {
      cd s = -cd(b[i].x, b[i].y);
      for (int k = a.rp[i]; k < a.rp[i + 1]; ++k) s += cd(zv[k].x, zv[k].y) * cd(x[a.ci[k]].x, x[a.ci[k]].y);
      r = std::max(r, std::abs(s));
    }
    check(r < 1e-12, "LU, complex: A x = b", r);
    cusolverSpDestroyCsrluInfoHost(zi);
  }
}

static void qr(bool device) {
  const char* where = device ? "device" : "host";
  for (int shape = 0; shape < 2; ++shape) {
    const int m = shape == 0 ? 6 : 8, n = 6;
    const double mu = shape == 0 ? 0.0 : 0.7;
    const Csr a = make(m, n, 3, shape == 0 ? 1.5 : 0.0);
    const int nnz = (int)a.v.size();
    V b(m);
    for (int i = 0; i < m; ++i) b[i] = 1.0 + 0.5 * i;
    V x(n), bq(m);
    int pos = -7;
    if (device) {
      csrqrInfo_t info;
      CK(cusolverSpCreateCsrqrInfo(&info));
      int* drp = upload(a.rp);
      int* dci = upload(a.ci);
      double* dv = upload(a.v);
      CK(cusolverSpXcsrqrAnalysis(h, m, n, nnz, descr, drp, dci, info));
      size_t in = 0, ws = 0;
      CK(cusolverSpDcsrqrBufferInfo(h, m, n, nnz, descr, dv, drp, dci, info, &in, &ws));
      void* buf = nullptr;
      cudaMalloc(&buf, ws + 16);
      CK(cusolverSpDcsrqrSetup(h, m, n, nnz, descr, dv, drp, dci, mu, info));
      double* db = upload(b);
      double* dx = upload(V(n));
      CK(cusolverSpDcsrqrFactor(h, m, n, nnz, db, dx, info, buf));  // factor and solve
      x = download(dx, n);
      bq = download(db, m);
      CK(cusolverSpDcsrqrZeroPivot(h, info, 1.5, &pos));
      for (void* p : {(void*)drp, (void*)dci, (void*)dv, buf, (void*)db, (void*)dx}) cudaFree(p);
      cusolverSpDestroyCsrqrInfo(info);
    } else {
      csrqrInfoHost_t info;
      CK(cusolverSpCreateCsrqrInfoHost(&info));
      CK(cusolverSpXcsrqrAnalysisHost(h, m, n, nnz, descr, a.rp.data(), a.ci.data(), info));
      size_t in = 0, ws = 0;
      CK(cusolverSpDcsrqrBufferInfoHost(h, m, n, nnz, descr, a.v.data(), a.rp.data(), a.ci.data(), info, &in, &ws));
      std::vector<char> buf(ws + 16);
      CK(cusolverSpDcsrqrSetupHost(h, m, n, nnz, descr, a.v.data(), a.rp.data(), a.ci.data(), mu, info));
      CK(cusolverSpDcsrqrFactorHost(h, m, n, nnz, nullptr, nullptr, info, buf.data()));
      bq = b;
      CK(cusolverSpDcsrqrSolveHost(h, m, n, bq.data(), x.data(), info, buf.data()));
      CK(cusolverSpDcsrqrZeroPivotHost(h, info, 1.5, &pos));
      cusolverSpDestroyCsrqrInfoHost(info);
    }
    std::vector<V> am = a.dense;
    for (int i = 0; i < n; ++i) am[i][i] -= mu;
    // The normal equations: (A - mu I)^T ((A - mu I) x - b) = 0.
    double ne = 0;
    for (int j = 0; j < n; ++j) {
      double s = 0;
      for (int i = 0; i < m; ++i) {
        double r = -b[i];
        for (int k = 0; k < n; ++k) r += am[i][k] * x[k];
        s += am[i][j] * r;
      }
      ne = std::max(ne, std::fabs(s));
    }
    double nb = 0, nq = 0;
    for (int i = 0; i < m; ++i) {
      nb += b[i] * b[i];
      nq += bq[i] * bq[i];
    }
    // What NVIDIA's returned for these: Q^T b's last entry (the square matrix's
    // reflects even its last column) and the zero pivot at tol 1.5.
    const double last = shape == 0 ? -2.65751 : 4.26982;
    const int want_pos = shape == 0 ? 2 : 1;
    char what[200];
    std::snprintf(what, sizeof what, "QR %s %dx%d mu %.1f: least squares x, b overwritten by Q^H b as NVIDIA's, zero pivot %d",
                  where, m, n, mu, want_pos);
    check(ne < 1e-10 && std::fabs(std::sqrt(nb) - std::sqrt(nq)) < 1e-10 && std::fabs(bq[m - 1] - last) < 1e-4 &&
              pos == want_pos,
          what, ne);
  }
}

static void chol(bool device) {
  const char* where = device ? "device" : "host";
  const int n = 14;
  // The probe's SPD matrix: a random pattern, diagonally dominant.
  std::vector<V> D(n, V(n, 0.0));
  unsigned seed = 7;
  auto rnd = [&]() {
    seed = seed * 1103515245u + 12345u;
    return (seed >> 8) % 1000 / 1000.0;
  };
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j)
      if (rnd() < 0.25) {
        const double v = rnd() - 0.5;
        D[i][j] = D[j][i] = v;
      }
  for (int i = 0; i < n; ++i) {
    double s = 1;
    for (int j = 0; j < n; ++j)
      if (j != i) s += std::fabs(D[i][j]);
    D[i][i] = s + 0.1 * i;
  }
  Csr a{n, n, {0}, {}, {}, D};
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      if (D[i][j] != 0) {
        a.ci.push_back(j);
        a.v.push_back(D[i][j]);
      }
    a.rp.push_back((int)a.ci.size());
  }
  const int nnz = (int)a.v.size();
  // NVIDIA's order for it: the elimination tree's postorder.
  const I p = {9, 2, 5, 0, 1, 3, 4, 6, 7, 8, 10, 11, 12, 13};
  V want(n);
  {
    std::vector<V> L(n, V(n, 0.0));
    for (int j = 0; j < n; ++j) {
      double s = D[p[j]][p[j]];
      for (int k = 0; k < j; ++k) s -= L[j][k] * L[j][k];
      L[j][j] = std::sqrt(s);
      for (int i = j + 1; i < n; ++i) {
        double t = D[p[i]][p[j]];
        for (int k = 0; k < j; ++k) t -= L[i][k] * L[j][k];
        L[i][j] = t / L[j][j];
      }
      want[j] = L[j][j];
    }
  }
  V b(n), x(n), diag(n);
  for (int i = 0; i < n; ++i) b[i] = std::cos(0.7 * i);
  int pos = -7, pos_bad = -7;
  V bad = a.v;  // the diagonal of row 3 made negative
  for (int k = a.rp[3]; k < a.rp[4]; ++k)
    if (a.ci[k] == 3) bad[k] = -1;
  if (device) {
    csrcholInfo_t info;
    CK(cusolverSpCreateCsrcholInfo(&info));
    int* drp = upload(a.rp);
    int* dci = upload(a.ci);
    double* dv = upload(a.v);
    CK(cusolverSpXcsrcholAnalysis(h, n, nnz, descr, drp, dci, info));
    size_t in = 0, ws = 0;
    CK(cusolverSpDcsrcholBufferInfo(h, n, nnz, descr, dv, drp, dci, info, &in, &ws));
    void* buf = nullptr;
    cudaMalloc(&buf, ws + 16);
    CK(cusolverSpDcsrcholFactor(h, n, nnz, descr, dv, drp, dci, info, buf));
    CK(cusolverSpDcsrcholZeroPivot(h, info, 1e-12, &pos));
    double* db = upload(b);
    double* dx = upload(V(n));
    double* dd = upload(V(n));
    CK(cusolverSpDcsrcholSolve(h, n, db, dx, info, buf));
    CK(cusolverSpDcsrcholDiag(h, info, dd));
    x = download(dx, n);
    diag = download(dd, n);
    cudaMemcpy(dv, bad.data(), bad.size() * sizeof(double), cudaMemcpyHostToDevice);
    CK(cusolverSpDcsrcholFactor(h, n, nnz, descr, dv, drp, dci, info, buf));
    CK(cusolverSpDcsrcholZeroPivot(h, info, 1e-12, &pos_bad));
    for (void* q : {(void*)drp, (void*)dci, (void*)dv, buf, (void*)db, (void*)dx, (void*)dd}) cudaFree(q);
    cusolverSpDestroyCsrcholInfo(info);
  } else {
    csrcholInfoHost_t info;
    CK(cusolverSpCreateCsrcholInfoHost(&info));
    CK(cusolverSpXcsrcholAnalysisHost(h, n, nnz, descr, a.rp.data(), a.ci.data(), info));
    size_t in = 0, ws = 0;
    CK(cusolverSpDcsrcholBufferInfoHost(h, n, nnz, descr, a.v.data(), a.rp.data(), a.ci.data(), info, &in, &ws));
    std::vector<char> buf(ws + 16);
    CK(cusolverSpDcsrcholFactorHost(h, n, nnz, descr, a.v.data(), a.rp.data(), a.ci.data(), info, buf.data()));
    CK(cusolverSpDcsrcholZeroPivotHost(h, info, 1e-12, &pos));
    CK(cusolverSpDcsrcholSolveHost(h, n, b.data(), x.data(), info, buf.data()));
    CK(cusolverSpDcsrcholFactorHost(h, n, nnz, descr, bad.data(), a.rp.data(), a.ci.data(), info, buf.data()));
    CK(cusolverSpDcsrcholZeroPivotHost(h, info, 1e-12, &pos_bad));
    diag = want;  // the host form has no Diag
    cusolverSpDestroyCsrcholInfoHost(info);
  }
  double derr = 0;
  for (int k = 0; k < n; ++k) derr = std::max(derr, std::fabs(diag[k] - want[k]));
  const double res = residual(D, x, b);
  // Row 3 is fifth (index 5) in the order: the host names 5, the device 6.
  const int want_bad = device ? 6 : 5;
  char what[200];
  std::snprintf(what, sizeof what, "Cholesky %s: A x = b; L's diagonal in the etree postorder; not positive definite at %d",
                where, want_bad);
  check(pos == -1 && res < 1e-12 && derr < 1e-12 && pos_bad == want_bad, what, std::max(res, derr));
}

int main() {
  if (cusolverSpCreate(&h) || cusparseCreateMatDescr(&descr)) {
    std::printf("FAIL: could not create the handles\n");
    return 1;
  }
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  lu();
  qr(true);
  qr(false);
  chol(true);
  chol(false);
  cusparseDestroyMatDescr(descr);
  cusolverSpDestroy(h);
  std::printf(failures ? "FAIL: %d low-level sparse checks\n" : "PASS: every low-level sparse check\n", failures);
  return failures ? 1 : 0;
}
