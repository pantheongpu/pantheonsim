// cusolverRf, the refactorization module: a system whose LU factors were
// found once (P A Q^T = L U), refactored for new values of A and solved.
// Checked by the solves' residuals and the factors, and by what an RTX 3060
// (CUDA 13.0) does where the documentation is silent:
//
//   - B(i, j) = A(P[i], Q[j]); setup loads B's values into M = L + U - I;
//     a Refactor before Analyze does nothing, and a second Refactor without
//     new values factors the factors in place
//   - the documented defaults, as the getters report them
//   - a zero pivot: ZERO_PIVOT, the boost report USED, and a Solve that then
//     divides by it; with a boost, the pivot becomes the boost
//   - the unit diagonal formats: ExtractSplitFactorsHost's L with or without
//     its unit diagonal, and L D, D^-1 U for the U forms
//   - Solve with two right-hand sides: INVALID_VALUE
//
// The batched forms are checked on the simulator only: NVIDIA's crashed in
// cusolverRfBatchAnalyze on that card (CUDA 13.0 and 13.2) for every input tried.
#include <cuda_runtime.h>
#include <cusolverRf.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static int failures = 0;
static const bool on_sim = std::getenv("VGPU_GPU") != nullptr;
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

using V = std::vector<double>;
using I = std::vector<int>;
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

const int n = 8;
const I P = {2, 0, 7, 5, 1, 3, 6, 4}, Q = {1, 3, 0, 5, 2, 7, 4, 6};

// A sparse matrix, row-major dense here; its B = P A Q^T is diagonally dominant.
static std::vector<V> make_a(double s) {
  std::vector<V> a(n, V(n, 0.0));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if ((i * 7 + j * 3) % 5 == 0) a[i][j] = std::sin(i + 2.0 * j + s) + 0.5;
  a[0][5] = 1.5 + s;
  a[7][1] = -2.0;
  for (int i = 0; i < n; ++i) a[P[i]][Q[i]] += 10.0 + i + s;
  return a;
}
struct Csr {
  I rp, ci;
  V v;
};
static Csr csr(const std::vector<V>& m, const std::vector<std::vector<int>>* pattern = nullptr) {
  Csr c;
  c.rp.push_back(0);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j)
      if (pattern ? (*pattern)[i][j] : m[i][j] != 0) {
        c.ci.push_back(j);
        c.v.push_back(m[i][j]);
      }
    c.rp.push_back((int)c.ci.size());
  }
  return c;
}
// The LU of B without pivoting, with its fill pattern: L unit lower, U upper.
static void lu(const std::vector<V>& a, std::vector<V>* L, std::vector<V>* U, std::vector<std::vector<int>>* pl,
               std::vector<std::vector<int>>* pu) {
  std::vector<V> b(n, V(n));
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) b[i][j] = a[P[i]][Q[j]];
  *L = std::vector<V>(n, V(n, 0.0));
  *U = b;
  *pl = std::vector<std::vector<int>>(n, std::vector<int>(n, 0));
  *pu = *pl;
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      if (b[i][j] != 0) (i > j ? (*pl)[i][j] : (*pu)[i][j]) = 1;
  for (int k = 0; k < n; ++k) {
    (*L)[k][k] = 1;
    (*pl)[k][k] = 1;
    (*pu)[k][k] = 1;
    for (int i = k + 1; i < n; ++i) {
      if (!(*pl)[i][k]) continue;
      const double l = (*U)[i][k] / (*U)[k][k];
      (*L)[i][k] = l;
      (*U)[i][k] = 0;
      for (int j = k + 1; j < n; ++j)
        if ((*pu)[k][j]) {
          (*U)[i][j] -= l * (*U)[k][j];
          (i > j ? (*pl)[i][j] : (*pu)[i][j]) = 1;
        }
    }
  }
}
static double residual(const std::vector<V>& a, const V& x, const V& f) {
  double r = 0;
  for (int i = 0; i < n; ++i) {
    double s = -f[i];
    for (int j = 0; j < n; ++j) s += a[i][j] * x[j];
    r = std::max(r, std::fabs(s));
  }
  return r;
}

struct Problem {
  std::vector<V> a, L, U;
  std::vector<std::vector<int>> pl, pu;
  Csr ca, cl, cu;
  explicit Problem(double s) {
    a = make_a(s);
    lu(a, &L, &U, &pl, &pu);
    ca = csr(a);
    cl = csr(L, &pl);
    cu = csr(U, &pu);
  }
};

static cusolverStatus_t setup_host(cusolverRfHandle_t h, Problem& p) {
  I pp = P, qq = Q;
  return cusolverRfSetupHost(n, (int)p.ca.v.size(), p.ca.rp.data(), p.ca.ci.data(), p.ca.v.data(), (int)p.cl.v.size(),
                             p.cl.rp.data(), p.cl.ci.data(), p.cl.v.data(), (int)p.cu.v.size(), p.cu.rp.data(),
                             p.cu.ci.data(), p.cu.v.data(), pp.data(), qq.data(), h);
}

static void basic() {
  Problem p(0.0);
  cusolverRfHandle_t h;
  CK(cusolverRfCreate(&h));
  cusolverRfMatrixFormat_t fmt;
  cusolverRfUnitDiagonal_t dg;
  double zero = -1, boost = -1;
  cusolverRfFactorization_t fa;
  cusolverRfTriangularSolve_t sa;
  cusolverRfResetValuesFastMode_t fm;
  cusolverRfNumericBoostReport_t rep;
  CK(cusolverRfGetMatrixFormat(h, &fmt, &dg));
  CK(cusolverRfGetNumericProperties(h, &zero, &boost));
  CK(cusolverRfGetAlgs(h, &fa, &sa));
  CK(cusolverRfGetResetValuesFastMode(h, &fm));
  CK(cusolverRfGetNumericBoostReport(h, &rep));
  check(fmt == CUSOLVERRF_MATRIX_FORMAT_CSR && dg == CUSOLVERRF_UNIT_DIAGONAL_STORED_L && zero == 0 && boost == 0 &&
            fa == CUSOLVERRF_FACTORIZATION_ALG0 && sa == CUSOLVERRF_TRIANGULAR_SOLVE_ALG1 &&
            fm == CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF && rep == CUSOLVERRF_NUMERIC_BOOST_NOT_USED,
        "defaults: CSR, unit L stored, zero = boost = 0, ALG0/ALG1, fast mode off", 0);
  CK(setup_host(h, p));
  int* dP = upload(P);
  int* dQ = upload(Q);
  // A Refactor before Analyze does nothing: M still holds B.
  CK(cusolverRfRefactor(h));
  int nnzM = -1;
  int *Mp = nullptr, *Mi = nullptr;
  double* Mx = nullptr;
  CK(cusolverRfAccessBundledFactorsDevice(h, &nnzM, &Mp, &Mi, &Mx));
  const auto mp = download(Mp, n + 1);
  const auto mi = download(Mi, nnzM);
  const auto m0 = download(Mx, nnzM);
  int want_nnz = 0;  // L strictly lower plus U
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) want_nnz += (i > j && p.pl[i][j]) || (i <= j && p.pu[i][j]);
  bool holds_b = mp[n] == nnzM && nnzM == want_nnz;
  for (int i = 0; i < n && holds_b; ++i)
    for (int k = mp[i]; k < mp[i + 1]; ++k) holds_b = holds_b && m0[k] == p.a[P[i]][Q[mi[k]]];
  check(holds_b, "setup: M's pattern is L + U - I, its values B = P A Q^T; Refactor before Analyze leaves them", 0);
  CK(cusolverRfAnalyze(h));
  CK(cusolverRfRefactor(h));
  V f(n);
  for (int i = 0; i < n; ++i) f[i] = 1.0 + i;
  double* dX = upload(f);
  double* dT = upload(V(n));
  CK(cusolverRfSolve(h, dP, dQ, 1, dT, n, dX, n));
  double res = residual(p.a, download(dX, n), f);
  // The split factors are the LU of B.
  int nl = 0, nu = 0;
  int *lp, *li, *up, *ui;
  double *lx, *ux;
  CK(cusolverRfExtractSplitFactorsHost(h, &nl, &lp, &li, &lx, &nu, &up, &ui, &ux));
  double ferr = 0;
  for (int i = 0; i < n; ++i) {
    for (int k = lp[i]; k < lp[i + 1]; ++k) ferr = std::max(ferr, std::fabs(lx[k] - p.L[i][li[k]]));
    for (int k = up[i]; k < up[i + 1]; ++k) ferr = std::max(ferr, std::fabs(ux[k] - p.U[i][ui[k]]));
  }
  const bool sizes = nl == (int)p.cl.v.size() && nu == (int)p.cu.v.size();
  for (void* q : {(void*)lp, (void*)li, (void*)lx, (void*)up, (void*)ui, (void*)ux}) std::free(q);
  check(res < 1e-12 && ferr < 1e-12 && sizes, "Analyze, Refactor, Solve: A x = f; the split factors are B's L (unit) and U",
        std::max(res, ferr));
  // New values, the same pattern.
  Problem p2(0.3);
  int* dArp = upload(p2.ca.rp);
  int* dAci = upload(p2.ca.ci);
  double* dAv = upload(p2.ca.v);
  CK(cusolverRfResetValues(n, (int)p2.ca.v.size(), dArp, dAci, dAv, dP, dQ, h));
  CK(cusolverRfRefactor(h));
  cudaMemcpy(dX, f.data(), n * sizeof(double), cudaMemcpyHostToDevice);
  CK(cusolverRfSolve(h, dP, dQ, 1, dT, n, dX, n));
  res = residual(p2.a, download(dX, n), f);
  check(res < 1e-12, "ResetValues, Refactor, Solve: the new system", res);
  // A second Refactor factors the factors, in place.
  const auto before = download(Mx, nnzM);
  CK(cusolverRfRefactor(h));
  const auto after = download(Mx, nnzM);
  bool again = true;
  for (int i = 0; i < n; ++i)
    for (int k = mp[i]; k < mp[i + 1]; ++k)
      if (mi[k] < i) {
        int dk = mp[mi[k]];
        while (mi[dk] != mi[k]) ++dk;
        // L(i, c) is divided by U(c, c) once more, when nothing else changed it.
        if (mi[k] == 0) again = again && std::fabs(after[k] - before[k] / before[dk]) < 1e-12;
      }
  check(again, "a second Refactor without new values refactors M in place (L divided by U's pivots again)", 0);
  // Two right-hand sides.
  double* dX2 = upload(V(2 * n, 1.0));
  double* dT2 = upload(V(2 * n));
  const int two = cusolverRfSolve(h, dP, dQ, 2, dT2, n, dX2, n);
  check(two == CUSOLVER_STATUS_INVALID_VALUE, "Solve with nrhs = 2: INVALID_VALUE", two);
  // A zero pivot: B(0, 0) = 0.
  {
    V v = p2.ca.v;
    for (int k = p2.ca.rp[P[0]]; k < p2.ca.rp[P[0] + 1]; ++k)
      if (p2.ca.ci[k] == Q[0]) v[k] = 0;
    cudaMemcpy(dAv, v.data(), v.size() * sizeof(double), cudaMemcpyHostToDevice);
    CK(cusolverRfResetValues(n, (int)v.size(), dArp, dAci, dAv, dP, dQ, h));
    const int st = cusolverRfRefactor(h);
    CK(cusolverRfGetNumericBoostReport(h, &rep));
    cudaMemcpy(dX, f.data(), n * sizeof(double), cudaMemcpyHostToDevice);
    CK(cusolverRfSolve(h, dP, dQ, 1, dT, n, dX, n));
    bool nonfinite = false;
    for (double x : download(dX, n)) nonfinite = nonfinite || !std::isfinite(x);
    check(st == CUSOLVER_STATUS_ZERO_PIVOT && rep == CUSOLVERRF_NUMERIC_BOOST_USED && nonfinite,
          "a zero pivot: Refactor ZERO_PIVOT, the boost report USED, Solve divides by it", st);
    CK(cusolverRfSetNumericProperties(h, 1e-12, 1e-3));
    CK(cusolverRfResetValues(n, (int)v.size(), dArp, dAci, dAv, dP, dQ, h));
    const int st2 = cusolverRfRefactor(h);
    CK(cusolverRfGetNumericBoostReport(h, &rep));
    const auto mx = download(Mx, nnzM);
    check(st2 == 0 && rep == CUSOLVERRF_NUMERIC_BOOST_USED && mx[0] == 1e-3,
          "with zero 1e-12 and boost 1e-3: SUCCESS, the pivot becomes the boost", mx[0]);
  }
  for (void* q : {(void*)dP, (void*)dQ, (void*)dX, (void*)dT, (void*)dArp, (void*)dAci, (void*)dAv, (void*)dX2, (void*)dT2})
    cudaFree(q);
  cusolverRfDestroy(h);
}

// The unit diagonal formats, set up from the device.
static void formats() {
  Problem p(0.0);
  int* dP = upload(P);
  int* dQ = upload(Q);
  V f(n);
  for (int i = 0; i < n; ++i) f[i] = 2.0 - i;
  for (auto dg : {CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L, CUSOLVERRF_UNIT_DIAGONAL_STORED_U, CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U}) {
    cusolverRfHandle_t h;
    CK(cusolverRfCreate(&h));
    CK(cusolverRfSetMatrixFormat(h, CUSOLVERRF_MATRIX_FORMAT_CSR, dg));
    // L without its diagonal for ASSUMED_L; U without its for ASSUMED_U (the patterns are what matter).
    std::vector<std::vector<int>> pl = p.pl, pu = p.pu;
    for (int i = 0; i < n; ++i) {
      if (dg == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L) pl[i][i] = 0;
      if (dg == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_U) pu[i][i] = 0;
    }
    const Csr cl = csr(p.L, &pl), cu = csr(p.U, &pu);
    int* d1 = upload(p.ca.rp);
    int* d2 = upload(p.ca.ci);
    double* d3 = upload(p.ca.v);
    int* d4 = upload(cl.rp);
    int* d5 = upload(cl.ci);
    double* d6 = upload(cl.v);
    int* d7 = upload(cu.rp);
    int* d8 = upload(cu.ci);
    double* d9 = upload(cu.v);
    CK(cusolverRfSetupDevice(n, (int)p.ca.v.size(), d1, d2, d3, (int)cl.v.size(), d4, d5, d6, (int)cu.v.size(), d7, d8, d9,
                             dP, dQ, h));
    CK(cusolverRfAnalyze(h));
    CK(cusolverRfRefactor(h));
    double* dX = upload(f);
    double* dT = upload(V(n));
    CK(cusolverRfSolve(h, dP, dQ, 1, dT, n, dX, n));
    const double res = residual(p.a, download(dX, n), f);
    int nl, nu;
    int *lp, *li, *up, *ui;
    double *lx, *ux;
    CK(cusolverRfExtractSplitFactorsHost(h, &nl, &lp, &li, &lx, &nu, &up, &ui, &ux));
    // ASSUMED_L: L without the unit diagonal. U forms: L D (pivots on L) and D^-1 U.
    double ferr = 0;
    for (int i = 0; i < n; ++i) {
      for (int k = lp[i]; k < lp[i + 1]; ++k) {
        const int j = li[k];
        const double want = dg == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L ? p.L[i][j] : p.L[i][j] * p.U[j][j];
        ferr = std::max(ferr, std::fabs(lx[k] - want));
      }
      for (int k = up[i]; k < up[i + 1]; ++k) {
        const int j = ui[k];
        const double want = dg == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L ? p.U[i][j] : p.U[i][j] / p.U[i][i];
        ferr = std::max(ferr, std::fabs(ux[k] - want));
      }
    }
    int strict = 0, diag_u = 0;
    for (int i = 0; i < n; ++i)
      for (int j = 0; j < n; ++j) {
        strict += i > j && p.pl[i][j];
        diag_u += i < j && p.pu[i][j];
      }
    const bool sizes = dg == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L ? nl == strict && nu == (int)p.cu.v.size()
                     : dg == CUSOLVERRF_UNIT_DIAGONAL_STORED_U ? nl == strict + n && nu == diag_u + n
                                                               : nl == strict + n && nu == diag_u;
    for (void* q : {(void*)lp, (void*)li, (void*)lx, (void*)up, (void*)ui, (void*)ux}) std::free(q);
    char what[160];
    std::snprintf(what, sizeof what, "SetupDevice, unit diagonal %s: A x = f; the split factors in that form",
                  dg == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L ? "ASSUMED_L" : dg == CUSOLVERRF_UNIT_DIAGONAL_STORED_U ? "STORED_U" : "ASSUMED_U");
    check(res < 1e-12 && ferr < 1e-12 && sizes, what, std::max(res, ferr));
    for (void* q : {(void*)d1, (void*)d2, (void*)d3, (void*)d4, (void*)d5, (void*)d6, (void*)d7, (void*)d8, (void*)d9,
                    (void*)dX, (void*)dT})
      cudaFree(q);
    cusolverRfDestroy(h);
  }
  cudaFree(dP);
  cudaFree(dQ);
}

static void batched() {
  if (!on_sim) {
    std::printf("ok   batched forms: not run on NVIDIA's, which crashes in cusolverRfBatchAnalyze on an RTX 3060\n");
    return;
  }
  const int batch = 3;
  std::vector<Problem> ps;
  for (int b = 0; b < batch; ++b) ps.emplace_back(0.1 * b);
  std::vector<double*> hv;
  for (auto& p : ps) hv.push_back(p.ca.v.data());
  cusolverRfHandle_t h;
  CK(cusolverRfCreate(&h));
  I pp = P, qq = Q;
  Problem& p0 = ps[0];
  CK(cusolverRfBatchSetupHost(batch, n, (int)p0.ca.v.size(), p0.ca.rp.data(), p0.ca.ci.data(), hv.data(),
                              (int)p0.cl.v.size(), p0.cl.rp.data(), p0.cl.ci.data(), p0.cl.v.data(), (int)p0.cu.v.size(),
                              p0.cu.rp.data(), p0.cu.ci.data(), p0.cu.v.data(), pp.data(), qq.data(), h));
  CK(cusolverRfBatchAnalyze(h));
  CK(cusolverRfBatchRefactor(h));
  int pos[batch] = {7, 7, 7};
  const int zp = cusolverRfBatchZeroPivot(h, pos);
  V f(n);
  for (int i = 0; i < n; ++i) f[i] = 1.0 + 0.5 * i;
  int* dP = upload(P);
  int* dQ = upload(Q);
  std::vector<double*> xs;
  for (int b = 0; b < batch; ++b) xs.push_back(upload(f));
  double* dT = upload(V(n));
  CK(cusolverRfBatchSolve(h, dP, dQ, 1, dT, n, xs.data(), n));
  double res = 0;
  for (int b = 0; b < batch; ++b) res = std::max(res, residual(ps[b].a, download(xs[b], n), f));
  check(zp == 0 && pos[0] == -1 && pos[1] == -1 && pos[2] == -1 && res < 1e-12,
        "batched: three systems refactored and solved, no zero pivot", res);
  // New values, the second singular at B(0, 0).
  std::vector<double*> dv;
  for (int b = 0; b < batch; ++b) {
    Problem q(0.2 + 0.1 * b);
    V v = q.ca.v;
    if (b == 1)
      for (int k = q.ca.rp[P[0]]; k < q.ca.rp[P[0] + 1]; ++k)
        if (q.ca.ci[k] == Q[0]) v[k] = 0;
    dv.push_back(upload(v));
  }
  int* da = upload(p0.ca.rp);
  int* dc = upload(p0.ca.ci);
  CK(cusolverRfBatchResetValues(batch, n, (int)p0.ca.v.size(), da, dc, dv.data(), dP, dQ, h));
  const int rst = cusolverRfBatchRefactor(h);
  const int zp2 = cusolverRfBatchZeroPivot(h, pos);
  check(rst == 0 && zp2 == CUSOLVER_STATUS_ZERO_PIVOT && pos[0] == -1 && pos[1] == 0 && pos[2] == -1,
        "batched: a singular member reported by BatchZeroPivot (row 0), not by BatchRefactor", rst);
  for (double* x : xs) cudaFree(x);
  for (double* x : dv) cudaFree(x);
  for (void* q : {(void*)dP, (void*)dQ, (void*)dT, (void*)da, (void*)dc}) cudaFree(q);
  cusolverRfDestroy(h);
}

int main() {
  basic();
  formats();
  batched();
  std::printf(failures ? "FAIL: %d refactorization checks\n" : "PASS: every refactorization check\n", failures);
  return failures ? 1 : 0;
}
