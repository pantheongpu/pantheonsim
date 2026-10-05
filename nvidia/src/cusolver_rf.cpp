// cusolverRf, the refactorization module, built into libcusolver.so.12 beside
// cusolver_api.cpp and cusolver_sp.cpp.
//
// A system A x = f whose LU factors were found once (P A Q^T = L U, by any
// means) is refactored for new values of A with the same pattern: the
// pattern of M = L + U - I is kept, B = P A Q^T is scattered into it (fill
// zero), and LU without pivoting runs in place on M, dropping whatever falls
// outside the pattern. Solve then permutes, substitutes and permutes back.
// Everything is computed on the host in double; Mx is mirrored to the device
// for AccessBundledFactorsDevice.
//
// The conventions, as an RTX 3060's CUDA 13.0 showed them (and its
// documentation states): B(i, j) = A(P[i], Q[j]); x = Q^T U^-1 L^-1 P f.
// Setup and ResetValues load B's values (not L's and U's) into M, fill zero;
// Refactor factors M in place, every call -- a second Refactor without new
// values factors the factors (as NVIDIA's does), and a Solve before the
// first Refactor uses B itself. M is always unit-lower L plus U; the unit
// diagonal format (STORED_L, ASSUMED_L, STORED_U, ASSUMED_U) says which
// input diagonal to drop and how ExtractSplitFactorsHost hands the factors
// back (L D and D^-1 U for the U forms). A pivot with |pivot| <= the
// numeric "zero" (0 by default) is passed over -- the entries below it are
// neither divided nor used -- and at the end replaced by the boost, when one
// is set; the boost report says USED whenever such a pivot occurred, and
// with no boost Refactor returns ZERO_PIVOT (Solve then divides by the zero).
// Solve takes one right-hand side (nrhs > 1, ldt < n or ldxf < n:
// INVALID_VALUE); before any setup, Analyze, Refactor and Solve return
// SUCCESS and do nothing, and so does a Refactor before Analyze. A fresh
// handle holds the documented defaults (CSR, STORED_L, ALG0/ALG1, zero =
// boost = 0, fast mode off), as NVIDIA's does.
//
// The batched forms follow the documentation and the same arithmetic: on
// that card NVIDIA's own crashed in cusolverRfBatchAnalyze/BatchRefactor (a
// cudaFree of an invalid pointer) on every input tried, CUDA 13.0's and
// 13.2's libraries alike, so they could not be measured.
#include <cusolverRf.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#include <cuda_runtime.h>

namespace {

std::mutex& g_mu = *new std::mutex;
std::set<const void*>& g_live = *new std::set<const void*>;

struct Rf {
  int format = CUSOLVERRF_MATRIX_FORMAT_CSR;
  int diag = CUSOLVERRF_UNIT_DIAGONAL_STORED_L;
  double zero = 0, boost = 0;
  int fact_alg = CUSOLVERRF_FACTORIZATION_ALG0, solve_alg = CUSOLVERRF_TRIANGULAR_SOLVE_ALG1;
  int fast = CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF;
  int boost_used = CUSOLVERRF_NUMERIC_BOOST_NOT_USED;

  bool setup = false, batched = false;
  bool analyzed = false;                    // Refactor does nothing before Analyze
  int n = 0;
  bool unit_l = true;                       // the format's unit side: L (STORED_L, ASSUMED_L) or U
  std::vector<int> mp, mi, diag_pos;        // M's pattern, CSR, and each row's diagonal
  std::vector<int> a_to_m;                  // A's k-th value goes to M position a_to_m[k] (-1: outside)
  int nnz_a = 0;
  std::vector<std::vector<double>> avals;   // per matrix, the values of A last loaded
  std::vector<std::vector<double>> mx;      // per matrix, M
  std::vector<int> zero_pos;                // per matrix: -1, or the zero pivot's row
  // AccessBundledFactorsDevice's device copy.
  int* d_mp = nullptr;
  int* d_mi = nullptr;
  double* d_mx = nullptr;
  ~Rf() {
    cudaFree(d_mp);
    cudaFree(d_mi);
    cudaFree(d_mx);
  }
};

Rf* rf(cusolverRfHandle_t h) {
  std::lock_guard<std::mutex> l(g_mu);
  return h && g_live.count(h) ? reinterpret_cast<Rf*>(h) : nullptr;
}

template <class T> bool fetch(const T* p, size_t n, bool device, std::vector<T>* out) {
  out->resize(n);
  if (!n) return true;
  if (!p) return false;
  if (device) return cudaMemcpy(out->data(), p, n * sizeof(T), cudaMemcpyDeviceToHost) == cudaSuccess;
  std::memcpy(out->data(), p, n * sizeof(T));
  return true;
}

bool valid_csr(int n, int nnz, const std::vector<int>& rp, const std::vector<int>& ci) {
  if ((int)rp.size() != n + 1 || rp[0] != 0 || rp[(size_t)n] != nnz) return false;
  for (int i = 0; i < n; ++i)
    if (rp[(size_t)i + 1] < rp[(size_t)i]) return false;
  for (int c : ci)
    if (c < 0 || c >= n) return false;
  return true;
}
bool valid_perm(int n, const std::vector<int>& p) {
  std::vector<char> seen((size_t)n, 0);
  for (int v : p) {
    if (v < 0 || v >= n || seen[(size_t)v]) return false;
    seen[(size_t)v] = 1;
  }
  return true;
}

// Builds M's pattern from L and U (their unit diagonal, stored or assumed,
// dropped) and A's map into it.
cusolverStatus_t build(Rf* r, int n, int nnzA, const std::vector<int>& arp, const std::vector<int>& aci, int nnzL,
                       const std::vector<int>& lrp, const std::vector<int>& lci, int nnzU, const std::vector<int>& urp,
                       const std::vector<int>& uci, const std::vector<int>& P, const std::vector<int>& Q) {
  if (!valid_csr(n, nnzA, arp, aci) || !valid_csr(n, nnzL, lrp, lci) || !valid_csr(n, nnzU, urp, uci) ||
      !valid_perm(n, P) || !valid_perm(n, Q))
    return CUSOLVER_STATUS_INVALID_VALUE;
  r->unit_l = r->diag == CUSOLVERRF_UNIT_DIAGONAL_STORED_L || r->diag == CUSOLVERRF_UNIT_DIAGONAL_ASSUMED_L;
  std::vector<std::vector<int>> rows((size_t)n);
  for (int i = 0; i < n; ++i) {
    for (int k = lrp[(size_t)i]; k < lrp[(size_t)i + 1]; ++k) {
      const int j = lci[(size_t)k];
      if (j >= i) continue;  // L's diagonal is never in M: unit, or the U forms' pivots, which M holds as U's
      rows[(size_t)i].push_back(j);
    }
    for (int k = urp[(size_t)i]; k < urp[(size_t)i + 1]; ++k) {
      const int j = uci[(size_t)k];
      if (j < i) continue;
      rows[(size_t)i].push_back(j);
    }
    rows[(size_t)i].push_back(i);  // the pivot is always in M
    std::sort(rows[(size_t)i].begin(), rows[(size_t)i].end());
    rows[(size_t)i].erase(std::unique(rows[(size_t)i].begin(), rows[(size_t)i].end()), rows[(size_t)i].end());
  }
  r->n = n;
  r->mp.assign(1, 0);
  r->mi.clear();
  r->diag_pos.assign((size_t)n, -1);
  for (int i = 0; i < n; ++i) {
    for (int j : rows[(size_t)i]) {
      if (j == i) r->diag_pos[(size_t)i] = (int)r->mi.size();
      r->mi.push_back(j);
    }
    r->mp.push_back((int)r->mi.size());
  }
  // A(row, col) is B(Pinv[row], Qinv[col]).
  std::vector<int> pinv((size_t)n), qinv((size_t)n);
  for (int i = 0; i < n; ++i) {
    pinv[(size_t)P[(size_t)i]] = i;
    qinv[(size_t)Q[(size_t)i]] = i;
  }
  r->nnz_a = nnzA;
  r->a_to_m.assign((size_t)nnzA, -1);
  for (int row = 0; row < n; ++row)
    for (int k = arp[(size_t)row]; k < arp[(size_t)row + 1]; ++k) {
      const int i = pinv[(size_t)row], j = qinv[(size_t)aci[(size_t)k]];
      const auto b = r->mi.begin() + r->mp[(size_t)i], e = r->mi.begin() + r->mp[(size_t)i + 1];
      const auto it = std::lower_bound(b, e, j);
      if (it != e && *it == j) r->a_to_m[(size_t)k] = (int)(it - r->mi.begin());
    }
  r->setup = true;
  r->analyzed = false;
  return CUSOLVER_STATUS_SUCCESS;
}

// M := B's values, fill zero.
void scatter(Rf* r, int b) {
  auto& m = r->mx[(size_t)b];
  m.assign(r->mi.size(), 0.0);
  const auto& a = r->avals[(size_t)b];
  for (int k = 0; k < r->nnz_a; ++k)
    if (r->a_to_m[(size_t)k] >= 0) m[(size_t)r->a_to_m[(size_t)k]] += a[(size_t)k];
}

// The device copy AccessBundledFactorsDevice handed out follows M, as
// NVIDIA's pointers do (they are its working buffers).
void sync_device(Rf* r) {
  if (!r->d_mx || r->mx.empty()) return;
  cudaMemcpy(r->d_mx, r->mx[0].data(), r->mi.size() * sizeof(double), cudaMemcpyHostToDevice);
}

// In-place LU of M on its pattern, unit L, rows in order (IKJ). A pivot
// with |pivot| <= zero is flagged: later rows pass over its column, and at
// the end it becomes the boost when one is set. Returns the first flagged
// row when there is no boost (a zero pivot), else -1.
int factor(Rf* r, int b) {
  auto& m = r->mx[(size_t)b];
  const int n = r->n;
  std::vector<int> where((size_t)n, -1);
  std::vector<char> flagged((size_t)n, 0);
  int first = -1;
  for (int i = 0; i < n; ++i) {
    for (int k = r->mp[(size_t)i]; k < r->mp[(size_t)i + 1]; ++k) where[(size_t)r->mi[(size_t)k]] = k;
    for (int k = r->mp[(size_t)i]; k < r->mp[(size_t)i + 1]; ++k) {
      const int c = r->mi[(size_t)k];
      if (c >= i) break;
      if (flagged[(size_t)c]) continue;
      m[(size_t)k] /= m[(size_t)r->diag_pos[(size_t)c]];
      const double l = m[(size_t)k];
      for (int kk = r->diag_pos[(size_t)c] + 1; kk < r->mp[(size_t)c + 1]; ++kk) {
        const int pos = where[(size_t)r->mi[(size_t)kk]];
        if (pos >= 0) m[(size_t)pos] -= l * m[(size_t)kk];
      }
    }
    if (!(std::fabs(m[(size_t)r->diag_pos[(size_t)i]]) > r->zero)) {
      flagged[(size_t)i] = 1;
      if (first < 0) first = i;
    }
    for (int k = r->mp[(size_t)i]; k < r->mp[(size_t)i + 1]; ++k) where[(size_t)r->mi[(size_t)k]] = -1;
  }
  r->boost_used = first >= 0 ? CUSOLVERRF_NUMERIC_BOOST_USED : CUSOLVERRF_NUMERIC_BOOST_NOT_USED;
  if (r->boost != 0)
    for (int i = 0; i < n; ++i)
      if (flagged[(size_t)i]) m[(size_t)r->diag_pos[(size_t)i]] = r->boost;
  return r->boost != 0 ? -1 : first;
}

// One matrix's refactorization, in place.
void refactor_one(Rf* r, int b) {
  r->zero_pos[(size_t)b] = factor(r, b);
  if (b == 0) sync_device(r);
}

// x = Q^T U^-1 L^-1 P f, in place on one right-hand side.
void solve_one(const Rf* r, int b, const std::vector<int>& P, const std::vector<int>& Q, std::vector<double>& x,
               std::vector<double>* temp) {
  const int n = r->n;
  const auto& m = r->mx[(size_t)b];
  std::vector<double> y((size_t)n);
  for (int i = 0; i < n; ++i) y[(size_t)i] = x[(size_t)P[(size_t)i]];
  for (int i = 0; i < n; ++i)  // L, unit
    for (int k = r->mp[(size_t)i]; k < r->diag_pos[(size_t)i]; ++k) y[(size_t)i] -= m[(size_t)k] * y[(size_t)r->mi[(size_t)k]];
  if (temp) *temp = y;
  for (int i = n - 1; i >= 0; --i) {  // U
    double s = y[(size_t)i];
    for (int k = r->diag_pos[(size_t)i] + 1; k < r->mp[(size_t)i + 1]; ++k) s -= m[(size_t)k] * y[(size_t)r->mi[(size_t)k]];
    y[(size_t)i] = s / m[(size_t)r->diag_pos[(size_t)i]];
  }
  for (int j = 0; j < n; ++j) x[(size_t)Q[(size_t)j]] = y[(size_t)j];
}

cusolverStatus_t do_setup(cusolverRfHandle_t h, bool device, int batch, int n, int nnzA, const int* arp, const int* aci,
                          const double* const* avals, int nnzL, const int* lrp, const int* lci, int nnzU,
                          const int* urp, const int* uci, const int* P, const int* Q) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (n < 0 || nnzA < 0 || nnzL < 0 || nnzU < 0 || batch < 1) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<int> ar, ac, lr, lc, ur, uc, p, q;
  if (!fetch(arp, (size_t)n + 1, device, &ar) || !fetch(aci, (size_t)nnzA, device, &ac) ||
      !fetch(lrp, (size_t)n + 1, device, &lr) || !fetch(lci, (size_t)nnzL, device, &lc) ||
      !fetch(urp, (size_t)n + 1, device, &ur) || !fetch(uci, (size_t)nnzU, device, &uc) ||
      !fetch(P, (size_t)n, device, &p) || !fetch(Q, (size_t)n, device, &q))
    return CUSOLVER_STATUS_INVALID_VALUE;
  if (const cusolverStatus_t st = build(r, n, nnzA, ar, ac, nnzL, lr, lc, nnzU, ur, uc, p, q);
      st != CUSOLVER_STATUS_SUCCESS)
    return st;
  r->batched = false;
  r->avals.assign((size_t)batch, {});
  for (int b = 0; b < batch; ++b)
    if (!fetch(avals[b], (size_t)nnzA, device, &r->avals[(size_t)b])) return CUSOLVER_STATUS_INVALID_VALUE;
  r->mx.assign((size_t)batch, {});
  r->zero_pos.assign((size_t)batch, -1);
  for (int b = 0; b < batch; ++b) scatter(r, b);  // B itself until the first Refactor
  if (r->d_mx) {  // a new pattern: the device copy is made afresh on the next access
    cudaFree(r->d_mp);
    cudaFree(r->d_mi);
    cudaFree(r->d_mx);
    r->d_mp = r->d_mi = nullptr;
    r->d_mx = nullptr;
  }
  r->boost_used = CUSOLVERRF_NUMERIC_BOOST_NOT_USED;
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t do_reset(cusolverRfHandle_t h, int batch, int n, int nnzA, const double* const* dvals) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup || n != r->n || nnzA != r->nnz_a || batch != (int)r->avals.size() || !dvals)
    return CUSOLVER_STATUS_INVALID_VALUE;
  for (int b = 0; b < batch; ++b) {
    if (!fetch(dvals[b], (size_t)nnzA, true, &r->avals[(size_t)b])) return CUSOLVER_STATUS_INVALID_VALUE;
    scatter(r, b);
  }
  sync_device(r);
  return CUSOLVER_STATUS_SUCCESS;
}

cusolverStatus_t do_solve(cusolverRfHandle_t h, const int* dP, const int* dQ, int nrhs, double* temp, int ldt,
                          double* const* xf, int ldxf, int batch) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup) return CUSOLVER_STATUS_SUCCESS;
  const int n = r->n;
  if (nrhs != 1 || ldt < n || ldxf < n || !xf) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<int> p, q;
  if (!fetch(dP, (size_t)n, true, &p) || !fetch(dQ, (size_t)n, true, &q) || !valid_perm(n, p) || !valid_perm(n, q))
    return CUSOLVER_STATUS_INVALID_VALUE;
  for (int b = 0; b < batch; ++b) {
    std::vector<double> x;
    if (!fetch<double>(xf[b], (size_t)n, true, &x)) return CUSOLVER_STATUS_EXECUTION_FAILED;
    std::vector<double> t;
    solve_one(r, b, p, q, x, &t);
    if (cudaMemcpy(xf[b], x.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess)
      return CUSOLVER_STATUS_EXECUTION_FAILED;
    if (temp && b == 0 && n)
      cudaMemcpy(temp, t.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice);
  }
  return CUSOLVER_STATUS_SUCCESS;
}

// Splits M by the format: the unit diagonal on L (stored or not), or, for the
// U forms, L D and D^-1 U with D the pivots (U's unit diagonal stored or not).
void split(const Rf* r, int b, std::vector<int>* lp, std::vector<int>* li, std::vector<double>* lx, std::vector<int>* up,
           std::vector<int>* ui, std::vector<double>* ux) {
  const bool store_unit = r->diag == CUSOLVERRF_UNIT_DIAGONAL_STORED_L || r->diag == CUSOLVERRF_UNIT_DIAGONAL_STORED_U;
  const auto& m = r->mx[(size_t)b];
  lp->assign(1, 0);
  up->assign(1, 0);
  li->clear(), lx->clear(), ui->clear(), ux->clear();
  for (int i = 0; i < r->n; ++i) {
    const double piv = m[(size_t)r->diag_pos[(size_t)i]];
    for (int k = r->mp[(size_t)i]; k < r->mp[(size_t)i + 1]; ++k) {
      const int j = r->mi[(size_t)k];
      if (j < i) {
        li->push_back(j);
        lx->push_back(r->unit_l ? m[(size_t)k] : m[(size_t)k] * m[(size_t)r->diag_pos[(size_t)j]]);
      } else if (j == i) {
        if (r->unit_l) {
          if (store_unit) li->push_back(i), lx->push_back(1.0);
          ui->push_back(i), ux->push_back(piv);
        } else {
          li->push_back(i), lx->push_back(piv);
          if (store_unit) ui->push_back(i), ux->push_back(1.0);
        }
      } else {
        ui->push_back(j);
        ux->push_back(r->unit_l ? m[(size_t)k] : m[(size_t)k] / piv);
      }
    }
    lp->push_back((int)li->size());
    up->push_back((int)ui->size());
  }
}

template <class T> T* host_copy(const std::vector<T>& v) {
  T* p = static_cast<T*>(std::malloc(std::max<size_t>(1, v.size()) * sizeof(T)));
  if (p && !v.empty()) std::memcpy(p, v.data(), v.size() * sizeof(T));
  return p;
}

}  // namespace

#define VGPU_EXPORT extern "C" __attribute__((visibility("default")))

VGPU_EXPORT cusolverStatus_t cusolverRfCreate(cusolverRfHandle_t* h) {
  if (!h) return CUSOLVER_STATUS_INVALID_VALUE;
  Rf* r = new Rf();
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.insert(r);
  }
  *h = reinterpret_cast<cusolverRfHandle_t>(r);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfDestroy(cusolverRfHandle_t h) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  {
    std::lock_guard<std::mutex> l(g_mu);
    g_live.erase(r);
  }
  delete r;
  return CUSOLVER_STATUS_SUCCESS;
}

VGPU_EXPORT cusolverStatus_t cusolverRfGetMatrixFormat(cusolverRfHandle_t h, cusolverRfMatrixFormat_t* f,
                                                       cusolverRfUnitDiagonal_t* d) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (f) *f = (cusolverRfMatrixFormat_t)r->format;
  if (d) *d = (cusolverRfUnitDiagonal_t)r->diag;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfSetMatrixFormat(cusolverRfHandle_t h, cusolverRfMatrixFormat_t f,
                                                       cusolverRfUnitDiagonal_t d) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if ((f != CUSOLVERRF_MATRIX_FORMAT_CSR && f != CUSOLVERRF_MATRIX_FORMAT_CSC) || (int)d < 0 || (int)d > 3)
    return CUSOLVER_STATUS_INVALID_VALUE;
  r->format = f;
  r->diag = d;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfSetNumericProperties(cusolverRfHandle_t h, double zero, double boost) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  r->zero = zero;
  r->boost = boost;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfGetNumericProperties(cusolverRfHandle_t h, double* zero, double* boost) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (zero) *zero = r->zero;
  if (boost) *boost = r->boost;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfGetNumericBoostReport(cusolverRfHandle_t h, cusolverRfNumericBoostReport_t* rep) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (rep) *rep = (cusolverRfNumericBoostReport_t)r->boost_used;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfSetAlgs(cusolverRfHandle_t h, cusolverRfFactorization_t fa,
                                               cusolverRfTriangularSolve_t sa) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if ((int)fa < 0 || (int)fa > 2 || (int)sa < 1 || (int)sa > 3) return CUSOLVER_STATUS_INVALID_VALUE;
  r->fact_alg = fa;
  r->solve_alg = sa;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfGetAlgs(cusolverRfHandle_t h, cusolverRfFactorization_t* fa,
                                               cusolverRfTriangularSolve_t* sa) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (fa) *fa = (cusolverRfFactorization_t)r->fact_alg;
  if (sa) *sa = (cusolverRfTriangularSolve_t)r->solve_alg;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfGetResetValuesFastMode(cusolverRfHandle_t h, cusolverRfResetValuesFastMode_t* m) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m) *m = (cusolverRfResetValuesFastMode_t)r->fast;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfSetResetValuesFastMode(cusolverRfHandle_t h, cusolverRfResetValuesFastMode_t m) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (m != CUSOLVERRF_RESET_VALUES_FAST_MODE_OFF && m != CUSOLVERRF_RESET_VALUES_FAST_MODE_ON)
    return CUSOLVER_STATUS_INVALID_VALUE;
  r->fast = m;
  return CUSOLVER_STATUS_SUCCESS;
}

VGPU_EXPORT cusolverStatus_t cusolverRfSetupHost(int n, int nnzA, int* arp, int* aci, double* av, int nnzL, int* lrp,
                                                 int* lci, double*, int nnzU, int* urp, int* uci, double*, int* P,
                                                 int* Q, cusolverRfHandle_t h) {
  const double* vals[1] = {av};
  return do_setup(h, false, 1, n, nnzA, arp, aci, vals, nnzL, lrp, lci, nnzU, urp, uci, P, Q);
}
VGPU_EXPORT cusolverStatus_t cusolverRfSetupDevice(int n, int nnzA, int* arp, int* aci, double* av, int nnzL, int* lrp,
                                                   int* lci, double*, int nnzU, int* urp, int* uci, double*, int* P,
                                                   int* Q, cusolverRfHandle_t h) {
  const double* vals[1] = {av};
  return do_setup(h, true, 1, n, nnzA, arp, aci, vals, nnzL, lrp, lci, nnzU, urp, uci, P, Q);
}
VGPU_EXPORT cusolverStatus_t cusolverRfResetValues(int n, int nnzA, int*, int*, double* av, int*, int*,
                                                   cusolverRfHandle_t h) {
  const double* vals[1] = {av};
  Rf* r = rf(h);
  if (r && r->batched) return CUSOLVER_STATUS_INVALID_VALUE;
  return do_reset(h, 1, n, nnzA, vals);
}
static cusolverStatus_t analyze(cusolverRfHandle_t h) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (r->setup) r->analyzed = true;
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfAnalyze(cusolverRfHandle_t h) { return analyze(h); }
VGPU_EXPORT cusolverStatus_t cusolverRfRefactor(cusolverRfHandle_t h) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup || !r->analyzed) return CUSOLVER_STATUS_SUCCESS;
  refactor_one(r, 0);
  return r->zero_pos[0] >= 0 ? CUSOLVER_STATUS_ZERO_PIVOT : CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfSolve(cusolverRfHandle_t h, int* P, int* Q, int nrhs, double* temp, int ldt,
                                             double* xf, int ldxf) {
  double* const x[1] = {xf};
  return do_solve(h, P, Q, nrhs, temp, ldt, x, ldxf, 1);
}

VGPU_EXPORT cusolverStatus_t cusolverRfAccessBundledFactorsDevice(cusolverRfHandle_t h, int* nnzM, int** Mp, int** Mi,
                                                                  double** Mx) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup || !nnzM || !Mp || !Mi || !Mx) return CUSOLVER_STATUS_INVALID_VALUE;
  const size_t nnz = r->mi.size();
  if (!r->d_mp) {
    cudaMalloc(&r->d_mp, ((size_t)r->n + 1) * sizeof(int));
    cudaMalloc(&r->d_mi, std::max<size_t>(1, nnz) * sizeof(int));
    cudaMalloc(&r->d_mx, std::max<size_t>(1, nnz) * sizeof(double));
  }
  if (cudaMemcpy(r->d_mp, r->mp.data(), r->mp.size() * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess ||
      (nnz && cudaMemcpy(r->d_mi, r->mi.data(), nnz * sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) ||
      (nnz && cudaMemcpy(r->d_mx, r->mx[0].data(), nnz * sizeof(double), cudaMemcpyHostToDevice) != cudaSuccess))
    return CUSOLVER_STATUS_EXECUTION_FAILED;
  *nnzM = (int)nnz;
  *Mp = r->d_mp;
  *Mi = r->d_mi;
  *Mx = r->d_mx;
  return CUSOLVER_STATUS_SUCCESS;
}
// Host copies the caller frees (malloc'd).
VGPU_EXPORT cusolverStatus_t cusolverRfExtractBundledFactorsHost(cusolverRfHandle_t h, int* nnzM, int** Mp, int** Mi,
                                                                 double** Mx) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup || !nnzM || !Mp || !Mi || !Mx) return CUSOLVER_STATUS_INVALID_VALUE;
  *nnzM = (int)r->mi.size();
  *Mp = host_copy(r->mp);
  *Mi = host_copy(r->mi);
  *Mx = host_copy(r->mx[0]);
  return *Mp && *Mi && *Mx ? CUSOLVER_STATUS_SUCCESS : CUSOLVER_STATUS_ALLOC_FAILED;
}
VGPU_EXPORT cusolverStatus_t cusolverRfExtractSplitFactorsHost(cusolverRfHandle_t h, int* nnzL, int** Lp, int** Li,
                                                               double** Lx, int* nnzU, int** Up, int** Ui,
                                                               double** Ux) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup || !nnzL || !Lp || !Li || !Lx || !nnzU || !Up || !Ui || !Ux) return CUSOLVER_STATUS_INVALID_VALUE;
  std::vector<int> lp, li, up, ui;
  std::vector<double> lx, ux;
  split(r, 0, &lp, &li, &lx, &up, &ui, &ux);
  *nnzL = (int)li.size();
  *nnzU = (int)ui.size();
  *Lp = host_copy(lp);
  *Li = host_copy(li);
  *Lx = host_copy(lx);
  *Up = host_copy(up);
  *Ui = host_copy(ui);
  *Ux = host_copy(ux);
  return CUSOLVER_STATUS_SUCCESS;
}

/* ---- batched ---- */

VGPU_EXPORT cusolverStatus_t cusolverRfBatchSetupHost(int batch, int n, int nnzA, int* arp, int* aci, double* av[],
                                                      int nnzL, int* lrp, int* lci, double*, int nnzU, int* urp,
                                                      int* uci, double*, int* P, int* Q, cusolverRfHandle_t h) {
  if (batch < 1 || !av) return rf(h) ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
  std::vector<const double*> vals(av, av + batch);
  const cusolverStatus_t st = do_setup(h, false, batch, n, nnzA, arp, aci, vals.data(), nnzL, lrp, lci, nnzU, urp, uci, P, Q);
  if (st == CUSOLVER_STATUS_SUCCESS) rf(h)->batched = true;
  return st;
}
VGPU_EXPORT cusolverStatus_t cusolverRfBatchAnalyze(cusolverRfHandle_t h) { return analyze(h); }
VGPU_EXPORT cusolverStatus_t cusolverRfBatchResetValues(int batch, int n, int nnzA, int*, int*, double* av[], int*, int*,
                                                        cusolverRfHandle_t h) {
  if (!av || batch < 1) return rf(h) ? CUSOLVER_STATUS_INVALID_VALUE : CUSOLVER_STATUS_NOT_INITIALIZED;
  std::vector<const double*> vals(av, av + batch);
  return do_reset(h, batch, n, nnzA, vals.data());
}
// Failures are reported by BatchZeroPivot, not here, as NVIDIA documents.
VGPU_EXPORT cusolverStatus_t cusolverRfBatchRefactor(cusolverRfHandle_t h) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!r->setup || !r->analyzed) return CUSOLVER_STATUS_SUCCESS;
  for (size_t b = 0; b < r->mx.size(); ++b) refactor_one(r, (int)b);
  return CUSOLVER_STATUS_SUCCESS;
}
VGPU_EXPORT cusolverStatus_t cusolverRfBatchSolve(cusolverRfHandle_t h, int* P, int* Q, int nrhs, double* temp, int ldt,
                                                  double* xf[], int ldxf) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  return do_solve(h, P, Q, nrhs, temp, ldt, xf, ldxf, (int)r->mx.size());
}
// position[j]: -1, or the row of matrix j's zero pivot; ZERO_PIVOT when any has one.
VGPU_EXPORT cusolverStatus_t cusolverRfBatchZeroPivot(cusolverRfHandle_t h, int* position) {
  Rf* r = rf(h);
  if (!r) return CUSOLVER_STATUS_NOT_INITIALIZED;
  if (!position) return CUSOLVER_STATUS_INVALID_VALUE;
  bool any = false;
  for (size_t b = 0; b < r->zero_pos.size(); ++b) {
    position[b] = r->zero_pos[b];
    any = any || r->zero_pos[b] >= 0;
  }
  return any ? CUSOLVER_STATUS_ZERO_PIVOT : CUSOLVER_STATUS_SUCCESS;
}
