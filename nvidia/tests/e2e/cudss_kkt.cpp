// cuDSS as SCS's GPU direct backend calls it (linsys/cudss/direct/private.c):
// a quasi-definite KKT matrix [P + Rx, A^T; A, -Ry] passed as its lower
// triangle in CSR (CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, zero-based, no
// row-end array), the default configuration, ANALYSIS then FACTORIZATION,
// the inertia read back -- SCS gives up unless n of it is positive -- then a
// SOLVE per iteration, and, when the diagonal changes, the same arrays handed
// back through cudssMatrixSetCsrPointers and a REFACTORIZATION. In every
// index and value width SCS builds with: 32-bit indices and doubles by
// default, 64-bit under DLONG, floats under SFLOAT.
//
// Built against VirtualGPU's own declarations, which follow NVIDIA's ABI, so
// the same program also runs against NVIDIA's libcudss on a real GPU: that is
// how these expectations were checked.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

#include "../../include/vgpu_cudss.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

// The KKT matrix as SCS forms it, kept on the host in lower CSR alongside the
// positions of its diagonal (SCS's diag_r_idxs).
struct Kkt {
  int n, m;
  std::vector<int64_t> rp, ci;
  std::vector<double> val;
  std::vector<int64_t> diag;  // where each row's diagonal entry is
  std::vector<double> p_diag;
};

static Kkt make_kkt(int n, int m, std::mt19937_64& g) {
  std::uniform_real_distribution<double> u(-1, 1);
  std::uniform_int_distribution<int> pick(0, n - 1);
  // P: symmetric, sparse, diagonally dominant (so positive semidefinite).
  std::vector<std::vector<std::pair<int, double>>> lower(n + m);
  std::vector<double> rowsum(n, 0);
  for (int i = 0; i < n; ++i)
    for (int k = 0; k < 2; ++k) {
      const int j = pick(g);
      if (j >= i) continue;
      const double v = u(g);
      lower[i].push_back({j, v});
      rowsum[i] += std::fabs(v);
      rowsum[j] += std::fabs(v);
    }
  Kkt K{n, m, {0}, {}, {}, {}, std::vector<double>(n)};
  for (int i = 0; i < n; ++i) K.p_diag[i] = rowsum[i] + (i % 3 == 0 ? 0.0 : 0.5);  // some of P's rows only PSD
  // A: three entries a row, one of them in a column of its own so A has full
  // row rank whatever the rest.
  for (int r = 0; r < m; ++r) {
    std::vector<int> cols = {r % n, pick(g), pick(g)};
    std::sort(cols.begin(), cols.end());
    cols.erase(std::unique(cols.begin(), cols.end()), cols.end());
    for (int c : cols) lower[n + r].push_back({c, u(g) + (c == r % n ? 2.0 : 0.0)});
  }
  for (int i = 0; i < n + m; ++i) {
    auto& row = lower[i];
    std::sort(row.begin(), row.end());
    for (auto& [c, v] : row) K.ci.push_back(c), K.val.push_back(v);
    K.diag.push_back((int64_t)K.ci.size());
    K.ci.push_back(i);
    K.val.push_back(0);  // set by set_diagonal
    K.rp.push_back((int64_t)K.ci.size());
  }
  return K;
}

// SCS's scs_update_lin_sys_diag_r: P's diagonal plus rho_x on top, -rho_y below.
static void set_diagonal(Kkt& K, double rho_x, double rho_y_scale) {
  for (int i = 0; i < K.n; ++i) K.val[K.diag[i]] = K.p_diag[i] + rho_x;
  for (int r = 0; r < K.m; ++r) K.val[K.diag[K.n + r]] = -rho_y_scale * (1 + (r % 4));
}

// ||K x - b|| relative to ||K|| ||x|| + ||b||, with K the full symmetric matrix.
static double residual(const Kkt& K, const std::vector<double>& x, const std::vector<double>& b) {
  const int N = K.n + K.m;
  std::vector<double> kx(N, 0);
  double knorm = 0;
  for (int i = 0; i < N; ++i)
    for (int64_t k = K.rp[i]; k < K.rp[i + 1]; ++k) {
      const int j = (int)K.ci[k];
      kx[i] += K.val[k] * x[j];
      if (j != i) kx[j] += K.val[k] * x[i];
      knorm = std::max(knorm, std::fabs(K.val[k]));
    }
  double r = 0, xn = 0, bn = 0;
  for (int i = 0; i < N; ++i) {
    r = std::max(r, std::fabs(kx[i] - b[i]));
    xn = std::max(xn, std::fabs(x[i]));
    bn = std::max(bn, std::fabs(b[i]));
  }
  return r / (knorm * xn * N + bn);
}

template <class I, class V>
static void scs_like(const char* label, cudssDataType_t itype, cudssDataType_t vtype, int n, int m, double tol) {
  std::printf("-- %s: n = %d, m = %d\n", label, n, m);
  std::mt19937_64 g(1234 + n * 7 + m);
  Kkt K = make_kkt(n, m, g);
  set_diagonal(K, 1e-6, 0.1);
  const int N = n + m;
  const int64_t nnz = K.rp[N];
  for (auto& v : K.val) v = (double)(V)v;  // the matrix is what the device holds

  std::vector<I> rp(K.rp.begin(), K.rp.end()), ci(K.ci.begin(), K.ci.end());
  std::vector<V> val(K.val.begin(), K.val.end());
  I *d_rp, *d_ci;
  V *d_val, *d_b, *d_x;
  cudaMalloc((void**)&d_rp, rp.size() * sizeof(I));
  cudaMalloc((void**)&d_ci, ci.size() * sizeof(I));
  cudaMalloc((void**)&d_val, val.size() * sizeof(V));
  cudaMalloc((void**)&d_b, N * sizeof(V));
  cudaMalloc((void**)&d_x, N * sizeof(V));
  cudaMemcpy(d_rp, rp.data(), rp.size() * sizeof(I), cudaMemcpyHostToDevice);
  cudaMemcpy(d_ci, ci.data(), ci.size() * sizeof(I), cudaMemcpyHostToDevice);
  cudaMemcpy(d_val, val.data(), val.size() * sizeof(V), cudaMemcpyHostToDevice);

  cudssHandle_t h = nullptr;
  cudssConfig_t cfg = nullptr;
  cudssData_t data = nullptr;
  cudssMatrix_t A = nullptr, B = nullptr, X = nullptr;
  IS(cudssCreate(&h), CUDSS_STATUS_SUCCESS);
  IS(cudssConfigCreate(&cfg), CUDSS_STATUS_SUCCESS);
  IS(cudssDataCreate(h, &data), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixCreateCsr(&A, N, N, nnz, d_rp, nullptr, d_ci, d_val, itype, itype, vtype, CUDSS_MTYPE_SYMMETRIC,
                          CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO),
     CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixCreateDn(&B, N, 1, N, d_b, vtype, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixCreateDn(&X, N, 1, N, d_x, vtype, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(h, CUDSS_PHASE_ANALYSIS, cfg, data, A, X, B), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(h, CUDSS_PHASE_FACTORIZATION, cfg, data, A, X, B), CUDSS_STATUS_SUCCESS);

  // The inertia, as SCS reads it: two of the index type, positive first.
  auto inertia_is = [&](const char* when) {
    I inertia[2] = {-1, -1};
    size_t written = 0;
    IS(cudssDataGet(h, data, CUDSS_DATA_INERTIA, inertia, sizeof inertia, &written), CUDSS_STATUS_SUCCESS);
    char what[160];
    std::snprintf(what, sizeof what, "%s: inertia (%lld, %lld), %zu bytes written, is (n, m) = (%d, %d)", when,
                  (long long)inertia[0], (long long)inertia[1], written, n, m);
    check(inertia[0] == n && inertia[1] == m && written == sizeof inertia, what);
  };
  inertia_is("after factorization");
  size_t need = 0;
  IS(cudssDataGet(h, data, CUDSS_DATA_INERTIA, nullptr, 0, &need), CUDSS_STATUS_SUCCESS);
  check(need == 2 * sizeof(I), "a zero size asks for the inertia's size: two of the index type");
  I wrong[3];
  IS(cudssDataGet(h, data, CUDSS_DATA_INERTIA, wrong, sizeof wrong, &need), CUDSS_STATUS_INVALID_VALUE);
  I npivots = -1;
  IS(cudssDataGet(h, data, CUDSS_DATA_NPIVOTS, &npivots, sizeof npivots, nullptr), CUDSS_STATUS_SUCCESS);
  check(npivots == 0, "no pivot needed the epsilon");
  int info = -1;
  IS(cudssDataGet(h, data, CUDSS_DATA_INFO, &info, sizeof info, nullptr), CUDSS_STATUS_SUCCESS);
  check(info == 0, "INFO is 0");

  auto solve_and_check = [&](int iteration) {
    std::vector<double> b(N);
    for (int i = 0; i < N; ++i) b[i] = std::sin(1.0 + i * (iteration + 1)) * (1 + i % 5);
    std::vector<V> bv(b.begin(), b.end()), xv(N, V(-77));
    cudaMemcpy(d_b, bv.data(), N * sizeof(V), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, xv.data(), N * sizeof(V), cudaMemcpyHostToDevice);
    IS(cudssExecute(h, CUDSS_PHASE_SOLVE, cfg, data, A, X, B), CUDSS_STATUS_SUCCESS);
    cudaMemcpy(xv.data(), d_x, N * sizeof(V), cudaMemcpyDeviceToHost);
    std::vector<double> x(xv.begin(), xv.end());
    std::vector<double> kb(bv.begin(), bv.end());
    const double r = residual(K, x, kb);
    char what[120];
    std::snprintf(what, sizeof what, "solve %d: relative residual %.2e < %.0e", iteration, r, tol);
    check(r < tol, what);
  };
  for (int it = 0; it < 3; ++it) solve_and_check(it);

  // The diagonal changes: new values in the same arrays, the pointers handed
  // back, and a refactorization on the analysed pattern.
  set_diagonal(K, 1e-3, 2.5);
  for (int64_t k = 0; k < nnz; ++k) val[k] = (V)K.val[k];
  for (int64_t k = 0; k < nnz; ++k) K.val[k] = (double)val[k];  // what the device holds
  cudaMemcpy(d_val, val.data(), val.size() * sizeof(V), cudaMemcpyHostToDevice);
  IS(cudssMatrixSetCsrPointers(A, d_rp, nullptr, d_ci, d_val), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(h, CUDSS_PHASE_REFACTORIZATION, cfg, data, A, X, B), CUDSS_STATUS_SUCCESS);
  inertia_is("after refactorization");
  for (int it = 3; it < 5; ++it) solve_and_check(it);

  // SCS's teardown order: solver data and config, matrices, then the handle.
  IS(cudssDataDestroy(h, data), CUDSS_STATUS_SUCCESS);
  IS(cudssConfigDestroy(cfg), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixDestroy(A), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixDestroy(B), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixDestroy(X), CUDSS_STATUS_SUCCESS);
  IS(cudssDestroy(h), CUDSS_STATUS_SUCCESS);
  cudaFree(d_rp);
  cudaFree(d_ci);
  cudaFree(d_val);
  cudaFree(d_b);
  cudaFree(d_x);
}

int main() {
  // SCS checks for cuDSS 0.8's API by version.
  check(CUDSS_VERSION >= 800, "CUDSS_VERSION selects the 0.8 signatures");
  scs_like<int32_t, double>("32-bit indices, double (SCS's default build)", CUDSS_R_32I, CUDSS_R_64F, 30, 45, 1e-13);
  scs_like<int64_t, double>("64-bit indices, double (DLONG)", CUDSS_R_64I, CUDSS_R_64F, 30, 45, 1e-13);
  scs_like<int32_t, float>("32-bit indices, float (SFLOAT)", CUDSS_R_32I, CUDSS_R_32F, 30, 45, 1e-5);
  scs_like<int64_t, float>("64-bit indices, float (DLONG SFLOAT)", CUDSS_R_64I, CUDSS_R_32F, 30, 45, 1e-5);
  scs_like<int32_t, double>("a larger one", CUDSS_R_32I, CUDSS_R_64F, 160, 220, 1e-13);
  std::printf(failures ? "FAIL: %d checks failed\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
