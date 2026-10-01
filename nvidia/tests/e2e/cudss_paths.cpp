// The rest of cuDSS: every matrix type (LU, LDL^T, LDL^H, Cholesky), views,
// one-based and 64-bit indices, real and complex in both precisions, several
// right-hand sides with a leading dimension, CSR arrays in host memory, a
// user permutation, the solve sub-phases chained by hand, iterative
// refinement, the pivot epsilon, a matrix that is not positive definite,
// uniform and non-uniform batches, a solve captured into a CUDA graph, and
// the argument checks, defaults and Get/Set protocol.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's
// libcudss on an RTX 3060, and every status, default and size below is what
// that library answered. Values that depend on the reordering (the
// permutation, DIAG, LU_NNZ) are only checked where the ordering is the
// caller's or cannot matter; matrices that need pivoting across supernodes
// are left to the solver's unit test, since NVIDIA's diagonal pivoting
// perturbs them.
#include <cuda_runtime_api.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

#include "../../include/vgpu_cudss.h"

using cd = std::complex<double>;
using cf = std::complex<float>;

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

template <class T>
static T* upload(const std::vector<T>& v) {
  T* p = nullptr;
  cudaMalloc((void**)&p, v.size() * sizeof(T) + 16);
  cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice);
  return p;
}

template <class T>
static std::vector<T> download(const T* p, size_t n) {
  std::vector<T> v(n);
  cudaDeviceSynchronize();
  cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost);
  return v;
}

static double mag(double v) { return std::fabs(v); }
static double mag(float v) { return std::fabs(v); }
static double mag(cd v) { return std::abs(v); }
static double mag(cf v) { return std::abs(cd(v.real(), v.imag())); }
static cd wide(double v) { return v; }
static cd wide(float v) { return v; }
static cd wide(cd v) { return v; }
static cd wide(cf v) { return cd(v.real(), v.imag()); }

// A dense matrix (row-major, complex double) and its CSR, of one triangle or all.
struct Dense {
  int n;
  std::vector<cd> a;
  cd at(int i, int j) const { return a[i * n + j]; }
};

template <class I, class V>
struct Csr {
  std::vector<I> rp, ci;
  std::vector<V> v;
};

template <class I, class V>
static Csr<I, V> to_csr(const Dense& d, cudssMatrixViewType_t view, int base) {
  Csr<I, V> c;
  c.rp.push_back((I)base);
  for (int i = 0; i < d.n; ++i) {
    for (int j = 0; j < d.n; ++j) {
      if (d.at(i, j) == cd(0) && i != j) continue;
      if ((view == CUDSS_MVIEW_LOWER && j > i) || (view == CUDSS_MVIEW_UPPER && j < i)) continue;
      c.ci.push_back((I)(j + base));
      if constexpr (std::is_same_v<V, cd>) c.v.push_back(d.at(i, j));
      else if constexpr (std::is_same_v<V, cf>) c.v.push_back(cf((float)d.at(i, j).real(), (float)d.at(i, j).imag()));
      else c.v.push_back((V)d.at(i, j).real());
    }
    c.rp.push_back((I)(c.ci.size() + base));
  }
  return c;
}

template <class V>
static double residual(const Dense& d, const std::vector<V>& x, const std::vector<V>& b, int off_x = 0,
                       int off_b = 0) {
  double r = 0, an = 0, xn = 0, bn = 0;
  for (int i = 0; i < d.n; ++i) {
    cd s = -wide(b[off_b + i]);
    for (int j = 0; j < d.n; ++j) s += d.at(i, j) * wide(x[off_x + j]), an = std::max(an, std::abs(d.at(i, j)));
    r = std::max(r, std::abs(s));
    xn = std::max(xn, mag(x[off_x + i]));
    bn = std::max(bn, mag(b[off_b + i]));
  }
  return r / (an * xn * d.n + bn);
}

struct Session {
  cudssHandle_t h = nullptr;
  cudssConfig_t cfg = nullptr;
  cudssData_t data = nullptr;
  Session() {
    cudssCreate(&h);
    cudssConfigCreate(&cfg);
    cudssDataCreate(h, &data);
  }
  ~Session() {
    cudssDataDestroy(h, data);
    cudssConfigDestroy(cfg);
    cudssDestroy(h);
  }
};

// Solves d x = b for nrhs right-hand sides with a leading dimension, through
// matrix type mt, view, base and the index and value types I and V, and
// checks the residual of each. Returns the session for further questions.
template <class I, class V>
static void solve_case(const char* what, const Dense& d, cudssMatrixType_t mt, cudssMatrixViewType_t view,
                       int base, cudssDataType_t it, cudssDataType_t vt, int nrhs, double tol,
                       long long want_pos = -1, long long want_neg = -1, bool host_csr = false) {
  std::printf("-- %s\n", what);
  Session s;
  auto c = to_csr<I, V>(d, view, base);
  const int n = d.n, ld = n + 2;
  std::vector<V> b((size_t)ld * nrhs, V(-5)), x((size_t)ld * nrhs, V(-9));
  for (int j = 0; j < nrhs; ++j)
    for (int i = 0; i < n; ++i) b[j * ld + i] = V(std::cos(1.0 + i + 3 * j) * (1 + j));
  I* rp = host_csr ? c.rp.data() : upload(c.rp);
  I* ci = host_csr ? c.ci.data() : upload(c.ci);
  V* vv = host_csr ? c.v.data() : upload(c.v);
  V* db = upload(b);
  V* dx = upload(x);
  cudssMatrix_t A, B, X;
  IS(cudssMatrixCreateCsr(&A, n, n, (int64_t)c.ci.size(), rp, nullptr, ci, vv, it, it, vt, mt, view,
                          (cudssIndexBase_t)base),
     CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixCreateDn(&B, n, nrhs, ld, db, vt, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixCreateDn(&X, n, nrhs, ld, dx, vt, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B),
     CUDSS_STATUS_SUCCESS);
  x = download(dx, x.size());
  for (int j = 0; j < nrhs; ++j) {
    const double r = residual(d, x, b, j * ld, j * ld);
    char msg[120];
    std::snprintf(msg, sizeof msg, "right-hand side %d: relative residual %.2e < %.0e", j, r, tol);
    check(r < tol, msg);
    // The rows between n and the leading dimension are not the solution's.
    check(mag(x[j * ld + n]) == 9 && mag(x[j * ld + n + 1]) == 9, "the padding rows are left alone");
  }
  if (want_pos >= 0) {
    I in[2] = {-1, -1};
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INERTIA, in, sizeof in, nullptr), CUDSS_STATUS_SUCCESS);
    char msg[120];
    std::snprintf(msg, sizeof msg, "inertia (%lld, %lld) is (%lld, %lld)", (long long)in[0], (long long)in[1],
                  want_pos, want_neg);
    check(in[0] == want_pos && in[1] == want_neg, msg);
  }
  cudssMatrixDestroy(A);
  cudssMatrixDestroy(B);
  cudssMatrixDestroy(X);
  if (!host_csr) cudaFree(rp), cudaFree(ci), cudaFree(vv);
  cudaFree(db);
  cudaFree(dx);
}

// A sparse, diagonally dominant matrix with the given diagonal signs: its
// inertia is the signs', and any pivot order factors it without pivoting.
static Dense dominant(int n, const char* signs, int kind, bool complex) {  // kind 0 general, 1 symmetric, 2 Hermitian
  Dense d{n, std::vector<cd>((size_t)n * n)};
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < i; ++j) {
      if ((i * 7 + j * 3) % 4) continue;
      const double im = complex ? 0.2 * ((i + j) % 2) - 0.1 : 0.0;
      const cd v = kind == 0 ? cd(0.3 * ((i + j) % 3) - 0.4, im) : cd(0.5 - 0.1 * ((i * j) % 5), im);
      d.a[i * n + j] = v;
      d.a[j * n + i] = kind == 2 ? std::conj(v) : kind == 1 ? v : cd(0.25 * ((i + 2 * j) % 3), 0);
    }
  for (int i = 0; i < n; ++i) d.a[i * n + i] = cd(signs[i % std::strlen(signs)] == '+' ? n : -n, 0);
  return d;
}

// The positive diagonal entries among n, cycling through signs.
static long long positives(int n, const char* signs) {
  long long p = 0;
  for (int i = 0; i < n; ++i) p += signs[i % std::strlen(signs)] == '+';
  return p;
}

static Dense from_rows(int n, std::vector<double> v) {
  Dense d{n, std::vector<cd>(v.begin(), v.end())};
  return d;
}

static void defaults_and_protocol() {
  std::printf("-- configuration defaults and the Get/Set protocol\n");
  cudssConfig_t cfg;
  IS(cudssConfigCreate(&cfg), CUDSS_STATUS_SUCCESS);
  auto geti = [&](cudssConfigParam_t p) {
    int v = -12345;
    cudssConfigGet(cfg, p, &v, sizeof v, nullptr);
    return v;
  };
  auto getd = [&](cudssConfigParam_t p) {
    double v = -12345;
    cudssConfigGet(cfg, p, &v, sizeof v, nullptr);
    return v;
  };
  auto getl = [&](cudssConfigParam_t p) {
    int64_t v = -12345;
    cudssConfigGet(cfg, p, &v, sizeof v, nullptr);
    return v;
  };
  check(geti(CUDSS_CONFIG_REORDERING_ALG) == 0 && geti(CUDSS_CONFIG_FACTORIZATION_ALG) == 0 &&
            geti(CUDSS_CONFIG_SOLVE_ALG) == 0 && geti(CUDSS_CONFIG_MATCHING_ALG) == 0 &&
            geti(CUDSS_CONFIG_PIVOT_TYPE) == 0 && geti(CUDSS_CONFIG_IR_N_STEPS) == 0,
        "algorithms default to 0, no refinement");
  check(getd(CUDSS_CONFIG_IR_TOL) == 0 && getd(CUDSS_CONFIG_PIVOT_THRESHOLD) == 1 &&
            getd(CUDSS_CONFIG_PIVOT_EPSILON) == -1,
        "IR_TOL 0, PIVOT_THRESHOLD 1, PIVOT_EPSILON -1 (the type's default)");
  check(getl(CUDSS_CONFIG_MAX_LU_NNZ) == -1 && getl(CUDSS_CONFIG_HYBRID_DEVICE_MEMORY_LIMIT) == -1,
        "MAX_LU_NNZ and HYBRID_DEVICE_MEMORY_LIMIT are 64-bit, -1");
  check(geti(CUDSS_CONFIG_USE_CUDA_REGISTER_MEMORY) == 1 && geti(CUDSS_CONFIG_HOST_NTHREADS) == -1 &&
            geti(CUDSS_CONFIG_ND_NLEVELS) == 10 && geti(CUDSS_CONFIG_UBATCH_SIZE) == 1 &&
            geti(CUDSS_CONFIG_UBATCH_INDEX) == -1 && geti(CUDSS_CONFIG_USE_SUPERPANELS) == 1 &&
            geti(CUDSS_CONFIG_DEVICE_COUNT) == 1 && geti(CUDSS_CONFIG_ND_UBFACTOR) == 20,
        "the rest of the defaults");
  size_t w = 0;
  IS(cudssConfigGet(cfg, CUDSS_CONFIG_IR_TOL, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
  check(w == 8, "a zero size asks for the size: IR_TOL is a double");
  int big[4];
  IS(cudssConfigGet(cfg, CUDSS_CONFIG_IR_N_STEPS, big, sizeof big, &w), CUDSS_STATUS_INVALID_VALUE);
  int one = 1;
  IS(cudssConfigGet(cfg, CUDSS_CONFIG_IR_N_STEPS, nullptr, 4, &w), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigSet(cfg, CUDSS_CONFIG_IR_N_STEPS, &one, 8), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigSet(cfg, CUDSS_CONFIG_IR_N_STEPS, nullptr, 4), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigSet(cfg, (cudssConfigParam_t)26, &one, 4), CUDSS_STATUS_INTERNAL_ERROR);
  int bk = CUDSS_PIVOT_BUNCH_KAUFMAN, levels = 0;
  IS(cudssConfigSet(cfg, CUDSS_CONFIG_PIVOT_TYPE, &bk, 4), CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssConfigSet(cfg, CUDSS_CONFIG_ND_NLEVELS, &levels, 4), CUDSS_STATUS_INVALID_VALUE);
  double tol = -1;
  IS(cudssConfigSet(cfg, CUDSS_CONFIG_IR_TOL, &tol, 8), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigSet(cfg, CUDSS_CONFIG_IR_N_STEPS, &one, 4), CUDSS_STATUS_SUCCESS);
  check(geti(CUDSS_CONFIG_IR_N_STEPS) == 1, "a setting reads back");
  int devices[1] = {-7};
  IS(cudssConfigGet(cfg, CUDSS_CONFIG_DEVICE_INDICES, devices, sizeof devices, &w), CUDSS_STATUS_SUCCESS);
  check(devices[0] == 0 && w == 4, "DEVICE_INDICES defaults to device 0");
  IS(cudssConfigGet(nullptr, CUDSS_CONFIG_IR_N_STEPS, &one, 4, &w), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigDestroy(cfg), CUDSS_STATUS_SUCCESS);
}

static void handles_and_objects() {
  std::printf("-- handles, properties, memory handler, logger, matrix objects\n");
  IS(cudssCreate(nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDestroy(nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigCreate(nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssConfigDestroy(nullptr), CUDSS_STATUS_INVALID_VALUE);
  int v[3] = {-1, -1, -1};
  IS(cudssGetProperty(MAJOR_VERSION, &v[0]), CUDSS_STATUS_SUCCESS);
  IS(cudssGetProperty(MINOR_VERSION, &v[1]), CUDSS_STATUS_SUCCESS);
  IS(cudssGetProperty(PATCH_LEVEL, &v[2]), CUDSS_STATUS_SUCCESS);
  check(v[0] == 0 && v[1] == 8 && v[2] == 0, "version 0.8.0");
  IS(cudssGetProperty((libraryPropertyType)7, &v[0]), CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssGetProperty(MAJOR_VERSION, nullptr), CUDSS_STATUS_INVALID_VALUE);

  cudssHandle_t h;
  IS(cudssCreate(&h), CUDSS_STATUS_SUCCESS);
  cudssData_t d;
  IS(cudssDataCreate(nullptr, &d), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataCreate(h, nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssSetStream(nullptr, nullptr), CUDSS_STATUS_INVALID_VALUE);
  cudssDeviceMemHandler_t mh, got;
  std::memset(&mh, 0, sizeof mh);
  std::memset(&got, 0, sizeof got);
  std::strcpy(mh.name, "test pool");
  IS(cudssGetDeviceMemHandler(h, &got), CUDSS_STATUS_INVALID_VALUE);  // none set yet
  IS(cudssSetDeviceMemHandler(h, &mh), CUDSS_STATUS_SUCCESS);
  IS(cudssGetDeviceMemHandler(h, &got), CUDSS_STATUS_SUCCESS);
  check(std::strcmp(got.name, "test pool") == 0, "the memory handler reads back");
  IS(cudssSetDeviceMemHandler(h, nullptr), CUDSS_STATUS_SUCCESS);
  cudaStream_t streams[2] = {nullptr, nullptr};
  IS(cudssSetMgStreams(h, streams, 1), CUDSS_STATUS_SUCCESS);
  IS(cudssDestroy(h), CUDSS_STATUS_SUCCESS);
  IS(cudssCreateMg(&h, 0, nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssCreateMg(&h, 17, nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssCreateMg(&h, 1, nullptr), CUDSS_STATUS_SUCCESS);
  IS(cudssSetMgStreams(h, streams, 2), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssSetMgStreams(h, nullptr, 1), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssSetMgStreams(h, streams, 1), CUDSS_STATUS_SUCCESS);
  IS(cudssDestroy(h), CUDSS_STATUS_SUCCESS);
  IS(cudssLoggerSetLevel(9), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssLoggerSetLevel(0), CUDSS_STATUS_SUCCESS);
  IS(cudssLoggerSetCallback(nullptr), CUDSS_STATUS_SUCCESS);

  double* p;
  cudaMalloc((void**)&p, 64);
  int* ip;
  cudaMalloc((void**)&ip, 64);
  cudssMatrix_t m;
  IS(cudssMatrixCreateDn(&m, 2, 3, 2, p, CUDSS_R_64F, CUDSS_LAYOUT_ROW_MAJOR), CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssMatrixCreateDn(&m, 2, 1, 2, p, CUDSS_R_32I, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixCreateDn(&m, -2, 1, 2, p, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixCreateCsr(&m, 2, 2, 2, ip, nullptr, ip, p, CUDSS_R_32I, CUDSS_R_64I, CUDSS_R_64F,
                          CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
     CUDSS_STATUS_INVALID_VALUE);  // 32-bit offsets cannot hold 64-bit indices' worth
  IS(cudssMatrixCreateCsr(&m, 2, 2, 2, ip, nullptr, ip, p, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_32I,
                          CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
     CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixCreateCsr(&m, -1, 2, 2, ip, nullptr, ip, p, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                          CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
     CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixDestroy(nullptr), CUDSS_STATUS_INVALID_VALUE);

  cudssMatrix_t csr, dn;
  IS(cudssMatrixCreateCsr(&csr, 4, 4, 5, ip, nullptr, ip, p, CUDSS_R_64I, CUDSS_R_32I, CUDSS_R_64F,
                          CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER, CUDSS_BASE_ONE),
     CUDSS_STATUS_SUCCESS);  // 64-bit offsets with 32-bit indices are fine
  IS(cudssMatrixCreateDn(&dn, 4, 2, 5, p, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR), CUDSS_STATUS_SUCCESS);
  int fmt = 0;
  IS(cudssMatrixGetFormat(csr, &fmt), CUDSS_STATUS_SUCCESS);
  check(fmt == CUDSS_MFORMAT_CSR, "a CSR matrix's format");
  IS(cudssMatrixGetFormat(dn, &fmt), CUDSS_STATUS_SUCCESS);
  check(fmt == CUDSS_MFORMAT_DENSE, "a dense matrix's format");
  IS(cudssMatrixGetDn(csr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixGetCsr(dn, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
                       nullptr, nullptr, nullptr, nullptr),
     CUDSS_STATUS_INVALID_VALUE);
  int64_t r = 0, c = 0, nz = 0, ld = 0, f = -1, l = -1;
  void *rs, *re = (void*)1, *cs, *vs;
  cudssDataType_t ot, it, vt;
  cudssMatrixType_t mt;
  cudssMatrixViewType_t mv;
  cudssIndexBase_t ib;
  IS(cudssMatrixGetCsr(csr, &r, &c, &nz, &rs, &re, &cs, &vs, &ot, &it, &vt, &mt, &mv, &ib), CUDSS_STATUS_SUCCESS);
  check(r == 4 && c == 4 && nz == 5 && rs == ip && re == nullptr && cs == ip && vs == p && ot == CUDSS_R_64I &&
            it == CUDSS_R_32I && vt == CUDSS_R_64F && mt == CUDSS_MTYPE_SPD && mv == CUDSS_MVIEW_UPPER &&
            ib == CUDSS_BASE_ONE,
        "a CSR matrix's description reads back");
  cudssLayout_t lay;
  IS(cudssMatrixGetDn(dn, &r, &c, &ld, &vs, &vt, &lay), CUDSS_STATUS_SUCCESS);
  check(r == 4 && c == 2 && ld == 5 && vs == p && vt == CUDSS_R_64F && lay == CUDSS_LAYOUT_COL_MAJOR,
        "a dense matrix's description reads back");
  IS(cudssMatrixGetDistributionRow1d(csr, &f, &l), CUDSS_STATUS_SUCCESS);
  check(f == 0 && l == 3, "an undistributed matrix spans rows 0 .. n - 1");
  IS(cudssMatrixSetCsrPointers(csr, ip, nullptr, nullptr, p), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixSetCsrPointers(dn, ip, nullptr, ip, p), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixSetValues(csr, nullptr), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixSetValues(dn, p), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixGetBatchDn(dn, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
     CUDSS_STATUS_INVALID_VALUE);
  IS(cudssMatrixDestroy(csr), CUDSS_STATUS_SUCCESS);
  IS(cudssMatrixDestroy(dn), CUDSS_STATUS_SUCCESS);
  cudaFree(p);
  cudaFree(ip);
}

// The order phases must come in, and what can be asked of the data before.
static void phases() {
  std::printf("-- phase order and the data object's states\n");
  Session s;
  const Dense d = from_rows(3, {4, 1, 0, 1, 5, 2, 0, 2, 6});
  auto c = to_csr<int, double>(d, CUDSS_MVIEW_FULL, 0);
  int* rp = upload(c.rp);
  int* ci = upload(c.ci);
  double* vv = upload(c.v);
  double* db = upload(std::vector<double>{1, 2, 3, 0, 0, 0});
  double* dx = upload(std::vector<double>(6, 0));
  float* fb = upload(std::vector<float>{1, 2, 3});
  cudssMatrix_t A, B, X, Bf, B2, Bld;
  cudssMatrixCreateCsr(&A, 3, 3, (int64_t)c.ci.size(), rp, nullptr, ci, vv, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                       CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
  cudssMatrixCreateDn(&B, 3, 1, 3, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&X, 3, 1, 3, dx, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&Bf, 3, 1, 3, fb, CUDSS_R_32F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&B2, 2, 1, 2, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&Bld, 3, 1, 2, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);

  int info = -1;
  size_t w = 77;
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INFO, &info, sizeof info, &w), CUDSS_STATUS_NOT_INITIALIZED);
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_USER_PERM, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
  check(w == 0, "no user permutation set: zero bytes");
  int zero = 0;
  int64_t lz = 0;
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_INFO, &zero, sizeof zero), CUDSS_STATUS_NOT_INITIALIZED);
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_LU_NNZ, &lz, sizeof lz), CUDSS_STATUS_NOT_INITIALIZED);
  IS(cudssDataGet(nullptr, s.data, CUDSS_DATA_INFO, &info, sizeof info, &w), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataGet(s.h, nullptr, CUDSS_DATA_INFO, &info, sizeof info, &w), CUDSS_STATUS_INVALID_VALUE);

  IS(cudssExecute(nullptr, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, nullptr, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, s.cfg, nullptr, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, nullptr, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, B, X, B), CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssExecute(s.h, 0, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, 1 << 12, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_REORDERING | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B),
     CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssExecute(s.h, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_REFACTORIZATION, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);

  // The analysis needs neither right-hand side nor solution.
  IS(cudssExecute(s.h, CUDSS_PHASE_REORDERING, s.cfg, s.data, A, nullptr, nullptr), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, s.cfg, s.data, A, nullptr, nullptr),
     CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_SYMBOLIC_FACTORIZATION, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INFO, &info, sizeof info, &w), CUDSS_STATUS_SUCCESS);
  check(info == 0 && w == 4, "INFO after the analysis is 0");
  int64_t i64 = -1;
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_LU_NNZ, &i64, sizeof i64, &w), CUDSS_STATUS_SUCCESS);
  check(i64 > 0, "LU_NNZ is predicted by the analysis");
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_FLOPS, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
  check(w == 8, "FLOPS is 64-bit");
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_MEMORY_ESTIMATES, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
  check(w == 128, "MEMORY_ESTIMATES is sixteen 64-bit values");
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_NPIVOTS, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
  check(w == 4, "NPIVOTS has the index type");
  int npiv = -1;
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_NPIVOTS, &npiv, sizeof npiv, &w), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INFO, nullptr, 4, &w), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataGet(s.h, s.data, (cudssDataParam_t)27, &info, 4, &w), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_DIAG, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
  check(w == 3 * sizeof(double), "DIAG has n of the value type");
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_REFACTORIZATION, s.cfg, s.data, A, X, B),
     CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, Bf), CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B2), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, Bld), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, nullptr, X, B), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_LU_NNZ, &lz, sizeof lz), CUDSS_STATUS_NOT_SUPPORTED);
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_INFO, &zero, 8), CUDSS_STATUS_INVALID_VALUE);
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_INFO, &zero, 4), CUDSS_STATUS_SUCCESS);
  IS(cudssDataSet(s.h, s.data, (cudssDataParam_t)27, &zero, 4), CUDSS_STATUS_INVALID_VALUE);

  // Factorization and solve in one call, and a refactorization and solve.
  IS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  auto x = download(dx, 3);
  const double want[3] = {0.20408163265306123, 0.18367346938775511, 0.43877551020408162};
  check(std::fabs(x[0] - want[0]) < 1e-14 && std::fabs(x[1] - want[1]) < 1e-14 && std::fabs(x[2] - want[2]) < 1e-14,
        "x = (10, 9, 21.5) / 49");
  IS(cudssExecute(s.h, CUDSS_PHASE_REFACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B),
     CUDSS_STATUS_SUCCESS);
  // In place: the solution overwrites the right-hand side.
  double* dio = upload(std::vector<double>{1, 2, 3});
  cudssMatrix_t IO;
  cudssMatrixCreateDn(&IO, 3, 1, 3, dio, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, IO, IO), CUDSS_STATUS_SUCCESS);
  x = download(dio, 3);
  check(std::fabs(x[0] - want[0]) < 1e-14 && std::fabs(x[2] - want[2]) < 1e-14, "a solve in place");

  // The sub-phases, chained by hand: each call reads its right-hand side and
  // writes its solution, so the second half takes the first half's output.
  cudaMemset(dx, 0, 3 * sizeof(double));
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE_FWD_PERM | CUDSS_PHASE_SOLVE_FWD | CUDSS_PHASE_SOLVE_DIAG, s.cfg, s.data, A,
                  X, B),
     CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE_BWD | CUDSS_PHASE_SOLVE_BWD_PERM, s.cfg, s.data, A, X, X),
     CUDSS_STATUS_SUCCESS);
  x = download(dx, 3);
  check(std::fabs(x[0] - want[0]) < 1e-14 && std::fabs(x[1] - want[1]) < 1e-14 && std::fabs(x[2] - want[2]) < 1e-14,
        "forward then backward sub-phases make the solution");

  // Refinement on its own starts from what the solution holds.
  int two = 2;
  IS(cudssConfigSet(s.cfg, CUDSS_CONFIG_IR_N_STEPS, &two, sizeof two), CUDSS_STATUS_SUCCESS);
  cudaMemset(dx, 0, 3 * sizeof(double));
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE_REFINEMENT, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  x = download(dx, 3);
  int steps = -1;
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_IR_N_STEPS, &steps, sizeof steps, nullptr), CUDSS_STATUS_SUCCESS);
  check(steps == 2 && std::fabs(x[1] - want[1]) < 1e-14, "two refinement steps from zero reach the solution");
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_IR_N_STEPS, &steps, sizeof steps, nullptr), CUDSS_STATUS_SUCCESS);
  check(steps == 2, "without a tolerance, exactly IR_N_STEPS steps");
  double tol = 1e-8;
  IS(cudssConfigSet(s.cfg, CUDSS_CONFIG_IR_TOL, &tol, sizeof tol), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_IR_N_STEPS, &steps, sizeof steps, nullptr), CUDSS_STATUS_SUCCESS);
  check(steps == 0, "a solution already within the tolerance takes no step");

  // The host can interrupt a call. NVIDIA's library takes the flag's own
  // address as the value, and reads the flag at each call.
  int stop = 0;
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_USER_HOST_INTERRUPT, &stop, sizeof(int*)), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  stop = 1;
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_EXECUTION_FAILED);
  stop = 0;
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  stop = 1;
  IS(cudssDataSet(s.h, s.data, CUDSS_DATA_USER_HOST_INTERRUPT, nullptr, 0), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);

  // Results can be written to device memory too.
  int* dpiv = upload(std::vector<int>{-1});
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_NPIVOTS, dpiv, sizeof(int), nullptr), CUDSS_STATUS_SUCCESS);
  check(download(dpiv, 1)[0] == 0, "NPIVOTS written to device memory");
  int in[2] = {-1, -1};
  IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INERTIA, in, sizeof in, nullptr), CUDSS_STATUS_SUCCESS);
  check(in[0] == 0 && in[1] == 0, "a general matrix has no inertia: (0, 0)");

  for (auto m : {A, B, X, Bf, B2, Bld, IO}) cudssMatrixDestroy(m);
  for (void* p : {(void*)rp, (void*)ci, (void*)vv, (void*)db, (void*)dx, (void*)fb, (void*)dio, (void*)dpiv})
    cudaFree(p);
}

// Small systems whose answers do not depend on the ordering.
static void pivots_and_permutations() {
  std::printf("-- user permutation, DIAG, a matrix that is not positive definite, the pivot epsilon\n");
  {
    // SPD, one-based, upper triangle, with the caller's order 3, 1, 2:
    // Cholesky's diagonal is then fixed, and cuDSS reports it by original row.
    Session s;
    const Dense d = from_rows(3, {4, 1, 0, 1, 5, 2, 0, 2, 6});
    auto c = to_csr<int, double>(d, CUDSS_MVIEW_UPPER, 1);
    int* rp = upload(c.rp);
    int* ci = upload(c.ci);
    double* vv = upload(c.v);
    double* db = upload(std::vector<double>{1, 2, 3});
    double* dx = upload(std::vector<double>(3, 0));
    cudssMatrix_t A, B, X;
    cudssMatrixCreateCsr(&A, 3, 3, (int64_t)c.ci.size(), rp, nullptr, ci, vv, CUDSS_R_32I, CUDSS_R_32I,
                         CUDSS_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_UPPER, CUDSS_BASE_ONE);
    cudssMatrixCreateDn(&B, 3, 1, 3, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&X, 3, 1, 3, dx, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    const int perm[3] = {3, 1, 2};
    IS(cudssDataSet(s.h, s.data, CUDSS_DATA_USER_PERM, perm, sizeof perm), CUDSS_STATUS_SUCCESS);
    int got[3] = {0, 0, 0};
    size_t w = 0;
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_USER_PERM, got, sizeof got, &w), CUDSS_STATUS_SUCCESS);
    check(got[0] == 3 && got[2] == 2 && w == 12, "the user permutation reads back");
    IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_PERM_REORDER_ROW, got, sizeof got, &w), CUDSS_STATUS_SUCCESS);
    check(got[0] == 3 && got[1] == 1 && got[2] == 2, "PERM_REORDER_ROW is the user's, one-based");
    IS(cudssExecute(s.h, CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B),
       CUDSS_STATUS_SUCCESS);
    auto x = download(dx, 3);
    check(std::fabs(x[0] * 49 - 10) < 1e-13 && std::fabs(x[1] * 49 - 9) < 1e-13 && std::fabs(x[2] * 49 - 21.5) < 1e-13,
          "x = (10, 9, 21.5) / 49 from the upper triangle");
    double diag[3];
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_DIAG, diag, sizeof diag, &w), CUDSS_STATUS_SUCCESS);
    check(std::fabs(diag[0] - 2) < 1e-14 && std::fabs(diag[1] - std::sqrt(4.0 + 1.0 / 12)) < 1e-14 &&
              std::fabs(diag[2] - std::sqrt(6.0)) < 1e-14,
          "Cholesky's diagonal in original order: 2, sqrt(49/12), sqrt(6)");
    int in[2] = {-1, -1};
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INERTIA, in, sizeof in, &w), CUDSS_STATUS_SUCCESS);
    check(in[0] == 0 && in[1] == 0, "SPD matrices report no inertia: (0, 0)");
    const int short_perm[2] = {1, 2};
    IS(cudssDataSet(s.h, s.data, CUDSS_DATA_USER_PERM, short_perm, sizeof short_perm), CUDSS_STATUS_SUCCESS);
    IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS, s.cfg, s.data, A, X, B), CUDSS_STATUS_INVALID_VALUE);
    for (auto m : {A, B, X}) cudssMatrixDestroy(m);
    for (void* p : {(void*)rp, (void*)ci, (void*)vv, (void*)db, (void*)dx}) cudaFree(p);
  }
  auto two_by_two = [](const char* what, std::vector<double> a, cudssMatrixType_t mt, double eps,
                       auto&& after) {
    Session s;
    const Dense d = from_rows(2, a);
    auto c = to_csr<int, double>(d, CUDSS_MVIEW_FULL, 0);
    int* rp = upload(c.rp);
    int* ci = upload(c.ci);
    double* vv = upload(c.v);
    double* db = upload(std::vector<double>{1, 2});
    double* dx = upload(std::vector<double>(2, 0));
    cudssMatrix_t A, B, X;
    cudssMatrixCreateCsr(&A, 2, 2, (int64_t)c.ci.size(), rp, nullptr, ci, vv, CUDSS_R_32I, CUDSS_R_32I,
                         CUDSS_R_64F, mt, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO);
    cudssMatrixCreateDn(&B, 2, 1, 2, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&X, 2, 1, 2, dx, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    if (eps > 0) IS(cudssConfigSet(s.cfg, CUDSS_CONFIG_PIVOT_EPSILON, &eps, sizeof eps), CUDSS_STATUS_SUCCESS);
    std::printf("   %s\n", what);
    IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A,
                    X, B),
       CUDSS_STATUS_SUCCESS);
    after(s, download(dx, 2));
    for (auto m : {A, B, X}) cudssMatrixDestroy(m);
    for (void* p : {(void*)rp, (void*)ci, (void*)vv, (void*)db, (void*)dx}) cudaFree(p);
  };
  two_by_two("[0 1; 1 0], general: pivoting swaps the rows", {0, 1, 1, 0}, CUDSS_MTYPE_GENERAL, 0,
             [](Session& s, std::vector<double> x) {
               check(x[0] == 2 && x[1] == 1, "x = (2, 1)");
               int npiv = -1;
               cudssDataGet(s.h, s.data, CUDSS_DATA_NPIVOTS, &npiv, sizeof npiv, nullptr);
               check(npiv == 0, "no pivot was perturbed");
             });
  two_by_two("[1 2; 2 1] passed as SPD", {1, 2, 2, 1}, CUDSS_MTYPE_SPD, 0, [](Session& s, std::vector<double>) {
    int info = 0;
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INFO, &info, sizeof info, nullptr), CUDSS_STATUS_SUCCESS);
    check(info == 2, "INFO names the second pivot, the first non-positive minor");
    int zero = 0;
    IS(cudssDataSet(s.h, s.data, CUDSS_DATA_INFO, &zero, sizeof zero), CUDSS_STATUS_SUCCESS);
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INFO, &info, sizeof info, nullptr), CUDSS_STATUS_SUCCESS);
    check(info == 0, "INFO can be cleared");
  });
  two_by_two("diag(-1e-4, 1) with PIVOT_EPSILON 1e-3", {-1e-4, 0, 0, 1}, CUDSS_MTYPE_GENERAL, 1e-3,
             [](Session& s, std::vector<double> x) {
               check(std::fabs(x[0] + 1000) < 1e-9 && x[1] == 2, "the small pivot became -1e-3: x = (-1000, 2)");
               int npiv = -1;
               cudssDataGet(s.h, s.data, CUDSS_DATA_NPIVOTS, &npiv, sizeof npiv, nullptr);
               check(npiv == 1, "NPIVOTS counts the perturbed pivot");
               double diag[2];
               IS(cudssDataGet(s.h, s.data, CUDSS_DATA_DIAG, diag, sizeof diag, nullptr), CUDSS_STATUS_SUCCESS);
               check(diag[0] == -1e-3 && diag[1] == 1, "DIAG shows it");
             });
  two_by_two("[1 1; 1 1], symmetric: singular", {1, 1, 1, 1}, CUDSS_MTYPE_SYMMETRIC, 0,
             [](Session& s, std::vector<double>) {
               int npiv = -1, in[2] = {-1, -1};
               cudssDataGet(s.h, s.data, CUDSS_DATA_NPIVOTS, &npiv, sizeof npiv, nullptr);
               cudssDataGet(s.h, s.data, CUDSS_DATA_INERTIA, in, sizeof in, nullptr);
               check(npiv == 1 && in[0] == 2 && in[1] == 0,
                     "the zero pivot is perturbed to +epsilon and counted as positive");
             });
}

static void batches() {
  std::printf("-- a uniform batch and a non-uniform batch\n");
  {
    Session s;
    int ub = 2;
    IS(cudssConfigSet(s.cfg, CUDSS_CONFIG_UBATCH_SIZE, &ub, sizeof ub), CUDSS_STATUS_SUCCESS);
    int* rp = upload(std::vector<int>{0, 1, 3, 5});
    int* ci = upload(std::vector<int>{0, 0, 1, 1, 2});
    double* vv = upload(std::vector<double>{4, 1, 5, 2, 6, /* the second system */ 2, 0, 3, 0, -4});
    double* db = upload(std::vector<double>{1, 2, 3, 4, 5, 6});
    double* dx = upload(std::vector<double>(6, 0));
    cudssMatrix_t A, B, X;
    cudssMatrixCreateCsr(&A, 3, 3, 5, rp, nullptr, ci, vv, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                         CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO);
    cudssMatrixCreateDn(&B, 3, 1, 3, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    cudssMatrixCreateDn(&X, 3, 1, 3, dx, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
    IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X,
                    B),
       CUDSS_STATUS_SUCCESS);
    auto x = download(dx, 6);
    check(std::fabs(x[0] * 49 - 10) < 1e-13 && std::fabs(x[2] * 49 - 21.5) < 1e-13 && std::fabs(x[3] - 2) < 1e-14 &&
              std::fabs(x[4] - 5.0 / 3) < 1e-14 && std::fabs(x[5] + 1.5) < 1e-14,
          "each system solved with its own values and right-hand side");
    int in[2] = {-1, -1};
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_INERTIA, in, sizeof in, nullptr), CUDSS_STATUS_SUCCESS);
    check(in[0] == 5 && in[1] == 1, "the batch's inertia adds up: (3, 0) + (2, 1)");
    int idx = 1;
    IS(cudssConfigSet(s.cfg, CUDSS_CONFIG_UBATCH_INDEX, &idx, sizeof idx), CUDSS_STATUS_SUCCESS);
    cudaMemset(dx, 0, 6 * sizeof(double));
    IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
    x = download(dx, 6);
    check(x[0] == 0.5 && std::fabs(x[1] - 2.0 / 3) < 1e-15 && x[2] == -0.75 && x[3] == 0,
          "UBATCH_INDEX 1 solves the second system alone, with a single right-hand side");
    for (auto m : {A, B, X}) cudssMatrixDestroy(m);
    for (void* p : {(void*)rp, (void*)ci, (void*)vv, (void*)db, (void*)dx}) cudaFree(p);
  }
  {
    // Sizes 2 and 3; the per-matrix pointer arrays in device memory.
    Session s;
    int nr[2] = {2, 3}, nz[2] = {3, 5}, one[2] = {1, 1};
    void* rps[2] = {upload(std::vector<int>{0, 2, 3}), upload(std::vector<int>{0, 1, 3, 5})};
    void* cis[2] = {upload(std::vector<int>{0, 1, 1}), upload(std::vector<int>{0, 0, 1, 1, 2})};
    void* vs[2] = {upload(std::vector<double>{2, 1, 4}), upload(std::vector<double>{4, 1, 5, 2, 6})};
    void* bs[2] = {upload(std::vector<double>{1, 2}), upload(std::vector<double>{1, 2, 3})};
    void* xs[2] = {upload(std::vector<double>{0, 0}), upload(std::vector<double>{0, 0, 0})};
    auto ptrs = [](void* const* p) { return (const void* const*)upload(std::vector<void*>{p[0], p[1]}); };
    const void* const* drp = ptrs(rps);
    const void* const* dci = ptrs(cis);
    const void* const* dv = ptrs(vs);
    const void* const* db = ptrs(bs);
    const void* const* dx = ptrs(xs);
    cudssMatrix_t A, B, X;
    IS(cudssMatrixCreateBatchCsr(&A, 2, nr, nr, nz, drp, nullptr, dci, dv, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                                 CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO),
       CUDSS_STATUS_SUCCESS);
    IS(cudssMatrixCreateBatchDn(&B, 2, nr, one, nr, db, CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
       CUDSS_STATUS_SUCCESS);
    IS(cudssMatrixCreateBatchDn(&X, 2, nr, one, nr, dx, CUDSS_R_32I, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR),
       CUDSS_STATUS_SUCCESS);
    int fmt = 0;
    cudssMatrixGetFormat(A, &fmt);
    check(fmt == (CUDSS_MFORMAT_CSR | CUDSS_MFORMAT_BATCH), "a batch of CSR matrices: CSR | BATCH");
    cudssMatrixGetFormat(B, &fmt);
    check(fmt == (CUDSS_MFORMAT_DENSE | CUDSS_MFORMAT_BATCH), "a batch of dense matrices: DENSE | BATCH");
    int64_t count = 0;
    void* got_rows = nullptr;
    IS(cudssMatrixGetBatchDn(B, &count, &got_rows, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr),
       CUDSS_STATUS_SUCCESS);
    check(count == 2 && got_rows == (void*)nr, "the batch reads back");
    IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION | CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X,
                    B),
       CUDSS_STATUS_SUCCESS);
    auto x0 = download((double*)xs[0], 2);
    auto x1 = download((double*)xs[1], 3);
    check(x0[0] == 0.25 && x0[1] == 0.5, "the first system: x = (1/4, 1/2)");
    check(x1[0] == 0.25 && std::fabs(x1[1] - 0.35) < 1e-15 && std::fabs(x1[2] - 2.3 / 6) < 1e-15,
          "the second, a lower-triangular general matrix: x = (1/4, 0.35, 2.3/6)");
    size_t w = 0;
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_PERM_REORDER_ROW, nullptr, 0, &w), CUDSS_STATUS_NOT_SUPPORTED);
    IS(cudssDataGet(s.h, s.data, CUDSS_DATA_DIAG, nullptr, 0, &w), CUDSS_STATUS_SUCCESS);
    check(w == 5 * sizeof(double), "DIAG covers every system: 2 + 3 values");
    for (auto m : {A, B, X}) cudssMatrixDestroy(m);
    for (void* p : {rps[0], rps[1], cis[0], cis[1], vs[0], vs[1], bs[0], bs[1], xs[0], xs[1]}) cudaFree(p);
    for (const void* const* p : {drp, dci, dv, db, dx}) cudaFree((void*)p);
  }
}

// A solve recorded into a CUDA graph reads the right-hand side when the graph
// runs, not when it was captured.
static void graph_capture() {
  std::printf("-- a solve captured into a CUDA graph\n");
  Session s;
  const Dense d = from_rows(3, {4, 1, 0, 1, 5, 2, 0, 2, 6});
  auto c = to_csr<int, double>(d, CUDSS_MVIEW_LOWER, 0);
  int* rp = upload(c.rp);
  int* ci = upload(c.ci);
  double* vv = upload(c.v);
  double* db = upload(std::vector<double>{0, 0, 0});
  double* dx = upload(std::vector<double>(3, 0));
  cudssMatrix_t A, B, X;
  cudssMatrixCreateCsr(&A, 3, 3, (int64_t)c.ci.size(), rp, nullptr, ci, vv, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                       CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, CUDSS_BASE_ZERO);
  cudssMatrixCreateDn(&B, 3, 1, 3, db, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudssMatrixCreateDn(&X, 3, 1, 3, dx, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR);
  cudaStream_t st;
  cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking);
  IS(cudssSetStream(s.h, st), CUDSS_STATUS_SUCCESS);
  IS(cudssExecute(s.h, CUDSS_PHASE_ANALYSIS | CUDSS_PHASE_FACTORIZATION, s.cfg, s.data, A, X, B),
     CUDSS_STATUS_SUCCESS);
  cudaStreamSynchronize(st);
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
  check(cudaStreamBeginCapture(st, cudaStreamCaptureModeThreadLocal) == cudaSuccess, "capture begins");
  IS(cudssExecute(s.h, CUDSS_PHASE_SOLVE, s.cfg, s.data, A, X, B), CUDSS_STATUS_SUCCESS);
  check(cudaStreamEndCapture(st, &graph) == cudaSuccess, "capture ends");
  check(cudaGraphInstantiate(&exec, graph, 0) == cudaSuccess, "the graph instantiates");
  const std::vector<std::vector<double>> rhs = {{1, 2, 3}, {4, 1, 6}};
  for (const auto& b : rhs) {
    cudaMemcpy(db, b.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);
    check(cudaGraphLaunch(exec, st) == cudaSuccess, "the graph launches");
    cudaStreamSynchronize(st);
    auto x = download(dx, 3);
    check(residual(d, x, b) < 1e-15, "the replay solved for the right-hand side of the moment");
  }
  cudaGraphExecDestroy(exec);
  cudaGraphDestroy(graph);
  cudaStreamDestroy(st);
  for (auto m : {A, B, X}) cudssMatrixDestroy(m);
  for (void* p : {(void*)rp, (void*)ci, (void*)vv, (void*)db, (void*)dx}) cudaFree(p);
}

int main() {
  defaults_and_protocol();
  handles_and_objects();
  phases();
  pivots_and_permutations();

  // Every factorization, view, base, width and value type. Inertia is the
  // diagonal's signs for the indefinite types, and (0, 0) for the others.
  solve_case<int, double>("general, real double, three right-hand sides", dominant(9, "+-+", 0, false),
                          CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, 0, CUDSS_R_32I, CUDSS_R_64F, 3, 1e-14, 0, 0);
  solve_case<int, float>("general, float, CSR in host memory", dominant(7, "+", 0, false), CUDSS_MTYPE_GENERAL,
                         CUDSS_MVIEW_FULL, 0, CUDSS_R_32I, CUDSS_R_32F, 2, 1e-6, -1, -1, true);
  solve_case<int64_t, double>("symmetric indefinite, upper triangle, one-based, 64-bit",
                              dominant(11, "+--+-", 1, false), CUDSS_MTYPE_SYMMETRIC, CUDSS_MVIEW_UPPER, 1,
                              CUDSS_R_64I, CUDSS_R_64F, 2, 1e-14, positives(11, "+--+-"),
                              11 - positives(11, "+--+-"));
  solve_case<int, double>("symmetric indefinite, full storage", dominant(10, "-+", 1, false), CUDSS_MTYPE_SYMMETRIC,
                          CUDSS_MVIEW_FULL, 0, CUDSS_R_32I, CUDSS_R_64F, 1, 1e-14, 5, 5);
  solve_case<int, double>("SPD", dominant(12, "+", 1, false), CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER, 0, CUDSS_R_32I,
                          CUDSS_R_64F, 2, 1e-14, 0, 0);
  solve_case<int, cd>("Hermitian indefinite, complex double", dominant(10, "++-", 2, true), CUDSS_MTYPE_HERMITIAN,
                      CUDSS_MVIEW_LOWER, 0, CUDSS_R_32I, CUDSS_C_64F, 2, 1e-14, positives(10, "++-"),
                      10 - positives(10, "++-"));
  solve_case<int64_t, cf>("HPD, complex float, 64-bit", dominant(8, "+", 2, true), CUDSS_MTYPE_HPD,
                          CUDSS_MVIEW_UPPER, 0, CUDSS_R_64I, CUDSS_C_32F, 1, 1e-6);
  solve_case<int, cd>("general, complex double, one-based", dominant(9, "+-", 0, true), CUDSS_MTYPE_GENERAL,
                      CUDSS_MVIEW_FULL, 1, CUDSS_R_32I, CUDSS_C_64F, 2, 1e-14);
  solve_case<int, cd>("complex symmetric (not Hermitian)", dominant(6, "+-", 1, true), CUDSS_MTYPE_SYMMETRIC,
                      CUDSS_MVIEW_LOWER, 0, CUDSS_R_32I, CUDSS_C_64F, 1, 1e-14);

  batches();
  graph_capture();
  std::printf(failures ? "FAIL: %d checks failed\n" : "PASS\n", failures);
  return failures ? 1 : 0;
}
