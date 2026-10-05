// AMD's hipSOLVER, unmodified (over rocSOLVER and rocBLAS), on a simulated
// MI300X, in double precision. Each answer is checked on the host by what it
// must satisfy, not against another library's digits:
//
//   getrf and getrs: x solves A x = b
//   potrf: L L^T gives back the symmetric positive definite A
//   syevd: each eigenpair has A v = lambda v, in ascending order
//   gesvd: U S V^T gives back A, with S descending
//
// Built ahead of time by build.sh, from hipSOLVER's documented API.
#include <hip/hip_runtime.h>
#include <hipsolver/hipsolver.h>

#include <cmath>
#include <cstdio>
#include <vector>

namespace {

int ok = 0, total = 0;
void check(const char* what, bool good) {
  ++total;
  ok += good;
  if (!good) std::printf("wrong: %s\n", what);
}

// Deterministic values in [-1, 1).
double value(size_t i) { return double((i * 2654435761u) % 2001) / 1000.0 - 1.0; }

// Device memory holding `h`, freed with it.
struct Buf {
  double* p = nullptr;
  explicit Buf(const std::vector<double>& h) {
    if (hipMalloc(&p, h.size() * 8) != hipSuccess || hipMemcpy(p, h.data(), h.size() * 8, hipMemcpyHostToDevice) != hipSuccess)
      p = nullptr;
  }
  explicit Buf(size_t n) {
    if (hipMalloc(&p, std::max<size_t>(n, 1) * 8) != hipSuccess) p = nullptr;
  }
  ~Buf() { (void)hipFree(p); }
  std::vector<double> get(size_t n) const {
    std::vector<double> h(n);
    if (hipMemcpy(h.data(), p, n * 8, hipMemcpyDeviceToHost) != hipSuccess) h.assign(n, NAN);
    return h;
  }
};
int info_of(const int* d) {
  int i = -1;
  return hipMemcpy(&i, d, 4, hipMemcpyDeviceToHost) == hipSuccess ? i : -1;
}

bool ran(hipsolverStatus_t s, const char* what) {
  if (s != HIPSOLVER_STATUS_SUCCESS) std::printf("%s: hipSOLVER error %d\n", what, int(s));
  return s == HIPSOLVER_STATUS_SUCCESS;
}

}  // namespace

int main() {
  hipsolverHandle_t h;
  if (!ran(hipsolverCreate(&h), "create")) return 1;
  int* info;
  if (hipMalloc(&info, 4) != hipSuccess) return 1;

  {  // LU with partial pivoting, then a solve, n = 48, two right-hand sides.
    const int n = 48, nrhs = 2;
    std::vector<double> a(n * n), b(n * nrhs);
    for (int i = 0; i < n * n; ++i) a[i] = value(i);
    for (int i = 0; i < n * nrhs; ++i) b[i] = value(9000 + i);
    Buf da(a), db(b), dw(0);
    int* ipiv;
    int lwork = 0, lwork2 = 0;
    bool good = hipMalloc(&ipiv, n * 4) == hipSuccess &&
                ran(hipsolverDgetrf_bufferSize(h, n, n, da.p, n, &lwork), "getrf size");
    Buf work(lwork);
    good = good && ran(hipsolverDgetrf(h, n, n, da.p, n, work.p, lwork, ipiv, info), "getrf") && info_of(info) == 0 &&
           ran(hipsolverDgetrs_bufferSize(h, HIPSOLVER_OP_N, n, nrhs, da.p, n, ipiv, db.p, n, &lwork2), "getrs size");
    Buf work2(lwork2);
    good = good && ran(hipsolverDgetrs(h, HIPSOLVER_OP_N, n, nrhs, da.p, n, ipiv, db.p, n, work2.p, lwork2, info), "getrs") &&
           info_of(info) == 0;
    const std::vector<double> x = db.get(n * nrhs);
    double err = 0, scale = 0;
    for (int c = 0; c < nrhs; ++c)
      for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += a[r + k * n] * x[k + c * n];
        err = std::max(err, std::abs(s - b[r + c * n]));
        scale = std::max(scale, std::abs(b[r + c * n]));
      }
    check("getrf and getrs solve A x = b", good && err <= 1e-9 * scale);
    (void)hipFree(ipiv);
  }

  {  // Cholesky of A = M M^T + n I, n = 40, lower.
    const int n = 40;
    std::vector<double> m(n * n), a(n * n);
    for (int i = 0; i < n * n; ++i) m[i] = value(3 * i + 1);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c < n; ++c) {
        double s = r == c ? n : 0;
        for (int k = 0; k < n; ++k) s += m[r + k * n] * m[c + k * n];
        a[r + c * n] = s;
      }
    Buf da(a);
    int lwork = 0;
    bool good = ran(hipsolverDpotrf_bufferSize(h, HIPSOLVER_FILL_MODE_LOWER, n, da.p, n, &lwork), "potrf size");
    Buf work(lwork);
    good = good && ran(hipsolverDpotrf(h, HIPSOLVER_FILL_MODE_LOWER, n, da.p, n, work.p, lwork, info), "potrf") &&
           info_of(info) == 0;
    const std::vector<double> l = da.get(n * n);
    double err = 0;
    for (int r = 0; r < n; ++r)
      for (int c = 0; c <= r; ++c) {
        double s = 0;
        for (int k = 0; k <= c; ++k) s += l[r + k * n] * l[c + k * n];
        err = std::max(err, std::abs(s - a[r + c * n]));
      }
    check("potrf: L L^T is A", good && err <= 1e-9 * n * n);
  }

  {  // Eigenvalues and vectors of a symmetric 32 x 32.
    const int n = 32;
    std::vector<double> a(n * n);
    for (int r = 0; r < n; ++r)
      for (int c = 0; c <= r; ++c) a[r + c * n] = a[c + r * n] = value(r * 97 + c * 31);
    Buf da(a), dw(n);
    int lwork = 0;
    bool good = ran(hipsolverDsyevd_bufferSize(h, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_UPPER, n, da.p, n,
                                               dw.p, &lwork),
                    "syevd size");
    Buf work(lwork);
    good = good &&
           ran(hipsolverDsyevd(h, HIPSOLVER_EIG_MODE_VECTOR, HIPSOLVER_FILL_MODE_UPPER, n, da.p, n, dw.p, work.p, lwork,
                               info),
               "syevd") &&
           info_of(info) == 0;
    const std::vector<double> v = da.get(n * n), w = dw.get(n);
    double err = 0;
    for (int j = 0; j < n; ++j) {
      if (j && w[j] < w[j - 1]) good = false;
      for (int r = 0; r < n; ++r) {
        double s = 0;
        for (int k = 0; k < n; ++k) s += a[r + k * n] * v[k + j * n];
        err = std::max(err, std::abs(s - w[j] * v[r + j * n]));
      }
    }
    check("syevd: A v = lambda v, ascending", good && err <= 1e-10 * n);
  }

  {  // Singular values and vectors of a 40 x 24, the thin U and V^T.
    const int m = 40, n = 24;
    std::vector<double> a(m * n);
    for (int i = 0; i < m * n; ++i) a[i] = value(5 * i + 2);
    Buf da(a), ds(n), du(m * n), dvt(n * n), drwork(n);
    int lwork = 0;
    bool good = ran(hipsolverDgesvd_bufferSize(h, 'S', 'S', m, n, &lwork), "gesvd size");
    Buf work(lwork);
    good = good && ran(hipsolverDgesvd(h, 'S', 'S', m, n, da.p, m, ds.p, du.p, m, dvt.p, n, work.p, lwork, drwork.p, info),
                       "gesvd") &&
           info_of(info) == 0;
    const std::vector<double> s = ds.get(n), u = du.get(m * n), vt = dvt.get(n * n);
    double err = 0;
    for (int j = 1; j < n; ++j)
      if (s[j] > s[j - 1]) good = false;
    for (int r = 0; r < m; ++r)
      for (int c = 0; c < n; ++c) {
        double t = 0;
        for (int k = 0; k < n; ++k) t += u[r + k * m] * s[k] * vt[k + c * n];
        err = std::max(err, std::abs(t - a[r + c * m]));
      }
    check("gesvd: U S V^T is A, S descending", good && err <= 1e-10 * m);
  }

  (void)hipFree(info);
  hipsolverDestroy(h);
  std::printf("hipSOLVER: %d of %d factorizations hold\n", ok, total);
  return ok == total ? 0 : 1;
}
