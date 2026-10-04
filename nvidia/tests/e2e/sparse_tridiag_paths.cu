// cuSPARSE's tridiagonal and pentadiagonal solvers, checked against the
// systems they solve (the residual) and against what an RTX 3060's cuSPARSE
// 13.0 does where the answer depends on the algorithm:
//
//   gtsv2                 LU with partial pivoting: a zero leading pivot is
//                         fine, a singular system comes back NaN, and dl, d,
//                         du are left alone
//   gtsv2_nopivot,        parallel cyclic reduction up to 2048 unknowns: with
//   gtsv2StridedBatch     d[0] = 0 every odd unknown is NaN and every even one
//                         right; at 4096, cyclic reduction first, and all NaN
//   gtsvInterleavedBatch  Thomas (all NaN on that system, du left holding the
//                         modified upper diagonal), LU (1/u_ii and u_{i,i+1}
//                         left in d and du) and Householder QR (R's first two
//                         diagonals left in d and du) -- the last two only for
//                         real values
//   gpsvInterleavedBatch  Householder QR, R's first three diagonals left in d,
//                         du and dw
//   refusals              the statuses NVIDIA's gives for sizes, strides,
//                         algorithms and leading dimensions
//
// In S, D, C and Z. Single precision agrees with the card to rounding (it
// computes in float, the simulator in double).
#include <cuComplex.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <type_traits>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;
// run_lib_check.sh runs this on a simulated GPU it names in VGPU_GPU.
static const bool on_sim = std::getenv("VGPU_GPU") != nullptr;

static void check(bool ok, const char* what, double err = 0) {
  std::printf("%-4s %s (%.2e)\n", ok ? "ok" : "FAIL", what, err);
  if (!ok) ++failures;
}
static void expect(int got, int want, const char* what) {
  std::printf("%-4s %s (status %d, expected %d)\n", got == want ? "ok" : "FAIL", what, got, want);
  if (got != want) ++failures;
}

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

// The four element types, seen as std::complex<double>.
template <class T> struct Num;
template <> struct Num<float> {
  static cdouble get(float v) { return v; }
  static float put(cdouble v) { return (float)v.real(); }
  static constexpr double tol = 2e-5;
  static constexpr bool cplx = false;
  static constexpr const char* name = "S";
};
template <> struct Num<double> {
  static cdouble get(double v) { return v; }
  static double put(cdouble v) { return v.real(); }
  static constexpr double tol = 1e-12;
  static constexpr bool cplx = false;
  static constexpr const char* name = "D";
};
template <> struct Num<cuComplex> {
  static cdouble get(cuComplex v) { return {v.x, v.y}; }
  static cuComplex put(cdouble v) { return make_cuComplex((float)v.real(), (float)v.imag()); }
  static constexpr double tol = 2e-5;
  static constexpr bool cplx = true;
  static constexpr const char* name = "C";
};
template <> struct Num<cuDoubleComplex> {
  static cdouble get(cuDoubleComplex v) { return {v.x, v.y}; }
  static cuDoubleComplex put(cdouble v) { return make_cuDoubleComplex(v.real(), v.imag()); }
  static constexpr double tol = 1e-12;
  static constexpr bool cplx = true;
  static constexpr const char* name = "Z";
};

// The cuSPARSE entry points for each type.
template <class T> struct Api;
#define VGPU_API(P, T)                                                                                            \
  template <> struct Api<T> {                                                                                     \
    static constexpr auto gtsv2_bs = cusparse##P##gtsv2_bufferSizeExt;                                            \
    static constexpr auto gtsv2 = cusparse##P##gtsv2;                                                             \
    static constexpr auto nopiv_bs = cusparse##P##gtsv2_nopivot_bufferSizeExt;                                    \
    static constexpr auto nopiv = cusparse##P##gtsv2_nopivot;                                                     \
    static constexpr auto strided_bs = cusparse##P##gtsv2StridedBatch_bufferSizeExt;                              \
    static constexpr auto strided = cusparse##P##gtsv2StridedBatch;                                               \
    static constexpr auto inter_bs = cusparse##P##gtsvInterleavedBatch_bufferSizeExt;                             \
    static constexpr auto inter = cusparse##P##gtsvInterleavedBatch;                                              \
    static constexpr auto gpsv_bs = cusparse##P##gpsvInterleavedBatch_bufferSizeExt;                              \
    static constexpr auto gpsv = cusparse##P##gpsvInterleavedBatch;                                               \
  };
VGPU_API(S, float)
VGPU_API(D, double)
VGPU_API(C, cuComplex)
VGPU_API(Z, cuDoubleComplex)
#undef VGPU_API

static cusparseHandle_t h;
static void* work;  // one workspace, larger than anything asked for

static double hash01(int i, int seed) {
  return std::fmod(std::fabs(std::sin(12.9898 * (i + 1) + 78.233 * seed)) * 43758.5453, 1.0);
}
// A value for diagonal `k` of row i; the main diagonal dominates.
template <class T> T entry(int i, int k, int seed, bool dominant = true) {
  const double re = hash01(i * 7 + k, seed) - 0.5, imv = Num<T>::cplx ? hash01(i * 11 + k, seed + 3) - 0.5 : 0.0;
  cdouble v(re, imv);
  if (k == 0 && dominant) v += cdouble(4.0 + hash01(i, seed + 9), 0.0);
  return Num<T>::put(v);
}

// || A x - b ||_inf / (|| A ||_inf || x ||_inf + || b ||_inf) for a banded A
// given by diagonals diag[k][i] = A(i, i + k - lower).
static double residual(const std::vector<std::vector<cdouble>>& diag, int lower, const std::vector<cdouble>& x,
                       const std::vector<cdouble>& b) {
  const int m = (int)x.size();
  double r = 0, an = 0, xn = 0, bn = 0;
  for (int i = 0; i < m; ++i) {
    cdouble s = 0;
    double row = 0;
    for (size_t k = 0; k < diag.size(); ++k) {
      const int j = i + (int)k - lower;
      if (j < 0 || j >= m) continue;
      s += diag[k][(size_t)i] * x[(size_t)j];
      row += std::abs(diag[k][(size_t)i]);
    }
    r = std::max(r, std::abs(s - b[(size_t)i]));
    an = std::max(an, row);
    xn = std::max(xn, std::abs(x[(size_t)i]));
    bn = std::max(bn, std::abs(b[(size_t)i]));
  }
  return r / (an * xn + bn + 1e-300);
}
template <class T> std::vector<cdouble> widen(const std::vector<T>& v, size_t first, size_t n, size_t step = 1) {
  std::vector<cdouble> out(n);
  for (size_t i = 0; i < n; ++i) out[i] = Num<T>::get(v[first + i * step]);
  return out;
}
static bool finite(cdouble v) { return std::isfinite(v.real()) && std::isfinite(v.imag()); }

// Householder QR as NVIDIA's makes it (beta = -sgn(alpha) * norm, every
// column reflected), returning R's diagonals 0..lower+... at rows j.
static std::vector<std::vector<cdouble>> householder_r(std::vector<std::vector<cdouble>> diag, int lower, int upper) {
  const int m = (int)diag[(size_t)lower].size(), width = 2 * lower + upper + 1;
  std::vector<cdouble> rows((size_t)m * width);
  auto at = [&](int i, int col) -> cdouble& { return rows[(size_t)i * width + (size_t)(col - i + lower)]; };
  for (int i = 0; i < m; ++i)
    for (int k = 0; k <= lower + upper; ++k)
      if (i + k - lower >= 0 && i + k - lower < m) at(i, i + k - lower) = diag[(size_t)k][(size_t)i];
  for (int j = 0; j < m; ++j) {
    const int last = std::min(m - 1, j + lower), lastc = std::min(m - 1, j + lower + upper);
    double n2 = 0;
    for (int i = j; i <= last; ++i) n2 += std::norm(at(i, j));
    if (n2 == 0) continue;
    const cdouble alpha = at(j, j), sgn = std::abs(alpha) == 0 ? 1.0 : alpha / std::abs(alpha);
    const cdouble beta = -sgn * std::sqrt(n2), tau = (beta - alpha) / beta;
    std::vector<cdouble> v((size_t)(last - j + 1));
    v[0] = 1;
    for (int i = j + 1; i <= last; ++i) v[(size_t)(i - j)] = at(i, j) / (alpha - beta);
    for (int col = j + 1; col <= lastc; ++col) {
      cdouble w = 0;
      for (int i = j; i <= last; ++i) w += std::conj(v[(size_t)(i - j)]) * at(i, col);
      w *= tau;
      for (int i = j; i <= last; ++i) at(i, col) -= v[(size_t)(i - j)] * w;
    }
    at(j, j) = beta;
  }
  std::vector<std::vector<cdouble>> r((size_t)lower + 1, std::vector<cdouble>((size_t)m));
  for (int j = 0; j < m; ++j)
    for (int k = 0; k <= lower; ++k)
      if (j + k < m) r[(size_t)k][(size_t)j] = at(j, j + k);
  return r;
}

// LU with partial pivoting of a tridiagonal matrix, returning 1/u_ii and
// u_{i,i+1} at row i, as gtsvInterleavedBatch's algo 1 leaves them.
static void lu_factors(std::vector<cdouble> dl, std::vector<cdouble> d, std::vector<cdouble> du,
                       std::vector<cdouble>* inv, std::vector<cdouble>* up) {
  const size_t m = d.size();
  inv->assign(m, 0);
  up->assign(m, 0);
  du[m - 1] = 0;
  for (size_t i = 0; i + 1 < m; ++i) {
    const cdouble l = dl[i + 1];
    auto mag = [](cdouble v) { return std::fabs(v.real()) + std::fabs(v.imag()); };
    if (mag(d[i]) >= mag(l)) {
      d[i + 1] -= l / d[i] * du[i];
    } else {
      const cdouble f = d[i] / l, t = d[i + 1];
      d[i] = l;
      d[i + 1] = du[i] - f * t;
      if (i + 2 < m) du[i + 1] = -f * du[i + 1];
      du[i] = t;
    }
    (*inv)[i] = 1.0 / d[i];
    (*up)[i] = du[i];
  }
}

static double rel(cdouble got, cdouble want) { return std::abs(got - want) / std::max(1.0, std::abs(want)); }

/* ---- gtsv2, gtsv2_nopivot and gtsv2StridedBatch on well-conditioned systems ---- */

template <class T> void solves() {
  using A = Api<T>;
  const char* P = Num<T>::name;
  char what[160];
  const int m = 37, n = 3, ldb = m + 2;
  std::vector<T> dl(m), d(m), du(m), B((size_t)ldb * n);
  for (int i = 0; i < m; ++i) {
    dl[i] = entry<T>(i, -1, 1);
    d[i] = entry<T>(i, 0, 1);
    du[i] = entry<T>(i, 1, 1);
  }
  for (size_t i = 0; i < B.size(); ++i) B[i] = Num<T>::put(cdouble(hash01((int)i, 5) - 0.3, Num<T>::cplx ? hash01((int)i, 6) : 0));
  // What a solver reads: dl[0] and du[m-1] are outside the matrix and must not
  // count. Garbage there checks that they are ignored.
  std::vector<T> dlg = dl, dug = du;
  dlg[0] = Num<T>::put(cdouble(9.0, 0));
  dug[m - 1] = Num<T>::put(cdouble(-7.0, 0));
  std::vector<std::vector<cdouble>> diag{widen(dl, 0, m), widen(d, 0, m), widen(du, 0, m)};
  diag[0][0] = 0;
  diag[2][(size_t)m - 1] = 0;
  T *ddl = upload(dlg), *dd = upload(d), *ddu = upload(dug);
  for (int pivot = 1; pivot >= 0; --pivot) {
    T* dB = upload(B);
    size_t bytes = 0;
    const int s0 = pivot ? A::gtsv2_bs(h, m, n, ddl, dd, ddu, dB, ldb, &bytes) : A::nopiv_bs(h, m, n, ddl, dd, ddu, dB, ldb, &bytes);
    const int s1 = pivot ? A::gtsv2(h, m, n, ddl, dd, ddu, dB, ldb, work) : A::nopiv(h, m, n, ddl, dd, ddu, dB, ldb, work);
    std::snprintf(what, sizeof what, "%sgtsv2%s: buffer size and solve succeed", P, pivot ? "" : "_nopivot");
    check(s0 == 0 && s1 == 0 && bytes > 0, what);
    const auto X = download(dB, B.size());
    double worst = 0;
    for (int j = 0; j < n; ++j)
      worst = std::max(worst, residual(diag, 1, widen(X, (size_t)j * ldb, m), widen(B, (size_t)j * ldb, m)));
    std::snprintf(what, sizeof what, "%sgtsv2%s: %d right-hand sides solved, ldb > m", P, pivot ? "" : "_nopivot", n);
    check(worst < 10 * Num<T>::tol, what, worst);
    bool padding = true;
    for (int j = 0; j < n; ++j)
      for (int i = m; i < ldb; ++i) padding = padding && std::memcmp(&X[(size_t)j * ldb + i], &B[(size_t)j * ldb + i], sizeof(T)) == 0;
    std::snprintf(what, sizeof what, "%sgtsv2%s: B's rows past m untouched", P, pivot ? "" : "_nopivot");
    check(padding, what);
    cudaFree(dB);
  }
  {
    const auto a = download(ddl, m), b = download(dd, m), c = download(ddu, m);
    std::snprintf(what, sizeof what, "%sgtsv2: dl, d and du left as they were", P);
    check(!std::memcmp(a.data(), dlg.data(), m * sizeof(T)) && !std::memcmp(b.data(), d.data(), m * sizeof(T)) &&
              !std::memcmp(c.data(), dug.data(), m * sizeof(T)),
          what);
  }
  // A strided batch of three systems, one right-hand side each.
  const int count = 3, stride = m + 5;
  std::vector<T> sl((size_t)stride * count), sd(sl.size()), su(sl.size()), sx(sl.size());
  for (int k = 0; k < count; ++k)
    for (int i = 0; i < m; ++i) {
      sl[(size_t)k * stride + i] = entry<T>(i, -1, 10 + k);
      sd[(size_t)k * stride + i] = entry<T>(i, 0, 10 + k);
      su[(size_t)k * stride + i] = entry<T>(i, 1, 10 + k);
      sx[(size_t)k * stride + i] = Num<T>::put(cdouble(1.0 + i % 4, Num<T>::cplx ? 0.5 : 0));
    }
  T *a = upload(sl), *b = upload(sd), *c = upload(su), *x = upload(sx);
  size_t bytes = 0;
  const int s0 = A::strided_bs(h, m, a, b, c, x, count, stride, &bytes), s1 = A::strided(h, m, a, b, c, x, count, stride, work);
  std::snprintf(what, sizeof what, "%sgtsv2StridedBatch: buffer size and solve succeed", P);
  check(s0 == 0 && s1 == 0 && bytes > 0, what);
  const auto X = download(x, sx.size());
  double worst = 0;
  for (int k = 0; k < count; ++k) {
    std::vector<std::vector<cdouble>> dg{widen(sl, (size_t)k * stride, m), widen(sd, (size_t)k * stride, m),
                                         widen(su, (size_t)k * stride, m)};
    dg[0][0] = 0;
    dg[2][(size_t)m - 1] = 0;
    worst = std::max(worst, residual(dg, 1, widen(X, (size_t)k * stride, m), widen(sx, (size_t)k * stride, m)));
  }
  std::snprintf(what, sizeof what, "%sgtsv2StridedBatch: %d systems solved, stride > m", P, count);
  check(worst < 10 * Num<T>::tol, what, worst);
  cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(x);
  cudaFree(ddl); cudaFree(dd); cudaFree(ddu);
}

/* ---- the interleaved solvers ---- */

template <class T> void interleaved() {
  using A = Api<T>;
  const char* P = Num<T>::name;
  char what[200];
  const int m = 9, count = 4;
  const size_t total = (size_t)m * count;
  std::vector<T> dl(total), d(total), du(total), x(total);
  for (int i = 0; i < m; ++i)
    for (int k = 0; k < count; ++k) {
      dl[(size_t)i * count + k] = entry<T>(i, -1, 20 + k);
      d[(size_t)i * count + k] = entry<T>(i, 0, 20 + k);
      du[(size_t)i * count + k] = entry<T>(i, 1, 20 + k);
      x[(size_t)i * count + k] = Num<T>::put(cdouble(0.5 + i, Num<T>::cplx ? -0.25 * k : 0));
    }
  for (int algo = 0; algo < 3; ++algo) {
    T *a = upload(dl), *b = upload(d), *c = upload(du), *xx = upload(x);
    size_t bytes = 0;
    const int s0 = A::inter_bs(h, algo, m, a, b, c, xx, count, &bytes), s1 = A::inter(h, algo, m, a, b, c, xx, count, work);
    std::snprintf(what, sizeof what, "%sgtsvInterleavedBatch algo %d: succeeds, asking for %s bytes", P, algo,
                  algo == 0 ? "128" : "m * batchCount elements' worth of");
    check(s0 == 0 && s1 == 0 && bytes == (algo == 0 ? 128 : total * sizeof(T)), what, (double)bytes);
    const auto X = download(xx, total), Dn = download(b, total), Un = download(c, total), Ln = download(a, total);
    double worst = 0, over = 0;
    bool d_same = true, l_same = !std::memcmp(Ln.data(), dl.data(), total * sizeof(T));
    for (int k = 0; k < count; ++k) {
      std::vector<std::vector<cdouble>> dg{widen(dl, k, m, count), widen(d, k, m, count), widen(du, k, m, count)};
      dg[0][0] = 0;
      dg[2][(size_t)m - 1] = 0;
      worst = std::max(worst, residual(dg, 1, widen(X, k, m, count), widen(x, k, m, count)));
      // What each algorithm leaves behind in d and du.
      if (algo == 0) {
        cdouble cp = 0;
        for (int i = 0; i + 1 < m; ++i) {
          cp = dg[2][(size_t)i] / (dg[1][(size_t)i] - dg[0][(size_t)i] * cp);
          over = std::max(over, rel(Num<T>::get(Un[(size_t)i * count + k]), cp));
        }
        d_same = d_same && widen(Dn, k, m, count) == widen(d, k, m, count);
      } else {
        std::vector<cdouble> want_d, want_u;
        if (algo == 1) {
          lu_factors(dg[0], dg[1], dg[2], &want_d, &want_u);
        } else {
          const auto r = householder_r(dg, 1, 1);
          want_d = r[0];
          want_u = r[1];
        }
        for (int i = 0; i < m; ++i) {
          const bool written = m >= 5 && i < m - 2;
          const cdouble wd = written ? want_d[(size_t)i] : dg[1][(size_t)i];
          const cdouble wu = written ? want_u[(size_t)i] : Num<T>::get(du[(size_t)i * count + k]);
          over = std::max(over, rel(Num<T>::get(Dn[(size_t)i * count + k]), wd));
          over = std::max(over, rel(Num<T>::get(Un[(size_t)i * count + k]), wu));
        }
      }
    }
    std::snprintf(what, sizeof what, "%sgtsvInterleavedBatch algo %d: %d systems solved", P, algo, count);
    check(worst < 10 * Num<T>::tol, what, worst);
    if (algo == 0)
      std::snprintf(what, sizeof what, "%sgtsvInterleavedBatch algo 0: du holds Thomas's c'_i, d and dl untouched", P);
    else
      std::snprintf(what, sizeof what, "%sgtsvInterleavedBatch algo %d: d and du hold %s for i < m - 2, dl untouched", P,
                    algo, algo == 1 ? "1/u_ii and u_{i,i+1}" : "R's diagonal and superdiagonal");
    check(over < 50 * Num<T>::tol && d_same && l_same, what, over);
    cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(xx);
  }

  // The pentadiagonal QR.
  const int pm = 11, pc = 3;
  const size_t pt = (size_t)pm * pc;
  std::vector<T> ps(pt), pl(pt), pd(pt), pu(pt), pw(pt), px(pt);
  for (int i = 0; i < pm; ++i)
    for (int k = 0; k < pc; ++k) {
      const size_t e = (size_t)i * pc + k;
      ps[e] = entry<T>(i, -2, 30 + k);
      pl[e] = entry<T>(i, -1, 30 + k);
      pd[e] = entry<T>(i, 0, 30 + k);
      pu[e] = entry<T>(i, 1, 30 + k);
      pw[e] = entry<T>(i, 2, 30 + k);
      px[e] = Num<T>::put(cdouble(1.0 - 0.1 * i, Num<T>::cplx ? 0.2 * k : 0));
    }
  T *a = upload(ps), *b = upload(pl), *c = upload(pd), *dd = upload(pu), *e = upload(pw), *xx = upload(px);
  size_t bytes = 0;
  const int s0 = A::gpsv_bs(h, 0, pm, a, b, c, dd, e, xx, pc, &bytes), s1 = A::gpsv(h, 0, pm, a, b, c, dd, e, xx, pc, work);
  std::snprintf(what, sizeof what, "%sgpsvInterleavedBatch: succeeds, asking for 2 m batchCount elements' worth", P);
  check(s0 == 0 && s1 == 0 && bytes == 2 * pt * sizeof(T), what, (double)bytes);
  const auto X = download(xx, pt), Dn = download(c, pt), Un = download(dd, pt), Wn = download(e, pt);
  double worst = 0, over = 0;
  for (int k = 0; k < pc; ++k) {
    std::vector<std::vector<cdouble>> dg{widen(ps, k, pm, pc), widen(pl, k, pm, pc), widen(pd, k, pm, pc),
                                         widen(pu, k, pm, pc), widen(pw, k, pm, pc)};
    dg[0][0] = dg[0][1] = dg[1][0] = 0;
    dg[3][(size_t)pm - 1] = dg[4][(size_t)pm - 1] = dg[4][(size_t)pm - 2] = 0;
    worst = std::max(worst, residual(dg, 2, widen(X, k, pm, pc), widen(px, k, pm, pc)));
    const auto r = householder_r(dg, 2, 2);
    for (int i = 0; i < pm; ++i) {
      const bool written = pm >= 5 && i < pm - 3;
      over = std::max(over, rel(Num<T>::get(Dn[(size_t)i * pc + k]), written ? r[0][(size_t)i] : Num<T>::get(pd[(size_t)i * pc + k])));
      over = std::max(over, rel(Num<T>::get(Un[(size_t)i * pc + k]), written ? r[1][(size_t)i] : Num<T>::get(pu[(size_t)i * pc + k])));
      over = std::max(over, rel(Num<T>::get(Wn[(size_t)i * pc + k]), written ? r[2][(size_t)i] : Num<T>::get(pw[(size_t)i * pc + k])));
    }
  }
  std::snprintf(what, sizeof what, "%sgpsvInterleavedBatch: %d systems solved", P, pc);
  check(worst < 10 * Num<T>::tol, what, worst);
  std::snprintf(what, sizeof what, "%sgpsvInterleavedBatch: d, du and dw hold R's first three diagonals for i < m - 3", P);
  check(over < 50 * Num<T>::tol, what, over);
  cudaFree(a); cudaFree(b); cudaFree(c); cudaFree(dd); cudaFree(e); cudaFree(xx);
}

/* ---- pivoting: what each algorithm makes of a zero pivot ---- */

template <class T> void pivoting() {
  using A = Api<T>;
  const char* P = Num<T>::name;
  char what[200];
  for (int m : {7, 512, 513, 2048, 2049}) {
    std::vector<T> dl(m), d(m), du(m), b(m);
    for (int i = 0; i < m; ++i) {
      dl[i] = entry<T>(i, -1, 40);
      d[i] = entry<T>(i, 0, 40);
      du[i] = entry<T>(i, 1, 40);
      b[i] = Num<T>::put(cdouble(1.0 + 0.37 * (i % 11), 0));
    }
    d[0] = Num<T>::put(0.0);  // needs a row interchange
    std::vector<std::vector<cdouble>> dg{widen(dl, 0, m), widen(d, 0, m), widen(du, 0, m)};
    dg[0][0] = 0;
    dg[2][(size_t)m - 1] = 0;
    T *a = upload(dl), *dd = upload(d), *c = upload(du);
    T* x = upload(b);
    A::gtsv2(h, m, 1, a, dd, c, x, m, work);
    const auto pivoted = download(x, m);
    double res = residual(dg, 1, widen(pivoted, 0, m), widen(b, 0, m));
    std::snprintf(what, sizeof what, "%sgtsv2, m = %d, d[0] = 0: solved by pivoting", P, m);
    check(res < 10 * Num<T>::tol, what, res);
    cudaMemcpy(x, b.data(), m * sizeof(T), cudaMemcpyHostToDevice);
    A::nopiv(h, m, 1, a, dd, c, x, m, work);
    const auto np = download(x, m);
    int odd_bad = 0, even_bad = 0;
    double even_err = 0;
    for (int i = 0; i < m; ++i) {
      const cdouble v = Num<T>::get(np[i]);
      if (!finite(v)) (i % 2 ? odd_bad : even_bad)++;
      else if (i % 2 == 0) even_err = std::max(even_err, rel(v, Num<T>::get(pivoted[i])));
    }
    if (m <= 2048) {  // PCR
      std::snprintf(what, sizeof what,
                    "%sgtsv2_nopivot, m = %d, d[0] = 0: PCR spoils every odd unknown and gets every even one", P, m);
      check(odd_bad == m / 2 && even_bad == 0 && even_err < 1e3 * Num<T>::tol, what, even_err);
    } else {
      std::snprintf(what, sizeof what, "%sgtsv2_nopivot, m = %d, d[0] = 0: cyclic reduction first spoils everything", P, m);
      check(odd_bad + even_bad == m, what);
    }
    cudaMemcpy(x, b.data(), m * sizeof(T), cudaMemcpyHostToDevice);
    A::strided(h, m, a, dd, c, x, 1, m, work);
    const auto st = download(x, m);
    int s_odd = 0, s_even = 0;
    for (int i = 0; i < m; ++i)
      if (!finite(Num<T>::get(st[i]))) (i % 2 ? s_odd : s_even)++;
    std::snprintf(what, sizeof what, "%sgtsv2StridedBatch, m = %d, d[0] = 0: %s", P, m,
                  m <= 512 ? "PCR spoils every odd unknown only" : "cyclic reduction first spoils everything");
    check(m <= 512 ? (s_odd == m / 2 && s_even == 0) : s_odd + s_even == m, what);
    if (m == 7) {
      for (int algo = 0; algo < 3; ++algo) {
        T *ia = upload(dl), *ib = upload(d), *ic = upload(du), *ix = upload(b);
        A::inter(h, algo, m, ia, ib, ic, ix, 1, work);
        const auto r = download(ix, m);
        int bad = 0;
        for (int i = 0; i < m; ++i) bad += !finite(Num<T>::get(r[i]));
        if (algo == 0) {
          std::snprintf(what, sizeof what, "%sgtsvInterleavedBatch Thomas, d[0] = 0: every unknown NaN", P);
          check(bad == m, what);
        } else {
          const double rr = residual(dg, 1, widen(r, 0, m), widen(b, 0, m));
          std::snprintf(what, sizeof what, "%sgtsvInterleavedBatch %s, d[0] = 0: solved", P, algo == 1 ? "LU" : "QR");
          check(bad == 0 && rr < 10 * Num<T>::tol, what, rr);
        }
        cudaFree(ia); cudaFree(ib); cudaFree(ic); cudaFree(ix);
      }
    }
    cudaFree(a); cudaFree(dd); cudaFree(c); cudaFree(x);
  }
  // Singular (rows 0 and 1 equal): gtsv2's answer is NaN throughout.
  std::vector<T> dl{Num<T>::put(0.0), Num<T>::put(1.0), Num<T>::put(0.0)}, d(3, Num<T>::put(1.0)),
      du{Num<T>::put(1.0), Num<T>::put(0.0), Num<T>::put(0.0)}, b{Num<T>::put(1.0), Num<T>::put(2.0), Num<T>::put(3.0)};
  T *a = upload(dl), *dd = upload(d), *c = upload(du), *x = upload(b);
  A::gtsv2(h, 3, 1, a, dd, c, x, 3, work);
  const auto r = download(x, 3);
  bool all_nan = true;
  for (const auto& v : r) all_nan = all_nan && std::isnan(Num<T>::get(v).real());
  std::snprintf(what, sizeof what, "%sgtsv2 on a singular system: NaN throughout", P);
  check(all_nan, what);
  cudaFree(a); cudaFree(dd); cudaFree(c); cudaFree(x);
}

/* ---- refusals ---- */

template <class T> void refusals() {
  using A = Api<T>;
  const char* P = Num<T>::name;
  char what[160];
  std::vector<T> z(64, Num<T>::put(1.0));
  T* p = upload(z);
  size_t bytes = 0;
  auto ex = [&](int got, int want, const char* fmt) {
    std::snprintf(what, sizeof what, fmt, P);
    expect(got, want, what);
  };
  ex(A::gtsv2(h, 2, 1, p, p, p, p, 2, work), 3, "%sgtsv2 m = 2");
  ex(A::nopiv(h, 2, 1, p, p, p, p, 2, work), 3, "%sgtsv2_nopivot m = 2");
  ex(A::gtsv2(h, 4, -1, p, p, p, p, 4, work), 3, "%sgtsv2 n < 0");
  ex(A::gtsv2(h, 4, 1, p, p, p, p, 3, work), 3, "%sgtsv2 ldb < m");
  ex(A::nopiv(h, 4, 1, p, p, p, p, 3, work), 3, "%sgtsv2_nopivot ldb < m");
  ex(A::gtsv2(h, 4, 0, p, p, p, p, 4, work), 0, "%sgtsv2 n = 0 does nothing");
  ex(A::gtsv2_bs(h, 2, 1, p, p, p, p, 1, &bytes), 0, "%sgtsv2_bufferSizeExt checks nothing (m = 2, ldb < m)");
  ex(A::strided(h, 2, p, p, p, p, 1, 2, work), 3, "%sgtsv2StridedBatch m = 2");
  ex(A::strided(h, 4, p, p, p, p, 0, 4, work), 3, "%sgtsv2StridedBatch batchCount = 0");
  ex(A::strided(h, 4, p, p, p, p, 2, 3, work), 3, "%sgtsv2StridedBatch batchStride < m");
  ex(A::strided_bs(h, 4, p, p, p, p, 2, 3, &bytes), 3, "%sgtsv2StridedBatch_bufferSizeExt batchStride < m");
  ex(A::inter(h, 3, 4, p, p, p, p, 1, work), 3, "%sgtsvInterleavedBatch algo 3");
  ex(A::inter_bs(h, 3, 4, p, p, p, p, 1, &bytes), 3, "%sgtsvInterleavedBatch_bufferSizeExt algo 3");
  ex(A::inter(h, 0, -1, p, p, p, p, 1, work), 3, "%sgtsvInterleavedBatch m < 0");
  ex(A::inter(h, 0, 4, p, p, p, p, -1, work), 3, "%sgtsvInterleavedBatch batchCount < 0");
  ex(A::inter(h, 1, 0, p, p, p, p, 1, work), 0, "%sgtsvInterleavedBatch m = 0 does nothing");
  ex(A::gpsv(h, 0, -1, p, p, p, p, p, p, 1, work), 3, "%sgpsvInterleavedBatch m < 0");
  ex(A::gpsv(h, 0, 4, p, p, p, p, p, p, -1, work), 6, "%sgpsvInterleavedBatch batchCount < 0");
  ex(A::gpsv_bs(h, 5, 4, p, p, p, p, p, p, 1, &bytes), 0, "%sgpsvInterleavedBatch_bufferSizeExt takes any algo");
  // NVIDIA documents algo 0 only; its 13.0 takes another, does nothing and
  // reports success. The simulator refuses it.
  {
    std::vector<T> x0 = z;
    T* x = upload(x0);
    const int st = A::gpsv(h, 1, 4, p, p, p, p, p, x, 1, work);
    const auto after = download(x, 4);
    const bool untouched = !std::memcmp(after.data(), x0.data(), 4 * sizeof(T));
    std::snprintf(what, sizeof what, "%sgpsvInterleavedBatch algo 1: %s (status %d)", P,
                  on_sim ? "NOT_SUPPORTED" : "success, x untouched (hardware)", st);
    check(on_sim ? st == 10 : (st == 0 && untouched), what);
    cudaFree(x);
  }
  ex(A::gtsv2(nullptr, 4, 1, p, p, p, p, 4, work), 1, "%sgtsv2 without a handle");
  cudaFree(p);
}

int main() {
  if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS) {
    std::printf("FAIL cusparseCreate\n");
    return 1;
  }
  cudaMalloc(&work, 1 << 22);
  solves<float>();
  solves<double>();
  solves<cuComplex>();
  solves<cuDoubleComplex>();
  interleaved<float>();
  interleaved<double>();
  interleaved<cuComplex>();
  interleaved<cuDoubleComplex>();
  pivoting<float>();
  pivoting<double>();
  pivoting<cuComplex>();
  pivoting<cuDoubleComplex>();
  refusals<float>();
  refusals<double>();
  refusals<cuComplex>();
  refusals<cuDoubleComplex>();
  cudaFree(work);
  cusparseDestroy(h);
  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
