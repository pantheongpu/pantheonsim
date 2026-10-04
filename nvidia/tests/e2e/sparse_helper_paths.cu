// cuSPARSE's legacy helpers and the rest of the generic API, checked against
// host arithmetic and against what an RTX 3060's cuSPARSE 13.0 answers:
//
//   pruning        pruneDense2csr and pruneCsr2csr (|a| > threshold, read host
//                  or device, one-based output) and their ByPercentage forms
//                  (threshold = the ceil(N p / 100)-th smallest magnitude), in
//                  half, single and double
//   nnz            per row and per column; NaN counts, -0 does not
//   compression    nnz_compress and csr2csr_compress (|a| > tol). NVIDIA's
//                  csr2csr_compress keeps a > tol for real values, leaving
//                  the slots of negative entries unwritten, so only
//                  non-negative real values are compared
//   unsorted CSR   csru2csr, csr2csru, and the protocol their info needs
//   general BSR    gebsr2gebsr in both block directions and several block
//                  shapes, gebsr2gebsc numeric and symbolic
//   csrcolor       a proper coloring and a reordering grouped by color
//                  (NVIDIA's colors are randomized, so only those properties)
//   SpGEMMreuse    the whole protocol, recomputed with new values and beta;
//                  SpGEMM's product count and ALG3 memory estimate
//   SpMMOp         refused: its operators are LTO-IR
//   SpSV/SpSM      updateMatrix, general and diagonal
//   the rest       the logger's refusals, the CSC sort
//
// CUDA 12.0's header has no SpSV/SpSM updateMatrix, so those are looked up
// by name in the cuSPARSE the program runs against.
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <set>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;
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

static cusparseHandle_t h;
static cusparseMatDescr_t md, md1;  // general, base zero and base one
static void* work;

/* ---- pruning ---- */

template <class T> struct Prune;
template <> struct Prune<float> {
  static constexpr auto dnnz = cusparseSpruneDense2csrNnz;
  static constexpr auto dense = cusparseSpruneDense2csr;
  static constexpr auto dbs = cusparseSpruneDense2csr_bufferSizeExt;
  static constexpr auto pnnz = cusparseSpruneDense2csrNnzByPercentage;
  static constexpr auto pdense = cusparseSpruneDense2csrByPercentage;
  static constexpr auto cnnz = cusparseSpruneCsr2csrNnz;
  static constexpr auto csr = cusparseSpruneCsr2csr;
  static constexpr auto cpnnz = cusparseSpruneCsr2csrNnzByPercentage;
  static constexpr auto cpcsr = cusparseSpruneCsr2csrByPercentage;
  static float put(double v) { return (float)v; }
  static double get(float v) { return v; }
  static constexpr const char* name = "S";
};
template <> struct Prune<double> {
  static constexpr auto dnnz = cusparseDpruneDense2csrNnz;
  static constexpr auto dense = cusparseDpruneDense2csr;
  static constexpr auto dbs = cusparseDpruneDense2csr_bufferSizeExt;
  static constexpr auto pnnz = cusparseDpruneDense2csrNnzByPercentage;
  static constexpr auto pdense = cusparseDpruneDense2csrByPercentage;
  static constexpr auto cnnz = cusparseDpruneCsr2csrNnz;
  static constexpr auto csr = cusparseDpruneCsr2csr;
  static constexpr auto cpnnz = cusparseDpruneCsr2csrNnzByPercentage;
  static constexpr auto cpcsr = cusparseDpruneCsr2csrByPercentage;
  static double put(double v) { return v; }
  static double get(double v) { return v; }
  static constexpr const char* name = "D";
};
template <> struct Prune<__half> {
  static constexpr auto dnnz = cusparseHpruneDense2csrNnz;
  static constexpr auto dense = cusparseHpruneDense2csr;
  static constexpr auto dbs = cusparseHpruneDense2csr_bufferSizeExt;
  static constexpr auto pnnz = cusparseHpruneDense2csrNnzByPercentage;
  static constexpr auto pdense = cusparseHpruneDense2csrByPercentage;
  static constexpr auto cnnz = cusparseHpruneCsr2csrNnz;
  static constexpr auto csr = cusparseHpruneCsr2csr;
  static constexpr auto cpnnz = cusparseHpruneCsr2csrNnzByPercentage;
  static constexpr auto cpcsr = cusparseHpruneCsr2csrByPercentage;
  static __half put(double v) { return __float2half((float)v); }
  static double get(__half v) { return __half2float(v); }
  static constexpr const char* name = "H";
};

// The CSR (zero-based, columns ascending) of the entries of a dense m x n
// column-major matrix whose magnitude exceeds th.
struct HostCsr {
  std::vector<int> off{0}, col;
  std::vector<double> val;
};
static HostCsr keep(int m, int n, const std::vector<double>& a, double th) {
  HostCsr c;
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j)
      if (std::fabs(a[(size_t)j * m + i]) > th) {
        c.col.push_back(j);
        c.val.push_back(a[(size_t)j * m + i]);
      }
    c.off.push_back((int)c.col.size());
  }
  return c;
}
static double percentile(std::vector<double> mags, double pct) {
  std::sort(mags.begin(), mags.end());
  long long pos = (long long)std::ceil(mags.size() * pct / 100.0) - 1;
  pos = std::max(0LL, std::min(pos, (long long)mags.size() - 1));
  return mags[(size_t)pos];
}

template <class T> void pruning() {
  using P = Prune<T>;
  char what[200];
  const int m = 4, n = 5;
  const std::vector<double> a{0.5, -2, 0, 1, 3, 0, -0.5, 0, 0, 1, 2, -3, -1, 0.25, 0, 4, 2, -2, 1, 0};
  std::vector<T> av(a.size());
  for (size_t i = 0; i < a.size(); ++i) av[i] = P::put(a[i]);
  T* dA = upload(av);
  auto run_dense = [&](double th, bool device, cusparseMatDescr_t dc, const char* label) {
    const int base = dc == md1 ? 1 : 0;
    const T thr = P::put(th);
    T* dth = upload(std::vector<T>{thr});
    int* off;
    cudaMalloc(&off, (m + 1) * sizeof(int));
    int nnz = -1, *dnnz;
    cudaMalloc(&dnnz, sizeof(int));
    size_t bytes = 0;
    P::dbs(h, m, n, dA, m, &thr, dc, nullptr, off, nullptr, &bytes);
    if (device) cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_DEVICE);
    const int s1 = P::dnnz(h, m, n, dA, m, device ? dth : &thr, dc, off, device ? dnnz : &nnz, work);
    if (device) nnz = download(dnnz, 1)[0];
    int* col;
    T* val;
    cudaMalloc(&col, std::max(nnz, 1) * sizeof(int));
    cudaMalloc(&val, std::max(nnz, 1) * sizeof(T));
    const int s2 = P::dense(h, m, n, dA, m, device ? dth : &thr, dc, val, off, col, work);
    cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST);
    const HostCsr want = keep(m, n, a, th);
    bool ok = s1 == 0 && s2 == 0 && nnz == (int)want.col.size();
    if (ok) {
      const auto o = download(off, m + 1), c = download(col, nnz);
      const auto v = download(val, nnz);
      for (int i = 0; i <= m; ++i) ok = ok && o[i] == want.off[i] + base;
      for (int k = 0; k < nnz; ++k) ok = ok && c[k] == want.col[k] + base && P::get(v[k]) == P::get(P::put(want.val[k]));
    }
    std::snprintf(what, sizeof what, "%spruneDense2csr threshold %g%s: keeps |a| > threshold (%d entries)", P::name, th,
                  label, nnz);
    check(ok, what);
    cudaFree(dth); cudaFree(off); cudaFree(dnnz); cudaFree(col); cudaFree(val);
  };
  run_dense(0.5, false, md, "");
  run_dense(1.0, false, md, "");
  run_dense(0.0, false, md, "");
  run_dense(0.5, true, md, ", read from the device");
  run_dense(0.5, false, md1, ", one-based output");
  {
    int* off;
    cudaMalloc(&off, (m + 1) * sizeof(int));
    int nnz;
    const T thr = P::put(0.5);
    std::snprintf(what, sizeof what, "%spruneDense2csrNnz lda < m", P::name);
    expect(P::dnnz(h, m, n, dA, m - 2, &thr, md, off, &nnz, work), 3, what);
    cudaFree(off);
  }
  // By percentage of the m * n magnitudes.
  pruneInfo_t info;
  cusparseCreatePruneInfo(&info);
  std::vector<double> mags;
  for (double v : a) mags.push_back(std::fabs(v));
  for (double pct : {0.0, 25.0, 50.0, 55.0, 90.0, 100.0}) {
    int* off;
    cudaMalloc(&off, (m + 1) * sizeof(int));
    int nnz = -1;
    const int s1 = P::pnnz(h, m, n, dA, m, (float)pct, md, off, &nnz, info, work);
    int* col;
    T* val;
    cudaMalloc(&col, std::max(nnz, 1) * sizeof(int));
    cudaMalloc(&val, std::max(nnz, 1) * sizeof(T));
    const int s2 = P::pdense(h, m, n, dA, m, (float)pct, md, val, off, col, info, work);
    const HostCsr want = keep(m, n, a, percentile(mags, pct));
    bool ok = s1 == 0 && s2 == 0 && nnz == (int)want.col.size();
    if (ok && nnz > 0) ok = download(col, nnz) == want.col;
    std::snprintf(what, sizeof what, "%spruneDense2csrByPercentage %g%%: %d entries left", P::name, pct, nnz);
    check(ok, what);
    cudaFree(off); cudaFree(col); cudaFree(val);
  }
  {
    int* off;
    cudaMalloc(&off, (m + 1) * sizeof(int));
    int nnz;
    std::snprintf(what, sizeof what, "%spruneDense2csrNnzByPercentage 101%%", P::name);
    expect(P::pnnz(h, m, n, dA, m, 101.0f, md, off, &nnz, info, work), 3, what);
    std::snprintf(what, sizeof what, "%spruneDense2csrNnzByPercentage -1%%", P::name);
    expect(P::pnnz(h, m, n, dA, m, -1.0f, md, off, &nnz, info, work), 3, what);
    cudaFree(off);
  }
  // CSR to CSR, by threshold and by percentage of the nnzA magnitudes.
  const std::vector<int> ao{0, 3, 5, 5, 8}, ac{0, 2, 4, 1, 3, 0, 2, 4};
  const std::vector<double> avv{0.5, -1, 2, 0, -3, 1, 0.75, -0.75};
  std::vector<T> cv(avv.size());
  for (size_t i = 0; i < avv.size(); ++i) cv[i] = P::put(avv[i]);
  int *dao = upload(ao), *dac = upload(ac);
  T* dav = upload(cv);
  std::vector<double> cm;
  for (double v : avv) cm.push_back(std::fabs(v));
  for (int byp = 0; byp < 2; ++byp)
    for (double t : byp ? std::vector<double>{0, 20, 50, 60, 100} : std::vector<double>{0.5, 0.75, 0}) {
      const double th = byp ? percentile(cm, t) : t;
      int* off;
      cudaMalloc(&off, 5 * sizeof(int));
      int nnz = -1;
      const T thr = P::put(t);
      const int s1 = byp ? P::cpnnz(h, 4, 5, 8, md, dav, dao, dac, (float)t, md, off, &nnz, info, work)
                         : P::cnnz(h, 4, 5, 8, md, dav, dao, dac, &thr, md, off, &nnz, work);
      int* col;
      T* val;
      cudaMalloc(&col, std::max(nnz, 1) * sizeof(int));
      cudaMalloc(&val, std::max(nnz, 1) * sizeof(T));
      const int s2 = byp ? P::cpcsr(h, 4, 5, 8, md, dav, dao, dac, (float)t, md, val, off, col, info, work)
                         : P::csr(h, 4, 5, 8, md, dav, dao, dac, &thr, md, val, off, col, work);
      std::vector<int> wc, wo{0};
      for (int i = 0; i < 4; ++i) {
        for (int k = ao[i]; k < ao[i + 1]; ++k)
          if (std::fabs(avv[k]) > th) wc.push_back(ac[k]);
        wo.push_back((int)wc.size());
      }
      bool ok = s1 == 0 && s2 == 0 && nnz == (int)wc.size() && download(off, 5) == wo;
      if (ok && nnz > 0) ok = download(col, nnz) == wc;
      std::snprintf(what, sizeof what, "%spruneCsr2csr%s %g: %d entries left", P::name, byp ? "ByPercentage" : "", t, nnz);
      check(ok, what);
      cudaFree(off); cudaFree(col); cudaFree(val);
    }
  cusparseDestroyPruneInfo(info);
  cudaFree(dA); cudaFree(dao); cudaFree(dac); cudaFree(dav);
}

/* ---- nnz and compression ---- */

static void counting() {
  const int m = 4, n = 5;
  std::vector<float> a{0.5f, -2, 0, 1, 3, 0, -0.5f, 0, 0, 1, 2, -3, -1, 0.25f, 0, 4, 2, -2, 1, 0};
  a[1] = NAN;   // counts
  a[2] = -0.0f; // does not
  float* dA = upload(a);
  int *per, total = -1;
  cudaMalloc(&per, 8 * sizeof(int));
  check(cusparseSnnz(h, CUSPARSE_DIRECTION_ROW, m, n, md, dA, m, per, &total) == 0 && total == 14 &&
            download(per, m) == std::vector<int>{4, 4, 3, 3},
        "Snnz by row: NaN counts, -0 does not");
  check(cusparseSnnz(h, CUSPARSE_DIRECTION_COLUMN, m, n, md, dA, m, per, &total) == 0 && total == 14 &&
            download(per, n) == std::vector<int>{3, 2, 3, 3, 3},
        "Snnz by column");
  expect(cusparseSnnz(h, CUSPARSE_DIRECTION_ROW, m, n, md, dA, 2, per, &total), 3, "Snnz lda < m");
  expect(cusparseSnnz(h, (cusparseDirection_t)5, m, n, md, dA, m, per, &total), 3, "Snnz bad direction");
  cusparseMatDescr_t sym;
  cusparseCreateMatDescr(&sym);
  cusparseSetMatType(sym, CUSPARSE_MATRIX_TYPE_SYMMETRIC);
  expect(cusparseSnnz(h, CUSPARSE_DIRECTION_ROW, m, n, sym, dA, m, per, &total), 8, "Snnz of a symmetric matrix");
  {
    std::vector<cuComplex> z(6, make_cuComplex(0, 0));
    z[1] = make_cuComplex(0, 1);
    z[4] = make_cuComplex(2, 0);
    cuComplex* dz = upload(z);
    check(cusparseCnnz(h, CUSPARSE_DIRECTION_ROW, 2, 3, md, dz, 2, per, &total) == 0 && total == 2 &&
              download(per, 2) == std::vector<int>{1, 1},
          "Cnnz: an imaginary entry counts");
  }
  // Compression, with non-negative values (see the top of the file).
  const std::vector<int> ro{0, 3, 5, 5, 8}, ci{0, 2, 4, 1, 3, 0, 2, 4};
  const std::vector<double> v{0.5, 1, 2, 0, 3, 1, 0.75, 0.25};
  int *dro = upload(ro), *dci = upload(ci);
  for (int dbl = 0; dbl < 2; ++dbl)
    for (double tol : {0.5, 0.75, 0.0}) {
      std::vector<int> wo{0}, wc, wper;
      std::vector<double> wv;
      for (int i = 0; i < 4; ++i) {
        for (int k = ro[i]; k < ro[i + 1]; ++k)
          if (v[k] > tol) {
            wc.push_back(ci[k]);
            wv.push_back(v[k]);
          }
        wo.push_back((int)wc.size());
        wper.push_back(wo.back() - wo[(size_t)i]);
      }
      int nnzC = -1, s1, s2;
      std::vector<int> gc, go;
      std::vector<double> gv;
      int *rc, *cc;
      cudaMalloc(&rc, 5 * sizeof(int));
      cudaMalloc(&cc, 8 * sizeof(int));
      if (dbl) {
        double* dv = upload(v);
        double* cv;
        cudaMalloc(&cv, 8 * sizeof(double));
        s1 = cusparseDnnz_compress(h, 4, md, dv, dro, per, &nnzC, tol);
        s2 = cusparseDcsr2csr_compress(h, 4, 5, md, dv, dci, dro, 8, per, cv, cc, rc, tol);
        const auto x = download(cv, std::max(nnzC, 0));
        gv.assign(x.begin(), x.end());
      } else {
        std::vector<float> fv(v.begin(), v.end());
        float* dv = upload(fv);
        float* cv;
        cudaMalloc(&cv, 8 * sizeof(float));
        s1 = cusparseSnnz_compress(h, 4, md, dv, dro, per, &nnzC, (float)tol);
        s2 = cusparseScsr2csr_compress(h, 4, 5, md, dv, dci, dro, 8, per, cv, cc, rc, (float)tol);
        const auto x = download(cv, std::max(nnzC, 0));
        gv.assign(x.begin(), x.end());
      }
      char what[160];
      std::snprintf(what, sizeof what, "%snnz_compress and csr2csr_compress tol %g: |a| > tol kept", dbl ? "D" : "S", tol);
      const bool ok = s1 == 0 && s2 == 0 && nnzC == (int)wc.size() && download(per, 4) == wper &&
                      download(rc, 5) == wo && download(cc, (size_t)std::max(nnzC, 0)) == wc && gv == wv;
      check(ok, what);
      cudaFree(rc); cudaFree(cc);
    }
  {
    std::vector<float> fv(v.begin(), v.end());
    float* dv = upload(fv);
    int nnzC;
    expect(cusparseSnnz_compress(h, 4, md, dv, dro, per, &nnzC, -1.0f), 3, "Snnz_compress negative tol");
  }
  // Complex: |a| against tol. (NVIDIA's nnz_compress reads only tol's real
  // part and its csr2csr_compress tol's magnitude, so a tol off the real axis
  // makes the two disagree there; this one is real.)
  {
    std::vector<cuComplex> z{{0.5f, 0}, {0, -1}, {2, 2}, {0, 0}, {-3, 0}, {0.2f, 0.1f}, {0.75f, 0}, {0, 0.9f}};
    cuComplex* dz = upload(z);
    int nnzC = -1;
    const int s1 = cusparseCnnz_compress(h, 4, md, dz, dro, per, &nnzC, make_cuComplex(0.6f, 0.0f));
    int *rc, *cc;
    cuComplex* cv;
    cudaMalloc(&rc, 5 * sizeof(int));
    cudaMalloc(&cc, 8 * sizeof(int));
    cudaMalloc(&cv, 8 * sizeof(cuComplex));
    const int s2 = cusparseCcsr2csr_compress(h, 4, 5, md, dz, dci, dro, 8, per, cv, cc, rc, make_cuComplex(0.6f, 0.0f));
    check(s1 == 0 && s2 == 0 && nnzC == 5 && download(per, 4) == std::vector<int>{2, 1, 0, 2} &&
              download(cc, 5) == std::vector<int>{2, 4, 3, 2, 4},
          "Cnnz_compress and Ccsr2csr_compress: |a| > tol");
  }
}

/* ---- unsorted CSR ---- */

template <class T, class F1, class F2, class F3>
void unsorted(const char* name, F1 bufsize, F2 to_sorted, F3 to_unsorted, T (*mk)(double)) {
  char what[160];
  const std::vector<int> ro{0, 3, 5, 5, 8}, ci{4, 0, 2, 3, 1, 4, 0, 2};
  std::vector<T> v(8);
  for (int i = 0; i < 8; ++i) v[i] = mk(1 + i);
  int *dro = upload(ro), *dci = upload(ci);
  T* dv = upload(v);
  csru2csrInfo_t info;
  cusparseCreateCsru2csrInfo(&info);
  std::snprintf(what, sizeof what, "%scsru2csr without csru2csr_bufferSizeExt first", name);
  expect(to_sorted(h, 4, 5, 8, md, dv, dro, dci, info, work), 3, what);
  std::snprintf(what, sizeof what, "%scsr2csru before any csru2csr", name);
  expect(to_unsorted(h, 4, 5, 8, md, dv, dro, dci, info, work), 3, what);
  size_t bytes = 0;
  const int s0 = bufsize(h, 4, 5, 8, dv, dro, dci, info, &bytes);
  const int s1 = to_sorted(h, 4, 5, 8, md, dv, dro, dci, info, work);
  const auto sc = download(dci, 8);
  const auto sv = download(dv, 8);
  bool ok = s0 == 0 && s1 == 0 && sc == std::vector<int>{0, 2, 4, 1, 3, 0, 2, 4};
  const int order[8] = {1, 2, 0, 4, 3, 6, 7, 5};  // where each sorted entry came from
  for (int k = 0; k < 8; ++k) ok = ok && !std::memcmp(&sv[k], &v[order[k]], sizeof(T));
  std::snprintf(what, sizeof what, "%scsru2csr sorts each row, values alongside", name);
  check(ok, what);
  std::vector<T> nv(8);
  for (int i = 0; i < 8; ++i) nv[i] = mk(10 * (i + 1));
  cudaMemcpy(dv, nv.data(), 8 * sizeof(T), cudaMemcpyHostToDevice);
  const int s2 = to_unsorted(h, 4, 5, 8, md, dv, dro, dci, info, work);
  const auto uc = download(dci, 8);
  const auto uv = download(dv, 8);
  ok = s2 == 0 && uc == ci;
  for (int k = 0; k < 8; ++k) ok = ok && !std::memcmp(&uv[order[k]], &nv[k], sizeof(T));
  std::snprintf(what, sizeof what, "%scsr2csru restores the order, carrying new values back", name);
  check(ok, what);
  csru2csrInfo_t other;
  cusparseCreateCsru2csrInfo(&other);
  bufsize(h, 4, 5, 4, dv, dro, dci, other, &bytes);
  std::snprintf(what, sizeof what, "%scsru2csr with nnz other than the info was sized for", name);
  expect(to_sorted(h, 4, 5, 8, md, dv, dro, dci, other, work), 3, what);
  cusparseDestroyCsru2csrInfo(info);
  cusparseDestroyCsru2csrInfo(other);
}
static float mk_s(double v) { return (float)v; }
static cuDoubleComplex mk_z(double v) { return make_cuDoubleComplex(v, -0.5 * v); }

/* ---- general BSR conversions ---- */

static void gebsr() {
  // A: 2 x 3 blocks of 2 x 3; block row 0 holds block columns 0 and 2, block
  // row 1 block column 1. One stored zero.
  const int mb = 2, nb = 3, rbd = 2, cbd = 3, nnzb = 3;
  const std::vector<int> ro{0, 2, 3}, ci{0, 2, 1};
  std::vector<float> v(18);
  for (int i = 0; i < 18; ++i) v[i] = 1.0f + i;
  v[4] = 0;
  int *dro = upload(ro), *dci = upload(ci);
  float* dv = upload(v);
  for (int dir = 0; dir < 2; ++dir) {
    const cusparseDirection_t D = dir ? CUSPARSE_DIRECTION_COLUMN : CUSPARSE_DIRECTION_ROW;
    // The element matrix and which elements are stored.
    const int M = mb * rbd, N = nb * cbd;
    std::vector<double> dense((size_t)M * N, 0);
    std::vector<char> stored((size_t)M * N, 0);
    for (int i = 0; i < mb; ++i)
      for (int k = ro[i]; k < ro[i + 1]; ++k)
        for (int r = 0; r < rbd; ++r)
          for (int c = 0; c < cbd; ++c) {
            const size_t e = (size_t)k * rbd * cbd + (dir ? (size_t)c * rbd + r : (size_t)r * cbd + c);
            dense[(size_t)(i * rbd + r) * N + ci[k] * cbd + c] = v[e];
            stored[(size_t)(i * rbd + r) * N + ci[k] * cbd + c] = 1;
          }
    for (auto [rc, cc] : std::vector<std::pair<int, int>>{{2, 2}, {1, 1}, {4, 3}, {3, 2}, {1, 3}}) {
      const int mbC = (M + rc - 1) / rc, nbC = (N + cc - 1) / cc;
      std::vector<int> wo{0}, wc;
      std::vector<double> wv;
      for (int I = 0; I < mbC; ++I) {
        for (int J = 0; J < nbC; ++J) {
          bool any = false;
          for (int r = 0; r < rc; ++r)
            for (int c = 0; c < cc; ++c) {
              const int i = I * rc + r, j = J * cc + c;
              if (i < M && j < N && stored[(size_t)i * N + j]) any = true;
            }
          if (!any) continue;
          wc.push_back(J);
          for (int e = 0; e < rc * cc; ++e) {
            const int r = dir ? e % rc : e / cc, c = dir ? e / rc : e % cc;
            const int i = I * rc + r, j = J * cc + c;
            wv.push_back(i < M && j < N ? dense[(size_t)i * N + j] : 0.0);
          }
        }
        wo.push_back((int)wc.size());
      }
      int bytes = -1, nnzC = -1;
      int* roC;
      cudaMalloc(&roC, (mbC + 1) * sizeof(int));
      const int s0 = cusparseSgebsr2gebsr_bufferSize(h, D, mb, nb, nnzb, md, dv, dro, dci, rbd, cbd, rc, cc, &bytes);
      const int s1 = cusparseXgebsr2gebsrNnz(h, D, mb, nb, nnzb, md, dro, dci, rbd, cbd, md, roC, rc, cc, &nnzC, work);
      int* ciC;
      float* vC;
      cudaMalloc(&ciC, std::max(nnzC, 1) * sizeof(int));
      cudaMalloc(&vC, std::max(nnzC, 1) * rc * cc * sizeof(float));
      const int s2 = cusparseSgebsr2gebsr(h, D, mb, nb, nnzb, md, dv, dro, dci, rbd, cbd, md, vC, roC, ciC, rc, cc, work);
      const auto gv = download(vC, (size_t)std::max(nnzC, 0) * rc * cc);
      const bool ok = s0 == 0 && s1 == 0 && s2 == 0 && bytes > 0 && nnzC == (int)wc.size() &&
                      download(roC, mbC + 1) == wo && download(ciC, nnzC) == wc &&
                      std::vector<double>(gv.begin(), gv.end()) == wv;
      char what[160];
      std::snprintf(what, sizeof what, "Sgebsr2gebsr %s blocks 2x3 -> %dx%d (%d blocks)", dir ? "column-major" : "row-major",
                    rc, cc, nnzC);
      check(ok, what);
      cudaFree(roC); cudaFree(ciC); cudaFree(vC);
    }
  }
  // gebsr2gebsc moves whole blocks.
  int bytes = -1;
  check(cusparseSgebsr2gebsc_bufferSize(h, mb, nb, nnzb, dv, dro, dci, rbd, cbd, &bytes) == 0 && bytes > 0,
        "Sgebsr2gebsc_bufferSize");
  int *cp, *ri;
  float* bv;
  cudaMalloc(&cp, (nb + 1) * sizeof(int));
  cudaMalloc(&ri, nnzb * sizeof(int));
  cudaMalloc(&bv, 18 * sizeof(float));
  const int s = cusparseSgebsr2gebsc(h, mb, nb, nnzb, dv, dro, dci, rbd, cbd, bv, ri, cp, CUSPARSE_ACTION_NUMERIC,
                                     CUSPARSE_INDEX_BASE_ZERO, work);
  std::vector<float> want(18);
  for (int e = 0; e < 6; ++e) {
    want[e] = v[e];            // block column 0: A's block 0
    want[6 + e] = v[12 + e];   // block column 1: A's block 2
    want[12 + e] = v[6 + e];   // block column 2: A's block 1
  }
  check(s == 0 && download(cp, nb + 1) == std::vector<int>{0, 1, 2, 3} && download(ri, 3) == std::vector<int>{0, 1, 0} &&
            download(bv, 18) == want,
        "Sgebsr2gebsc numeric: blocks moved whole");
  const std::vector<int> ro1{1, 3, 4}, ci1{1, 3, 2};
  int *dro1 = upload(ro1), *dci1 = upload(ci1);
  std::vector<float> junk(18, -1.0f);
  cudaMemcpy(bv, junk.data(), 18 * sizeof(float), cudaMemcpyHostToDevice);
  const int s2 = cusparseSgebsr2gebsc(h, mb, nb, nnzb, dv, dro1, dci1, rbd, cbd, bv, ri, cp, CUSPARSE_ACTION_SYMBOLIC,
                                      CUSPARSE_INDEX_BASE_ONE, work);
  check(s2 == 0 && download(cp, nb + 1) == std::vector<int>{1, 2, 3, 4} && download(ri, 3) == std::vector<int>{1, 2, 1} &&
            download(bv, 18) == junk,
        "Sgebsr2gebsc symbolic, one-based in and out: values untouched");
  expect(cusparseSgebsr2gebsc(h, mb, nb, nnzb, dv, dro, dci, rbd, cbd, bv, ri, cp, (cusparseAction_t)5,
                              CUSPARSE_INDEX_BASE_ZERO, work),
         3, "Sgebsr2gebsc bad action");
  expect(cusparseSgebsr2gebsc(h, mb, nb, nnzb, dv, dro, dci, 0, cbd, bv, ri, cp, CUSPARSE_ACTION_NUMERIC,
                              CUSPARSE_INDEX_BASE_ZERO, work),
         3, "Sgebsr2gebsc block dimension 0");
}

/* ---- csrcolor ---- */

static void coloring() {
  // An 8 x 8 grid's 5-point Laplacian.
  const int N = 8, M = N * N;
  std::vector<int> ro{0}, ci;
  for (int i = 0; i < N; ++i)
    for (int j = 0; j < N; ++j) {
      const int k = i * N + j;
      if (i > 0) ci.push_back(k - N);
      if (j > 0) ci.push_back(k - 1);
      ci.push_back(k);
      if (j < N - 1) ci.push_back(k + 1);
      if (i < N - 1) ci.push_back(k + N);
      ro.push_back((int)ci.size());
    }
  std::vector<float> v(ci.size(), 1.0f);
  int *dro = upload(ro), *dci = upload(ci);
  float* dv = upload(v);
  int *col, *reo, nc = -1;
  cudaMalloc(&col, M * sizeof(int));
  cudaMalloc(&reo, M * sizeof(int));
  cusparseColorInfo_t info;
  cusparseCreateColorInfo(&info);
  const float frac = 1.0f;
  const int s = cusparseScsrcolor(h, M, (int)ci.size(), md, dv, dro, dci, &frac, &nc, col, reo, info);
  const auto c = download(col, M), r = download(reo, M);
  bool proper = s == 0 && nc > 0, inrange = true;
  for (int i = 0; i < M; ++i) {
    inrange = inrange && c[i] >= 0 && c[i] < nc;
    for (int k = ro[i]; k < ro[i + 1]; ++k)
      if (ci[k] != i && c[ci[k]] == c[i]) proper = false;
  }
  std::set<int> perm(r.begin(), r.end());
  bool grouped = (int)perm.size() == M && *perm.begin() == 0 && *perm.rbegin() == M - 1;
  for (int i = 1; i < M && grouped; ++i) grouped = c[r[i - 1]] <= c[r[i]];
  check(proper && inrange, "Scsrcolor: neighbours never share a color", nc);
  check(grouped, "Scsrcolor: reordering is a permutation grouped by color");
  check(cusparseScsrcolor(h, M, (int)ci.size(), md, dv, dro, dci, &frac, &nc, col, nullptr, info) == 0,
        "Scsrcolor without a reordering");
  cusparseDestroyColorInfo(info);
}

/* ---- SpGEMMreuse, and SpGEMM's product count and memory estimate ---- */

static void spgemm_reuse() {
  const std::vector<int> ao{0, 2, 3, 5}, ac{0, 2, 1, 0, 3}, bo{0, 1, 3, 4, 6}, bc{1, 0, 2, 1, 0, 2};
  const std::vector<float> av{1, 2, 3, 4, 5}, bv{1, 2, -1, 3, 2, 4};
  int *dao = upload(ao), *dac = upload(ac), *dbo = upload(bo), *dbc = upload(bc);
  float *dav = upload(av), *dbv = upload(bv);
  cusparseSpMatDescr_t A, B, C;
  cusparseCreateCsr(&A, 3, 4, 5, dao, dac, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateCsr(&B, 4, 3, 6, dbo, dbc, dbv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  int* dco;
  cudaMalloc(&dco, 4 * sizeof(int));
  cusparseCreateCsr(&C, 3, 3, 0, dco, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_32F);
  const cusparseOperation_t N = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpGEMMDescr_t g;
  cusparseSpGEMM_createDescr(&g);
  size_t b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0;
  void *p1, *p2, *p3, *p4, *p5;
  bool ok = cusparseSpGEMMreuse_workEstimation(h, N, N, A, B, C, CUSPARSE_SPGEMM_DEFAULT, g, &b1, nullptr) == 0;
  cudaMalloc(&p1, b1 + 1);
  ok = ok && cusparseSpGEMMreuse_workEstimation(h, N, N, A, B, C, CUSPARSE_SPGEMM_DEFAULT, g, &b1, p1) == 0;
  ok = ok && cusparseSpGEMMreuse_nnz(h, N, N, A, B, C, CUSPARSE_SPGEMM_DEFAULT, g, &b2, nullptr, &b3, nullptr, &b4,
                                     nullptr) == 0;
  cudaMalloc(&p2, b2 + 1);
  cudaMalloc(&p3, b3 + 1);
  cudaMalloc(&p4, b4 + 1);
  ok = ok && cusparseSpGEMMreuse_nnz(h, N, N, A, B, C, CUSPARSE_SPGEMM_DEFAULT, g, &b2, p2, &b3, p3, &b4, p4) == 0;
  int64_t r, c, nnz = 0;
  cusparseSpMatGetSize(C, &r, &c, &nnz);
  check(ok && nnz == 6, "SpGEMMreuse workEstimation and nnz: C's nnz", (double)nnz);
  int* dcc;
  float* dcv;
  cudaMalloc(&dcc, nnz * sizeof(int));
  cudaMalloc(&dcv, nnz * sizeof(float));
  const std::vector<float> hundred(nnz, 100.0f);
  cudaMemcpy(dcv, hundred.data(), nnz * sizeof(float), cudaMemcpyHostToDevice);
  cusparseCsrSetPointers(C, dco, dcc, dcv);
  ok = cusparseSpGEMMreuse_copy(h, N, N, A, B, C, CUSPARSE_SPGEMM_DEFAULT, g, &b5, nullptr) == 0;
  cudaMalloc(&p5, b5 + 1);
  ok = ok && cusparseSpGEMMreuse_copy(h, N, N, A, B, C, CUSPARSE_SPGEMM_DEFAULT, g, &b5, p5) == 0;
  check(ok && download(dco, 4) == std::vector<int>{0, 1, 3, 6} && download(dcc, nnz) == std::vector<int>{1, 0, 2, 0, 1, 2} &&
            download(dcv, nnz) == hundred,
        "SpGEMMreuse copy: C's structure written, its values left alone");
  float al = 2, be = 0;
  ok = cusparseSpGEMMreuse_compute(h, N, N, &al, A, B, &be, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g) == 0;
  check(ok && download(dcv, nnz) == std::vector<float>{14, 12, -6, 20, 8, 40}, "SpGEMMreuse compute: C = 2 A B");
  be = 1;
  ok = cusparseSpGEMMreuse_compute(h, N, N, &al, A, B, &be, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g) == 0;
  check(ok && download(dcv, nnz) == std::vector<float>{28, 24, -12, 40, 16, 80}, "SpGEMMreuse compute: C = 2 A B + C");
  const std::vector<float> zero(5, 0.0f);
  cudaMemcpy(dav, zero.data(), 5 * sizeof(float), cudaMemcpyHostToDevice);
  be = 0;
  ok = cusparseSpGEMMreuse_compute(h, N, N, &al, A, B, &be, C, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g) == 0;
  check(ok && download(dcv, nnz) == std::vector<float>(6, 0.0f), "SpGEMMreuse compute again with new values in A");
  cudaMemcpy(dav, av.data(), 5 * sizeof(float), cudaMemcpyHostToDevice);
  expect(cusparseSpGEMMreuse_compute(h, CUSPARSE_OPERATION_TRANSPOSE, N, &al, A, B, &be, C, CUDA_R_32F,
                                     CUSPARSE_SPGEMM_DEFAULT, g),
         10, "SpGEMMreuse compute op(A) = A^T");
  const double ald = 2, bed = 0;
  expect(cusparseSpGEMMreuse_compute(h, N, N, &ald, A, B, &bed, C, CUDA_R_64F, CUSPARSE_SPGEMM_DEFAULT, g), 10,
         "SpGEMMreuse compute, single values in double");
  int64_t prods = -1;
  check(cusparseSpGEMM_getNumProducts(g, &prods) == 0 && prods == 7, "SpGEMM_getNumProducts after reuse",
        (double)prods);
  cusparseSpGEMM_destroyDescr(g);

  // SpGEMM with ALG3: the product count and the memory estimate.
  int* dco2;
  cudaMalloc(&dco2, 4 * sizeof(int));
  cusparseSpMatDescr_t C2;
  cusparseCreateCsr(&C2, 3, 3, 0, dco2, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                    CUDA_R_32F);
  cusparseSpGEMMDescr_t g2;
  cusparseSpGEMM_createDescr(&g2);
  prods = -1;
  const int sp0 = cusparseSpGEMM_getNumProducts(g2, &prods);
  check(sp0 == 0 && prods == 0, "SpGEMM_getNumProducts before workEstimation: 0", (double)prods);
  b1 = b2 = b3 = 0;
  al = 1;
  ok = cusparseSpGEMM_workEstimation(h, N, N, &al, A, B, &be, C2, CUDA_R_32F, CUSPARSE_SPGEMM_ALG3, g2, &b1, nullptr) == 0;
  cudaMalloc(&p1, b1 + 1);
  ok = ok && cusparseSpGEMM_workEstimation(h, N, N, &al, A, B, &be, C2, CUDA_R_32F, CUSPARSE_SPGEMM_ALG3, g2, &b1, p1) == 0;
  ok = ok && cusparseSpGEMM_getNumProducts(g2, &prods) == 0 && prods == 7;
  check(ok, "SpGEMM_getNumProducts after workEstimation: A B's scalar products", (double)prods);
  ok = cusparseSpGEMM_estimateMemory(h, N, N, &al, A, B, &be, C2, CUDA_R_32F, CUSPARSE_SPGEMM_ALG3, g2, 0.2f, &b3, nullptr,
                                     nullptr) == 0 && b3 > 0;
  cudaMalloc(&p3, b3 + 1);
  ok = ok && cusparseSpGEMM_estimateMemory(h, N, N, &al, A, B, &be, C2, CUDA_R_32F, CUSPARSE_SPGEMM_ALG3, g2, 0.2f, &b3, p3,
                                           &b2) == 0 && b2 > 0;
  cudaMalloc(&p2, b2 + 1);
  ok = ok && cusparseSpGEMM_compute(h, N, N, &al, A, B, &be, C2, CUDA_R_32F, CUSPARSE_SPGEMM_ALG3, g2, &b2, p2) == 0;
  cusparseSpMatGetSize(C2, &r, &c, &nnz);
  check(ok && nnz == 6, "SpGEMM ALG3 with estimateMemory", (double)nnz);
  expect(cusparseSpGEMM_estimateMemory(h, N, N, &al, A, B, &be, C2, CUDA_R_32F, CUSPARSE_SPGEMM_DEFAULT, g2, 0.2f, &b3,
                                       nullptr, nullptr),
         3, "SpGEMM_estimateMemory with the default algorithm");
  expect(cusparseSpGEMM_getNumProducts(nullptr, &prods), 3, "SpGEMM_getNumProducts NULL descriptor");
  cusparseSpGEMM_destroyDescr(g2);

  // SpMMOp: the operators are LTO-IR.
  std::vector<float> Bv(4 * 2, 1), Cv(3 * 2, 0);
  float *dB = upload(Bv), *dC = upload(Cv);
  cusparseDnMatDescr_t Bd, Cd;
  cusparseCreateDnMat(&Bd, 4, 2, 2, dB, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseCreateDnMat(&Cd, 3, 2, 2, dC, CUDA_R_32F, CUSPARSE_ORDER_ROW);
  cusparseSpMMOpPlan_t plan = nullptr;
  size_t ws = 0;
  expect(cusparseSpMMOp_createPlan(h, &plan, N, N, A, Bd, Cd, CUDA_R_32F, CUSPARSE_SPMM_OP_ALG_DEFAULT, nullptr, 0, nullptr,
                                   0, nullptr, 0, &ws),
         3, "SpMMOp_createPlan without operators");
  const char junk[64] = {1, 2, 3};
  expect(cusparseSpMMOp_createPlan(h, &plan, N, N, A, Bd, Cd, CUDA_R_32F, CUSPARSE_SPMM_OP_ALG_DEFAULT, junk, 64, junk, 64,
                                   junk, 64, &ws),
         7, "SpMMOp_createPlan with operators that are not LTO-IR nvJitLink takes");
  expect(cusparseSpMMOp(nullptr, nullptr), 3, "SpMMOp without a plan");
}

/* ---- SpSV and SpSM updateMatrix ---- */

using UpdateFn = cusparseStatus_t (*)(cusparseHandle_t, void*, void*, int);

static void update_matrix() {
  auto sv_update = reinterpret_cast<UpdateFn>(dlsym(RTLD_DEFAULT, "cusparseSpSV_updateMatrix"));
  auto sm_update = reinterpret_cast<UpdateFn>(dlsym(RTLD_DEFAULT, "cusparseSpSM_updateMatrix"));
  if (!sv_update || !sm_update) {
    std::printf("SKIP updateMatrix: this cuSPARSE has none\n");
    return;
  }
  const std::vector<int> lo{0, 1, 3, 6}, lc{0, 0, 1, 0, 1, 2};
  const std::vector<float> lv{2, 1, 4, 1, 1, 5};
  int *dlo = upload(lo), *dlc = upload(lc);
  float* dlv = upload(lv);
  cusparseSpMatDescr_t L;
  cusparseCreateCsr(&L, 3, 3, 6, dlo, dlc, dlv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseFillMode_t fm = CUSPARSE_FILL_MODE_LOWER;
  cusparseSpMatSetAttribute(L, CUSPARSE_SPMAT_FILL_MODE, &fm, sizeof(fm));
  const std::vector<float> x{2, 5, 7};
  float *dx = upload(x), *dy;
  cudaMalloc(&dy, 3 * sizeof(float));
  cusparseDnVecDescr_t X, Y;
  cusparseCreateDnVec(&X, 3, dx, CUDA_R_32F);
  cusparseCreateDnVec(&Y, 3, dy, CUDA_R_32F);
  cusparseSpSVDescr_t d;
  cusparseSpSV_createDescr(&d);
  size_t bytes = 0;
  const float one = 1;
  const cusparseOperation_t N = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpSV_bufferSize(h, N, &one, L, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d, &bytes);
  void* buf;
  cudaMalloc(&buf, bytes + 16);
  cusparseSpSV_analysis(h, N, &one, L, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d, buf);
  bool ok = cusparseSpSV_solve(h, N, &one, L, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d) == 0 &&
            download(dy, 3) == std::vector<float>{1, 1, 1};
  check(ok, "SpSV solve before any update");
  const std::vector<float> ones(6, 1.0f);
  float* dl2 = upload(ones);
  ok = sv_update(h, d, dl2, 0) == 0 && cusparseSpSV_solve(h, N, &one, L, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d) == 0 &&
       download(dy, 3) == std::vector<float>{2, 3, 2};
  void* now = nullptr;
  cusparseSpMatGetValues(L, &now);
  check(ok && now == dlv && download(dlv, 6) == lv, "SpSV_updateMatrix general: solved with the new values, A untouched");
  float* ddg = upload(std::vector<float>{4, 8, 10});
  ok = sv_update(h, d, ddg, 1) == 0 && cusparseSpSV_solve(h, N, &one, L, X, Y, CUDA_R_32F, CUSPARSE_SPSV_ALG_DEFAULT, d) == 0;
  const auto y = download(dy, 3);
  const double e = std::fabs(y[0] - 0.5) + std::fabs(y[1] - 0.5625) + std::fabs(y[2] - 0.59375);
  check(ok && e < 1e-6 && download(dl2, 6) == ones, "SpSV_updateMatrix diagonal: on top of the general update", e);
  expect(sv_update(h, d, ddg, 7), 3, "SpSV_updateMatrix bad part");
  expect(sv_update(h, d, nullptr, 0), 3, "SpSV_updateMatrix NULL values");
  // SpSM, diagonal only.
  const std::vector<float> bm{2, 5, 7, 1, 1, 1};
  float *dbm = upload(bm), *dcm;
  cudaMalloc(&dcm, 6 * sizeof(float));
  cusparseDnMatDescr_t Bm, Cm;
  cusparseCreateDnMat(&Bm, 3, 2, 3, dbm, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&Cm, 3, 2, 3, dcm, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseSpSMDescr_t sm;
  cusparseSpSM_createDescr(&sm);
  cusparseSpSM_bufferSize(h, N, N, &one, L, Bm, Cm, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, sm, &bytes);
  void* buf2;
  cudaMalloc(&buf2, bytes + 16);
  cusparseSpSM_analysis(h, N, N, &one, L, Bm, Cm, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, sm, buf2);
  ok = sm_update(h, sm, ddg, 1) == 0 &&
       cusparseSpSM_solve(h, N, N, &one, L, Bm, Cm, CUDA_R_32F, CUSPARSE_SPSM_ALG_DEFAULT, sm) == 0;
  const auto cm = download(dcm, 6);
  // L with diagonal 4, 8, 10: column 0 of B is (2, 5, 7), column 1 is (1, 1, 1).
  const double want[6] = {0.5, 0.5625, 0.59375, 0.25, 0.09375, 0.065625};
  double err = 0;
  for (int i = 0; i < 6; ++i) err = std::max(err, std::fabs(cm[i] - want[i]));
  check(ok && err < 1e-6, "SpSM_updateMatrix diagonal", err);
}

/* ---- the logger, the CSC sort ---- */

static void the_rest() {
  expect(cusparseLoggerSetLevel(-1), 3, "LoggerSetLevel(-1)");
  expect(cusparseLoggerSetLevel(0), 0, "LoggerSetLevel(0)");
  expect(cusparseLoggerSetMask(0), 0, "LoggerSetMask(0)");
  expect(cusparseLoggerOpenFile("/nonexistent-dir/x/cusparse.log"), 3, "LoggerOpenFile where no file can be made");
  expect(cusparseLoggerSetCallback(nullptr), 0, "LoggerSetCallback(NULL)");
  const std::vector<int> co{0, 2, 3, 5}, ri{2, 0, 1, 2, 0};
  int *dco = upload(co), *dri = upload(ri), *P;
  cudaMalloc(&P, 5 * sizeof(int));
  cusparseCreateIdentityPermutation(h, 5, P);
  size_t bytes = 0;
  const int s0 = cusparseXcscsort_bufferSizeExt(h, 3, 3, 5, dco, dri, &bytes);
  const int s1 = cusparseXcscsort(h, 3, 3, 5, md, dco, dri, P, work);
  check(s0 == 0 && s1 == 0 && download(dri, 5) == std::vector<int>{0, 2, 1, 0, 2} &&
            download(P, 5) == std::vector<int>{1, 0, 2, 4, 3},
        "Xcscsort sorts each column's rows, P alongside");
}

int main() {
  if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS) {
    std::printf("FAIL cusparseCreate\n");
    return 1;
  }
  cusparseCreateMatDescr(&md);
  cusparseCreateMatDescr(&md1);
  cusparseSetMatIndexBase(md1, CUSPARSE_INDEX_BASE_ONE);
  cudaMalloc(&work, 1 << 22);
  pruning<float>();
  pruning<double>();
  pruning<__half>();
  counting();
  unsorted<float>("S", cusparseScsru2csr_bufferSizeExt, cusparseScsru2csr, cusparseScsr2csru, mk_s);
  unsorted<cuDoubleComplex>("Z", cusparseZcsru2csr_bufferSizeExt, cusparseZcsru2csr, cusparseZcsr2csru, mk_z);
  gebsr();
  coloring();
  spgemm_reuse();
  update_matrix();
  the_rest();
  (void)on_sim;
  cudaFree(work);
  cusparseDestroy(h);
  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
