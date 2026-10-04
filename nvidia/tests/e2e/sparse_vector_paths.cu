// cuSPARSE's sparse vectors, the deprecated gemvi, and the two ELL storage
// formats, checked against dense arithmetic done here and against the
// statuses an RTX 3060's cuSPARSE 13.0 answers with:
//
//   SpVec        create/get/set and their refusals (size < 0, nnz > size,
//                a bad base, NULL outputs)
//   SpVV         op(x) . y for A and A^H, real, half (computed in single),
//                int8 (into int32 and single) and complex; host and device
//                results; A^T, mixed types and sizes refused
//   Axpby, Gather, Scatter, Rot
//                in real and complex, with one-based indices; a repeated
//                index lands its first entry
//   gemvi        y = alpha op(A) x + beta y, A dense, x sparse, A and A^T
//   Blocked-ELL  SpMM (column- and row-major B and C, op(B) = B^T, half, int8)
//                against a dense product, a padding block counting as block
//                column 0, DenseToSparse filling it, GetSize's nnz, and what
//                NVIDIA's refuses (SpMV, SparseToDense, op(A) = A^T, complex,
//                another SpMM algorithm)
//   sliced ELL   SpMV with A, A^T and A^H, one-based (offsets included),
//                padding slots skipped, and what NVIDIA's refuses (another
//                SpMV algorithm, SpMM, both conversions, SDDMM)
//
// CUDA 12.0's header has no sliced ELL, so cusparseCreateSlicedEll is looked
// up by name in the cuSPARSE the program runs against.
#include <cuComplex.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <dlfcn.h>

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

using cdouble = std::complex<double>;

static int failures = 0;

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
static double max_abs_diff(const std::vector<double>& a, const std::vector<double>& b) {
  double e = 0;
  for (size_t i = 0; i < a.size(); ++i) e = std::max(e, std::fabs(a[i] - b[i]));
  return e;
}

static cusparseHandle_t h;
static void* work;

/* ---- SpVec descriptors ---- */

static void descriptors() {
  std::vector<int> idx{1, 4, 6};
  std::vector<float> xv{2, 3, 5};
  int* di = upload(idx);
  float* dx = upload(xv);
  cusparseSpVecDescr_t t = nullptr;
  expect(cusparseCreateSpVec(&t, -1, 0, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateSpVec size < 0");
  expect(cusparseCreateSpVec(&t, 8, -1, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateSpVec nnz < 0");
  expect(cusparseCreateSpVec(&t, 2, 3, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateSpVec nnz > size");
  expect(cusparseCreateSpVec(&t, 8, 3, di, dx, CUSPARSE_INDEX_32I, (cusparseIndexBase_t)5, CUDA_R_32F), 3,
         "CreateSpVec index base 5");
  expect(cusparseCreateSpVec(nullptr, 8, 3, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateSpVec NULL descriptor");
  expect(cusparseCreateSpVec(&t, 8, 3, nullptr, nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 0,
         "CreateSpVec takes NULL arrays");
  cusparseDestroySpVec(t);
  expect(cusparseCreateSpVec(&t, 8, 3, di, dx, CUSPARSE_INDEX_16U, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 0,
         "CreateSpVec takes 16-bit indices");
  cusparseDestroySpVec(t);
  cusparseSpVecDescr_t X;
  if (cusparseCreateSpVec(&X, 8, 3, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F)) {
    check(false, "CreateSpVec");
    return;
  }
  int64_t size = 0, nnz = 0;
  void *i = nullptr, *v = nullptr;
  cusparseIndexType_t it;
  cusparseIndexBase_t b;
  cudaDataType vt;
  const int st = cusparseSpVecGet(X, &size, &nnz, &i, &v, &it, &b, &vt);
  check(st == 0 && size == 8 && nnz == 3 && i == di && v == dx && it == CUSPARSE_INDEX_32I &&
            b == CUSPARSE_INDEX_BASE_ONE && vt == CUDA_R_32F,
        "SpVecGet returns what was given");
  expect(cusparseSpVecGet(X, nullptr, &nnz, &i, &v, &it, &b, &vt), 3, "SpVecGet NULL output");
  expect(cusparseSpVecGetIndexBase(X, nullptr), 3, "SpVecGetIndexBase NULL output");
  expect(cusparseSpVecGetValues(X, nullptr), 3, "SpVecGetValues NULL output");
  float* other = upload(xv);
  void* got = nullptr;
  check(cusparseSpVecSetValues(X, other) == 0 && cusparseSpVecGetValues(X, &got) == 0 && got == other,
        "SpVecSetValues and GetValues");
  const void* cgot = nullptr;
  check(cusparseConstSpVecGetValues(X, &cgot) == 0 && cgot == other, "ConstSpVecGetValues");
  cusparseConstSpVecDescr_t C;
  check(cusparseCreateConstSpVec(&C, 8, 3, di, dx, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) == 0,
        "CreateConstSpVec");
  const void *ci = nullptr, *cv = nullptr;
  check(cusparseConstSpVecGet(C, &size, &nnz, &ci, &cv, &it, &b, &vt) == 0 && ci == di && cv == dx &&
            it == CUSPARSE_INDEX_64I && vt == CUDA_R_64F,
        "ConstSpVecGet");
  cusparseDestroySpVec(C);
  cusparseDestroySpVec(X);
  cudaFree(di); cudaFree(dx); cudaFree(other);
}

/* ---- SpVV ---- */

static void spvv() {
  std::vector<int> idx{1, 4, 6};
  int* di = upload(idx);
  std::vector<float> xv{2, 3, 5}, yv{1, 2, 3, 4, 5, 6, 7, 8};
  float *dx = upload(xv), *dy = upload(yv);
  cusparseSpVecDescr_t X;
  cusparseDnVecDescr_t Y;
  cusparseCreateSpVec(&X, 8, 3, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnVec(&Y, 8, dy, CUDA_R_32F);
  const double want = 2 * 2 + 3 * 5 + 5 * 7;
  float r = -1;
  size_t bytes = 0;
  check(cusparseSpVV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, X, Y, &r, CUDA_R_32F, &bytes) == 0,
        "SpVV_bufferSize");
  check(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, X, Y, &r, CUDA_R_32F, work) == 0 && r == want,
        "SpVV single, result on the host", r - want);
  float* dr;
  cudaMalloc(&dr, sizeof(float));
  cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_DEVICE);
  const int sd = cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, X, Y, dr, CUDA_R_32F, work);
  cusparseSetPointerMode(h, CUSPARSE_POINTER_MODE_HOST);
  check(sd == 0 && download(dr, 1)[0] == want, "SpVV single, result on the device");
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_TRANSPOSE, X, Y, &r, CUDA_R_32F, work), 3, "SpVV op = A^T");
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, X, Y, &r, CUDA_R_32F, work), 3,
         "SpVV op = A^H of a real vector");
  double rd = 0;
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, X, Y, &rd, CUDA_R_64F, work), 10,
         "SpVV single vectors, double compute");
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, X, Y, nullptr, CUDA_R_32F, work), 3, "SpVV NULL result");
  cusparseDnVecDescr_t Ys;
  cusparseCreateDnVec(&Ys, 5, dy, CUDA_R_32F);
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, X, Ys, &r, CUDA_R_32F, work), 3,
         "SpVV vectors of different sizes");
  // Half precision computes in single; int8 into int32 or single.
  std::vector<__half> hx{__float2half(2), __float2half(3), __float2half(5)}, hy(8);
  for (int i = 0; i < 8; ++i) hy[i] = __float2half((float)(i + 1));
  __half *dhx = upload(hx), *dhy = upload(hy);
  cusparseSpVecDescr_t Xh;
  cusparseDnVecDescr_t Yh;
  cusparseCreateSpVec(&Xh, 8, 3, di, dhx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);
  cusparseCreateDnVec(&Yh, 8, dhy, CUDA_R_16F);
  r = -1;
  check(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, Xh, Yh, &r, CUDA_R_32F, work) == 0 && r == want,
        "SpVV half vectors, single compute");
  __half hr;
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, Xh, Yh, &hr, CUDA_R_16F, work), 10,
         "SpVV half compute");
  expect(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, Xh, Y, &r, CUDA_R_32F, work), 10,
         "SpVV half x, single y");
  std::vector<signed char> ix{2, 3, 5}, iy{1, 2, 3, 4, 5, 6, 7, 8};
  signed char *dix = upload(ix), *diy = upload(iy);
  cusparseSpVecDescr_t Xi;
  cusparseDnVecDescr_t Yi;
  cusparseCreateSpVec(&Xi, 8, 3, di, dix, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_8I);
  cusparseCreateDnVec(&Yi, 8, diy, CUDA_R_8I);
  int ri = -1;
  r = -1;
  check(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, Xi, Yi, &ri, CUDA_R_32I, work) == 0 && ri == (int)want,
        "SpVV int8 vectors, int32 result");
  check(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, Xi, Yi, &r, CUDA_R_32F, work) == 0 && r == want,
        "SpVV int8 vectors, single result");
  // Complex: A multiplies x as it is, A^H conjugates it.
  std::vector<cuComplex> cx{{1, 2}, {3, -1}, {0, 1}}, cy(8);
  for (int i = 0; i < 8; ++i) cy[i] = make_cuComplex((float)i, (float)(1 - i));
  cuComplex *dcx = upload(cx), *dcy = upload(cy);
  cusparseSpVecDescr_t Xc;
  cusparseDnVecDescr_t Yc;
  cusparseCreateSpVec(&Xc, 8, 3, di, dcx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateDnVec(&Yc, 8, dcy, CUDA_C_32F);
  cdouble dot = 0, dotc = 0;
  for (int k = 0; k < 3; ++k) {
    const cdouble a(cx[k].x, cx[k].y), b(cy[idx[k]].x, cy[idx[k]].y);
    dot += a * b;
    dotc += std::conj(a) * b;
  }
  cuComplex rc;
  check(cusparseSpVV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, Xc, Yc, &rc, CUDA_C_32F, work) == 0 &&
            std::abs(cdouble(rc.x, rc.y) - dot) < 1e-5,
        "SpVV complex, op = A");
  check(cusparseSpVV(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, Xc, Yc, &rc, CUDA_C_32F, work) == 0 &&
            std::abs(cdouble(rc.x, rc.y) - dotc) < 1e-5,
        "SpVV complex, op = A^H conjugates x");
  cusparseDestroySpVec(X); cusparseDestroySpVec(Xh); cusparseDestroySpVec(Xi); cusparseDestroySpVec(Xc);
  cusparseDestroyDnVec(Y); cusparseDestroyDnVec(Ys); cusparseDestroyDnVec(Yh); cusparseDestroyDnVec(Yi);
  cusparseDestroyDnVec(Yc);
}

/* ---- Axpby, Gather, Scatter, Rot ---- */

static void vector_ops() {
  const std::vector<float> y0{1, 2, 3, 4, 5, 6, 7, 8};
  const std::vector<float> xv{2, 3, 5};
  std::vector<int> idx{1, 4, 6}, idx1{2, 5, 7}, dup{1, 1, 6};
  int *di = upload(idx), *di1 = upload(idx1), *ddup = upload(dup);
  float *dx = upload(xv), *dy = upload(y0);
  cusparseSpVecDescr_t X, X1, Xd;
  cusparseDnVecDescr_t Y;
  cusparseCreateSpVec(&X, 8, 3, di, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateSpVec(&X1, 8, 3, di1, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE, CUDA_R_32F);
  cusparseCreateSpVec(&Xd, 8, 3, ddup, dx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
  cusparseCreateDnVec(&Y, 8, dy, CUDA_R_32F);
  auto reset = [&] { cudaMemcpy(dy, y0.data(), 32, cudaMemcpyHostToDevice); cudaMemcpy(dx, xv.data(), 12, cudaMemcpyHostToDevice); };
  auto yd = [&] { const auto v = download(dy, 8); return std::vector<double>(v.begin(), v.end()); };
  float al = 2, be = 3;
  const std::vector<double> want{3, 10, 9, 12, 21, 18, 31, 24};
  check(cusparseAxpby(h, &al, X, &be, Y) == 0 && yd() == want, "Axpby y = 2 x + 3 y");
  reset();
  check(cusparseAxpby(h, &al, X1, &be, Y) == 0 && yd() == want, "Axpby, one-based indices");
  reset();
  check(cusparseAxpby(h, &al, Xd, &be, Y) == 0 && yd() == std::vector<double>{3, 10, 9, 12, 15, 18, 31, 24},
        "Axpby with a repeated index: the first entry lands");
  reset();
  cusparseDnVecDescr_t Ys, Yd;
  cusparseCreateDnVec(&Ys, 7, dy, CUDA_R_32F);
  double* ddy;
  cudaMalloc(&ddy, 64);
  cusparseCreateDnVec(&Yd, 8, ddy, CUDA_R_64F);
  expect(cusparseAxpby(h, &al, X, &be, Ys), 3, "Axpby vectors of different sizes");
  expect(cusparseAxpby(h, &al, X, &be, Yd), 10, "Axpby vectors of different types");
  // Gather and Scatter.
  check(cusparseGather(h, Y, X) == 0 && download(dx, 3) == std::vector<float>{2, 5, 7}, "Gather x = y[idx]");
  const std::vector<float> sv{10, 20, 30};
  cudaMemcpy(dx, sv.data(), 12, cudaMemcpyHostToDevice);
  check(cusparseScatter(h, X, Y) == 0 && download(dy, 8) == std::vector<float>{1, 10, 3, 4, 20, 6, 30, 8},
        "Scatter y[idx] = x");
  reset();
  cudaMemcpy(dx, sv.data(), 12, cudaMemcpyHostToDevice);
  check(cusparseScatter(h, Xd, Y) == 0 && download(dy, 8) == std::vector<float>{1, 10, 3, 4, 5, 6, 30, 8},
        "Scatter with a repeated index: the first entry lands");
  expect(cusparseGather(h, Yd, X), 10, "Gather vectors of different types");
  expect(cusparseScatter(h, X, Yd), 10, "Scatter vectors of different types");
  expect(cusparseGather(h, Ys, X), 3, "Gather vectors of different sizes");
  reset();
  // Rot: x = c x + s y, y = c y - s x.
  float c = 0.6f, s = 0.8f;
  check(cusparseRot(h, &c, &s, X, Y) == 0, "Rot");
  const auto rx = download(dx, 3), ry = download(dy, 8);
  double err = 0;
  for (int k = 0; k < 3; ++k) {
    err = std::max(err, std::fabs(rx[k] - (0.6 * xv[k] + 0.8 * y0[idx[k]])));
    err = std::max(err, std::fabs(ry[idx[k]] - (0.6 * y0[idx[k]] - 0.8 * xv[k])));
  }
  check(err < 1e-6, "Rot: x = c x + s y, y = c y - s x", err);
  // Complex: Axpby, and Rot with complex c and s, multiplied as given.
  std::vector<cuComplex> cx{{1, 2}, {3, -1}, {0, 1}}, cy(8);
  for (int i = 0; i < 8; ++i) cy[i] = make_cuComplex((float)i, (float)(1 - i));
  cuComplex *dcx = upload(cx), *dcy = upload(cy);
  cusparseSpVecDescr_t Xc;
  cusparseDnVecDescr_t Yc;
  cusparseCreateSpVec(&Xc, 8, 3, di, dcx, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
  cusparseCreateDnVec(&Yc, 8, dcy, CUDA_C_32F);
  const cuComplex cc = make_cuComplex(0.6f, 0.5f), cs = make_cuComplex(0.8f, 0.25f);
  check(cusparseRot(h, &cc, &cs, Xc, Yc) == 0, "Rot complex");
  const auto zx = download(dcx, 3), zy = download(dcy, 8);
  err = 0;
  const cdouble C(0.6f, 0.5f), Sv(0.8f, 0.25f);
  for (int k = 0; k < 3; ++k) {
    const cdouble a(cx[k].x, cx[k].y), b(cy[idx[k]].x, cy[idx[k]].y);
    err = std::max(err, std::abs(cdouble(zx[k].x, zx[k].y) - (C * a + Sv * b)));
    err = std::max(err, std::abs(cdouble(zy[idx[k]].x, zy[idx[k]].y) - (C * b - Sv * a)));
  }
  check(err < 1e-5, "Rot complex: c and s complex, multiplied as given", err);
  cudaMemcpy(dcx, cx.data(), 24, cudaMemcpyHostToDevice);
  cudaMemcpy(dcy, cy.data(), 64, cudaMemcpyHostToDevice);
  const cuComplex ca = make_cuComplex(1, 1), cb = make_cuComplex(2, 0);
  check(cusparseAxpby(h, &ca, Xc, &cb, Yc) == 0, "Axpby complex");
  const auto ay = download(dcy, 8);
  err = 0;
  for (int i = 0; i < 8; ++i) {
    cdouble w = cdouble(2, 0) * cdouble(cy[i].x, cy[i].y);
    for (int k = 0; k < 3; ++k)
      if (idx[k] == i) w += cdouble(1, 1) * cdouble(cx[k].x, cx[k].y);
    err = std::max(err, std::abs(cdouble(ay[i].x, ay[i].y) - w));
  }
  check(err < 1e-5, "Axpby complex: y = alpha x + beta y", err);
  cusparseDestroySpVec(X); cusparseDestroySpVec(X1); cusparseDestroySpVec(Xd); cusparseDestroySpVec(Xc);
  cusparseDestroyDnVec(Y); cusparseDestroyDnVec(Ys); cusparseDestroyDnVec(Yd); cusparseDestroyDnVec(Yc);
}

/* ---- gemvi ---- */

static void gemvi() {
  const int m = 3, n = 8, lda = 4;
  std::vector<float> A((size_t)lda * n);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < lda; ++i) A[(size_t)j * lda + i] = 0.5f * i + j;
  float* dA = upload(A);
  std::vector<int> xi{1, 4, 6}, xt{0, 2, 1}, x1{2, 5, 7};
  std::vector<float> xv{2, 3, 5};
  int *dxi = upload(xi), *dxt = upload(xt), *dx1 = upload(x1);
  float* dxv = upload(xv);
  std::vector<float> y0(8, 1.0f);
  float* dy = upload(y0);
  int bytes = -1;
  check(cusparseSgemvi_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, 3, &bytes) == 0 && bytes >= 0,
        "gemvi_bufferSize");
  const float al = 2, be = 0.5f;
  check(cusparseSgemvi(h, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, &al, dA, lda, 3, dxv, dxi, &be, dy,
                       CUSPARSE_INDEX_BASE_ZERO, work) == 0,
        "gemvi op = A");
  auto y = download(dy, m);
  double err = 0;
  for (int i = 0; i < m; ++i) {
    double w = 0.5;
    for (int k = 0; k < 3; ++k) w += 2 * A[(size_t)xi[k] * lda + i] * xv[k];
    err = std::max(err, std::fabs(y[i] - w));
  }
  check(err < 1e-4, "gemvi: y = alpha A x + beta y", err);
  cudaMemcpy(dy, y0.data(), 32, cudaMemcpyHostToDevice);
  check(cusparseSgemvi(h, CUSPARSE_OPERATION_NON_TRANSPOSE, m, n, &al, dA, lda, 3, dxv, dx1, &be, dy,
                       CUSPARSE_INDEX_BASE_ONE, work) == 0 &&
            std::fabs(download(dy, 1)[0] - y[0]) < 1e-4,
        "gemvi, one-based indices");
  cudaMemcpy(dy, y0.data(), 32, cudaMemcpyHostToDevice);
  check(cusparseSgemvi(h, CUSPARSE_OPERATION_TRANSPOSE, m, n, &al, dA, lda, 3, dxv, dxt, &be, dy,
                       CUSPARSE_INDEX_BASE_ZERO, work) == 0,
        "gemvi op = A^T");
  y = download(dy, n);
  err = 0;
  for (int j = 0; j < n; ++j) {
    double w = 0.5;
    for (int k = 0; k < 3; ++k) w += 2 * A[(size_t)j * lda + xt[k]] * xv[k];
    err = std::max(err, std::fabs(y[j] - w));
  }
  check(err < 1e-4, "gemvi: y = alpha A^T x + beta y", err);
  expect(cusparseSgemvi(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, m, n, &al, dA, lda, 3, dxv, dxt, &be, dy,
                        CUSPARSE_INDEX_BASE_ZERO, work),
         3, "gemvi op = A^H of a real matrix");
}

/* ---- Blocked-ELL ---- */

// rows x cols, blocks of BS, ellCols wide; block row i's slot k is block
// column bci[i * slots + k] (-1: padding).
struct Bell {
  int rows, cols, bs, ell_cols;
  std::vector<int> bci;
  std::vector<double> vals;  // rows x ellCols, row-major
  std::vector<double> dense(bool pad_as_zero) const {
    std::vector<double> d((size_t)rows * cols, 0);
    const int slots = ell_cols / bs;
    for (int br = 0; br < rows / bs; ++br)
      for (int k = 0; k < slots; ++k) {
        int bc = bci[(size_t)br * slots + k];
        if (bc < 0) {
          if (pad_as_zero) continue;
          bc = 0;  // NVIDIA's multiplies a padding block as block column 0
        }
        for (int r = 0; r < bs; ++r)
          for (int c = 0; c < bs; ++c)
            d[(size_t)(br * bs + r) * cols + bc * bs + c] += vals[(size_t)(br * bs + r) * ell_cols + k * bs + c];
      }
    return d;
  }
};

template <class T> T from_d(double v);
template <> float from_d<float>(double v) { return (float)v; }
template <> double from_d<double>(double v) { return v; }
template <> __half from_d<__half>(double v) { return __float2half((float)v); }
template <> signed char from_d<signed char>(double v) { return (signed char)v; }
template <> int from_d<int>(double v) { return (int)v; }
template <class T> double to_d(T v) { return (double)v; }
template <> double to_d<__half>(__half v) { return (double)__half2float(v); }

template <class T, class TC>
void bell_spmm(const Bell& A, cudaDataType vt, cudaDataType ct_c, cudaDataType ct, const char* name, bool row_major,
               bool transB, double tol) {
  char what[200];
  std::vector<T> av(A.vals.size());
  for (size_t i = 0; i < av.size(); ++i) av[i] = from_d<T>(A.vals[i]);
  int* dbci = upload(A.bci);
  T* dav = upload(av);
  cusparseSpMatDescr_t M;
  cusparseCreateBlockedEll(&M, A.rows, A.cols, A.bs, A.ell_cols, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, vt);
  const int N = 3;
  // op(B) is cols x N.
  const int brows = transB ? N : A.cols, bcols = transB ? A.cols : N;
  std::vector<T> B((size_t)brows * bcols);
  for (size_t i = 0; i < B.size(); ++i) B[i] = from_d<T>((double)((int)(i % 5) - 2));
  std::vector<TC> C((size_t)A.rows * N, from_d<TC>(1));
  T* dB = upload(B);
  TC* dC = upload(C);
  cusparseDnMatDescr_t Bd, Cd;
  const cusparseOrder_t ord = row_major ? CUSPARSE_ORDER_ROW : CUSPARSE_ORDER_COL;
  cusparseCreateDnMat(&Bd, brows, bcols, row_major ? bcols : brows, dB, vt, ord);
  cusparseCreateDnMat(&Cd, A.rows, N, row_major ? N : A.rows, dC, ct_c, ord);
  const float alf = 2, bef = 0.5f;
  const double ald = 2, bed = 0.5;
  const int ali = 2, bei = 0;
  const void* pa = ct == CUDA_R_64F ? (const void*)&ald : ct == CUDA_R_32I ? (const void*)&ali : (const void*)&alf;
  const void* pb = ct == CUDA_R_64F ? (const void*)&bed : ct == CUDA_R_32I ? (const void*)&bei : (const void*)&bef;
  const double beta = ct == CUDA_R_32I ? 0 : 0.5;
  const cusparseOperation_t opB = transB ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
  size_t bytes = 0;
  const int s0 = cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, opB, pa, M, Bd, pb, Cd, ct,
                                         CUSPARSE_SPMM_BLOCKED_ELL_ALG1, &bytes);
  const int s1 = cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, opB, pa, M, Bd, pb, Cd, ct,
                              CUSPARSE_SPMM_BLOCKED_ELL_ALG1, work);
  const auto out = download(dC, C.size());
  const auto d = A.dense(false);
  double err = 0, scale = 1;
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < N; ++j) {
      double w = beta;
      for (int k = 0; k < A.cols; ++k) {
        const size_t bi = transB ? (row_major ? (size_t)j * bcols + k : (size_t)k * brows + j)
                                 : (row_major ? (size_t)k * bcols + j : (size_t)j * brows + k);
        w += 2 * d[(size_t)i * A.cols + k] * to_d(B[bi]);
      }
      const double g = to_d(out[row_major ? (size_t)i * N + j : (size_t)j * A.rows + i]);
      err = std::max(err, std::fabs(g - w));
      scale = std::max(scale, std::fabs(w));
    }
  std::snprintf(what, sizeof what, "Blocked-ELL SpMM %s, %s B and C%s (padding block as block column 0)", name,
                row_major ? "row-major" : "column-major", transB ? ", op(B) = B^T" : "");
  check(s0 == 0 && s1 == 0 && err / scale <= tol, what, err / scale);
  cusparseDestroySpMat(M);
  cusparseDestroyDnMat(Bd);
  cusparseDestroyDnMat(Cd);
  cudaFree(dbci); cudaFree(dav); cudaFree(dB); cudaFree(dC);
}

static void blocked_ell() {
  Bell A{6, 8, 2, 4, {0, 2, 1, -1, 3, 0}, {}};
  A.vals.resize((size_t)A.rows * A.ell_cols);
  for (size_t i = 0; i < A.vals.size(); ++i) A.vals[i] = 1 + (double)(i % 7);
  bell_spmm<float, float>(A, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, "single", false, false, 1e-6);
  bell_spmm<float, float>(A, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, "single", true, false, 1e-6);
  bell_spmm<float, float>(A, CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, "single", false, true, 1e-6);
  bell_spmm<double, double>(A, CUDA_R_64F, CUDA_R_64F, CUDA_R_64F, "double", false, false, 1e-12);
  bell_spmm<__half, __half>(A, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, "half, single compute", false, false, 2e-3);
  bell_spmm<__half, float>(A, CUDA_R_16F, CUDA_R_32F, CUDA_R_32F, "half into single", false, false, 1e-6);
  bell_spmm<signed char, int>(A, CUDA_R_8I, CUDA_R_32I, CUDA_R_32I, "int8 into int32", false, false, 0);

  std::vector<float> av(A.vals.begin(), A.vals.end());
  int* dbci = upload(A.bci);
  float* dav = upload(av);
  cusparseSpMatDescr_t M;
  cusparseCreateBlockedEll(&M, A.rows, A.cols, A.bs, A.ell_cols, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
  int64_t r = 0, c = 0, nnz = 0;
  cusparseSpMatGetSize(M, &r, &c, &nnz);
  check(r == 6 && c == 8 && nnz == 6 * 8 * 2, "Blocked-ELL GetSize: nnz is rows * cols * blockSize, as NVIDIA's says",
        (double)nnz);
  cusparseFormat_t f;
  check(cusparseSpMatGetFormat(M, &f) == 0 && f == CUSPARSE_FORMAT_BLOCKED_ELL, "Blocked-ELL format");
  int64_t gr, gc, gb, ge;
  void *gi, *gv;
  cusparseIndexType_t it;
  cusparseIndexBase_t gbase;
  cudaDataType gt;
  check(cusparseBlockedEllGet(M, &gr, &gc, &gb, &ge, &gi, &gv, &it, &gbase, &gt) == 0 && gb == 2 && ge == 4 &&
            gi == dbci && gv == dav,
        "BlockedEllGet");
  const void *cgi, *cgv;
  check(cusparseConstBlockedEllGet(M, &gr, &gc, &gb, &ge, &cgi, &cgv, &it, &gbase, &gt) == 0 && cgi == dbci,
        "ConstBlockedEllGet");
  void *p1, *p2, *p3;
  cusparseIndexType_t i1, i2;
  expect(cusparseCsrGet(M, &gr, &gc, &nnz, &p1, &p2, &p3, &i1, &i2, &gbase, &gt), 3, "CsrGet of a Blocked-ELL matrix");
  // DenseToSparse fills the values of the blocks the column indices name.
  std::vector<float> D((size_t)A.rows * A.cols);
  for (int i = 0; i < A.rows; ++i)
    for (int j = 0; j < A.cols; ++j) D[(size_t)j * A.rows + i] = (float)(10 * i + j);
  float* dD = upload(D);
  cusparseDnMatDescr_t Dd;
  cusparseCreateDnMat(&Dd, A.rows, A.cols, A.rows, dD, CUDA_R_32F, CUSPARSE_ORDER_COL);
  std::vector<float> junk(av.size(), -9.0f);
  float* dz = upload(junk);
  cusparseSpMatDescr_t E;
  cusparseCreateBlockedEll(&E, A.rows, A.cols, A.bs, A.ell_cols, dbci, dz, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                           CUDA_R_32F);
  size_t bytes = 0;
  const int d0 = cusparseDenseToSparse_bufferSize(h, Dd, E, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, &bytes);
  const int d1 = cusparseDenseToSparse_analysis(h, Dd, E, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, work);
  const int d2 = cusparseDenseToSparse_convert(h, Dd, E, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, work);
  const auto ev = download(dz, av.size());
  const std::vector<float> want{0, 1, 4, 5, 10, 11, 14, 15, 22, 23, 0, 0, 32, 33, 0, 0, 46, 47, 40, 41, 56, 57, 50, 51};
  check(d0 == 0 && d1 == 0 && d2 == 0 && ev == want, "DenseToSparse into Blocked-ELL, padding blocks zeroed");
  // What NVIDIA's refuses.
  std::vector<float> x(8, 1), y(6, 0);
  float *dx = upload(x), *dy = upload(y);
  cusparseDnVecDescr_t X, Y;
  cusparseCreateDnVec(&X, 8, dx, CUDA_R_32F);
  cusparseCreateDnVec(&Y, 6, dy, CUDA_R_32F);
  const float al = 1, be = 0;
  expect(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, M, X, &be, Y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, work),
         10, "Blocked-ELL SpMV");
  expect(cusparseSparseToDense(h, M, Dd, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, work), 10, "Blocked-ELL SparseToDense");
  std::vector<float> Bv(8 * 3, 1), Cv(6 * 3, 0);
  float *dB = upload(Bv), *dC = upload(Cv);
  cusparseDnMatDescr_t Bd, Cd, Bt, Ct;
  cusparseCreateDnMat(&Bd, 8, 3, 8, dB, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&Cd, 6, 3, 6, dC, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&Bt, 6, 3, 6, dB, CUDA_R_32F, CUSPARSE_ORDER_COL);
  cusparseCreateDnMat(&Ct, 8, 3, 8, dC, CUDA_R_32F, CUSPARSE_ORDER_COL);
  expect(cusparseSpMM(h, CUSPARSE_OPERATION_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, M, Bt, &be, Ct, CUDA_R_32F,
                      CUSPARSE_SPMM_BLOCKED_ELL_ALG1, work),
         10, "Blocked-ELL SpMM op(A) = A^T");
  expect(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, M, Bd, &be, Cd,
                      CUDA_R_32F, CUSPARSE_SPMM_CSR_ALG1, work),
         3, "Blocked-ELL SpMM with CUSPARSE_SPMM_CSR_ALG1");
  const double ald = 1, bed = 0;
  expect(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ald, M, Bd, &bed, Cd,
                      CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, work),
         10, "Blocked-ELL SpMM single values, double compute");
  {
    std::vector<cuComplex> cv(av.size(), make_cuComplex(1, 0)), cb(8 * 3, make_cuComplex(1, 0)), cc(6 * 3);
    cuComplex *dcv = upload(cv), *dcb = upload(cb), *dcc = upload(cc);
    cusparseSpMatDescr_t Mc;
    cusparseDnMatDescr_t Bc, Cc;
    cusparseCreateBlockedEll(&Mc, 6, 8, 2, 4, dbci, dcv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_C_32F);
    cusparseCreateDnMat(&Bc, 8, 3, 8, dcb, CUDA_C_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&Cc, 6, 3, 6, dcc, CUDA_C_32F, CUSPARSE_ORDER_COL);
    const cuComplex ca = make_cuComplex(1, 0), cz = make_cuComplex(0, 0);
    expect(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &ca, Mc, Bc, &cz, Cc,
                        CUDA_C_32F, CUSPARSE_SPMM_ALG_DEFAULT, work),
           10, "Blocked-ELL SpMM complex");
  }
  cusparseSpMatDescr_t T;
  expect(cusparseCreateBlockedEll(&T, 5, 8, 2, 4, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateBlockedEll rows not a multiple of the block size");
  expect(cusparseCreateBlockedEll(&T, 6, 7, 2, 4, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateBlockedEll cols not a multiple of the block size");
  expect(cusparseCreateBlockedEll(&T, 6, 8, 2, 3, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateBlockedEll ellCols not a multiple of the block size");
  expect(cusparseCreateBlockedEll(&T, 6, 8, 2, 10, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateBlockedEll ellCols > cols");
  expect(cusparseCreateBlockedEll(&T, 6, 8, 0, 4, dbci, dav, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F), 3,
         "CreateBlockedEll block size 0");
}

/* ---- sliced ELL ---- */

using CreateSellFn = cusparseStatus_t (*)(cusparseSpMatDescr_t*, int64_t, int64_t, int64_t, int64_t, int64_t, void*,
                                          void*, void*, cusparseIndexType_t, cusparseIndexType_t, cusparseIndexBase_t,
                                          cudaDataType);
static const cusparseSpMVAlg_t kSellAlg1 = static_cast<cusparseSpMVAlg_t>(5);  // CUSPARSE_SPMV_SELL_ALG1, CUDA 12.1+

static void sliced_ell() {
  auto create = reinterpret_cast<CreateSellFn>(dlsym(RTLD_DEFAULT, "cusparseCreateSlicedEll"));
  if (!create) {
    std::printf("SKIP sliced ELL: this cuSPARSE has no cusparseCreateSlicedEll\n");
    return;
  }
  // 5 x 6, slices of 2 rows, slots column by column within a slice.
  const int R = 5, C = 6, S = 2;
  const std::vector<std::vector<std::pair<int, double>>> rows{
      {{0, 1}, {3, 2}}, {{1, 3}}, {{0, 4}, {2, 5}, {5, 6}}, {}, {{4, 7}, {5, 8}}};
  std::vector<int> off{0, 4, 10, 14}, col{0, 1, 3, -1, 0, -1, 2, -1, 5, -1, 4, -1, 5, -1};
  std::vector<float> val{1, 3, 2, 0, 4, 0, 5, 0, 6, 0, 7, 0, 8, 0};
  val[3] = 99;  // a padding slot's value is never read
  int *doff = upload(off), *dcol = upload(col);
  float* dval = upload(val);
  cusparseSpMatDescr_t A;
  if (create(&A, R, C, 8, 14, S, doff, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
             CUDA_R_32F)) {
    check(false, "CreateSlicedEll");
    return;
  }
  std::vector<double> dense((size_t)R * C, 0);
  for (int r = 0; r < R; ++r)
    for (const auto& [c, v] : rows[(size_t)r]) dense[(size_t)r * C + c] = v;
  const float al = 2, be = 0.5f;
  for (int op = 0; op < 2; ++op) {
    const bool t = op == 1;
    const int xn = t ? R : C, yn = t ? C : R;
    std::vector<float> x(xn), y(yn, 1.0f);
    for (int i = 0; i < xn; ++i) x[i] = 0.5f + i;
    float *dx = upload(x), *dy = upload(y);
    cusparseDnVecDescr_t X, Y;
    cusparseCreateDnVec(&X, xn, dx, CUDA_R_32F);
    cusparseCreateDnVec(&Y, yn, dy, CUDA_R_32F);
    size_t bytes = 0;
    const cusparseOperation_t o = t ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const int s0 = cusparseSpMV_bufferSize(h, o, &al, A, X, &be, Y, CUDA_R_32F, kSellAlg1, &bytes);
    const int s1 = cusparseSpMV(h, o, &al, A, X, &be, Y, CUDA_R_32F, kSellAlg1, work);
    const auto out = download(dy, yn);
    std::vector<double> want(yn, 0.5), got(out.begin(), out.end());
    for (int r = 0; r < R; ++r)
      for (int c = 0; c < C; ++c) {
        const double a = dense[(size_t)r * C + c];
        if (t) want[c] += 2 * a * x[r];
        else want[r] += 2 * a * x[c];
      }
    check(s0 == 0 && s1 == 0 && max_abs_diff(got, want) < 1e-4,
          t ? "sliced ELL SpMV op = A^T" : "sliced ELL SpMV op = A, padding slots skipped", max_abs_diff(got, want));
    if (!t) {
      expect(cusparseSpMV(h, o, &al, A, X, &be, Y, CUDA_R_32F, CUSPARSE_SPMV_CSR_ALG1, work), 3,
             "sliced ELL SpMV with CUSPARSE_SPMV_CSR_ALG1");
      expect(cusparseSpMV(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &al, A, Y, &be, X, CUDA_R_32F, kSellAlg1, work), 3,
             "sliced ELL SpMV op = A^H of a real matrix");
      // One-based: column indices and slice offsets both.
      std::vector<int> off1(off), col1(col);
      for (int& v : off1) ++v;
      for (int& v : col1)
        if (v >= 0) ++v;
      int *doff1 = upload(off1), *dcol1 = upload(col1);
      cusparseSpMatDescr_t A1;
      create(&A1, R, C, 8, 14, S, doff1, dcol1, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ONE,
             CUDA_R_32F);
      cudaMemcpy(dy, y.data(), yn * sizeof(float), cudaMemcpyHostToDevice);
      const int s2 = cusparseSpMV(h, o, &al, A1, X, &be, Y, CUDA_R_32F, kSellAlg1, work);
      const auto o1 = download(dy, yn);
      std::vector<double> g1(o1.begin(), o1.end());
      check(s2 == 0 && max_abs_diff(g1, want) < 1e-4, "sliced ELL SpMV, one-based offsets and columns",
            max_abs_diff(g1, want));
      cusparseDestroySpMat(A1);
      // What NVIDIA's refuses sliced ELL in.
      std::vector<float> Bv(C * 2, 1), Cv(R * 2, 0), Dv(R * C, 0), Av(R * 3, 1), Bw(3 * C, 1);
      float *dB = upload(Bv), *dC = upload(Cv), *dD = upload(Dv), *dA3 = upload(Av), *dB3 = upload(Bw);
      cusparseDnMatDescr_t Bd, Cd, Dd, A3, B3;
      cusparseCreateDnMat(&Bd, C, 2, C, dB, CUDA_R_32F, CUSPARSE_ORDER_COL);
      cusparseCreateDnMat(&Cd, R, 2, R, dC, CUDA_R_32F, CUSPARSE_ORDER_COL);
      cusparseCreateDnMat(&Dd, R, C, R, dD, CUDA_R_32F, CUSPARSE_ORDER_COL);
      cusparseCreateDnMat(&A3, R, 3, R, dA3, CUDA_R_32F, CUSPARSE_ORDER_COL);
      cusparseCreateDnMat(&B3, 3, C, 3, dB3, CUDA_R_32F, CUSPARSE_ORDER_COL);
      expect(cusparseSpMM(h, o, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, A, Bd, &be, Cd, CUDA_R_32F,
                          CUSPARSE_SPMM_ALG_DEFAULT, work),
             10, "sliced ELL SpMM");
      expect(cusparseSparseToDense(h, A, Dd, CUSPARSE_SPARSETODENSE_ALG_DEFAULT, work), 10, "sliced ELL SparseToDense");
      expect(cusparseDenseToSparse_analysis(h, Dd, A, CUSPARSE_DENSETOSPARSE_ALG_DEFAULT, work), 10,
             "sliced ELL DenseToSparse");
      expect(cusparseSDDMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &al, A3, B3,
                                      &be, A, CUDA_R_32F, CUSPARSE_SDDMM_ALG_DEFAULT, &bytes),
             10, "sliced ELL SDDMM");
    }
  }
  // Complex, op = A^H.
  {
    std::vector<cuComplex> cv(14);
    for (int i = 0; i < 14; ++i) cv[i] = make_cuComplex(val[i], 0.5f * val[i]);
    cuComplex* dcv = upload(cv);
    cusparseSpMatDescr_t Ac;
    create(&Ac, R, C, 8, 14, S, doff, dcol, dcv, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
           CUDA_C_32F);
    std::vector<cuComplex> x(R), y(C, make_cuComplex(0, 0));
    for (int i = 0; i < R; ++i) x[i] = make_cuComplex(1.0f + i, 0.25f);
    cuComplex *dx = upload(x), *dy = upload(y);
    cusparseDnVecDescr_t X, Y;
    cusparseCreateDnVec(&X, R, dx, CUDA_C_32F);
    cusparseCreateDnVec(&Y, C, dy, CUDA_C_32F);
    const cuComplex one = make_cuComplex(1, 0), zero = make_cuComplex(0, 0);
    const int st = cusparseSpMV(h, CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE, &one, Ac, X, &zero, Y, CUDA_C_32F, kSellAlg1, work);
    const auto out = download(dy, C);
    double err = 0;
    for (int c = 0; c < C; ++c) {
      cdouble w = 0;
      for (int r = 0; r < R; ++r) {
        const double a = dense[(size_t)r * C + c];
        w += std::conj(cdouble(a, 0.5 * a)) * cdouble(x[r].x, x[r].y);
      }
      err = std::max(err, std::abs(cdouble(out[c].x, out[c].y) - w));
    }
    check(st == 0 && err < 1e-4, "sliced ELL SpMV complex, op = A^H", err);
  }
  int64_t r = 0, c = 0, nnz = 0;
  cusparseSpMatGetSize(A, &r, &c, &nnz);
  cusparseFormat_t f;
  cusparseSpMatGetFormat(A, &f);
  check(r == R && c == C && nnz == 8 && (int)f == 7, "sliced ELL GetSize and GetFormat");
  cusparseSpMatDescr_t T;
  expect(create(&T, R, C, 8, 7, S, doff, dcol, dval, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F),
         3, "CreateSlicedEll sellValuesSize < nnz");
  expect(create(&T, R, C, 8, 14, S, doff, dcol, dval, CUSPARSE_INDEX_64I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                CUDA_R_32F),
         10, "CreateSlicedEll offsets and columns of different index types");
  cusparseDestroySpMat(A);
}

int main() {
  if (cusparseCreate(&h) != CUSPARSE_STATUS_SUCCESS) {
    std::printf("FAIL cusparseCreate\n");
    return 1;
  }
  cudaMalloc(&work, 1 << 22);
  descriptors();
  spvv();
  vector_ops();
  gemvi();
  blocked_ell();
  sliced_ell();
  cudaFree(work);
  cusparseDestroy(h);
  std::printf("%s (%d failure%s)\n", failures ? "FAIL" : "PASS", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
