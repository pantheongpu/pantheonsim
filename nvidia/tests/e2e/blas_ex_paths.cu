// cuBLAS entry points HeCBench's programs reach for, checked against what an
// RTX 3060's cuBLAS answers: the typed level-1 routines (cublasDotEx_64 is
// f16sp's), the matrix add geam (geam-cuda) and batched least squares
// gelsBatched (gels-cuda). Every check passes on the card too.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <complex>
#include <cstdio>
#include <cstring>
#include <vector>

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

template <class T> static T* dev(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 16);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> host(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

// ---- the typed level-1 routines ----
static void level1(cublasHandle_t h) {
  const int n = 5;
  const float xf[n] = {1, 2, 3, 4, 5}, yf[n] = {0.5f, -1, 2, 0.25f, 3};   // x.y = 20.5
  std::vector<__half> xh(n), yh(n);
  std::vector<__nv_bfloat16> xb(n), yb(n);
  std::vector<double> xd(n), yd(n);
  std::vector<std::complex<float>> xc(n), yc(n);
  for (int i = 0; i < n; ++i) {
    xh[i] = __float2half(xf[i]);
    yh[i] = __float2half(yf[i]);
    xb[i] = __float2bfloat16(xf[i]);
    yb[i] = __float2bfloat16(yf[i]);
    xd[i] = xf[i];
    yd[i] = yf[i];
    xc[i] = {xf[i], (float)i};
    yc[i] = {yf[i], -1.0f};
  }
  __half* dxh = dev(xh);
  __half* dyh = dev(yh);
  __half rh;
  IS(cublasDotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == 20.5f, "half dot, accumulated in single precision");
  IS(cublasDotEx_64(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == 20.5f, "cublasDotEx_64");
  IS(cublasDotEx(h, n, dxh, CUDA_R_16F, -1, dyh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == 8.0f, "a negative increment walks x from its end");
  IS(cublasDotEx(h, 0, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == 0.0f, "an empty dot is 0");
  float rf = -1;
  IS(cublasDotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &rf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasDotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_16F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasDotEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_32F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasDotEx(h, n, dxh, CUDA_R_8I, 1, dyh, CUDA_R_8I, 1, &rf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  __nv_bfloat16* dxb = dev(xb);
  __nv_bfloat16* dyb = dev(yb);
  __nv_bfloat16 rb;
  IS(cublasDotEx(h, n, dxb, CUDA_R_16BF, 1, dyb, CUDA_R_16BF, 1, &rb, CUDA_R_16BF, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__bfloat162float(rb) == 20.5f, "bfloat16 dot");
  double* dxd = dev(xd);
  double* dyd = dev(yd);
  double rd = 0;
  IS(cublasDotEx(h, n, dxd, CUDA_R_64F, 1, dyd, CUDA_R_64F, 1, &rd, CUDA_R_64F, CUDA_R_64F), CUBLAS_STATUS_SUCCESS);
  check(rd == 20.5, "double dot");
  IS(cublasDotEx(h, n, dxd, CUDA_R_64F, 1, dyd, CUDA_R_64F, 1, &rd, CUDA_R_64F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  auto* dxc = dev(xc);
  auto* dyc = dev(yc);
  std::complex<float> rc;
  IS(cublasDotEx(h, n, dxc, CUDA_C_32F, 1, dyc, CUDA_C_32F, 1, &rc, CUDA_C_32F, CUDA_C_32F), CUBLAS_STATUS_SUCCESS);
  check(rc == std::complex<float>(30.5f, 0.75f), "complex dot");
  IS(cublasDotcEx(h, n, dxc, CUDA_C_32F, 1, dyc, CUDA_C_32F, 1, &rc, CUDA_C_32F, CUDA_C_32F), CUBLAS_STATUS_SUCCESS);
  check(rc == std::complex<float>(10.5f, -30.75f), "conjugated complex dot");
  IS(cublasDotcEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == 20.5f, "conjugated dot of real halves");

  IS(cublasNrm2Ex(h, n, dxh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == __half2float(__float2half(std::sqrt(55.0f))), "half norm");
  IS(cublasNrm2Ex(h, n, dxh, CUDA_R_16F, 1, &rf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasNrm2Ex(h, n, dxh, CUDA_R_16F, -1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(rh) == 0.0f, "no norm for a negative increment");
  IS(cublasNrm2Ex(h, n, dxc, CUDA_C_32F, 1, &rf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(std::fabs(rf - std::sqrt(85.0f)) < 1e-5f, "a complex norm is real");
  IS(cublasNrm2Ex(h, n, dxc, CUDA_C_32F, 1, &rf, CUDA_R_32F, CUDA_C_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasAsumEx(h, n, dxh, CUDA_R_16F, 1, &rf, CUDA_R_32F, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(rf == 15.0f, "half sum of magnitudes, into single precision");
  IS(cublasAsumEx(h, n, dxh, CUDA_R_16F, 1, &rh, CUDA_R_16F, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasAsumEx(h, n, dxc, CUDA_C_32F, 1, &rf, CUDA_R_32F, CUDA_C_32F), CUBLAS_STATUS_SUCCESS);
  check(rf == 25.0f, "complex sum of |re| + |im|");
  int idx = -1;
  IS(cublasIamaxEx(h, n, dyh, CUDA_R_16F, 1, &idx), CUBLAS_STATUS_SUCCESS);
  check(idx == 5, "half i?amax");
  IS(cublasIaminEx(h, n, dyh, CUDA_R_16F, 1, &idx), CUBLAS_STATUS_SUCCESS);
  check(idx == 4, "half i?amin");
  IS(cublasIamaxEx(h, n, dxh, CUDA_R_8I, 1, &idx), CUBLAS_STATUS_NOT_SUPPORTED);

  // y = 2x + y, in halves, the scalar in single precision.
  const float two = 2.0f;
  const __half two_h = __float2half(2.0f);
  IS(cublasAxpyEx(h, n, &two, CUDA_R_32F, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  auto y = host(dyh, n);
  check(__half2float(y[0]) == 2.5f && __half2float(y[4]) == 13.0f, "half axpy");
  IS(cublasAxpyEx(h, n, &two_h, CUDA_R_16F, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1, CUDA_R_32F),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasAxpyEx_64(h, n, &two, CUDA_R_32F, dxh, CUDA_R_16F, -1, dyh, CUDA_R_16F, 1, CUDA_R_32F),
     CUBLAS_STATUS_SUCCESS);
  y = host(dyh, n);
  check(__half2float(y[0]) == 12.5f && __half2float(y[4]) == 15.0f, "half axpy, x walked from its end");
  IS(cublasScalEx(h, n, &two, CUDA_R_32F, dxh, CUDA_R_16F, 1, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  auto x = host(dxh, n);
  check(__half2float(x[0]) == 2.0f && __half2float(x[4]) == 10.0f, "half scal");
  IS(cublasScalEx(h, n, &two, CUDA_R_32F, dxh, CUDA_R_16F, -1, CUDA_R_32F), CUBLAS_STATUS_SUCCESS);
  check(__half2float(host(dxh, 1)[0]) == 2.0f, "no scal for a negative increment");
  IS(cublasScalEx(h, n, &two_h, CUDA_R_16F, dxh, CUDA_R_16F, 1, CUDA_R_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasCopyEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_16F, 1), CUBLAS_STATUS_SUCCESS);
  check(__half2float(host(dyh, n)[4]) == 10.0f, "half copy");
  IS(cublasCopyEx(h, n, dxh, CUDA_R_16F, 1, dyh, CUDA_R_32F, 1), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSwapEx(h, n, dxd, CUDA_R_64F, 1, dyd, CUDA_R_64F, 1), CUBLAS_STATUS_SUCCESS);
  check(host(dxd, n)[0] == 0.5 && host(dyd, n)[0] == 1.0, "double swap");

  // The result through a device pointer.
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
  double* dres = nullptr;
  cudaMalloc(&dres, sizeof(double));
  IS(cublasDotEx(h, n, dxd, CUDA_R_64F, 1, dyd, CUDA_R_64F, 1, dres, CUDA_R_64F, CUDA_R_64F), CUBLAS_STATUS_SUCCESS);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
  check(host(dres, 1)[0] == 20.5, "the result through a device pointer");
  for (void* p : {(void*)dxh, (void*)dyh, (void*)dxb, (void*)dyb, (void*)dxd, (void*)dyd, (void*)dxc, (void*)dyc,
                  (void*)dres})
    cudaFree(p);
}

// ---- geam ----
static void geam(cublasHandle_t h) {
  const int m = 3, n = 2;
  std::vector<float> A(12), B(12), C(12, -1);
  for (int i = 0; i < 12; ++i) {
    A[i] = i + 1;
    B[i] = 100 + i;
  }
  float* dA = dev(A);
  float* dB = dev(B);
  float* dC = dev(C);
  auto reset = [&] {
    cudaMemcpy(dA, A.data(), 48, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), 48, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), 48, cudaMemcpyHostToDevice);
  };
  auto is = [&](float* d, std::vector<float> want, const char* what) { check(host(d, 12) == want, what); };
  const float al = 2, be = 1, zero = 0;
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  is(dC, {102, 105, 108, -1, 114, 117, 120, -1, -1, -1, -1, -1}, "C = 2A + B, padding untouched");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &al, dA, 2, &be, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  is(dC, {102, 107, 112, -1, 108, 113, 118, -1, -1, -1, -1, -1}, "C = 2A^T + B");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_T, m, n, &al, dA, 2, &be, dB, 2, dC, 3), CUBLAS_STATUS_SUCCESS);
  is(dC, {102, 108, 114, 105, 111, 117, -1, -1, -1, -1, -1, -1}, "C = 2A^T + B^T");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &zero, nullptr, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  is(dC, {2, 4, 6, -1, 10, 12, 14, -1, -1, -1, -1, -1}, "beta 0: B is not read, and may be NULL");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &zero, nullptr, 4, &be, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  is(dC, {100, 101, 102, -1, 104, 105, 106, -1, -1, -1, -1, -1}, "alpha 0: A is not read");
  std::vector<float> nan(12, NAN);
  cudaMemcpy(dB, nan.data(), 48, cudaMemcpyHostToDevice);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &zero, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  is(dC, {2, 4, 6, -1, 10, 12, 14, -1, -1, -1, -1, -1}, "NaN in B stays out of C when beta is 0");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dA, 4), CUBLAS_STATUS_SUCCESS);
  is(dA, {102, 105, 108, 4, 114, 117, 120, 8, 9, 10, 11, 12}, "in place over A");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dB, 4), CUBLAS_STATUS_SUCCESS);
  is(dB, {102, 105, 108, 103, 114, 117, 120, 107, 108, 109, 110, 111}, "in place over B");
  reset();
  IS(cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dA, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 3, &be, dB, 4, dA, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_T, m, n, &al, dA, 4, &be, dB, 4, dB, 4), CUBLAS_STATUS_INVALID_VALUE);
  is(dA, A, "and a refused one in place leaves A");
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 2, &be, dB, 4, dC, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_T, CUBLAS_OP_N, m, n, &al, dA, 1, &be, dB, 4, dC, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 2, dC, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dC, 2), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, -1, n, &al, dA, 4, &be, dB, 4, dC, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, (cublasOperation_t)7, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dC, 4),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, 0, n, &al, dA, 4, &be, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  IS(cublasSgeam(h, CUBLAS_OP_C, CUBLAS_OP_N, m, n, &al, dA, 2, &be, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  IS(cublasSgeam_64(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &al, dA, 4, &be, dB, 4, dC, 4), CUBLAS_STATUS_SUCCESS);
  // Doubles, and the conjugate transpose of a complex matrix.
  std::vector<double> Ad(A.begin(), A.end());
  double* dAd = dev(Ad);
  double* dCd = dev(std::vector<double>(12, -1));
  const double ald = 0.5;
  IS(cublasDgeam(h, CUBLAS_OP_N, CUBLAS_OP_N, m, n, &ald, dAd, 4, &ald, dAd, 4, dCd, 4), CUBLAS_STATUS_SUCCESS);
  check(host(dCd, 12)[5] == 6.0, "double geam, A + A halved twice over");
  std::vector<cuComplex> Ac(6), Bc(6, make_cuComplex(0, 0));
  for (int i = 0; i < 6; ++i) Ac[i] = make_cuComplex((float)i, 1);
  cuComplex* dAc = dev(Ac);
  cuComplex* dBc = dev(Bc);
  cuComplex* dCc = dev(Bc);
  const cuComplex one = make_cuComplex(1, 0), czero = make_cuComplex(0, 0);
  IS(cublasCgeam(h, CUBLAS_OP_C, CUBLAS_OP_N, 2, 3, &one, dAc, 3, &czero, dBc, 2, dCc, 2), CUBLAS_STATUS_SUCCESS);
  const auto cc = host(dCc, 6);
  const float re[6] = {0, 3, 1, 4, 2, 5};
  bool ok = true;
  for (int i = 0; i < 6; ++i) ok = ok && cc[i].x == re[i] && cc[i].y == -1;
  check(ok, "complex geam conjugates as it transposes");
  for (void* p : {(void*)dA, (void*)dB, (void*)dC, (void*)dAd, (void*)dCd, (void*)dAc, (void*)dBc, (void*)dCc})
    cudaFree(p);
}

// ---- gelsBatched ----
static void gels(cublasHandle_t h) {
  // Two 4 x 2 systems, lda 5 and ldc 6 with padding; the second matrix is
  // rank deficient (its second column is zero).
  const int m = 4, n = 2, nrhs = 2, lda = 5, ldc = 6;
  std::vector<float> A0 = {1, 2, 0, 1, -9, 0, 1, 3, -1, -9};   // full rank
  std::vector<float> A1 = {2, 0, 1, 3, -9, 0, 0, 0, 0, -9};
  // C0's columns are A0 (1, 2)^T and A0 (2, -1)^T, exactly.
  std::vector<float> C0 = {1, 4, 6, -1, -9, -9, 2, 3, -3, 3, -9, -9};
  std::vector<float> C1 = {1, 2, 3, 4, -9, -9, 1, 1, 1, 1, -9, -9};
  float* dA[2] = {dev(A0), dev(A1)};
  float* dC[2] = {dev(C0), dev(C1)};
  float** dAa = dev(std::vector<float*>(dA, dA + 2));
  float** dCa = dev(std::vector<float*>(dC, dC + 2));
  int* dinfo = dev(std::vector<int>{-7, -7});
  int info = -5;
  IS(cublasSgelsBatched(h, CUBLAS_OP_N, m, n, nrhs, dAa, lda, dCa, ldc, &info, dinfo, 2), CUBLAS_STATUS_SUCCESS);
  const auto di = host(dinfo, 2);
  check(info == 0 && di[0] == 0 && di[1] == 2, "info 0; devInfo flags the zero on the second R's diagonal");
  const auto x = host(dC[0], 12);
  check(std::fabs(x[0] - 1) < 1e-5f && std::fabs(x[1] - 2) < 1e-5f && std::fabs(x[6] - 2) < 1e-5f &&
            std::fabs(x[7] + 1) < 1e-5f && x[4] == -9 && x[11] == -9,
        "the least-squares solutions, in C's first n rows");
  const auto a = host(dA[0], 10);
  check(std::fabs(std::fabs(a[0]) - std::sqrt(6.0f)) < 1e-5f && a[4] == -9, "A holds R's diagonal");
  check(host(dA[1], 10)[5] == 0 && host(dA[1], 10)[6] == 0, "a zero column stays zero in R");
  auto call = [&](cublasOperation_t t, int mm, int nn, int rr, int la, int lc, int bb, int want_status,
                  int want_info, const char* what) {
    int inf = -5;
    const int s = cublasSgelsBatched(h, t, mm, nn, rr, dAa, la, dCa, lc, &inf, dinfo, bb);
    check(s == want_status && inf == want_info, what);
  };
  call(CUBLAS_OP_T, m, n, nrhs, lda, ldc, 2, CUBLAS_STATUS_NOT_SUPPORTED, 0, "OP_T is not supported");
  call(CUBLAS_OP_N, 2, 4, nrhs, lda, ldc, 2, CUBLAS_STATUS_NOT_SUPPORTED, 0, "m < n is not supported");
  call(CUBLAS_OP_N, -1, n, nrhs, lda, ldc, 2, CUBLAS_STATUS_INVALID_VALUE, -1, "m < 0: info -1");
  call(CUBLAS_OP_N, m, n, -1, lda, ldc, 2, CUBLAS_STATUS_INVALID_VALUE, -2, "nrhs < 0: info -2");
  call(CUBLAS_OP_N, m, -1, nrhs, lda, ldc, 2, CUBLAS_STATUS_INVALID_VALUE, -3, "n < 0: info -3");
  call(CUBLAS_OP_N, m, n, nrhs, 3, ldc, 2, CUBLAS_STATUS_INVALID_VALUE, -5, "lda < m: info -5");
  call(CUBLAS_OP_N, m, n, nrhs, lda, 3, 2, CUBLAS_STATUS_INVALID_VALUE, -7, "ldc < m: info -7");
  call(CUBLAS_OP_N, m, n, nrhs, lda, ldc, -1, CUBLAS_STATUS_INVALID_VALUE, -8, "a negative batch: info -8");
  call(CUBLAS_OP_N, m, n, nrhs, lda, ldc, 0, CUBLAS_STATUS_SUCCESS, 0, "an empty batch");
  IS(cublasSgelsBatched(h, CUBLAS_OP_N, m, n, nrhs, dAa, lda, dCa, ldc, nullptr, dinfo, 2), CUBLAS_STATUS_INVALID_VALUE);

  // A complex system, and a double one with no devInfoArray.
  std::vector<cuComplex> Ac = {{1, 1}, {2, 0}}, Cc = {{1, 0}, {0, 1}};
  cuComplex* dAc = dev(Ac);
  cuComplex* dCc = dev(Cc);
  cuComplex** dAca = dev(std::vector<cuComplex*>{dAc});
  cuComplex** dCca = dev(std::vector<cuComplex*>{dCc});
  IS(cublasCgelsBatched(h, CUBLAS_OP_N, 2, 1, 1, dAca, 2, dCca, 2, &info, dinfo, 1), CUBLAS_STATUS_SUCCESS);
  const auto xc = host(dCc, 1)[0];
  check(std::fabs(xc.x - 1.0f / 6) < 1e-6f && std::fabs(xc.y - 1.0f / 6) < 1e-6f, "complex least squares");
  IS(cublasCgelsBatched(h, CUBLAS_OP_C, 2, 1, 1, dAca, 2, dCca, 2, &info, dinfo, 1), CUBLAS_STATUS_NOT_SUPPORTED);
  std::vector<double> Ad = {3, 4}, Cd = {5, 10};
  double* dAd = dev(Ad);
  double* dCd = dev(Cd);
  double** dAda = dev(std::vector<double*>{dAd});
  double** dCda = dev(std::vector<double*>{dCd});
  IS(cublasDgelsBatched(h, CUBLAS_OP_N, 2, 1, 1, dAda, 2, dCda, 2, &info, nullptr, 1), CUBLAS_STATUS_SUCCESS);
  check(std::fabs(host(dCd, 1)[0] - 2.2) < 1e-12, "double least squares: (3*5 + 4*10) / 25");
}

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  level1(h);
  geam(h);
  gels(h);
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS checks\n" : "PASS: every cuBLAS check\n", failures);
  return failures ? 1 : 0;
}
