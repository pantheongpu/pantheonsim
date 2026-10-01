// cuBLAS entry points HeCBench's programs reach for, checked against what an
// RTX 3060's cuBLAS answers: the typed level-1 routines (cublasDotEx_64 is
// f16sp's), the matrix add geam (geam-cuda) and batched least squares
// gelsBatched (gels-cuda) and batched QR, geqrfBatched; and the GEMM family's Ex forms (GemmEx's type
// table, SgemmEx, and the complex CgemmEx, Cgemm3mEx, CherkEx, CsyrkEx and
// their 3m forms, over single-precision or int8 complex operands) and the
// grouped batched GEMMs. Every check passes on the card too.
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


// ---- geqrfBatched ----
// LAPACK's geqrf, as the card leaves it: R on and above the diagonal, the
// reflectors below it, min(m, n) scalars in Tau; the values are an RTX
// 3060's, to single-precision rounding.
static void geqrf(cublasHandle_t h) {
  const int m = 4, n = 3, lda = 5;
  const std::vector<float> A = {1, 2, 0, 1, -9, 0, 1, 3, -1, -9, 2, 2, 1, 0, -9};
  float* dA = dev(A);
  float* dT = dev(std::vector<float>(4, -7));
  float** pa = dev(std::vector<float*>{dA});
  float** pt = dev(std::vector<float*>{dT});
  int info = 99;
  IS(cublasSgeqrfBatched(h, m, n, pa, lda, pt, &info, 1), CUBLAS_STATUS_SUCCESS);
  const float wantA[15] = {-2.44948983f, 0.579795897f, 0, 0.289897949f, -9, -0.408248305f, -3.29140282f, 0.739881694f,
                           -0.275815666f, -9, -2.44948959f, -1.21528721f, -1.23412991f, -0.63189137f, -9};
  const float wantT[4] = {1.40824831f, 1.23190701f, 1.42929959f, -7};
  const auto ga = host(dA, 15), gt = host(dT, 4);
  bool ok = info == 0;
  for (int i = 0; i < 15; ++i) ok = ok && std::fabs(ga[i] - wantA[i]) < 2e-6f;
  for (int i = 0; i < 4; ++i) ok = ok && std::fabs(gt[i] - wantT[i]) < 2e-6f;
  check(ok, "sgeqrfBatched: R, the reflectors and tau as LAPACK leaves them; padding and tau[3] untouched");
  auto call = [&](int mm, int nn, int la, int bb, int want_status, int want_info, const char* what) {
    int inf = 99;
    const int st = cublasSgeqrfBatched(h, mm, nn, pa, la, pt, &inf, bb);
    check(st == want_status && inf == want_info, what);
  };
  call(-1, n, lda, 1, CUBLAS_STATUS_INVALID_VALUE, -1, "geqrf m < 0: info -1");
  call(m, -1, lda, 1, CUBLAS_STATUS_INVALID_VALUE, -2, "geqrf n < 0: info -2");
  call(m, n, 3, 1, CUBLAS_STATUS_INVALID_VALUE, -4, "geqrf lda < m: info -4");
  call(0, 0, 0, 1, CUBLAS_STATUS_INVALID_VALUE, -4, "geqrf lda 0, even for an empty matrix: info -4");
  call(m, n, lda, -1, CUBLAS_STATUS_INVALID_VALUE, -7, "geqrf a negative batch: info -7");
  call(m, 0, lda, 1, CUBLAS_STATUS_SUCCESS, 0, "geqrf with no columns");
  IS(cublasSgeqrfBatched(h, m, n, pa, lda, pt, nullptr, 1), CUBLAS_STATUS_INVALID_VALUE);
  // A complex 2 x 2, and m < n in double.
  std::vector<cuComplex> Ac = {{1, 1}, {2, 0}, {0, -1}, {3, 2}};
  cuComplex* dAc = dev(Ac);
  cuComplex* dTc = dev(std::vector<cuComplex>(2));
  cuComplex** pac = dev(std::vector<cuComplex*>{dAc});
  cuComplex** ptc = dev(std::vector<cuComplex*>{dTc});
  IS(cublasCgeqrfBatched(h, 2, 2, pac, 2, ptc, &info, 1), CUBLAS_STATUS_SUCCESS);
  const auto rc = host(dAc, 4), tc = host(dTc, 2);
  const float wc[12] = {-2.44948983f, 0, 0.534846902f, -0.155051008f, -2.04124141f, -1.22474468f, -2.88675141f, 0,
                        1.40824831f, 0.408248276f, 1.64896524f, 0.760818064f};
  ok = true;
  for (int i = 0; i < 4; ++i) ok = ok && std::fabs(rc[i].x - wc[2 * i]) < 2e-6f && std::fabs(rc[i].y - wc[2 * i + 1]) < 2e-6f;
  for (int i = 0; i < 2; ++i) ok = ok && std::fabs(tc[i].x - wc[8 + 2 * i]) < 2e-6f && std::fabs(tc[i].y - wc[9 + 2 * i]) < 2e-6f;
  check(ok, "cgeqrfBatched: a real R diagonal, complex tau");
  std::vector<double> Ad = {3, 4, 1, 2, 0, 5};   // 2 x 3
  double* dAd = dev(Ad);
  double* dTd = dev(std::vector<double>(3, -7));
  double** pad = dev(std::vector<double*>{dAd});
  double** ptd = dev(std::vector<double*>{dTd});
  IS(cublasDgeqrfBatched(h, 2, 3, pad, 2, ptd, &info, 1), CUBLAS_STATUS_SUCCESS);
  const auto rd = host(dAd, 6), td = host(dTd, 3);
  check(std::fabs(rd[0] + 5) < 1e-14 && std::fabs(td[0] - 1.6) < 1e-14 && std::fabs(rd[1] - 0.5) < 1e-14 &&
            td[2] == -7,
        "dgeqrfBatched with m < n: two reflectors");
  for (void* p : {(void*)dA, (void*)dT, (void*)pa, (void*)pt, (void*)dAc, (void*)dTc, (void*)pac, (void*)ptc,
                  (void*)dAd, (void*)dTd, (void*)pad, (void*)ptd})
    cudaFree(p);
}

// ---- the GEMM family's Ex forms ----
// Small integers throughout, so every product and sum is exact and the card's
// order of operations cannot show: the answers are compared exactly.
using cplx = std::complex<double>;
static cplx at(const std::vector<char2>& v, int i) { return {(double)v[i].x, (double)v[i].y}; }
static cplx at(const std::vector<cuComplex>& v, int i) { return {v[i].x, v[i].y}; }
template <class V>
static cplx opel(const V& a, int ld, cublasOperation_t op, int i, int j) {   // op(A)(i, j)
  const cplx v = op == CUBLAS_OP_N ? at(a, j * ld + i) : at(a, i * ld + j);
  return op == CUBLAS_OP_C ? std::conj(v) : v;
}
static bool same(const std::vector<cuComplex>& got, const std::vector<cplx>& want) {
  for (size_t i = 0; i < got.size(); ++i)
    if (got[i].x != (float)want[i].real() || got[i].y != (float)want[i].imag()) return false;
  return true;
}

static void gemm_ex(cublasHandle_t h) {
  const int m = 3, n = 4, k = 5, ld = 6;
  std::vector<char2> A8(ld * 6), B8(ld * 6);
  std::vector<cuComplex> Af(ld * 6), Bf(ld * 6), C0(ld * n);
  for (int i = 0; i < ld * 6; ++i) {
    A8[i] = make_char2((char)(i % 7 - 3), (char)(i % 5 - 2));
    B8[i] = make_char2((char)(i % 3 - 1), (char)(2 - i % 4));
    Af[i] = make_cuComplex(A8[i].x, A8[i].y);
    Bf[i] = make_cuComplex(B8[i].x, B8[i].y);
  }
  for (int i = 0; i < ld * n; ++i) C0[i] = make_cuComplex((float)(i % 9 - 4), (float)(i % 2));
  char2* dA8 = dev(A8);
  char2* dB8 = dev(B8);
  cuComplex* dAf = dev(Af);
  cuComplex* dBf = dev(Bf);
  cuComplex* dC = dev(C0);
  const cuComplex al = make_cuComplex(2, -1), be = make_cuComplex(1, 3);
  auto want_gemm = [&](cublasOperation_t ta, cublasOperation_t tb) {
    std::vector<cplx> c(ld * n);
    for (int i = 0; i < ld * n; ++i) c[i] = at(C0, i);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        cplx s = 0;
        for (int p = 0; p < k; ++p) s += opel(A8, ld, ta, i, p) * opel(B8, ld, tb, p, j);
        c[j * ld + i] = cplx(2, -1) * s + cplx(1, 3) * c[j * ld + i];
      }
    return c;
  };
  auto reset = [&] { cudaMemcpy(dC, C0.data(), C0.size() * sizeof(cuComplex), cudaMemcpyHostToDevice); };
  const cublasOperation_t N = CUBLAS_OP_N, T = CUBLAS_OP_T, C = CUBLAS_OP_C;
  reset();
  IS(cublasCgemmEx(h, N, N, m, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * n), want_gemm(N, N)), "CgemmEx over int8 pairs, padding rows untouched");
  reset();
  IS(cublasCgemmEx(h, C, T, m, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * n), want_gemm(C, T)), "CgemmEx, A conjugate-transposed and B transposed");
  reset();
  IS(cublasCgemmEx_64(h, T, C, m, n, k, &al, dAf, CUDA_C_32F, ld, dBf, CUDA_C_32F, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * n), want_gemm(T, C)), "cublasCgemmEx_64 over single-precision complex");
  reset();
  IS(cublasCgemm3mEx(h, N, C, m, n, k, &al, dAf, CUDA_C_32F, ld, dBf, CUDA_C_32F, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * n), want_gemm(N, C)), "Cgemm3mEx: the same product");
  reset();
  IS(cublasGemmEx(h, N, N, m, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * n), want_gemm(N, N)), "GemmEx takes the int8 complex form too");
  // The types the complex Ex GEMMs take: C_32F or C_8I in, C_32F out.
  IS(cublasCgemmEx(h, N, N, m, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_64F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasCgemmEx(h, N, N, m, n, k, &al, dA8, CUDA_C_8I, ld, dBf, CUDA_C_32F, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasCgemm3mEx(h, N, N, m, n, k, &al, dA8, CUDA_R_32F, ld, dB8, CUDA_R_32F, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  // A bad handle comes before a bad type, and a bad type before a bad size.
  IS(cublasCgemmEx(nullptr, N, N, m, n, k, &al, dA8, CUDA_R_32F, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_INITIALIZED);
  IS(cublasCgemmEx(h, N, N, -1, n, k, &al, dA8, CUDA_R_32F, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasCgemmEx(h, N, N, -1, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasCgemmEx(h, T, N, m, n, 7, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, 7, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_INVALID_VALUE);   // lda 6 is short of op(A)'s 7 rows
  IS(cublasCgemmEx(h, N, N, m, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, 2),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasCgemmEx(h, N, N, 0, n, k, &al, dA8, CUDA_C_8I, 0, dB8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, 1),
     CUBLAS_STATUS_INVALID_VALUE);   // a leading dimension is at least 1, even for an empty matrix
  IS(cublasCgemmEx(h, (cublasOperation_t)5, N, m, n, k, &al, dA8, CUDA_C_8I, ld, dB8, CUDA_C_8I, ld, &be, dC,
                   CUDA_C_32F, ld),
     CUBLAS_STATUS_INVALID_VALUE);

  // herk and syrk: only C's `uplo` triangle written; herk's diagonal real.
  // herk takes T as C, and syrk C as T, as the card does.
  const int nn = 4, kk = 3;
  auto want_rk = [&](bool herm, bool lower, cublasOperation_t t, cplx alpha, cplx beta) {
    std::vector<cplx> c(ld * nn);
    for (int i = 0; i < ld * nn; ++i) c[i] = at(C0, i);
    for (int j = 0; j < nn; ++j)
      for (int i = lower ? j : 0; i < (lower ? nn : j + 1); ++i) {
        cplx s = 0;
        for (int p = 0; p < kk; ++p) {
          const cplx x = t == CUBLAS_OP_N ? at(A8, p * ld + i) : at(A8, i * ld + p);
          const cplx y = t == CUBLAS_OP_N ? at(A8, p * ld + j) : at(A8, j * ld + p);
          s += herm ? (t == CUBLAS_OP_N ? x * std::conj(y) : std::conj(x) * y) : x * y;
        }
        c[j * ld + i] = alpha * s + beta * c[j * ld + i];
        if (herm && i == j) c[j * ld + i] = c[j * ld + i].real();
      }
    return c;
  };
  const float fa = 2, fb = -1;
  reset();
  IS(cublasCherkEx(h, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &fa, dA8, CUDA_C_8I, ld, &fb, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * nn), want_rk(true, true, N, 2, -1)), "CherkEx: A A^H into the lower triangle");
  reset();
  IS(cublasCherk3mEx(h, CUBLAS_FILL_MODE_UPPER, T, nn, kk, &fa, dAf, CUDA_C_32F, ld, &fb, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * nn), want_rk(true, false, C, 2, -1)), "Cherk3mEx with OP_T: A^H A, upper");
  reset();
  IS(cublasCsyrkEx(h, CUBLAS_FILL_MODE_UPPER, C, nn, kk, &al, dA8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * nn), want_rk(false, false, T, cplx(2, -1), cplx(1, 3))), "CsyrkEx with OP_C: A^T A");
  reset();
  IS(cublasCsyrk3mEx_64(h, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &al, dAf, CUDA_C_32F, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  check(same(host(dC, ld * nn), want_rk(false, true, N, cplx(2, -1), cplx(1, 3))), "cublasCsyrk3mEx_64: A A^T");
  IS(cublasCherkEx_64(h, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &fa, dA8, CUDA_C_8I, ld, &fb, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasCherkEx(h, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &fa, dA8, CUDA_R_8I, ld, &fb, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasCsyrkEx(h, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &al, dAf, CUDA_C_32F, ld, &be, dC, CUDA_C_64F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasCherkEx(h, CUBLAS_FILL_MODE_FULL, N, nn, kk, &fa, dA8, CUDA_C_8I, ld, &fb, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasCherkEx(h, CUBLAS_FILL_MODE_LOWER, T, nn, 7, &fa, dA8, CUDA_C_8I, ld, &fb, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_INVALID_VALUE);   // op(A) is k x n: lda 6 is short of 7
  IS(cublasCsyrkEx(h, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &al, dA8, CUDA_C_8I, ld, &be, dC, CUDA_C_32F, 3),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasCsyrkEx(nullptr, CUBLAS_FILL_MODE_LOWER, N, nn, kk, &al, dA8, CUDA_R_8I, ld, &be, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_INITIALIZED);

  // SgemmEx: the real rows of GemmEx's table under single-precision compute.
  std::vector<int8_t> a8(ld * 6), b8(ld * 6);
  for (int i = 0; i < ld * 6; ++i) a8[i] = (int8_t)(i % 7 - 3), b8[i] = (int8_t)(i % 5 - 2);
  int8_t* da8 = dev(a8);
  int8_t* db8 = dev(b8);
  std::vector<float> c32(ld * n, 1);
  float* dc32 = dev(c32);
  const float two = 2, half_ = 0.5f;
  IS(cublasSgemmEx(h, T, N, m, n, k, &two, da8, CUDA_R_8I, ld, db8, CUDA_R_8I, ld, &half_, dc32, CUDA_R_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  {
    const auto got = host(dc32, ld * n);
    bool ok = true;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < ld; ++i) {
        float s = 0;
        for (int p = 0; p < k; ++p) s += (float)a8[i * ld + p] * b8[j * ld + p];
        ok = ok && got[j * ld + i] == (i < m ? 2 * s + 0.5f : 1.0f);
      }
    check(ok, "SgemmEx: int8 operands, a single-precision result");
  }
  std::vector<__half> ah(ld * 6), bh(ld * 6), ch(ld * n, __float2half(0));
  for (int i = 0; i < ld * 6; ++i) ah[i] = __float2half(a8[i]), bh[i] = __float2half(b8[i]);
  __half* dah = dev(ah);
  __half* dbh = dev(bh);
  __half* dch = dev(ch);
  IS(cublasSgemmEx(h, N, N, m, n, k, &two, dah, CUDA_R_16F, ld, dbh, CUDA_R_16F, ld, &half_, dch, CUDA_R_16F, ld),
     CUBLAS_STATUS_SUCCESS);
  {
    float s = 0;
    for (int p = 0; p < k; ++p) s += (float)a8[p * ld + 1] * b8[2 * ld + p];
    check(__half2float(host(dch, ld * n)[2 * ld + 1]) == 2 * s, "SgemmEx in half");
  }
  IS(cublasSgemmEx_64(h, N, N, m, n, k, &two, dah, CUDA_R_16F, ld, dbh, CUDA_R_16F, ld, &half_, dc32, CUDA_R_32F, ld),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasSgemmEx(h, N, N, m, n, k, &two, dc32, CUDA_R_32F, ld, dc32, CUDA_R_32F, ld, &half_, dch, CUDA_R_16F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSgemmEx(h, N, N, m, n, k, &two, dah, CUDA_R_16F, ld, dbh, CUDA_R_16BF, ld, &half_, dc32, CUDA_R_32F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSgemmEx(h, N, N, m, n, k, &two, dAf, CUDA_C_32F, ld, dBf, CUDA_C_32F, ld, &half_, dC, CUDA_C_32F, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSgemmEx(h, N, N, m, n, k, &two, da8, CUDA_R_8I, ld, db8, CUDA_R_8I, ld, &half_, dc32, CUDA_R_32I, ld),
     CUBLAS_STATUS_NOT_SUPPORTED);

  // GemmEx's table, a row at a time.
  auto ge = [&](cudaDataType at_, cudaDataType bt, cudaDataType ct, cublasComputeType_t comp, void* a, void* b,
                void* c) {
    const double one[2] = {1, 0}, zero[2] = {0, 0};
    return cublasGemmEx(h, N, N, 2, 2, 2, one, a, at_, 4, b, bt, 4, zero, c, ct, 4, comp, CUBLAS_GEMM_DEFAULT);
  };
  IS(ge(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, dah, dbh, dch), CUBLAS_STATUS_SUCCESS);
  IS(ge(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_16F, dah, dbh, dch), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ge(CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_16F, dah, dbh, dc32), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ge(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, dc32, dc32, dc32), CUBLAS_STATUS_SUCCESS);
  IS(ge(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_64F, dc32, dc32, dc32), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ge(CUDA_R_32F, CUDA_R_32F, CUDA_R_16F, CUBLAS_COMPUTE_32F, dc32, dc32, dch), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ge(CUDA_R_16F, CUDA_R_16BF, CUDA_R_32F, CUBLAS_COMPUTE_32F, dah, dbh, dc32), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ge(CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_32F_PEDANTIC, dAf, dBf, dC), CUBLAS_STATUS_SUCCESS);
  IS(ge(CUDA_C_32F, CUDA_C_32F, CUDA_C_32F, CUBLAS_COMPUTE_64F, dAf, dBf, dC), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ge(CUDA_R_8I, CUDA_R_8I, CUDA_R_32F, CUBLAS_COMPUTE_32F, da8, db8, dc32), CUBLAS_STATUS_SUCCESS);
  IS(ge(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, da8, db8, dc32), CUBLAS_STATUS_SUCCESS);
  IS(ge(CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32F, da8, db8, dc32), CUBLAS_STATUS_NOT_SUPPORTED);
  // The int8 path with an int32 result: lda and ldb multiples of 4, A and B
  // 4-byte aligned, B not transposed. And no GEMM takes a misaligned matrix.
  const int i1 = 1, i0 = 0;
  IS(cublasGemmEx(h, T, N, 4, 4, 4, &i1, da8, CUDA_R_8I, 4, db8, CUDA_R_8I, 4, &i0, dc32, CUDA_R_32I, 4,
                  CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasGemmEx(h, N, N, 4, 4, 4, &i1, da8, CUDA_R_8I, 5, db8, CUDA_R_8I, 4, &i0, dc32, CUDA_R_32I, 4,
                  CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasGemmEx(h, N, T, 4, 4, 4, &i1, da8, CUDA_R_8I, 4, db8, CUDA_R_8I, 4, &i0, dc32, CUDA_R_32I, 4,
                  CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasGemmEx(h, N, N, 4, 4, 4, &i1, da8 + 1, CUDA_R_8I, 4, db8, CUDA_R_8I, 4, &i0, dc32, CUDA_R_32I, 4,
                  CUBLAS_COMPUTE_32I, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSgemm(h, N, N, 2, 2, 2, &two, (float*)((char*)dc32 + 2), 4, dc32, 4, &half_, dc32, 4),
     CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSgemm(h, N, N, 4, 4, 4, &two, dc32, 3, dc32, 4, &half_, dc32, 4), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasGemmEx(nullptr, N, N, 2, 2, 2, &two, dc32, CUDA_R_64F, 4, dc32, CUDA_R_32F, 4, &half_, dc32, CUDA_R_64F, 4,
                  CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_NOT_INITIALIZED);

  // Double complex, beta 0: C is not read, so NaN in it stays out.
  std::vector<cuDoubleComplex> az = {{1, 1}, {2, 0}, {0, -1}, {3, 2}}, cz(4, make_cuDoubleComplex(NAN, NAN));
  cuDoubleComplex* daz = dev(az);
  cuDoubleComplex* dcz = dev(cz);
  const cuDoubleComplex z1 = make_cuDoubleComplex(1, 0), z0 = make_cuDoubleComplex(0, 0);
  IS(cublasGemmEx(h, C, N, 2, 2, 2, &z1, daz, CUDA_C_64F, 2, daz, CUDA_C_64F, 2, &z0, dcz, CUDA_C_64F, 2,
                  CUBLAS_COMPUTE_64F, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_SUCCESS);
  const auto gz = host(dcz, 4);
  check(gz[0].x == 6 && gz[0].y == 0 && gz[2].x == 5 && gz[2].y == 3 && gz[3].x == 14 && gz[3].y == 0,
        "GemmEx in double complex: A^H A, NaN in C ignored under beta 0");
  // The strided-batched form with complex elements.
  reset();
  IS(cublasGemmStridedBatchedEx(h, N, N, 2, 2, 2, &al, dAf, CUDA_C_32F, ld, 2, dBf, CUDA_C_32F, ld, 2, &be, dC,
                                CUDA_C_32F, ld, 2 * ld, 2, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT),
     CUBLAS_STATUS_SUCCESS);
  {
    const auto got = host(dC, ld * n);
    const cplx s = at(Af, 2) * at(Bf, 2 + 2 * ld) + at(Af, 2 + ld) * at(Bf, 3 + 2 * ld);
    const cplx w = cplx(2, -1) * s + cplx(1, 3) * at(C0, 4 * ld - ld * 2);
    check(got[2 * ld].x == (float)w.real() && got[2 * ld].y == (float)w.imag(), "GemmStridedBatchedEx in C_32F");
  }
  for (void* p : {(void*)dA8, (void*)dB8, (void*)dAf, (void*)dBf, (void*)dC, (void*)da8, (void*)db8, (void*)dc32,
                  (void*)dah, (void*)dbh, (void*)dch, (void*)daz, (void*)dcz})
    cudaFree(p);
}

// ---- the grouped batched GEMMs (cuBLAS 12.5 on) ----
#if CUBLAS_VERSION >= 120500
static void grouped(cublasHandle_t h) {
  // Group 0: two 2 x 2 products C = 2 A B + C, A 2 x 3; group 1: one 3 x 1
  // product C = A^T B - C, A 2 x 3. Integers, so the answers are exact.
  const int ld = 3;
  std::vector<float> M(9 * 6);
  for (int i = 0; i < 9 * 6; ++i) M[i] = (float)((i * 5) % 7 - 3);
  float* d = dev(M);
  // The three GEMMs' matrices, group 0's two and then group 1's one; C all ones.
  std::vector<float*> pa = {d, d + 9, d + 18}, pb = {d + 27, d + 36, d + 45};
  std::vector<float*> pc = {dev(std::vector<float>(9, 1)), dev(std::vector<float>(9, 1)), dev(std::vector<float>(9, 1))};
  float** dpa = dev(pa);
  float** dpb = dev(pb);
  float** dpc = dev(pc);
  const cublasOperation_t ta[2] = {CUBLAS_OP_N, CUBLAS_OP_T}, tb[2] = {CUBLAS_OP_N, CUBLAS_OP_N};
  const int m[2] = {2, 3}, n[2] = {2, 1}, k[2] = {3, 2}, lda[2] = {ld, ld}, ldb[2] = {ld, ld}, ldc[2] = {ld, ld},
            size[2] = {2, 1};
  const float al[2] = {2, 1}, be[2] = {1, -1};
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, dpa, lda, dpb, ldb, be, dpc, ldc, 2, size),
     CUBLAS_STATUS_SUCCESS);
  bool ok = true;
  for (int g = 0, i = 0; g < 2; ++g)
    for (int j = 0; j < size[g]; ++j, ++i) {
      const auto got = host(pc[i], 9);
      const float* A = &M[9 * i];
      const float* B = &M[27 + 9 * i];
      for (int c = 0; c < n[g]; ++c)
        for (int r = 0; r < m[g]; ++r) {
          float s = 0;
          for (int p = 0; p < k[g]; ++p) s += (ta[g] == CUBLAS_OP_N ? A[p * ld + r] : A[r * ld + p]) * B[c * ld + p];
          ok = ok && got[c * ld + r] == al[g] * s + be[g] * 1;
        }
    }
  check(ok, "sgemmGroupedBatched: two groups, each with its own operations, sizes, alpha and beta");
  const int64_t m64[2] = {2, 3}, n64[2] = {2, 1}, k64[2] = {3, 2}, ld64[2] = {ld, ld}, size64[2] = {2, 1};
  IS(cublasGemmGroupedBatchedEx_64(h, ta, tb, m64, n64, k64, al, (const void* const*)dpa, CUDA_R_32F, ld64,
                                   (const void* const*)dpb, CUDA_R_32F, ld64, be, (void* const*)dpc, CUDA_R_32F, ld64,
                                   2, size64, CUBLAS_COMPUTE_32F_FAST_TF32),
     CUBLAS_STATUS_SUCCESS);
  // The types it takes, and what it refuses.
  auto ex = [&](cudaDataType t, cublasComputeType_t c) {
    return cublasGemmGroupedBatchedEx(h, ta, tb, m, n, k, al, (const void* const*)dpa, t, lda,
                                      (const void* const*)dpb, t, ldb, be, (void* const*)dpc, t, ldc, 2, size, c);
  };
  IS(ex(CUDA_R_16F, CUBLAS_COMPUTE_32F), CUBLAS_STATUS_SUCCESS);
  IS(ex(CUDA_R_16F, CUBLAS_COMPUTE_16F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ex(CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_16F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ex(CUDA_C_32F, CUBLAS_COMPUTE_32F), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(ex(CUDA_R_8I, CUBLAS_COMPUTE_32I), CUBLAS_STATUS_NOT_SUPPORTED);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, dpa, lda, dpb, ldb, be, dpc, ldc, 0, size),
     CUBLAS_STATUS_SUCCESS);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, dpa, lda, dpb, ldb, be, dpc, ldc, -1, size),
     CUBLAS_STATUS_INVALID_VALUE);
  const int bad_size[2] = {2, -1}, bad_lda[2] = {ld, 1};
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, dpa, lda, dpb, ldb, be, dpc, ldc, 2, bad_size),
     CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, al, dpa, bad_lda, dpb, ldb, be, dpc, ldc, 2, size),
     CUBLAS_STATUS_INVALID_VALUE);   // group 1's A^T is 3 x 2: lda must reach k = 2
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE);
  IS(cublasSgemmGroupedBatched(h, ta, tb, m, n, k, d, dpa, lda, dpb, ldb, d, dpc, ldc, 2, size),
     CUBLAS_STATUS_NOT_SUPPORTED);
  cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST);
  // Double precision.
  std::vector<double> Md(M.begin(), M.end());
  double* dd = dev(Md);
  double* dcd = dev(std::vector<double>(9, 0));
  double** dpad = dev(std::vector<double*>{dd});
  double** dpbd = dev(std::vector<double*>{dd + 27});
  double** dpcd = dev(std::vector<double*>{dcd});
  const double ald[1] = {1}, bed[1] = {0};
  const int one[1] = {1};
  IS(cublasDgemmGroupedBatched(h, ta, tb, m, n, k, ald, dpad, lda, dpbd, ldb, bed, dpcd, ldc, 1, one),
     CUBLAS_STATUS_SUCCESS);
  check(host(dcd, 1)[0] == Md[0] * Md[27] + Md[3] * Md[28] + Md[6] * Md[29], "dgemmGroupedBatched");
  for (void* p : {(void*)d, (void*)pc[0], (void*)pc[1], (void*)pc[2], (void*)dpa, (void*)dpb, (void*)dpc, (void*)dd,
                  (void*)dcd, (void*)dpad, (void*)dpbd, (void*)dpcd})
    cudaFree(p);
}
#endif

int main() {
  cublasHandle_t h;
  if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
    std::printf("FAIL: cublasCreate\n");
    return 1;
  }
  level1(h);
  geam(h);
  gels(h);
  geqrf(h);
  gemm_ex(h);
#if CUBLAS_VERSION >= 120500
  grouped(h);
#endif
  cublasDestroy(h);
  std::printf(failures ? "FAIL: %d cuBLAS checks\n" : "PASS: every cuBLAS check\n", failures);
  return failures ? 1 : 0;
}
