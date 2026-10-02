// The legacy cuBLAS API (cublas.h), which NVIDIA's libcublas.so.13 still
// exports: cublasInit/Shutdown, cublasGetError, cublasAlloc/Free,
// cublasGetVersion, cublasSetKernelStream and the handle-less BLAS routines
// with their char options and scalars by value. What an RTX 3060's cuBLAS
// 13.0 does is what is checked -- the routines run without cublasInit,
// cublasGetError reports the latest call (a success clears an earlier
// failure), gemv reads any trans but N as T -- and every check passes on the
// card too. Values are small integers, so every result is exact.
#include <cublas.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want)                                                                     \
  do {                                                                                     \
    const int got_ = (int)(call);                                                          \
    char what_[512];                                                                       \
    std::snprintf(what_, sizeof what_, "%s -> %s (got %d)", #call, #want, got_);           \
    check(got_ == (int)(want), what_);                                                     \
  } while (0)

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

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  const std::vector<float> x = {1, 2, 3, 4};
  float* dx = dev(x);
  // Before cublasInit: the routines run anyway.
  check(cublasSdot(4, dx, 1, dx, 1) == 30.0f, "sdot before cublasInit: 30");
  IS(cublasGetError(), CUBLAS_STATUS_SUCCESS);
  int v = -1;
  IS(cublasGetVersion(&v), CUBLAS_STATUS_SUCCESS);
  check(v == CUBLAS_VERSION, "cublasGetVersion before cublasInit");
  IS(cublasShutdown(), CUBLAS_STATUS_SUCCESS);
  IS(cublasInit(), CUBLAS_STATUS_SUCCESS);
  IS(cublasInit(), CUBLAS_STATUS_SUCCESS);
  v = -1;
  IS(cublasGetVersion(&v), CUBLAS_STATUS_SUCCESS);
  check(v == CUBLAS_VERSION, "cublasGetVersion after cublasInit");
  IS(cublasGetVersion(nullptr), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasSetKernelStream(0), CUBLAS_STATUS_SUCCESS);

  // Level 1: results returned by value.
  float* d34 = dev(std::vector<float>{3, 4});
  check(cublasSnrm2(2, d34, 1) == 5.0f, "snrm2");
  check(cublasSnrm2(4, dx, 0) == 0.0f && cublasGetError() == CUBLAS_STATUS_SUCCESS, "snrm2 with incx 0 is 0, no error");
  check(cublasSasum(4, dx, 1) == 10.0f, "sasum");
  check(cublasIsamax(4, dx, 1) == 4, "isamax");
  check(cublasIsamin(4, dx, 1) == 1, "isamin");
  float* dy = dev(std::vector<float>(4, 1));
  cublasSaxpy(4, 2.0f, dx, 1, dy, 1);
  check(host(dy, 4) == std::vector<float>({3, 5, 7, 9}), "saxpy with alpha by value");
  cublasSscal(4, -1.0f, dy, 1);
  check(host(dy, 4) == std::vector<float>({-3, -5, -7, -9}), "sscal");
  cublasSswap(4, dx, 1, dy, 1);
  cublasScopy(4, dy, 1, dx, 1);
  check(host(dx, 4) == x && host(dy, 4) == x, "sswap then scopy");
  const std::vector<double> xd = {1, -2, 3};
  double* dxd = dev(xd);
  check(cublasDdot(3, dxd, 1, dxd, 1) == 14.0 && cublasIdamax(3, dxd, 1) == 3, "ddot, idamax");
  const std::vector<cuComplex> xc = {{1, 1}, {2, -1}};
  cuComplex* dxc = dev(xc);
  const cuComplex dc = cublasCdotc(2, dxc, 1, dxc, 1), du = cublasCdotu(2, dxc, 1, dxc, 1);
  check(dc.x == 7 && dc.y == 0 && du.x == 3 && du.y == -2, "cdotc, cdotu returned by value");
  const cuDoubleComplex zd = cublasZdotu(1, (const cuDoubleComplex*)dev(std::vector<cuDoubleComplex>{{0, 2}}), 1,
                                         (const cuDoubleComplex*)dev(std::vector<cuDoubleComplex>{{0, 3}}), 1);
  check(zd.x == -6 && zd.y == 0, "zdotu");

  // Rotations: the scalars and param are host values.
  float a = 3, b = 4, c = 0, s = 0;
  cublasSrotg(&a, &b, &c, &s);
  check(a == 5 && c == 0.6f && s == 0.8f && cublasGetError() == CUBLAS_STATUS_SUCCESS, "srotg on host scalars");
  const float param[5] = {-1, 2, 0, 0, 2};   // H = 2 I
  float *rx = dev(x), *ry = dev(x);
  cublasSrotm(4, rx, 1, ry, 1, param);
  check(host(rx, 4) == std::vector<float>({2, 4, 6, 8}), "srotm with a host param array");
  cublasSrot(4, rx, 1, ry, 1, 0.0f, 1.0f);
  check(host(rx, 4) == std::vector<float>({2, 4, 6, 8}) && host(ry, 4) == std::vector<float>({-2, -4, -6, -8}),
        "srot with c and s by value");

  // Level 2: gemv reads any trans but N as T.
  const std::vector<float> A = {1, 2, 3, 4};   // [1 3; 2 4]
  float* dA = dev(A);
  float* dv = dev(std::vector<float>{1, 10});
  float* dout = dev(std::vector<float>(2, 0));
  const char modes[] = {'N', 'n', 'T', 't', 'X', 'q'};
  for (char t : modes) {
    cublasSgemv(t, 2, 2, 1.0f, dA, 2, dv, 1, 0.0f, dout, 1);
    const bool n = t == 'N' || t == 'n';
    const std::vector<float> want = n ? std::vector<float>{31, 42} : std::vector<float>{21, 43};
    char what[64];
    std::snprintf(what, sizeof what, "sgemv trans '%c' is %s", t, n ? "N" : "T");
    check(host(dout, 2) == want && cublasGetError() == CUBLAS_STATUS_SUCCESS, what);
  }
  cublasSgemv('N', 2, 2, 1.0f, dA, 1, dv, 1, 0.0f, dout, 1);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasGetError(), CUBLAS_STATUS_SUCCESS);   // read once, then cleared
  cublasSgemv('N', 2, 2, 1.0f, dA, 1, dv, 1, 0.0f, dout, 1);
  cublasSscal(2, 1.0f, dout, 1);
  IS(cublasGetError(), CUBLAS_STATUS_SUCCESS);   // the later success is what is kept
  cublasStrmv('u', 'n', 'n', 2, dA, 2, dv, 1);   // [1 3; 0 4] (1, 10) = (31, 40)
  check(host(dv, 2) == std::vector<float>({31, 40}) && cublasGetError() == CUBLAS_STATUS_SUCCESS,
        "strmv with lower-case options");
  cublasStrmv('Q', 'N', 'N', 2, dA, 2, dv, 1);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);
  cublasStrmv('L', 'N', 'Q', 2, dA, 2, dv, 1);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);
  cublasStrsv('L', 'X', 'N', 2, dA, 2, dv, 1);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);
  cublasSsyr('R', 2, 1.0f, dv, 1, dA, 2);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);

  // Level 3: gemm, and trmm in place on B.
  float* dB = dev(std::vector<float>{1, 0, 0, 1});
  float* dC = dev(std::vector<float>(4, 5));
  cublasSgemm('N', 'T', 2, 2, 2, 2.0f, dA, 2, dB, 2, 1.0f, dC, 2);
  check(host(dC, 4) == std::vector<float>({7, 9, 11, 13}), "sgemm, C = 2 A I^T + C");
  cublasSgemm('X', 'Y', 2, 2, 2, 1.0f, dA, 2, dB, 2, 0.0f, dC, 2);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);
  cublasStrmm('L', 'U', 'N', 'N', 2, 2, 1.0f, dA, 2, dB, 2);   // B = triu(A) I
  check(host(dB, 4) == std::vector<float>({1, 0, 3, 4}), "strmm writes B in place");
  cublasStrsm('l', 'L', 'N', 'N', 2, 2, 1.0f, dA, 2, dB, 2);
  IS(cublasGetError(), CUBLAS_STATUS_SUCCESS);
  cublasStrsm('Q', 'L', 'N', 'N', 2, 2, 1.0f, dA, 2, dB, 2);
  IS(cublasGetError(), CUBLAS_STATUS_INVALID_VALUE);
  const std::vector<cuDoubleComplex> Az = {{0, 1}, {1, 0}, {2, 0}, {0, -1}};
  cuDoubleComplex* dAz = dev(Az);
  cuDoubleComplex* dCz = dev(std::vector<cuDoubleComplex>(4, {0, 0}));
  cublasZgemm('C', 'N', 2, 2, 2, {1, 0}, dAz, 2, dAz, 2, {0, 0}, dCz, 2);
  const auto cz = host(dCz, 4);   // A^H A = [2, 2+i... ]: diag 2 and 5
  check(cz[0].x == 2 && cz[0].y == 0 && cz[3].x == 5 && cz[3].y == 0, "zgemm with 'C'");

  // cublasAlloc and cublasFree.
  void* p = nullptr;
  IS(cublasAlloc(0, 4, &p), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasAlloc(-1, 4, &p), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasAlloc(4, 0, &p), CUBLAS_STATUS_INVALID_VALUE);
  IS(cublasAlloc(4, 4, &p), CUBLAS_STATUS_SUCCESS);
  IS(cublasSetVector(4, 4, x.data(), 1, p, 1), CUBLAS_STATUS_SUCCESS);
  check(cublasSasum(4, (const float*)p, 1) == 10.0f, "a cublasAlloc buffer is device memory");
  IS(cublasFree(p), CUBLAS_STATUS_SUCCESS);
  IS(cublasFree(nullptr), CUBLAS_STATUS_SUCCESS);
  IS(cublasFree((void*)0x1234), CUBLAS_STATUS_INTERNAL_ERROR);
  cudaGetLastError();   // the refused free leaves an error behind on the card

  IS(cublasShutdown(), CUBLAS_STATUS_SUCCESS);
  IS(cublasShutdown(), CUBLAS_STATUS_SUCCESS);
  check(cublasSasum(4, dx, 1) == 10.0f, "a routine after cublasShutdown still runs");
  std::printf(failures ? "FAIL: %d legacy cuBLAS checks\n" : "PASS: every legacy cuBLAS check\n", failures);
  return failures ? 1 : 0;
}
