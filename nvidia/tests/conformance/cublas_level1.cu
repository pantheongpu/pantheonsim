// Differential conformance for cuBLAS level 1 and the triangular band product:
// axpy, scal, dot, nrm2, asum and i?amax in single and double precision, positive and
// negative increments (BLAS walks a negative-increment vector from its far
// end), and tbmv over both triangles, both operations, unit and non-unit
// diagonals and several band widths; and dgmm (a matrix times a diagonal
// one) from both sides, over every kind of increment, padded and in place.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
  printf("%s -> %d\n", #x, (int)s_); return; } } while (0)

template <class T> static T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T));
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> down(const T* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}
template <class T> static void print(const char* tag, const std::vector<T>& v) {
  printf("%-28s", tag);
  for (T x : v) printf(" %.6f", (double)x);
  printf("\n");
}

// Deterministic, sign-mixed values with distinct magnitudes.
template <class T> static std::vector<T> values(size_t n, int seed) {
  std::vector<T> v(n);
  for (size_t i = 0; i < n; ++i)
    v[i] = (T)(((int)((i * 7 + seed * 13) % 17) - 8) * 0.375 + 0.0625 * seed);
  return v;
}

cublasStatus_t axpy(cublasHandle_t h, int n, const float* a, const float* x, int ix, float* y, int iy) { return cublasSaxpy(h, n, a, x, ix, y, iy); }
cublasStatus_t axpy(cublasHandle_t h, int n, const double* a, const double* x, int ix, double* y, int iy) { return cublasDaxpy(h, n, a, x, ix, y, iy); }
cublasStatus_t scal(cublasHandle_t h, int n, const float* a, float* x, int ix) { return cublasSscal(h, n, a, x, ix); }
cublasStatus_t scal(cublasHandle_t h, int n, const double* a, double* x, int ix) { return cublasDscal(h, n, a, x, ix); }
cublasStatus_t dot(cublasHandle_t h, int n, const float* x, int ix, const float* y, int iy, float* r) { return cublasSdot(h, n, x, ix, y, iy, r); }
cublasStatus_t dot(cublasHandle_t h, int n, const double* x, int ix, const double* y, int iy, double* r) { return cublasDdot(h, n, x, ix, y, iy, r); }
cublasStatus_t nrm2(cublasHandle_t h, int n, const float* x, int ix, float* r) { return cublasSnrm2(h, n, x, ix, r); }
cublasStatus_t nrm2(cublasHandle_t h, int n, const double* x, int ix, double* r) { return cublasDnrm2(h, n, x, ix, r); }
cublasStatus_t asum(cublasHandle_t h, int n, const float* x, int ix, float* r) { return cublasSasum(h, n, x, ix, r); }
cublasStatus_t asum(cublasHandle_t h, int n, const double* x, int ix, double* r) { return cublasDasum(h, n, x, ix, r); }
cublasStatus_t iamax(cublasHandle_t h, int n, const float* x, int ix, int* r) { return cublasIsamax(h, n, x, ix, r); }
cublasStatus_t iamax(cublasHandle_t h, int n, const double* x, int ix, int* r) { return cublasIdamax(h, n, x, ix, r); }
cublasStatus_t tbmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n, int k, const float* A, int lda, float* x, int ix) { return cublasStbmv(h, u, t, d, n, k, A, lda, x, ix); }
cublasStatus_t tbmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n, int k, const double* A, int lda, double* x, int ix) { return cublasDtbmv(h, u, t, d, n, k, A, lda, x, ix); }
cublasStatus_t dgmm(cublasHandle_t h, cublasSideMode_t s, int m, int n, const float* A, int lda, const float* x, int ix, float* C, int ldc) { return cublasSdgmm(h, s, m, n, A, lda, x, ix, C, ldc); }
cublasStatus_t dgmm(cublasHandle_t h, cublasSideMode_t s, int m, int n, const double* A, int lda, const double* x, int ix, double* C, int ldc) { return cublasDdgmm(h, s, m, n, A, lda, x, ix, C, ldc); }

template <class T> static void diag(cublasHandle_t h, const char* ty) {
  const int m = 5, n = 4, lda = 7;   // padded: rows 5 and 6 of each column are not A's
  for (int left = 0; left <= 1; ++left)
    for (int inc : {1, 2, -1, 0}) {
      const int len = left ? m : n;
      const size_t lx = inc == 0 ? 1 : (size_t)(inc < 0 ? -inc : inc) * (len - 1) + 1;
      auto hA = values<T>((size_t)lda * n, 6);
      auto hx = values<T>(lx, 7);
      T* dA = up(hA);
      T* dx = up(hx);
      std::vector<T> zeros((size_t)lda * n, (T)0);
      T* dC = up(zeros);
      CB(dgmm(h, left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, m, n, dA, lda, dx, inc, dC, lda));
      char tag[64];
      snprintf(tag, sizeof tag, "%s dgmm %s inc %d", ty, left ? "L" : "R", inc);
      print(tag, down(dC, (size_t)lda * n));
      // In place: C is A.
      CB(dgmm(h, left ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT, m, n, dA, lda, dx, inc, dA, lda));
      snprintf(tag, sizeof tag, "%s dgmm %s inc %d in place", ty, left ? "L" : "R", inc);
      print(tag, down(dA, (size_t)lda * n));
      cudaFree(dA); cudaFree(dx); cudaFree(dC);
    }
}

template <class T> static void level1(cublasHandle_t h, const char* ty) {
  const int n = 7;
  const int incs[][2] = {{1, 1}, {2, 3}, {-1, 1}, {2, -2}};
  for (auto& in : incs) {
    const int ix = in[0], iy = in[1];
    const size_t lx = (size_t)(ix < 0 ? -ix : ix) * (n - 1) + 1;
    const size_t ly = (size_t)(iy < 0 ? -iy : iy) * (n - 1) + 1;
    auto hx = values<T>(lx, 1), hy = values<T>(ly, 2);
    T* dx = up(hx);
    T* dy = up(hy);
    char tag[64];
    const T alpha = (T)-1.5;
    CB(axpy(h, n, &alpha, dx, ix, dy, iy));
    snprintf(tag, sizeof tag, "%s axpy inc %d,%d", ty, ix, iy);
    print(tag, down(dy, ly));
    T r = 0;
    CB(dot(h, n, dx, ix, dy, iy, &r));
    printf("%s dot inc %d,%d = %.6f\n", ty, ix, iy, (double)r);
    CB(nrm2(h, n, dx, ix, &r));
    printf("%s nrm2 inc %d = %.6f\n", ty, ix, (double)r);
    CB(asum(h, n, dx, ix, &r));
    printf("%s asum inc %d = %.6f\n", ty, ix, (double)r);
    int im = -1;
    CB(iamax(h, n, dx, ix, &im));
    printf("%s iamax inc %d = %d\n", ty, ix, im);
    const T s = (T)0.5;
    CB(scal(h, n, &s, dx, ix));
    snprintf(tag, sizeof tag, "%s scal inc %d", ty, ix);
    print(tag, down(dx, lx));
    cudaFree(dx); cudaFree(dy);
  }
  // A tie in magnitude: the first index wins.
  std::vector<T> tie = {(T)1, (T)-3, (T)2, (T)3, (T)-3};
  T* dt = up(tie);
  int im = -1;
  CB(iamax(h, 5, dt, 1, &im));
  printf("%s iamax tie = %d\n", ty, im);
  cudaFree(dt);
}

template <class T> static void band(cublasHandle_t h, const char* ty) {
  const int n = 6;
  for (int k : {0, 1, 2}) {
    const int lda = k + 2;  // one row of padding between band columns
    for (int upper = 0; upper <= 1; ++upper)
      for (int trans = 0; trans <= 1; ++trans)
        for (int unit = 0; unit <= 1; ++unit)
          for (int inc : {1, -2}) {
            auto hA = values<T>((size_t)lda * n, 3 + k);
            const size_t lx = (size_t)(inc < 0 ? -inc : inc) * (n - 1) + 1;
            auto hx = values<T>(lx, 4);
            T* dA = up(hA);
            T* dx = up(hx);
            CB(tbmv(h, upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER,
                    trans ? CUBLAS_OP_T : CUBLAS_OP_N, unit ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT,
                    n, k, dA, lda, dx, inc));
            char tag[64];
            snprintf(tag, sizeof tag, "%s tbmv k%d %s %s %s inc %d", ty, k, upper ? "U" : "L",
                     trans ? "T" : "N", unit ? "unit" : "diag", inc);
            print(tag, down(dx, lx));
            cudaFree(dA); cudaFree(dx);
          }
  }
}

static void run() {
  cublasHandle_t h;
  CB(cublasCreate(&h));
  level1<float>(h, "s");
  level1<double>(h, "d");
  band<float>(h, "s");
  band<double>(h, "d");
  diag<float>(h, "s");
  diag<double>(h, "d");
  // Results written through a device pointer.
  CB(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_DEVICE));
  auto hx = values<double>(9, 5);
  double* dx = up(hx);
  double* dr = nullptr;
  int* di = nullptr;
  cudaMalloc(&dr, sizeof(double));
  cudaMalloc(&di, sizeof(int));
  CB(cublasDnrm2(h, 9, dx, 1, dr));
  CB(cublasIdamax(h, 9, dx, 1, di));
  printf("device-pointer nrm2 = %.6f iamax = %d\n", down(dr, 1)[0], down(di, 1)[0]);
  // The 64-bit-index forms, whose i?amax writes an int64_t; the device-pointer
  // one catches a result written 4 bytes wide.
  int64_t* dl = nullptr;
  cudaMalloc(&dl, sizeof(int64_t));
  cudaMemset(dl, 0xff, sizeof(int64_t));
  CB(cublasIdamax_64(h, 9, dx, 1, dl));
  printf("device-pointer iamax_64 = %lld\n", (long long)down(dl, 1)[0]);
  CB(cublasSetPointerMode(h, CUBLAS_POINTER_MODE_HOST));
  {
    auto hy = values<double>(9, 3);
    double* dy = up(hy);
    const double a = -1.5, s = 0.5;
    double r = 0;
    int64_t im = -1;
    CB(cublasDaxpy_64(h, 9, &a, dx, 1, dy, 1));
    print("d axpy_64", down(dy, 9));
    CB(cublasDdot_64(h, 9, dx, 1, dy, 1, &r));
    printf("d dot_64 = %.6f\n", r);
    CB(cublasDnrm2_64(h, 5, dx, 2, &r));
    printf("d nrm2_64 inc 2 = %.6f\n", r);
    CB(cublasDasum_64(h, 5, dx, 2, &r));
    printf("d asum_64 inc 2 = %.6f\n", r);
    CB(cublasIdamax_64(h, 9, dy, 1, &im));
    printf("d iamax_64 = %lld\n", (long long)im);
    CB(cublasDscal_64(h, 9, &s, dy, 1));
    print("d scal_64", down(dy, 9));
    auto fx = values<float>(9, 5), fy = values<float>(9, 3);
    float *sx = up(fx), *sy = up(fy);
    const float fa = -1.5f, fs = 0.5f;
    float fr = 0;
    CB(cublasSaxpy_64(h, 9, &fa, sx, 1, sy, 1));
    print("s axpy_64", down(sy, 9));
    CB(cublasSdot_64(h, 9, sx, 1, sy, 1, &fr));
    printf("s dot_64 = %.6f\n", (double)fr);
    CB(cublasSnrm2_64(h, 9, sx, 1, &fr));
    printf("s nrm2_64 = %.6f\n", (double)fr);
    CB(cublasSasum_64(h, 9, sx, 1, &fr));
    printf("s asum_64 = %.6f\n", (double)fr);
    CB(cublasIsamax_64(h, 9, sy, 1, &im));
    printf("s iamax_64 = %lld\n", (long long)im);
    CB(cublasSscal_64(h, 9, &fs, sy, 1));
    print("s scal_64", down(sy, 9));
    cudaFree(dy); cudaFree(sx); cudaFree(sy);
  }
  cudaFree(dx); cudaFree(dr); cudaFree(di); cudaFree(dl);
  cublasDestroy(h);
}

// The host <-> device copy helpers, with strides on both sides, then back.
static void host_copies() {
  std::vector<float> hx(10);
  for (int i = 0; i < 10; ++i) hx[i] = (float)(i + 1);
  float* dv = up(std::vector<float>(15, 0.0f));
  CB(cublasSetVector(5, sizeof(float), hx.data(), 2, dv, 3));   // x[0], x[2], ... to dv[0], dv[3], ...
  print("SetVector inc 2,3", down(dv, 15));
  std::vector<float> back(5, -1.0f);
  CB(cublasGetVector(5, sizeof(float), dv, 3, back.data(), 1));
  print("GetVector inc 3,1", back);
  // A 3 x 2 column-major matrix with leading dimension 4 on the host and 5 on the device.
  std::vector<double> ha(8);
  for (int i = 0; i < 8; ++i) ha[i] = 10.0 + i;
  double* dm = up(std::vector<double>(10, 0.0));
  CB(cublasSetMatrix(3, 2, sizeof(double), ha.data(), 4, dm, 5));
  print("SetMatrix 3x2 ld 4,5", down(dm, 10));
  std::vector<double> hb(6, -1.0);
  CB(cublasGetMatrixAsync(3, 2, sizeof(double), dm, 5, hb.data(), 3, 0));
  cudaStreamSynchronize(0);
  print("GetMatrixAsync ld 5,3", hb);
  printf("SetVector inc 0 -> %d\n", (int)cublasSetVector(5, sizeof(float), hx.data(), 0, dv, 1));
  printf("SetMatrix ld < rows -> %d\n", (int)cublasSetMatrix(3, 2, sizeof(double), ha.data(), 2, dm, 5));
  cudaFree(dv); cudaFree(dm);
}

int main() { run(); host_copies(); return 0; }
