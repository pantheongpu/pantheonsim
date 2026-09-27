// Differential conformance for cuBLAS level 1 and the triangular band product:
// axpy, scal, dot, nrm2 and i?amax in single and double precision, positive and
// negative increments (BLAS walks a negative-increment vector from its far
// end), and tbmv over both triangles, both operations, unit and non-unit
// diagonals and several band widths.
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
cublasStatus_t iamax(cublasHandle_t h, int n, const float* x, int ix, int* r) { return cublasIsamax(h, n, x, ix, r); }
cublasStatus_t iamax(cublasHandle_t h, int n, const double* x, int ix, int* r) { return cublasIdamax(h, n, x, ix, r); }
cublasStatus_t tbmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n, int k, const float* A, int lda, float* x, int ix) { return cublasStbmv(h, u, t, d, n, k, A, lda, x, ix); }
cublasStatus_t tbmv(cublasHandle_t h, cublasFillMode_t u, cublasOperation_t t, cublasDiagType_t d, int n, int k, const double* A, int lda, double* x, int ix) { return cublasDtbmv(h, u, t, d, n, k, A, lda, x, ix); }

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
  cudaFree(dx); cudaFree(dr); cudaFree(di);
  cublasDestroy(h);
}

int main() { run(); return 0; }
