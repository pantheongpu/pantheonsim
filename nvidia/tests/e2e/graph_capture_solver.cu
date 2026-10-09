// cuSOLVER's dense factorizations and solvers inside a captured CUDA graph (see
// graph_capture_common.h): buffer sizes are asked for beforehand, the matrices are written from a
// counter by kernels in the capture, and the graph is launched with new inputs.
// run_graph_capture.sh solver cusolver --card runs the same program on NVIDIA's cuSOLVER.
#include <cusolverDn.h>

#include <algorithm>
#include <functional>

#include "graph_capture_common.h"

using namespace gc;

#define OK(x) do { cusolverStatus_t s_ = (x); if (s_ != CUSOLVER_STATUS_SUCCESS) { \
  std::printf("     %s -> %d\n", #x, (int)s_); ok = false; } } while (0)

// A symmetric, diagonally dominant matrix (so positive definite): m(i, j) = m(j, i) from the counter.
template <class T>
__global__ void make_spd(T* a, int n, const int* counter) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= n || j >= n) return;
  const int lo = min(i, j), hi = max(i, j);
  T v = static_cast<T>(pattern(static_cast<size_t>(lo) * 31 + hi, *counter, 1, 0)) * static_cast<T>(0.1);
  if (i == j) v = static_cast<T>(n) + static_cast<T>(pattern(i, *counter, 2, 1)) * static_cast<T>(0.1);
  a[static_cast<size_t>(j) * n + i] = v;
}
// A general, diagonally dominant matrix.
template <class T>
__global__ void make_general(T* a, int m, int n, const int* counter) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x, j = blockIdx.y * blockDim.y + threadIdx.y;
  if (i >= m || j >= n) return;
  T v = static_cast<T>(pattern(static_cast<size_t>(i) * 31 + j * 7, *counter, 3, 0)) * static_cast<T>(0.2);
  if (i == j) v += static_cast<T>(m + 2);
  a[static_cast<size_t>(j) * m + i] = v;
}
template <class T>
static void spd(cudaStream_t st, const int* counter, T* a, int n) {
  make_spd<T><<<dim3((n + 7) / 8, (n + 7) / 8), dim3(8, 8), 0, st>>>(a, n, counter);
}
template <class T>
static void general(cudaStream_t st, const int* counter, T* a, int m, int n) {
  make_general<T><<<dim3((m + 7) / 8, (n + 7) / 8), dim3(8, 8), 0, st>>>(a, m, n, counter);
}

int main() {
  Runner r;
  cusolverDnHandle_t h;
  cusolverDnCreate(&h);
  cusolverDnSetStream(h, r.st);
  const int n = 12;
  const size_t nn = (size_t)n * n;
  {
    float *a = r.alloc<float>(nn), *b = r.alloc<float>(n * 2);
    int *info = r.alloc<int>(1);
    int lwork = 0;
    cusolverDnSpotrf_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, a, n, &lwork);
    float* work = r.alloc<float>(lwork + 1);
    r.run("cusolverDnSpotrf + Spotrs", [&] {
      bool ok = true;
      spd(r.st, r.counter, a, n);
      fill(r.st, r.counter, b, n * 2, 1.0f, 0, 5);
      OK(cusolverDnSpotrf(h, CUBLAS_FILL_MODE_LOWER, n, a, n, work, lwork, info));
      OK(cusolverDnSpotrs(h, CUBLAS_FILL_MODE_LOWER, n, 2, a, n, b, n, info));
      return ok;
    }, {{a, nn, Dt::F32}, {b, (size_t)n * 2, Dt::F32}, {info, 1, Dt::I32}}, 1e-4);
  }
  {
    double *a = r.alloc<double>(nn), *b = r.alloc<double>(n * 2);
    int *info = r.alloc<int>(1), *ipiv = r.alloc<int>(n);
    int lwork = 0;
    cusolverDnDgetrf_bufferSize(h, n, n, a, n, &lwork);
    double* work = r.alloc<double>(lwork + 1);
    r.run("cusolverDnDgetrf + Dgetrs", [&] {
      bool ok = true;
      general(r.st, r.counter, a, n, n);
      fill(r.st, r.counter, b, n * 2, 1.0f, 0, 5);
      OK(cusolverDnDgetrf(h, n, n, a, n, work, ipiv, info));
      OK(cusolverDnDgetrs(h, CUBLAS_OP_N, n, 2, a, n, ipiv, b, n, info));
      return ok;
    }, {{a, nn, Dt::F64}, {b, (size_t)n * 2, Dt::F64}, {ipiv, (size_t)n, Dt::I32}, {info, 1, Dt::I32}}, 1e-9);
  }
  {
    float *a = r.alloc<float>(nn), *tau = r.alloc<float>(n), *c = r.alloc<float>(nn);
    int *info = r.alloc<int>(1);
    int l1 = 0, l2 = 0, l3 = 0;
    cusolverDnSgeqrf_bufferSize(h, n, n, a, n, &l1);
    cusolverDnSorgqr_bufferSize(h, n, n, n, a, n, tau, &l2);
    cusolverDnSormqr_bufferSize(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, n, n, a, n, tau, c, n, &l3);
    float* work = r.alloc<float>(std::max(l1, std::max(l2, l3)) + 1);
    const int lw = std::max(l1, std::max(l2, l3));
    r.run("cusolverDnSgeqrf + Sormqr + Sorgqr", [&] {
      bool ok = true;
      general(r.st, r.counter, a, n, n);
      fill(r.st, r.counter, c, nn, 0.5f, 0, 7);
      OK(cusolverDnSgeqrf(h, n, n, a, n, tau, work, lw, info));
      OK(cusolverDnSormqr(h, CUBLAS_SIDE_LEFT, CUBLAS_OP_T, n, n, n, a, n, tau, c, n, work, lw, info));
      OK(cusolverDnSorgqr(h, n, n, n, a, n, tau, work, lw, info));
      return ok;
    }, {{a, nn, Dt::F32}, {tau, (size_t)n, Dt::F32}, {c, nn, Dt::F32}, {info, 1, Dt::I32}}, 1e-3);
  }
  {
    // Batched: pointers to the matrices in a device array.
    float *a0 = r.alloc<float>(nn), *a1 = r.alloc<float>(nn);
    float* host_array[2] = {a0, a1};
    float** arrays = r.alloc<float*>(2);
    cudaMemcpy(arrays, host_array, sizeof host_array, cudaMemcpyHostToDevice);
    int* info = r.alloc<int>(2);
    r.run("cusolverDnSpotrfBatched", [&] {
      bool ok = true;
      spd(r.st, r.counter, a0, n);
      spd(r.st, r.counter, a1, n);
      OK(cusolverDnSpotrfBatched(h, CUBLAS_FILL_MODE_LOWER, n, arrays, n, info, 2));
      return ok;
    }, {{a0, nn, Dt::F32}, {a1, nn, Dt::F32}, {info, 2, Dt::I32}}, 1e-4);
  }
  {
    // The Jacobi routines: batched forms are captured, the single-matrix forms are not (below).
    float *a = r.alloc<float>(nn), *w = r.alloc<float>(n);
    int *info = r.alloc<int>(1);
    syevjInfo_t params;
    cusolverDnCreateSyevjInfo(&params);
    int lwork = 0;
    cusolverDnSsyevjBatched_bufferSize(h, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, n, a, n, w, &lwork, params, 1);
    float* work = r.alloc<float>(lwork + 1);
    r.run("cusolverDnSsyevjBatched (the parameters object destroyed after the capture)", [&] {
      bool ok = true;
      spd(r.st, r.counter, a, n);
      syevjInfo_t p;
      cusolverDnCreateSyevjInfo(&p);
      cusolverDnXsyevjSetTolerance(p, 1e-7);
      cusolverDnXsyevjSetMaxSweeps(p, 50);
      OK(cusolverDnSsyevjBatched(h, CUSOLVER_EIG_MODE_NOVECTOR, CUBLAS_FILL_MODE_LOWER, n, a, n, w, work, lwork, info, p, 1));
      cusolverDnDestroySyevjInfo(p);
      return ok;
    }, {{w, (size_t)n, Dt::F32}, {info, 1, Dt::I32}}, 1e-3);
    cusolverDnDestroySyevjInfo(params);
  }
  {
    double *a = r.alloc<double>(nn), *tau = r.alloc<double>(n), *d = r.alloc<double>(n), *e = r.alloc<double>(n);
    int *info = r.alloc<int>(1);
    int lwork = 0;
    cusolverDnDsytrd_bufferSize(h, CUBLAS_FILL_MODE_LOWER, n, a, n, d, e, tau, &lwork);
    double* work = r.alloc<double>(lwork + 1);
    r.run("cusolverDnDsytrd", [&] {
      bool ok = true;
      spd(r.st, r.counter, a, n);
      OK(cusolverDnDsytrd(h, CUBLAS_FILL_MODE_LOWER, n, a, n, d, e, tau, work, lwork, info));
      return ok;
    }, {{a, nn, Dt::F64}, {d, (size_t)n, Dt::F64}, {e, (size_t)n, Dt::F64}, {tau, (size_t)n, Dt::F64}, {info, 1, Dt::I32}}, 1e-9);
  }
  {
    // The 64-bit LU with 64-bit pivots, and its host workspace.
    double *a = r.alloc<double>(nn), *b = r.alloc<double>(n);
    int64_t* ipiv = r.alloc<int64_t>(n);
    int* info = r.alloc<int>(1);
    size_t dev_bytes = 0, host_bytes = 0;
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    cusolverDnXgetrf_bufferSize(h, params, n, n, CUDA_R_64F, a, n, CUDA_R_64F, &dev_bytes, &host_bytes);
    void* dwork = r.alloc<char>(dev_bytes + 1);
    std::vector<char> hwork(host_bytes + 1);
    r.run("cusolverDnXgetrf + Xgetrs", [&] {
      bool ok = true;
      general(r.st, r.counter, a, n, n);
      fill(r.st, r.counter, b, n, 1.0f, 0, 5);
      cusolverDnParams_t p;
      cusolverDnCreateParams(&p);
      OK(cusolverDnXgetrf(h, p, n, n, CUDA_R_64F, a, n, ipiv, CUDA_R_64F, dwork, dev_bytes, hwork.data(), host_bytes, info));
      OK(cusolverDnXgetrs(h, p, CUBLAS_OP_N, n, 1, CUDA_R_64F, a, n, ipiv, CUDA_R_64F, b, n, info));
      cusolverDnDestroyParams(p);
      return ok;
    }, {{a, nn, Dt::F64}, {b, (size_t)n, Dt::F64}, {info, 1, Dt::I32}}, 1e-9);
    cusolverDnDestroyParams(params);
  }
  {
    // The 64-bit API, with a parameters object.
    double* a = r.alloc<double>(nn);
    int* info = r.alloc<int>(1);
    size_t dev_bytes = 0, host_bytes = 0;
    cusolverDnParams_t params;
    cusolverDnCreateParams(&params);
    cusolverDnXpotrf_bufferSize(h, params, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, a, n, CUDA_R_64F, &dev_bytes, &host_bytes);
    void *dwork = r.alloc<char>(dev_bytes + 1);
    std::vector<char> hwork(host_bytes + 1);
    r.run("cusolverDnXpotrf (a host workspace, parameters destroyed after)", [&] {
      bool ok = true;
      spd(r.st, r.counter, a, n);
      cusolverDnParams_t p;
      cusolverDnCreateParams(&p);
      OK(cusolverDnXpotrf(h, p, CUBLAS_FILL_MODE_LOWER, n, CUDA_R_64F, a, n, CUDA_R_64F, dwork, dev_bytes, hwork.data(), host_bytes, info));
      cusolverDnDestroyParams(p);
      return ok;
    }, {{a, nn, Dt::F64}, {info, 1, Dt::I32}}, 1e-9);
    cusolverDnDestroyParams(params);
  }
  // The routines NVIDIA's library cannot capture wait for the stream (the eigenvalue and singular value
  // solvers that iterate to convergence): the call answers an error and the capture is invalidated.
  // Measured on the RTX 3060 (each as the first call after a fresh start: once a capture has been
  // invalidated this way, the library's later calls in the process answer EXECUTION_FAILED, so this
  // program checks one -- last, in its own block): gesvd, syevd, syevdx, sygvd, sygvdx, sygvj, sytrf,
  // orgtr, gesvda and the 64-bit Xsyevd, Xsyevdx, Xgesvd, Xgesvdp, Xgeev answer INTERNAL_ERROR (7) and
  // invalidate the capture; syevj answers success and invalidates it. Captured: potrf, potrs, potri and
  // their batched forms, getrf, getrs, geqrf, orgqr, ormqr, sytrd, ormtr, orgbr, gebrd, sytri, laswp,
  // lauum, gesvdj, syevjBatched, gesvdjBatched, and the 64-bit Xpotrf, Xpotrs, Xgetrf, Xgetrs, Xgeqrf,
  // Xtrtri, Xsytrs, XsyevBatched.
  {
    struct Refusal {
      const char* name;
      std::function<int()> call;
      int status;
    };
    double *a, *b, *w, *u, *vt, *s;
    for (double** p : {&a, &b, &w, &u, &vt, &s}) cudaMalloc(p, 1 << 20);
    int* info = r.alloc<int>(4);
    int* ipiv = r.alloc<int>(n);
    double* work = r.alloc<double>(1 << 17);
    const int lw = 1 << 17;
    void* xwork = r.alloc<char>(1 << 20);
    std::vector<char> hwork(1 << 20);
    syevjInfo_t sj;
    cusolverDnCreateSyevjInfo(&sj);
    cusolverDnParams_t pr;
    cusolverDnCreateParams(&pr);
    auto matrices = [&] {   // symmetric, diagonally dominant
      std::vector<double> m(nn);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < n; ++i)
          m[j * n + i] = i == j ? n + 1.0 : 0.1 * ((std::min(i, j) * 31 + std::max(i, j)) % 5 - 2);
      cudaMemcpy(a, m.data(), nn * 8, cudaMemcpyHostToDevice);
      cudaMemcpy(b, m.data(), nn * 8, cudaMemcpyHostToDevice);
    };
    const std::vector<Refusal> refusals = {
      {"cusolverDnDgesvd", [&] { return (int)cusolverDnDgesvd(h, 'A', 'A', n, n, a, n, s, u, n, vt, n, work, lw, nullptr, info); }, 7},
    };
    for (const Refusal& f : refusals) {
      // A fresh handle for each: the library's own state does not carry a refused capture over.
      cusolverDnDestroy(h);
      cusolverDnCreate(&h);
      cusolverDnSetStream(h, r.st);
      matrices();
      const int eager = f.call();   // the same call, made eagerly, answers success
      cudaStreamSynchronize(r.st);
      expect(std::string(f.name) + " eagerly", eager == 0, eager);
      matrices();
      cudaGraph_t g = nullptr;
      const cudaError_t bc = cudaStreamBeginCapture(r.st, cudaStreamCaptureModeGlobal);
      const int rc = f.call();
      cudaStreamCaptureStatus status = cudaStreamCaptureStatusNone;
      cudaStreamIsCapturing(r.st, &status);
      const cudaError_t e = cudaStreamEndCapture(r.st, &g);
      if (g) cudaGraphDestroy(g);
      cudaGetLastError();
      cudaStreamSynchronize(r.st);
      expect(std::string(f.name) + " in a capture: refused, and the capture is invalidated",
             bc == cudaSuccess && rc == f.status && status == cudaStreamCaptureStatusInvalidated && e == cudaErrorStreamCaptureInvalidated,
             rc);
    }
    cusolverDnDestroyParams(pr);
    cusolverDnDestroySyevjInfo(sj);
    for (double* p : {a, b, w, u, vt, s}) cudaFree(p);
  }
  cusolverDnDestroy(h);
  return finish();
}
