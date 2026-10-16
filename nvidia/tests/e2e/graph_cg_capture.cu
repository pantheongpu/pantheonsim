// CUDA Samples' conjugateGradientCudaGraphs, small: the conjugate-gradient loop of cuBLAS and cuSPARSE calls run
// once call by call and once captured into a CUDA graph and launched for each iteration, which must give the
// same residuals and the same scalars (alpha, -alpha, beta, the dot products) after every iteration.
//
// In the sample the captured loop's residual grows (44, 1027, 2052, ...) where the eager one falls, and the
// program runs its 10000 iterations. This program shows after which iteration the two differ and in which
// scalar, so it prints what each held.
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusparse.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                  \
  do {                                                                         \
    const int r_ = (int)(x);                                                   \
    if (r_ != 0) {                                                             \
      std::printf("FAIL %s returned %d (line %d)\n", #x, r_, __LINE__);        \
      std::exit(1);                                                            \
    }                                                                          \
  } while (0)

static void tridiag(std::vector<int>& I, std::vector<int>& J, std::vector<float>& val, int N, int nz) {
  I.assign(N + 1, 0);
  J.assign(nz, 0);
  val.assign(nz, 0.f);
  I[0] = 0, J[0] = 0, J[1] = 1;
  val[0] = (float)rand() / RAND_MAX + 10.0f;
  val[1] = (float)rand() / RAND_MAX;
  for (int i = 1; i < N; i++) {
    if (i > 1) I[i] = I[i - 1] + 3;
    else I[1] = 2;
    const int start = (i - 1) * 3 + 2;
    J[start] = i - 1;
    J[start + 1] = i;
    if (i < N - 1) J[start + 2] = i + 1;
    val[start] = val[start - 1];
    val[start + 1] = (float)rand() / RAND_MAX + 10.0f;
    if (i < N - 1) val[start + 2] = (float)rand() / RAND_MAX;
  }
  I[N] = nz;
}

__global__ void initVectors(float* rhs, float* x, int N) {
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  for (size_t i = gid; i < (size_t)N; i += gridDim.x * blockDim.x) {
    rhs[i] = 1.0;
    x[i] = 0.0;
  }
}
__global__ void r1_div_x(float* r1, float* r0, float* b) {
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) b[0] = r1[0] / r0[0];
}
__global__ void a_minus(float* a, float* na) {
  if (blockIdx.x * blockDim.x + threadIdx.x == 0) na[0] = -(a[0]);
}

struct Scalars {
  float r1, a, na, b, dot, r0;
};
struct History {
  std::vector<Scalars> it;
};

static void run(bool graph, int iters, History& h) {
  srand(1);
  const int N = 4096, nz = (N - 2) * 3 + 4;
  std::vector<int> I, J;
  std::vector<float> val;
  tridiag(I, J, val, N, nz);
  float* d_x; float* d_r; float* d_p; float* d_Ax; float* d_val; int* d_col; int* d_row;
  CK(cudaMalloc(&d_col, nz * sizeof(int)));
  CK(cudaMalloc(&d_row, (N + 1) * sizeof(int)));
  CK(cudaMalloc(&d_val, nz * sizeof(float)));
  CK(cudaMalloc(&d_x, N * sizeof(float)));
  CK(cudaMalloc(&d_r, N * sizeof(float)));
  CK(cudaMalloc(&d_p, N * sizeof(float)));
  CK(cudaMalloc(&d_Ax, N * sizeof(float)));
  float *d_r1, *d_r0, *d_dot, *d_a, *d_na, *d_b;
  CK(cudaMalloc(&d_r1, 4)); CK(cudaMalloc(&d_r0, 4)); CK(cudaMalloc(&d_dot, 4));
  CK(cudaMalloc(&d_a, 4)); CK(cudaMalloc(&d_na, 4)); CK(cudaMalloc(&d_b, 4));
  for (float* p : {d_r1, d_r0, d_dot, d_a, d_na, d_b}) CK(cudaMemset(p, 0, 4));

  cublasHandle_t blas;
  cusparseHandle_t sp;
  CK(cublasCreate(&blas));
  CK(cusparseCreate(&sp));
  cudaStream_t stream1, streamForGraph;
  CK(cudaStreamCreate(&stream1));
  CK(cudaStreamCreate(&streamForGraph));

  cusparseSpMatDescr_t matA;
  CK(cusparseCreateCsr(&matA, N, N, nz, d_row, d_col, d_val, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));
  cusparseDnVecDescr_t vecx, vecp, vecAx;
  CK(cusparseCreateDnVec(&vecx, N, d_x, CUDA_R_32F));
  CK(cusparseCreateDnVec(&vecp, N, d_p, CUDA_R_32F));
  CK(cusparseCreateDnVec(&vecAx, N, d_Ax, CUDA_R_32F));
  float alpha = 1.0f, alpham1 = -1.0f, beta = 0.0f;
  size_t bufferSize = 0;
  CK(cusparseSpMV_bufferSize(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                             CUSPARSE_SPMV_ALG_DEFAULT, &bufferSize));
  void* buffer = nullptr;
  CK(cudaMalloc(&buffer, bufferSize ? bufferSize : 1));

  CK(cudaMemcpyAsync(d_col, J.data(), nz * sizeof(int), cudaMemcpyHostToDevice, stream1));
  CK(cudaMemcpyAsync(d_row, I.data(), (N + 1) * sizeof(int), cudaMemcpyHostToDevice, stream1));
  CK(cudaMemcpyAsync(d_val, val.data(), nz * sizeof(float), cudaMemcpyHostToDevice, stream1));
  initVectors<<<8, 256, 0, stream1>>>(d_r, d_x, N);

  CK(cusparseSetStream(sp, stream1));
  CK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecx, &beta, vecAx, CUDA_R_32F,
                  CUSPARSE_SPMV_ALG_DEFAULT, buffer));
  CK(cublasSetStream(blas, stream1));
  CK(cublasSaxpy(blas, N, &alpham1, d_Ax, 1, d_r, 1));
  CK(cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE));
  CK(cublasSdot(blas, N, d_r, 1, d_r, 1, d_r1));
  // The first iteration, eagerly, as the sample does.
  CK(cublasScopy(blas, N, d_r, 1, d_p, 1));
  CK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp, &beta, vecAx, CUDA_R_32F,
                  CUSPARSE_SPMV_ALG_DEFAULT, buffer));
  CK(cublasSdot(blas, N, d_p, 1, d_Ax, 1, d_dot));
  r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);
  CK(cublasSaxpy(blas, N, d_a, d_p, 1, d_x, 1));
  a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);
  CK(cublasSaxpy(blas, N, d_na, d_Ax, 1, d_r, 1));
  CK(cudaMemcpyAsync(d_r0, d_r1, 4, cudaMemcpyDeviceToDevice, stream1));
  CK(cublasSdot(blas, N, d_r, 1, d_r, 1, d_r1));
  float r1 = 0;
  CK(cudaMemcpyAsync(&r1, d_r1, 4, cudaMemcpyDeviceToHost, stream1));
  CK(cudaStreamSynchronize(stream1));

  auto snapshot = [&] {
    Scalars s{};
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(&s.a, d_a, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&s.na, d_na, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&s.b, d_b, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&s.dot, d_dot, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&s.r0, d_r0, 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(&s.r1, d_r1, 4, cudaMemcpyDeviceToHost));
    h.it.push_back(s);
  };
  snapshot();

  cudaGraph_t initGraph = nullptr;
  cudaGraphExec_t graphExec = nullptr;
  if (graph) {
    CK(cublasSetStream(blas, stream1));
    CK(cusparseSetStream(sp, stream1));
    CK(cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal));
    r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_r0, d_b);
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE);
    CK(cublasSscal(blas, N, d_b, d_p, 1));
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);
    CK(cublasSaxpy(blas, N, &alpha, d_r, 1, d_p, 1));
    cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE);
    CK(cusparseSetPointerMode(sp, CUSPARSE_POINTER_MODE_HOST));
    CK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp, &beta, vecAx, CUDA_R_32F,
                    CUSPARSE_SPMV_ALG_DEFAULT, buffer));
    CK(cudaMemsetAsync(d_dot, 0, 4, stream1));
    CK(cublasSdot(blas, N, d_p, 1, d_Ax, 1, d_dot));
    r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);
    CK(cublasSaxpy(blas, N, d_a, d_p, 1, d_x, 1));
    a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);
    CK(cublasSaxpy(blas, N, d_na, d_Ax, 1, d_r, 1));
    CK(cudaMemcpyAsync(d_r0, d_r1, 4, cudaMemcpyDeviceToDevice, stream1));
    CK(cudaMemsetAsync(d_r1, 0, 4, stream1));
    CK(cublasSdot(blas, N, d_r, 1, d_r, 1, d_r1));
    CK(cudaMemcpyAsync(&r1, d_r1, 4, cudaMemcpyDeviceToHost, stream1));
    CK(cudaStreamEndCapture(stream1, &initGraph));
    CK(cudaGraphInstantiate(&graphExec, initGraph, nullptr, nullptr, 0));
  }
  CK(cublasSetStream(blas, stream1));
  CK(cusparseSetStream(sp, stream1));

  for (int k = 2; k <= iters; ++k) {
    if (graph) {
      CK(cudaGraphLaunch(graphExec, streamForGraph));
      CK(cudaStreamSynchronize(streamForGraph));
    } else {
      r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_r0, d_b);
      cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE);
      CK(cublasSscal(blas, N, d_b, d_p, 1));
      cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_HOST);
      CK(cublasSaxpy(blas, N, &alpha, d_r, 1, d_p, 1));
      CK(cusparseSpMV(sp, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, matA, vecp, &beta, vecAx, CUDA_R_32F,
                      CUSPARSE_SPMV_ALG_DEFAULT, buffer));
      cublasSetPointerMode(blas, CUBLAS_POINTER_MODE_DEVICE);
      CK(cublasSdot(blas, N, d_p, 1, d_Ax, 1, d_dot));
      r1_div_x<<<1, 1, 0, stream1>>>(d_r1, d_dot, d_a);
      CK(cublasSaxpy(blas, N, d_a, d_p, 1, d_x, 1));
      a_minus<<<1, 1, 0, stream1>>>(d_a, d_na);
      CK(cublasSaxpy(blas, N, d_na, d_Ax, 1, d_r, 1));
      CK(cudaMemcpyAsync(d_r0, d_r1, 4, cudaMemcpyDeviceToDevice, stream1));
      CK(cublasSdot(blas, N, d_r, 1, d_r, 1, d_r1));
      CK(cudaMemcpyAsync(&r1, d_r1, 4, cudaMemcpyDeviceToHost, stream1));
      CK(cudaStreamSynchronize(stream1));
    }
    snapshot();
  }
  (void)r1;
  if (graph) {
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(initGraph);
  }
  cusparseDestroy(sp);
  cublasDestroy(blas);
  cudaStreamDestroy(stream1);
  cudaStreamDestroy(streamForGraph);
  for (void* p : {(void*)d_col, (void*)d_row, (void*)d_val, (void*)d_x, (void*)d_r, (void*)d_p, (void*)d_Ax,
                  (void*)d_r1, (void*)d_r0, (void*)d_dot, (void*)d_a, (void*)d_na, (void*)d_b, buffer})
    cudaFree(p);
}

int main() {
  const int iters = 6;
  History eager, graph;
  run(false, iters, eager);
  run(true, iters, graph);
  int bad = 0;
  for (size_t k = 0; k < eager.it.size(); ++k) {
    const Scalars& e = eager.it[k];
    const Scalars& g = graph.it[k];
    auto close = [](float a, float b) { return std::fabs(a - b) <= 1e-3f * std::fmax(1.0f, std::fmax(std::fabs(a), std::fabs(b))); };
    const bool same = close(e.r1, g.r1) && close(e.a, g.a) && close(e.na, g.na) && close(e.b, g.b) && close(e.dot, g.dot) &&
                      close(e.r0, g.r0);
    std::printf("%-4s iteration %zu: r1 %.6g/%.6g a %.6g/%.6g na %.6g/%.6g b %.6g/%.6g dot %.6g/%.6g r0 %.6g/%.6g\n",
                same ? "ok" : "FAIL", k + 1, e.r1, g.r1, e.a, g.a, e.na, g.na, e.b, g.b, e.dot, g.dot, e.r0, g.r0);
    if (!same) ++bad;
  }
  std::printf("%s\n", bad == 0 ? "PASS" : "FAIL");
  return bad == 0 ? 0 : 1;
}
