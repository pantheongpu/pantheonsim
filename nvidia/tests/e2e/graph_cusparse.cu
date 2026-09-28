// cuSPARSE calls inside a captured CUDA graph.
//
// This is the shape of HiGHS's HiPDLP solver: each iteration is a kernel that
// writes a vector, an SpMV that reads it, and a kernel that reads the product,
// captured once and replayed with new inputs. A cuSPARSE that computed at call
// time read the vector before any kernel had written it, and was absent from
// every replay: the solver went to NaN.
//
// Also checked, as on hardware: the graph keeps the descriptors and host-mode
// scalars as they were when captured, so changing either afterwards does not
// reach into it.
#include <cmath>
#include <cstdio>
#include <cstring>

#include <cuda_runtime.h>
#include <cusparse.h>

#define CK(x)                                                                     \
  do {                                                                            \
    cudaError_t e_ = (x);                                                         \
    if (e_) {                                                                     \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
      return 1;                                                                   \
    }                                                                             \
  } while (0)
#define CS(x)                                                                        \
  do {                                                                               \
    cusparseStatus_t s_ = (x);                                                       \
    if (s_) {                                                                        \
      std::printf("FAIL %s:%d: cuSPARSE status %d\n", __FILE__, __LINE__, (int)s_);   \
      return 1;                                                                      \
    }                                                                                \
  } while (0)

constexpr int N = 64;

// x[i] = k + i, with k read on the device, as HiPDLP reads its iteration count.
__global__ void set_x(double* x, const int* k) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) x[i] = *k + i;
}
__global__ void plus_one(double* z, const double* y) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) z[i] = y[i] + 1.0;
}

int main() {
  // A tridiagonal N x N matrix: 2 on the diagonal, -1 beside it.
  int rows[N + 1], cols[3 * N];
  double vals[3 * N];
  int nnz = 0;
  for (int r = 0; r < N; ++r) {
    rows[r] = nnz;
    for (int c = r - 1; c <= r + 1; ++c)
      if (c >= 0 && c < N) { cols[nnz] = c; vals[nnz] = c == r ? 2.0 : -1.0; ++nnz; }
  }
  rows[N] = nnz;

  int *d_rows, *d_cols, *d_k;
  double *d_vals, *d_x, *d_y, *d_z, *d_other, *d_mx, *d_my;
  CK(cudaMalloc(&d_rows, sizeof rows));
  CK(cudaMalloc(&d_cols, nnz * sizeof(int)));
  CK(cudaMalloc(&d_vals, nnz * sizeof(double)));
  CK(cudaMalloc(&d_k, sizeof(int)));
  for (double** p : {&d_x, &d_y, &d_z, &d_other, &d_mx, &d_my}) CK(cudaMalloc(p, 2 * N * sizeof(double)));
  CK(cudaMemcpy(d_rows, rows, sizeof rows, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_cols, cols, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(d_vals, vals, nnz * sizeof(double), cudaMemcpyHostToDevice));
  // Poisoned: any read of these before a kernel writes them is NaN.
  for (double* p : {d_x, d_y, d_z, d_mx, d_my}) CK(cudaMemset(p, 0xff, 2 * N * sizeof(double)));
  CK(cudaMemset(d_other, 0, 2 * N * sizeof(double)));

  cudaStream_t stream;
  CK(cudaStreamCreate(&stream));
  cusparseHandle_t h;
  CS(cusparseCreate(&h));
  CS(cusparseSetStream(h, stream));
  cusparseSpMatDescr_t A;
  cusparseDnVecDescr_t X, Y;
  cusparseDnMatDescr_t MX, MY;
  CS(cusparseCreateCsr(&A, N, N, nnz, d_rows, d_cols, d_vals, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                       CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
  CS(cusparseCreateDnVec(&X, N, d_x, CUDA_R_64F));
  CS(cusparseCreateDnVec(&Y, N, d_y, CUDA_R_64F));
  // Two columns: x and 2x, both written by the graph's kernel via SpMV below.
  CS(cusparseCreateDnMat(&MX, N, 2, N, d_mx, CUDA_R_64F, CUSPARSE_ORDER_COL));
  CS(cusparseCreateDnMat(&MY, N, 2, N, d_my, CUDA_R_64F, CUSPARSE_ORDER_COL));

  double alpha = 1.0, beta = 0.0;
  size_t bytes = 0, mbytes = 0;
  CS(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, X, &beta, Y, CUDA_R_64F,
                             CUSPARSE_SPMV_CSR_ALG2, &bytes));
  CS(cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                             A, MX, &beta, MY, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &mbytes));
  void *buf = nullptr, *mbuf = nullptr;
  CK(cudaMalloc(&buf, bytes ? bytes : 1));
  CK(cudaMalloc(&mbuf, mbytes ? mbytes : 1));

  cudaGraph_t graph;
  cudaGraphExec_t exec;
  CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
  set_x<<<1, N, 0, stream>>>(d_x, d_k);
  CS(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, X, &beta, Y, CUDA_R_64F,
                  CUSPARSE_SPMV_CSR_ALG2, buf));
  plus_one<<<1, N, 0, stream>>>(d_z, d_y);
  // SpMM over [x, 2x]: the columns are filled from x by copies in the graph.
  CK(cudaMemcpyAsync(d_mx, d_x, N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
  CK(cudaMemcpyAsync(d_mx + N, d_y, N * sizeof(double), cudaMemcpyDeviceToDevice, stream));
  CS(cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, MX, &beta,
                  MY, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, mbuf));
  CK(cudaStreamEndCapture(stream, &graph));
  CK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));

  // Neither of these may reach into the graph.
  alpha = 100.0;
  CS(cusparseDnVecSetValues(Y, d_other));

  int bad = 0;
  for (int k : {3, 10}) {
    CK(cudaMemcpyAsync(d_k, &k, sizeof k, cudaMemcpyHostToDevice, stream));
    CK(cudaGraphLaunch(exec, stream));
    CK(cudaStreamSynchronize(stream));
    double y[N], z[N], my[2 * N], other[N];
    CK(cudaMemcpy(y, d_y, sizeof y, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(z, d_z, sizeof z, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(my, d_my, sizeof my, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(other, d_other, sizeof other, cudaMemcpyDeviceToHost));
    for (int i = 0; i < N; ++i) {
      auto x = [&](int j) { return j < 0 || j >= N ? 0.0 : double(k + j); };
      const double want = 2 * x(i) - x(i - 1) - x(i + 1);
      // A applied to y, which is A x: second differences of a line are zero
      // inside, so only the ends are nonzero.
      double ay = 2 * want;
      if (i > 0) ay -= 2 * x(i - 1) - x(i - 2) - x(i);
      if (i < N - 1) ay -= 2 * x(i + 1) - x(i) - x(i + 2);
      if (y[i] != want) { if (bad++ < 5) std::printf("FAIL k=%d SpMV y[%d] = %g, want %g\n", k, i, y[i], want); }
      if (z[i] != want + 1) { if (bad++ < 5) std::printf("FAIL k=%d kernel after SpMV z[%d] = %g\n", k, i, z[i]); }
      if (my[i] != want) { if (bad++ < 5) std::printf("FAIL k=%d SpMM column 0 [%d] = %g, want %g\n", k, i, my[i], want); }
      if (my[N + i] != ay) { if (bad++ < 5) std::printf("FAIL k=%d SpMM column 1 [%d] = %g, want %g\n", k, i, my[N + i], ay); }
      if (other[i] != 0.0) { if (bad++ < 5) std::printf("FAIL k=%d a descriptor changed after capture reached the graph\n", k); }
    }
  }
  std::printf(bad ? "FAILED\n" : "PASS\n");
  // The graph is left for exit to release, as programs often do. Releasing its
  // recorded cuSPARSE calls then used to crash, after libcusparse's own state
  // had been destroyed.
  return bad ? 1 : 0;
}
