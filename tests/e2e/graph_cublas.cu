// A cuBLAS call inside a captured CUDA graph.
//
// This is the shape llama.cpp uses for batched attention: a kernel writes an
// array of device pointers, and the very next call is a batched GEMM that reads
// them. Inside a stream capture the kernel has not run yet -- it runs when the
// graph is launched -- so a cuBLAS implementation that computes at call time
// reads uninitialized memory and dereferences it as pointers.
//
// The arrays are poisoned before capture so that failing to record the GEMM is
// caught here rather than showing up as plausible-looking wrong numbers later.
#include <cmath>
#include <cstdio>

#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#define CK(x)                                                                        \
  do {                                                                               \
    cudaError_t e_ = (x);                                                            \
    if (e_) {                                                                        \
      std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_));    \
      return 1;                                                                      \
    }                                                                                \
  } while (0)
#define CB(x)                                                              \
  do {                                                                     \
    cublasStatus_t s_ = (x);                                               \
    if (s_) {                                                              \
      std::printf("FAIL %s:%d: cuBLAS status %d\n", __FILE__, __LINE__, (int)s_); \
      return 1;                                                            \
    }                                                                      \
  } while (0)

// The same pointer arithmetic ggml's k_compute_batched_ptrs does.
__global__ void fill_ptrs(const void* a, const void* b, char* d, const void** ps, void** pd,
                          long long ne12, long long ne13, long long ne23, size_t nb02, size_t nb03,
                          size_t nb12, size_t nb13, size_t nbd2, size_t nbd3) {
  const long long i13 = blockIdx.x * blockDim.x + threadIdx.x;
  const long long i12 = blockIdx.y * blockDim.y + threadIdx.y;
  if (i13 >= ne13 || i12 >= ne12) return;
  ps[0 * ne23 + i12 + i13 * ne12] = (const char*)a + i12 * nb02 + i13 * nb03;
  ps[1 * ne23 + i12 + i13 * ne12] = (const char*)b + i12 * nb12 + i13 * nb13;
  pd[0 * ne23 + i12 + i13 * ne12] = (char*)d + i12 * nbd2 + i13 * nbd3;
}

int main() {
  const int M = 8, N = 4, K = 8;
  const long long ne12 = 4, ne13 = 2, ne23 = ne12 * ne13;
  const size_t na = (size_t)M * K, nb = (size_t)K * N, nc = (size_t)M * N;

  cublasHandle_t h;
  CB(cublasCreate(&h));
  cudaStream_t st;
  CK(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking));
  CB(cublasSetStream(h, st));

  __half *A, *B;
  float* C;
  CK(cudaMalloc(&A, na * ne23 * sizeof(__half)));
  CK(cudaMalloc(&B, nb * ne23 * sizeof(__half)));
  CK(cudaMalloc(&C, nc * ne23 * sizeof(float)));

  __half hA[na * 8], hB[nb * 8];
  for (long long z = 0; z < ne23; ++z) {
    for (size_t i = 0; i < na; ++i) hA[z * na + i] = __float2half(float((i + z) % 7) * 0.25f);
    for (size_t i = 0; i < nb; ++i) hB[z * nb + i] = __float2half(float((i + 2 * z) % 5) * 0.5f);
  }
  CK(cudaMemcpy(A, hA, na * ne23 * sizeof(__half), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(B, hB, nb * ne23 * sizeof(__half), cudaMemcpyHostToDevice));

  const void** ps;
  void** pd;
  CK(cudaMalloc(&ps, 2 * ne23 * sizeof(void*)));
  CK(cudaMalloc(&pd, 1 * ne23 * sizeof(void*)));

  const dim3 blk(1, 1), grd((unsigned)ne13, (unsigned)ne12);
  const float alpha = 1.0f, beta = 0.0f;
  auto launch_fill = [&] {
    fill_ptrs<<<grd, blk, 0, st>>>(A, B, (char*)C, ps, pd, ne12, ne13, ne23, na * sizeof(__half),
                                   na * ne12 * sizeof(__half), nb * sizeof(__half),
                                   nb * ne12 * sizeof(__half), nc * sizeof(float),
                                   nc * ne12 * sizeof(float));
  };
  auto gemm = [&] {
    return cublasGemmBatchedEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, N, K, &alpha,
                               (const void* const*)(ps + 0 * ne23), CUDA_R_16F, K,
                               (const void* const*)(ps + 1 * ne23), CUDA_R_16F, K, &beta,
                               (void* const*)pd, CUDA_R_32F, M, (int)ne23, CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT);
  };

  // Reference: the same work run eagerly.
  CK(cudaMemset(C, 0, nc * ne23 * sizeof(float)));
  CK(cudaMemset(ps, 0x3f, 2 * ne23 * sizeof(void*)));
  CK(cudaMemset(pd, 0x3f, 1 * ne23 * sizeof(void*)));
  launch_fill();
  CK(cudaGetLastError());
  CB(gemm());
  CK(cudaStreamSynchronize(st));
  float eager[nc * 8];
  CK(cudaMemcpy(eager, C, nc * ne23 * sizeof(float), cudaMemcpyDeviceToHost));

  int wrong = 0;
  for (long long z = 0; z < ne23; ++z)
    for (int n = 0; n < N; ++n)
      for (int m = 0; m < M; ++m) {
        double acc = 0.0;
        for (int k = 0; k < K; ++k)
          acc += (double)__half2float(hA[z * na + m * K + k]) *
                 (double)__half2float(hB[z * nb + n * K + k]);
        if (std::fabs(eager[z * nc + n * M + m] - acc) > 1e-3) ++wrong;
      }
  if (wrong) {
    std::printf("FAIL: eager batched GEMM wrong in %d of %lld entries\n", wrong,
                (long long)(nc * ne23));
    return 1;
  }

  // The captured form must produce the same thing.
  CK(cudaMemset(C, 0, nc * ne23 * sizeof(float)));
  CK(cudaMemset(ps, 0x3f, 2 * ne23 * sizeof(void*)));
  CK(cudaMemset(pd, 0x3f, 1 * ne23 * sizeof(void*)));
  cudaGraph_t g;
  cudaGraphExec_t ge;
  CK(cudaStreamBeginCapture(st, cudaStreamCaptureModeRelaxed));
  launch_fill();
  CB(gemm());
  CK(cudaStreamEndCapture(st, &g));
  CK(cudaGraphInstantiate(&ge, g, 0));
  CK(cudaGraphLaunch(ge, st));
  CK(cudaStreamSynchronize(st));

  float replayed[nc * 8];
  CK(cudaMemcpy(replayed, C, nc * ne23 * sizeof(float), cudaMemcpyDeviceToHost));
  int differ = 0;
  for (size_t i = 0; i < nc * ne23; ++i)
    if (std::fabs(replayed[i] - eager[i]) > 1e-6) ++differ;
  if (differ) {
    std::printf("FAIL: graph replay differs from eager in %d of %lld entries\n", differ,
                (long long)(nc * ne23));
    return 1;
  }

  std::printf("PASS\n");
  return 0;
}
