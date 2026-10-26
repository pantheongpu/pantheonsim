// E5b / use case 5: library offload. cublasSgemm and cublasGemmEx(fp16 in, fp32 accumulate) through the SAME binary on the simulator
// (its clean-room cuBLAS) and on the real A10G (NVIDIA's cuBLAS). Prints time per call and a hash of C, so equal bits are visible.
// Build: nvcc -arch=sm_86 -O2 -cudart shared e5_gemm.cu -o e5_gemm -lcublas
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <time.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "FAIL %d %s\n", __LINE__, cudaGetErrorName(e_)); exit(1); } } while (0)
#define CB(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "FAIL cublas %d status %d\n", __LINE__, (int)s_); exit(1); } } while (0)
static double now() { timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec + t.tv_nsec * 1e-9; }
static unsigned fnv(const void* p, size_t n) { unsigned h = 2166136261u; const unsigned char* b = (const unsigned char*)p; for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 16777619u; return h; }
int main(int argc, char** argv) {
  cublasHandle_t h; CB(cublasCreate(&h));
  for (int n : {256, 512, 1024, 2048}) {
    size_t e = (size_t)n * n;
    std::vector<float> a(e), b(e), c(e);
    unsigned long long x = 88172645463325252ull; auto rnd = [&]() { x ^= x << 13; x ^= x >> 7; x ^= x << 17; return float((x >> 40) & 0xffff) / 32768.0f - 1.0f; };
    for (size_t i = 0; i < e; ++i) { a[i] = rnd(); b[i] = rnd(); }
    float *da, *db, *dc; CK(cudaMalloc(&da, e * 4)); CK(cudaMalloc(&db, e * 4)); CK(cudaMalloc(&dc, e * 4));
    CK(cudaMemcpy(da, a.data(), e * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), e * 4, cudaMemcpyHostToDevice));
    float al = 1.f, be = 0.f;
    CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, da, n, db, n, &be, dc, n)); CK(cudaDeviceSynchronize());
    double t0 = now(); CB(cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, da, n, db, n, &be, dc, n)); CK(cudaDeviceSynchronize()); double t1 = now();
    CK(cudaMemcpy(c.data(), dc, e * 4, cudaMemcpyDeviceToHost));
    printf("sgemm n=%d ms %.3f gflops %.1f hash %08x\n", n, (t1 - t0) * 1e3, 2.0 * n * n * n / (t1 - t0) * 1e-9, fnv(c.data(), e * 4)); fflush(stdout);
    // fp16 inputs, fp32 accumulate and output
    std::vector<__half> ah(e), bh(e);
    for (size_t i = 0; i < e; ++i) { ah[i] = __float2half(a[i]); bh[i] = __float2half(b[i]); }
    __half *dah, *dbh; CK(cudaMalloc(&dah, e * 2)); CK(cudaMalloc(&dbh, e * 2));
    CK(cudaMemcpy(dah, ah.data(), e * 2, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dbh, bh.data(), e * 2, cudaMemcpyHostToDevice));
    CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, dah, CUDA_R_16F, n, dbh, CUDA_R_16F, n, &be, dc, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize());
    t0 = now(); CB(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &al, dah, CUDA_R_16F, n, dbh, CUDA_R_16F, n, &be, dc, CUDA_R_32F, n, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));
    CK(cudaDeviceSynchronize()); t1 = now();
    CK(cudaMemcpy(c.data(), dc, e * 4, cudaMemcpyDeviceToHost));
    printf("gemmex_f16 n=%d ms %.3f gflops %.1f hash %08x\n", n, (t1 - t0) * 1e3, 2.0 * n * n * n / (t1 - t0) * 1e-9, fnv(c.data(), e * 4)); fflush(stdout);
    cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dah); cudaFree(dbh);
  }
  return 0;
}
