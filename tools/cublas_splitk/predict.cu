// What the simulator's model (nvidia/src/gemm_tensor_sum.hpp and cublas_splitk.hpp) answers for the products of
// nvidia/tests/e2e/cublas_tc_gemm.cu, computed on the host of a card machine and printed in that program's format, to be diffed against the card's
// own lines before the card's file is pinned (and so before the simulator is run at all): cublasGemmEx is replaced by the model.
//   nvcc -std=c++20 -I<repo>/include -I<repo>/nvidia/src -arch=sm_XX -Xcompiler -ffp-contract=off predict.cu -o predict -lcublas
// The device decides the table: compute capability 7.5 the T4 one, 8.6 the A10G one. Cases the model does not take (a BF16 product on a Turing
// card, the sequential sums of the host) print the unchanged C and are to be ignored.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <vector>

#include "cublas_splitk.hpp"
#include "gemm_input_rounding.hpp"
#include "gemm_tensor_sum.hpp"

static int g_dev_cc = 0;
static cublasStatus_t real_gemmex(cublasHandle_t h, cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, const void* al, const void* A, cudaDataType ta,
                                  int lda, const void* B, cudaDataType tb, int ldb, const void* be, void* C, cudaDataType tc, int ldc, cublasComputeType_t ct, cublasGemmAlgo_t a) {
  return cublasGemmEx(h, oa, ob, m, n, k, al, A, ta, lda, B, tb, ldb, be, C, tc, ldc, ct, a);
}
extern "C" int vgpu_tensor_sum_v2(int, unsigned* chunk, unsigned* chunk_tf32, int* cc, int* splitk) {
  *chunk = 8; *chunk_tf32 = 4; *cc = g_dev_cc;
  *splitk = g_dev_cc == 75 ? 1 : g_dev_cc == 86 ? 2 : 0;
  return g_dev_cc >= 75 ? 1 : 0;
}
extern "C" int vgpu_tensor_sum_v1(int d, unsigned* a, unsigned* b, int* c) { int s; return vgpu_tensor_sum_v2(d, a, b, c, &s); }

static float load(cudaDataType t, const uint8_t* p) {
  if (t == CUDA_R_16F) { __half h; std::memcpy(&h, p, 2); return __half2float(h); }
  if (t == CUDA_R_16BF) { __nv_bfloat16 h; std::memcpy(&h, p, 2); return __bfloat162float(h); }
  float x; std::memcpy(&x, p, 4); return x;
}
static void store(cudaDataType t, uint8_t* p, float x) {
  if (t == CUDA_R_16F) { const __half h = __float2half(x); std::memcpy(p, &h, 2); }
  else if (t == CUDA_R_16BF) { const __nv_bfloat16 h = __float2bfloat16(x); std::memcpy(p, &h, 2); }
  else std::memcpy(p, &x, 4);
}
static size_t bytes(cudaDataType t) { return t == CUDA_R_32F ? 4 : 2; }

static cublasStatus_t predict_gemmex(cublasHandle_t hh, cublasOperation_t oa, cublasOperation_t ob, int m, int n, int k, const void* alpha, const void* dA,
                                     cudaDataType ta_, int lda, const void* dB, cudaDataType, int ldb, const void* beta, void* dC, cudaDataType tc_, int ldc,
                                     cublasComputeType_t compute, cublasGemmAlgo_t) {
  using namespace vgpu::gemmtc;
  const bool ta = oa != CUBLAS_OP_N, tb = ob != CUBLAS_OP_N;
  const size_t na = (size_t)lda * (ta ? m : k), nb = (size_t)ldb * (tb ? k : n), nc = (size_t)ldc * n;
  std::vector<uint8_t> rA(na * bytes(ta_)), rB(nb * bytes(ta_)), rC(nc * bytes(tc_));
  cudaMemcpy(rA.data(), dA, rA.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rB.data(), dB, rB.size(), cudaMemcpyDeviceToHost);
  cudaMemcpy(rC.data(), dC, rC.size(), cudaMemcpyDeviceToHost);
  const bool tf32 = compute == CUBLAS_COMPUTE_32F_FAST_TF32;
  std::vector<float> fa(na), fb(nb), P;
  for (size_t i = 0; i < na; ++i) { fa[i] = load(ta_, &rA[i * bytes(ta_)]); if (tf32) fa[i] = vgpu::gemmround::to_tf32(fa[i]); }
  for (size_t i = 0; i < nb; ++i) { fb[i] = load(ta_, &rB[i * bytes(ta_)]); if (tf32) fb[i] = vgpu::gemmround::to_tf32(fb[i]); }
  const Format f = tf32 ? Format::Tf32 : ta_ == CUDA_R_16F ? Format::F16 : Format::Bf16;
  Model md = model_of(0);
  if (!md.takes(f)) return CUBLAS_STATUS_SUCCESS;
  const Plan plan = plan_of(static_cast<SplitTable>(md.splitk), f, tc_ != CUDA_R_32F, ta, tb, m, n, k);
  const int slice = plan.len;
  const float al = *static_cast<const float*>(alpha), be = *static_cast<const float*>(beta);
  std::vector<float> Cold((size_t)m * n), R;
  if (be != 0.f)
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) Cold[(size_t)i + (size_t)j * m] = load(tc_, &rC[((size_t)i + (size_t)j * ldc) * bytes(tc_)]);
  product_epilogue(md, f, ta, tb, m, n, k, fa.data(), lda, fb.data(), ldb, slice, plan.serial, al, be, Cold.data(), R);
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) store(tc_, &rC[((size_t)i + (size_t)j * ldc) * bytes(tc_)], R[(size_t)i + (size_t)j * m]);
  if (std::getenv("CMP")) {   // CMP=1: also run the card's cuBLAS on the same inputs and say where the model's output differs (to stderr)
    std::vector<uint8_t> cardC(rC.size());
    void* d2;
    cudaMalloc(&d2, rC.size());
    std::vector<uint8_t> c0(rC.size());
    cudaMemcpy(c0.data(), dC, rC.size(), cudaMemcpyDeviceToHost);   // the caller's C (unchanged until here)
    cudaMemcpy(d2, c0.data(), rC.size(), cudaMemcpyHostToDevice);
    real_gemmex(hh, oa, ob, m, n, k, alpha, dA, ta_, lda, dB, ta_, ldb, beta, d2, tc_, ldc, compute, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    cudaMemcpy(cardC.data(), d2, rC.size(), cudaMemcpyDeviceToHost);
    cudaFree(d2);
    long bad = 0; int shown = 0;
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i) {
        const size_t at = ((size_t)i + (size_t)j * ldc) * bytes(tc_);
        if (std::memcmp(&cardC[at], &rC[at], bytes(tc_))) {
          ++bad;
          if (shown++ < 6) std::fprintf(stderr, "  (%d,%d) card %.9g model %.9g slice %d\n", i, j, load(tc_, &cardC[at]), load(tc_, &rC[at]), slice);
        }
      }
    std::fprintf(stderr, "CASE %dx%dx%d ta%d tb%d in%d out%d alpha%g beta%g: differing %ld of %d, slice %d\n", m, n, k, (int)ta, (int)tb, (int)ta_, (int)tc_, *(const float*)alpha, *(const float*)beta, bad, m * n, slice);
  }
  cudaMemcpy(dC, rC.data(), rC.size(), cudaMemcpyHostToDevice);
  return CUBLAS_STATUS_SUCCESS;
}
#define cublasGemmEx predict_gemmex
#define main card_main
#include "cublas_tc_gemm.cu"
#undef main
int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  g_dev_cc = p.major * 10 + p.minor;
  return card_main();
}
