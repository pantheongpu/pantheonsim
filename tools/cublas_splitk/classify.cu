// Which epilogue does the card use for each split product of a table? For every row "<type> <ta> <tb> <m> <n> <k> <L>" on stdin (type as in
// cublas_splitk.hpp: 0 f16->f32, 1 f16->f16, 2 bf16->f32, 3 bf16->f16, 4 tf32->f32) the card's cublasGemmEx runs on random inputs with alpha = 1.3 and
// beta = 0.7 (which separate the fused from the unfused forms), and the model of nvidia/src/gemm_tensor_sum.hpp answers the same with the sum epilogue
// and, for two slices, the serial one. Prints the row and S (sum form matches bit for bit), A (serial form matches) or none.
//   g++ -x c++ -std=c++20 -O2 -pthread -ffp-contract=off -Iinclude -Inv_src -I<cuda>/include classify.cu simd/simd.cpp simd/lanes.cpp ... -lcublas -lcudart
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cublas_splitk.hpp"
#include "gemm_input_rounding.hpp"
#include "gemm_tensor_sum.hpp"

static int g_cc = 0;
extern "C" int vgpu_tensor_sum_v2(int, unsigned* chunk, unsigned* chunk_tf32, int* cc, int* splitk) {
  *chunk = 8; *chunk_tf32 = 4; *cc = g_cc; *splitk = 0;
  return g_cc >= 75 ? 1 : 0;
}
extern "C" int vgpu_tensor_sum_v1(int d, unsigned* a, unsigned* b, int* c) { int s; return vgpu_tensor_sum_v2(d, a, b, c, &s); }

static uint64_t g_state = 88172645463325252ull;
static float next_value() {
  g_state ^= g_state << 13; g_state ^= g_state >> 7; g_state ^= g_state << 17;
  return static_cast<float>((static_cast<double>(g_state >> 11) / 9007199254740992.0 - 0.5) * 4.0);
}
enum Fmt { F16, BF16, F32 };
static size_t fb(Fmt f) { return f == F32 ? 4 : 2; }
static void put(Fmt f, uint8_t* p, float x) {
  if (f == F16) { const __half h = __float2half(x); std::memcpy(p, &h, 2); }
  else if (f == BF16) { const __nv_bfloat16 h = __float2bfloat16(x); std::memcpy(p, &h, 2); }
  else std::memcpy(p, &x, 4);
}
static float get(Fmt f, const uint8_t* p) {
  if (f == F16) { __half h; std::memcpy(&h, p, 2); return __half2float(h); }
  if (f == BF16) { __nv_bfloat16 h; std::memcpy(&h, p, 2); return __bfloat162float(h); }
  float x; std::memcpy(&x, p, 4); return x;
}
static cudaDataType dt(Fmt f) { return f == F16 ? CUDA_R_16F : f == BF16 ? CUDA_R_16BF : CUDA_R_32F; }

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  g_cc = prop.major * 10 + prop.minor;
  cublasHandle_t h;
  cublasCreate(&h);
  using namespace vgpu::gemmtc;
  int type, ta, tb, m, n, k, L;
  while (std::scanf("%d %d %d %d %d %d %d", &type, &ta, &tb, &m, &n, &k, &L) == 7) {
    const Fmt in = type <= 1 ? F16 : type <= 3 ? BF16 : F32, out = (type == 1 || type == 3) ? F16 : F32;
    const bool tf32 = type == 4;
    const int lda = ta ? k : m, ldb = tb ? n : k, ldc = m;
    const size_t na = (size_t)lda * (ta ? m : k), nb = (size_t)ldb * (tb ? k : n), nc = (size_t)ldc * n;
    g_state = 88172645463325252ull ^ ((uint64_t)m << 40) ^ ((uint64_t)n << 20) ^ (uint64_t)k;
    std::vector<uint8_t> A(na * fb(in)), B(nb * fb(in)), C(nc * fb(out));
    for (size_t i = 0; i < na; ++i) put(in, &A[i * fb(in)], next_value());
    for (size_t i = 0; i < nb; ++i) put(in, &B[i * fb(in)], next_value());
    for (size_t i = 0; i < nc; ++i) put(out, &C[i * fb(out)], next_value());
    void *dA, *dB, *dC;
    cudaMalloc(&dA, A.size()); cudaMalloc(&dB, B.size()); cudaMalloc(&dC, C.size());
    cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size(), cudaMemcpyHostToDevice);
    const float alpha = 1.3f, beta = 0.7f;
    cublasSetMathMode(h, tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    const cublasStatus_t st = cublasGemmEx(h, ta ? CUBLAS_OP_T : CUBLAS_OP_N, tb ? CUBLAS_OP_T : CUBLAS_OP_N, m, n, k, &alpha, dA, dt(in), lda, dB, dt(in), ldb,
                                           &beta, dC, dt(out), ldc, tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    std::vector<uint8_t> card(C.size());
    cudaMemcpy(card.data(), dC, C.size(), cudaMemcpyDeviceToHost);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    if (st != CUBLAS_STATUS_SUCCESS) { std::printf("%d %d %d %d %d %d %d refused\n", type, ta, tb, m, n, k, L); std::fflush(stdout); continue; }
    std::vector<float> fa(na), fbv(nb), Cold(nc), R;
    for (size_t i = 0; i < na; ++i) { fa[i] = get(in, &A[i * fb(in)]); if (tf32) fa[i] = vgpu::gemmround::to_tf32(fa[i]); }
    for (size_t i = 0; i < nb; ++i) { fbv[i] = get(in, &B[i * fb(in)]); if (tf32) fbv[i] = vgpu::gemmround::to_tf32(fbv[i]); }
    for (size_t i = 0; i < nc; ++i) Cold[i] = get(out, &C[i * fb(out)]);
    Model md = model_of(0);
    const Format f = tf32 ? Format::Tf32 : in == F16 ? Format::F16 : Format::Bf16;
    auto matches = [&](bool serial) {
      product_epilogue(md, f, ta != 0, tb != 0, m, n, k, fa.data(), lda, fbv.data(), ldb, L, serial, alpha, beta, Cold.data(), R);
      for (int j = 0; j < n; ++j)
        for (int i = 0; i < m; ++i) {
          uint8_t buf[4];
          put(out, buf, R[(size_t)i + (size_t)j * m]);
          if (std::memcmp(buf, &card[((size_t)i + (size_t)j * ldc) * fb(out)], fb(out))) return false;
        }
      return true;
    };
    const bool two = L < k && (k + L - 1) / L == 2;
    const char* res = matches(false) ? "S" : (two && matches(true)) ? "A" : "none";
    std::printf("%d %d %d %d %d %d %d %s\n", type, ta, tb, m, n, k, L, res);
    std::fflush(stdout);
  }
  return 0;
}
