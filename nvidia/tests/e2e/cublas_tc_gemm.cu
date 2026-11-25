// cuBLAS tensor-core GEMMs (FP16, BF16 and TF32 inputs) on random data: one checksum per case, to be compared with what a real card
// printed (nvidia/tests/data/cublas_tc_gemm.<card>.txt, run by run_cublas_tc_gemm.sh).
//
// A real card's tensor core does not sum a K-long dot product in sequence: it adds a chunk of products and the accumulator in one
// step (docs/matrix-numerics.md), and the chunk, and so the bits of the answer, depend on the architecture (Hopper's chunk is twice
// Ampere's). cuBLAS on the simulator computes on the host; before issue "Cross-architecture (cuBLAS tensor-core GEMM)" it summed in
// sequence on every profile, so a half-precision GEMM printed the same bytes on a T4 profile and an H100 profile and neither was a
// card's. This program is the oracle that holds the library to the cards that were run.
//
// Not here, because the card's answer there is not a plain chain of chunks: cuBLAS splits some products over K, and the split is its
// heuristic's (the card's multiprocessor count among its inputs), which no rule here models. Seen on the cards that were run: K of 1024 and
// more on a T4 (f16 128x128x1024, 256x256x4096), an A10G's f16 96x80x512 into FP32 with alpha and beta (a T4's is the chain), and its TF32
// products of K 256 and 512 (K = 128 is the chain). Open: issue "Cross-architecture (cuBLAS tensor-core GEMM): products cuBLAS splits over K".
//
// Output, one line per case:  <name>: status <cublas status> checksum 0x<fnv1a-64 of the output bits> first <first 4 outputs, hex>
// Inputs: a fixed xorshift generator, values (r - 0.5) * 4 rounded to the input format; the same bytes on every machine.
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint64_t g_state = 0x9e3779b97f4a7c15ull;
static float next_value() {
  g_state ^= g_state << 13;
  g_state ^= g_state >> 7;
  g_state ^= g_state << 17;
  const double r = static_cast<double>(g_state >> 11) / 9007199254740992.0;   // [0, 1)
  return static_cast<float>((r - 0.5) * 4.0);
}

enum Fmt { F16, BF16, F32 };
static size_t fmt_bytes(Fmt f) { return f == F32 ? 4 : 2; }

static std::vector<uint8_t> make(Fmt f, size_t n) {
  std::vector<uint8_t> v(n * fmt_bytes(f));
  for (size_t i = 0; i < n; ++i) {
    const float x = next_value();
    if (f == F16) { const __half h = __float2half(x); std::memcpy(&v[i * 2], &h, 2); }
    else if (f == BF16) { const __nv_bfloat16 h = __float2bfloat16(x); std::memcpy(&v[i * 2], &h, 2); }
    else std::memcpy(&v[i * 4], &x, 4);
  }
  return v;
}

struct Case {
  const char* name;
  Fmt in, out;
  cublasComputeType_t compute;
  int m, n, k;
  bool ta, tb;
  float alpha, beta;
};

static uint64_t fnv(const std::vector<uint8_t>& v) {
  uint64_t h = 1469598103934665603ull;
  for (uint8_t b : v) { h ^= b; h *= 1099511628211ull; }
  return h;
}

static cudaDataType dt(Fmt f) { return f == F16 ? CUDA_R_16F : f == BF16 ? CUDA_R_16BF : CUDA_R_32F; }

static int g_cc = 0;   // compute capability times ten of the device

// The plain FP32 answer or the TF32 one? The product of the same inputs in double, the normwise error of the output against it: TF32 inputs
// (10 bits) give about 1e-4 to 1e-3, an FP32 kernel 1e-6 and below.
static const char* tf32_class(const Case& c, const std::vector<uint8_t>& A, const std::vector<uint8_t>& B, const std::vector<uint8_t>& C) {
  const int lda = c.ta ? c.k : c.m, ldb = c.tb ? c.n : c.k;
  auto f = [](const std::vector<uint8_t>& v, size_t i) { float x; std::memcpy(&x, &v[i * 4], 4); return static_cast<double>(x); };
  double err = 0, mag = 0;
  for (int j = 0; j < c.n; j += 7)
    for (int i = 0; i < c.m; i += 5) {
      double ref = 0;
      for (int p = 0; p < c.k; ++p)
        ref += f(A, c.ta ? p + static_cast<size_t>(i) * lda : i + static_cast<size_t>(p) * lda) * f(B, c.tb ? j + static_cast<size_t>(p) * ldb : p + static_cast<size_t>(j) * ldb);
      const double got = f(C, i + static_cast<size_t>(j) * c.m);
      err = std::max(err, std::fabs(got - ref));
      mag = std::max(mag, std::fabs(ref));
    }
  return err / mag < 2e-6 ? "plain fp32" : "tf32";
}

static void run(cublasHandle_t h, const Case& c) {
  g_state = 0x9e3779b97f4a7c15ull ^ (static_cast<uint64_t>(c.m) << 40) ^ (static_cast<uint64_t>(c.n) << 20) ^ static_cast<uint64_t>(c.k);
  const int lda = c.ta ? c.k : c.m, ldb = c.tb ? c.n : c.k, ldc = c.m;
  const size_t na = static_cast<size_t>(lda) * (c.ta ? c.m : c.k), nb = static_cast<size_t>(ldb) * (c.tb ? c.k : c.n),
               nc = static_cast<size_t>(ldc) * c.n;
  auto A = make(c.in, na), B = make(c.in, nb), C = make(c.out, nc);
  void *dA, *dB, *dC;
  cudaMalloc(&dA, A.size());
  cudaMalloc(&dB, B.size());
  cudaMalloc(&dC, C.size());
  cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dC, C.data(), C.size(), cudaMemcpyHostToDevice);
  cublasSetMathMode(h, c.compute == CUBLAS_COMPUTE_32F_FAST_TF32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
  const cublasStatus_t st = cublasGemmEx(h, c.ta ? CUBLAS_OP_T : CUBLAS_OP_N, c.tb ? CUBLAS_OP_T : CUBLAS_OP_N, c.m, c.n, c.k, &c.alpha,
                                         dA, dt(c.in), lda, dB, dt(c.in), ldb, &c.beta, dC, dt(c.out), ldc, c.compute, CUBLAS_GEMM_DEFAULT);
  cudaDeviceSynchronize();
  cudaMemcpy(C.data(), dC, C.size(), cudaMemcpyDeviceToHost);
  std::printf("%s: status %d", c.name, static_cast<int>(st));
  if (st == CUBLAS_STATUS_SUCCESS && c.compute == CUBLAS_COMPUTE_32F_FAST_TF32 && g_cc < 80) {
    // A card before Ampere has no TF32 tensor core: its answer is an ordinary FP32 kernel's, whose association is the library's.
    // Only that it is the plain answer (and not the TF32 one) is held to.
    std::printf(" class %s", tf32_class(c, A, B, C));
  } else if (st == CUBLAS_STATUS_SUCCESS) {
    if (c.compute == CUBLAS_COMPUTE_32F_FAST_TF32) std::printf(" class %s", tf32_class(c, A, B, C));
    std::printf(" checksum 0x%016llx first", static_cast<unsigned long long>(fnv(C)));
    for (int i = 0; i < 4 && static_cast<size_t>(i) < nc; ++i) {
      if (c.out == F32) { uint32_t u; std::memcpy(&u, &C[i * 4], 4); std::printf(" %08x", u); }
      else { uint16_t u; std::memcpy(&u, &C[i * 2], 2); std::printf(" %04x", u); }
    }
  }
  std::printf("\n");
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  g_cc = prop.major * 10 + prop.minor;
  cublasHandle_t h;
  cublasCreate(&h);
  const cublasComputeType_t C32 = CUBLAS_COMPUTE_32F, CT = CUBLAS_COMPUTE_32F_FAST_TF32;
  const Case cases[] = {
      // The shape of the application smoke's gemm_f16 / gemm_bf16 (PyTorch x.half() @ y.half()), and others in the same family.
      {"f16 f16 32x40x48 NN",       F16,  F16, C32, 32, 40, 48, false, false, 1.f, 0.f},
      {"bf16 bf16 32x40x48 NN",     BF16, BF16, C32, 32, 40, 48, false, false, 1.f, 0.f},
      {"f16 f32 32x40x48 NN",       F16,  F32, C32, 32, 40, 48, false, false, 1.f, 0.f},
      {"bf16 f32 32x40x48 NN",      BF16, F32, C32, 32, 40, 48, false, false, 1.f, 0.f},
      {"f16 f16 64x64x64 NN",       F16,  F16, C32, 64, 64, 64, false, false, 1.f, 0.f},
      {"f16 f16 64x64x64 TN",       F16,  F16, C32, 64, 64, 64, true,  false, 1.f, 0.f},
      {"f16 f16 64x64x64 NT",       F16,  F16, C32, 64, 64, 64, false, true,  1.f, 0.f},
      {"f16 f32 128x96x256 NN",     F16,  F32, C32, 128, 96, 256, false, false, 1.f, 0.f},
      {"f16 f32 128x96x256 TN",     F16,  F32, C32, 128, 96, 256, true,  false, 1.f, 0.f},
      {"bf16 f32 128x96x256 NN",    BF16, F32, C32, 128, 96, 256, false, false, 1.f, 0.f},
      {"f16 f16 96x80x512 alpha beta", F16, F16, C32, 96, 80, 512, false, false, 0.5f, 0.25f},
      {"f16 f32 64x64x16 NN",       F16,  F32, C32, 64, 64, 16, false, false, 1.f, 0.f},
      {"f16 f32 64x64x8 NN",        F16,  F32, C32, 64, 64, 8, false, false, 1.f, 0.f},
      // TF32: a product big enough for the card's tensor-core kernel (m n k >= 2^21).
      {"tf32 f32 128x128x128 NN",   F32,  F32, CT, 128, 128, 128, false, false, 1.f, 0.f},
      {"tf32 f32 128x128x128 TN",   F32,  F32, CT, 128, 128, 128, true,  false, 1.f, 0.f},
      {"tf32 f32 128x128x128 NT alpha beta", F32, F32, CT, 128, 128, 128, false, true, 0.5f, 0.25f},
  };
  for (const Case& c : cases) run(h, c);
  cublasDestroy(h);
  return 0;
}
