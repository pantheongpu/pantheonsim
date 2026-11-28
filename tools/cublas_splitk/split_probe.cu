// Summation-tree probe of a cuBLAS tensor-core GEMM (the FPRev idea): which products does cuBLAS add before which?
//
// For the output element (t, t) of an m x n x k product the program makes the dot product
//     sum_p A[t,p] * B[p,t]   with every product 1, except product a = +2^30 and product b = -2^30.
// (A 'one' is the product 1 * 2^-8, see kOne.) The two big products cancel, but every 1 that was added to one of them before they met is lost (a float, or a tensor core's aligned
// chunk, cannot hold 2^30 + 1). The result is therefore k - 2 - (the number of ones inside the smallest sub-sum that holds both a and b)
// + ... : exactly (k minus the size of the lowest common subtree of a and b). Scanning b for a fixed a draws the association tree:
// products in another K slice of a split-K kernel meet a's only when the slices are summed, so they print the smallest counts.
// Row t of A and column t of B carry the probe of element (t,t); the min(m,n) diagonal elements are min(m,n) probes in one GEMM.
//
// usage: split_probe <case> <a> [<b-first> <b-last>]    prints "a b count" lines (count = alpha-corrected integer, -1 when not an integer)
//        split_probe list                                prints the case names
// The cases are the shapes of the issue; the call is the one nvidia/tests/e2e/cublas_tc_gemm.cu makes (cublasGemmEx, CUBLAS_GEMM_DEFAULT).
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// A 'one' is the product 1 * kOne; kOne is small so that the sum of all of them (at most 4096 * 2^-8 = 16) stays far under half an ulp (64)
// of 2^30, which a float holding 2^30 + slice sum would otherwise round to: with plain ones a slice of 96 leaks 128 into the result.
static const float kOne = 0.00390625f;   // 2^-8
enum Fmt { F16, BF16, F32 };
struct Case {
  const char* name;
  Fmt in, out;
  bool tf32;
  int m, n, k;
  bool ta, tb;
  float alpha, beta;
};

static const Case kCases[] = {
    {"f16f32_96x80x512_ab", F16, F32, false, 96, 80, 512, false, false, 0.5f, 0.25f},
    {"f16f16_96x80x512_ab", F16, F16, false, 96, 80, 512, false, false, 0.5f, 0.25f},
    {"f16f32_128x128x1024", F16, F32, false, 128, 128, 1024, false, false, 1.f, 0.f},
    {"f16f32_256x256x4096", F16, F32, false, 256, 256, 4096, false, false, 1.f, 0.f},
    {"tf32_128x128x256_nn", F32, F32, true, 128, 128, 256, false, false, 1.f, 0.f},
    {"tf32_256x256x512_nn", F32, F32, true, 256, 256, 512, false, false, 1.f, 0.f},
    {"tf32_256x256x512_tn", F32, F32, true, 256, 256, 512, true, false, 1.f, 0.f},
    {"tf32_128x128x128_nn", F32, F32, true, 128, 128, 128, false, false, 1.f, 0.f},
    {"f16f32_128x96x256_nn", F16, F32, false, 128, 96, 256, false, false, 1.f, 0.f},
};

static cudaDataType dt(Fmt f) { return f == F16 ? CUDA_R_16F : f == BF16 ? CUDA_R_16BF : CUDA_R_32F; }
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

// The counts of GEMMs of one shape: probe t has the big products at a and bs[t]. The operands are built once (all ones) and the few entries of a
// probe patched in and out, so that a long scan does not rebuild millions of elements per GEMM.
struct Prober {
  Case c;
  int lda, ldb, ldc;
  size_t na, nb, nc;
  void *dA = nullptr, *dB = nullptr, *dC = nullptr;
  std::vector<uint8_t> A, B, C;
  explicit Prober(const Case& cc) : c(cc) {
    lda = c.ta ? c.k : c.m; ldb = c.tb ? c.n : c.k; ldc = c.m;
    na = (size_t)lda * (c.ta ? c.m : c.k); nb = (size_t)ldb * (c.tb ? c.k : c.n); nc = (size_t)ldc * c.n;
    cudaMalloc(&dA, na * fb(c.in)); cudaMalloc(&dB, nb * fb(c.in)); cudaMalloc(&dC, nc * fb(c.out));
    A.resize(na * fb(c.in)); B.resize(nb * fb(c.in)); C.resize(nc * fb(c.out));
    for (size_t i = 0; i < na; ++i) put(c.in, &A[i * fb(c.in)], 1.f);
    for (size_t i = 0; i < nb; ++i) put(c.in, &B[i * fb(c.in)], kOne);
  }
  ~Prober() { cudaFree(dA); cudaFree(dB); cudaFree(dC); }
  size_t Aat(int row, int p) const { return c.ta ? (size_t)p + (size_t)row * lda : (size_t)row + (size_t)p * lda; }
  size_t Bat(int p, int col) const { return c.tb ? (size_t)col + (size_t)p * ldb : (size_t)p + (size_t)col * ldb; }
  // false when cuBLAS refuses the call.
  bool counts(cublasHandle_t h, int a, const std::vector<int>& bs, std::vector<long>& out) {
    const float big = 32768.f;
    for (size_t t = 0; t < bs.size(); ++t) {
      put(c.in, &A[Aat((int)t, a) * fb(c.in)], big);
      put(c.in, &A[Aat((int)t, bs[t]) * fb(c.in)], big);
      put(c.in, &B[Bat(a, (int)t) * fb(c.in)], big);
      put(c.in, &B[Bat(bs[t], (int)t) * fb(c.in)], -big);
    }
    for (size_t i = 0; i < nc; ++i) put(c.out, &C[i * fb(c.out)], 0.f);
    cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size(), cudaMemcpyHostToDevice);
    for (size_t t = 0; t < bs.size(); ++t) {   // restore the ones
      put(c.in, &A[Aat((int)t, a) * fb(c.in)], 1.f);
      put(c.in, &A[Aat((int)t, bs[t]) * fb(c.in)], 1.f);
      put(c.in, &B[Bat(a, (int)t) * fb(c.in)], kOne);
      put(c.in, &B[Bat(bs[t], (int)t) * fb(c.in)], kOne);
    }
    cublasSetMathMode(h, c.tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    const cublasStatus_t st = cublasGemmEx(h, c.ta ? CUBLAS_OP_T : CUBLAS_OP_N, c.tb ? CUBLAS_OP_T : CUBLAS_OP_N, c.m, c.n, c.k, &c.alpha,
                                           dA, dt(c.in), lda, dB, dt(c.in), ldb, &c.beta, dC, dt(c.out), ldc,
                                           c.tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    if (st != CUBLAS_STATUS_SUCCESS) return false;
    cudaMemcpy(C.data(), dC, C.size(), cudaMemcpyDeviceToHost);
    out.clear();
    for (size_t t = 0; t < bs.size(); ++t) {
      const float v = get(c.out, &C[((size_t)t + (size_t)t * ldc) * fb(c.out)]) / (c.alpha * kOne);
      const long r = std::lround(v);
      out.push_back(std::fabs(v - (float)r) < 1e-3f ? r : -1L);
    }
    return true;
  }
};

// Survey mode: one shape per stdin line "<in f16|bf16|tf32> <out f16|f32> <m> <n> <k> <ta 0|1> <tb 0|1> <alpha> <beta>"; prints the K slice length
// cuBLAS used (k when it did not split) and the number of slices. The slice length is the first b = 16 j whose count with a = 0 is not the
// chain's (k minus the end of b's chunk of `chunk` products), the slices being summed one after the other (checked on the cases above).
static int survey() {
  cublasHandle_t h;
  cublasCreate(&h);
  char in[16], out[16];
  int m, n, k, ta, tb;
  float alpha, beta;
  while (std::scanf("%15s %15s %d %d %d %d %d %f %f", in, out, &m, &n, &k, &ta, &tb, &alpha, &beta) == 9) {
    Case c{"survey", F16, F32, false, m, n, k, ta != 0, tb != 0, alpha, beta};
    if (!std::strcmp(in, "bf16")) c.in = BF16;
    if (!std::strcmp(in, "tf32")) { c.in = F32; c.tf32 = true; }
    if (!std::strcmp(out, "f16")) c.out = F16;
    const int chunk = c.tf32 ? 4 : 8;
    const int probes = m < n ? m : n;
    long L = k;
    std::vector<int> bs;
    Prober pr(c);
    bool refused = false, done = false;
    for (int b = 16; b < k && !done;) {
      bs.clear();
      for (; b < k && (int)bs.size() < probes; b += 16) bs.push_back(b);
      std::vector<long> r;
      if (!pr.counts(h, 0, bs, r)) { refused = true; break; }
      for (size_t t = 0; t < bs.size(); ++t) {
        const long chain = k - ((bs[t] / chunk) + 1) * chunk;
        if (r[t] != chain) { L = bs[t]; done = true; break; }
      }
    }
    // Check the whole tree against the slice model for three values of a (the first product, one in the middle, the last): the count of ones
    // that survive with the big products at a and b is k minus the end of the later one's chunk (counted from the start of its slice) when both are in one slice, and k minus the end
    // of the later one's slice otherwise (slices added in order). A product whose scan disagrees anywhere is not a plain split ("odd").
    bool odd = false;
    if (!refused) {
      const int as[3] = {0, (k / 2) | 1, k - 1};
      for (int ai = 0; ai < 3 && !odd && !refused; ++ai) {
        const int a = as[ai];
        for (int b = 0; b < k && !odd && !refused;) {
          bs.clear();
          for (; b < k && (int)bs.size() < probes; b += 4) if (b != a) bs.push_back(b);
          std::vector<long> r;
          if (bs.empty()) break;
          if (!pr.counts(h, a, bs, r)) { refused = true; break; }
          for (size_t t = 0; t < bs.size(); ++t) {
            const int hi = bs[t] > a ? bs[t] : a, lo = bs[t] > a ? a : bs[t];
            long want;
            if (hi / (int)L == lo / (int)L) want = k - (long)(((hi % (int)L) / chunk + 1) * chunk);
            else { const long end = (long)(hi / (int)L + 1) * L; want = k - (end < k ? end : k); }
            if (r[t] != want) { odd = true; break; }
          }
        }
      }
    }
    if (refused) std::printf("%s %s %d %d %d %d %d %g %g refused\n", in, out, m, n, k, ta, tb, alpha, beta);
    else std::printf("%s %s %d %d %d %d %d %g %g L %ld slices %ld %s\n", in, out, m, n, k, ta, tb, alpha, beta, L, (k + L - 1) / L, odd ? "odd" : "ok");
    std::fflush(stdout);
  }
  return 0;
}

int main(int argc, char** argv) {
  if (argc >= 2 && !std::strcmp(argv[1], "list")) {
    for (const Case& c : kCases) std::printf("%s\n", c.name);
    return 0;
  }
  if (argc >= 2 && !std::strcmp(argv[1], "survey")) return survey();
  if (argc >= 12 && !std::strcmp(argv[1], "shape")) {
    // shape <f16|bf16|tf32> <f16|f32> m n k ta tb a b-first b-last: the scan of one arbitrary product (alpha 1, beta 0)
    Case c{"shape", F16, F32, false, std::atoi(argv[4]), std::atoi(argv[5]), std::atoi(argv[6]), std::atoi(argv[7]) != 0, std::atoi(argv[8]) != 0, 1.f, 0.f};
    if (!std::strcmp(argv[2], "bf16")) c.in = BF16;
    if (!std::strcmp(argv[2], "tf32")) { c.in = F32; c.tf32 = true; }
    if (!std::strcmp(argv[3], "f16")) c.out = F16;
    cublasHandle_t h;
    cublasCreate(&h);
    const int a = std::atoi(argv[9]), b0 = std::atoi(argv[10]), b1 = std::atoi(argv[11]);
    const int probes = c.m < c.n ? c.m : c.n;
    Prober pr(c);
    for (int b = b0; b <= b1;) {
      std::vector<int> bs;
      for (; b <= b1 && (int)bs.size() < probes; ++b) if (b != a) bs.push_back(b);
      std::vector<long> r;
      if (bs.empty() || !pr.counts(h, a, bs, r)) break;
      for (size_t t = 0; t < bs.size(); ++t) std::printf("%d %d %ld\n", a, bs[t], r[t]);
    }
    return 0;
  }
  if (argc < 3) { std::fprintf(stderr, "usage: split_probe <case> <a> [b-first b-last]\n"); return 2; }
  const Case* cs = nullptr;
  for (const Case& c : kCases) if (!std::strcmp(c.name, argv[1])) cs = &c;
  if (!cs) { std::fprintf(stderr, "no such case\n"); return 2; }
  const Case& c = *cs;
  const int a = std::atoi(argv[2]);
  int b0 = 0, b1 = c.k - 1;
  if (argc >= 5) { b0 = std::atoi(argv[3]); b1 = std::atoi(argv[4]); }
  cublasHandle_t h;
  cublasCreate(&h);
  const int lda = c.ta ? c.k : c.m, ldb = c.tb ? c.n : c.k, ldc = c.m;
  const size_t na = (size_t)lda * (c.ta ? c.m : c.k), nb = (size_t)ldb * (c.tb ? c.k : c.n), nc = (size_t)ldc * c.n;
  void *dA, *dB, *dC;
  cudaMalloc(&dA, na * fb(c.in)); cudaMalloc(&dB, nb * fb(c.in)); cudaMalloc(&dC, nc * fb(c.out));
  const int probes = c.m < c.n ? c.m : c.n;
  const float big = 32768.f;
  std::vector<uint8_t> A(na * fb(c.in)), B(nb * fb(c.in)), C(nc * fb(c.out));
  for (int bstart = b0; bstart <= b1;) {
    // Probe t uses b = bstart + t (skipping b == a).
    std::vector<int> bs;
    for (int b = bstart; b <= b1 && (int)bs.size() < probes; ++b) if (b != a) bs.push_back(b);
    bstart = bs.empty() ? b1 + 1 : bs.back() + 1;
    if (bs.empty()) break;
    for (size_t i = 0; i < na; ++i) put(c.in, &A[i * fb(c.in)], 1.f);
    for (size_t i = 0; i < nb; ++i) put(c.in, &B[i * fb(c.in)], kOne);
    for (size_t i = 0; i < nc; ++i) put(c.out, &C[i * fb(c.out)], 0.f);
    auto Aat = [&](int row, int p) { return c.ta ? (size_t)p + (size_t)row * lda : (size_t)row + (size_t)p * lda; };
    auto Bat = [&](int p, int col) { return c.tb ? (size_t)col + (size_t)p * ldb : (size_t)p + (size_t)col * ldb; };
    for (size_t t = 0; t < bs.size(); ++t) {
      put(c.in, &A[Aat((int)t, a) * fb(c.in)], big);
      put(c.in, &A[Aat((int)t, bs[t]) * fb(c.in)], big);
      put(c.in, &B[Bat(a, (int)t) * fb(c.in)], big);
      put(c.in, &B[Bat(bs[t], (int)t) * fb(c.in)], -big);
    }
    cudaMemcpy(dA, A.data(), A.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size(), cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size(), cudaMemcpyHostToDevice);
    cublasSetMathMode(h, c.tf32 ? CUBLAS_TF32_TENSOR_OP_MATH : CUBLAS_DEFAULT_MATH);
    const cublasStatus_t st = cublasGemmEx(h, c.ta ? CUBLAS_OP_T : CUBLAS_OP_N, c.tb ? CUBLAS_OP_T : CUBLAS_OP_N, c.m, c.n, c.k, &c.alpha,
                                           dA, dt(c.in), lda, dB, dt(c.in), ldb, &c.beta, dC, dt(c.out), ldc,
                                           c.tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    cudaDeviceSynchronize();
    if (st != CUBLAS_STATUS_SUCCESS) { std::fprintf(stderr, "cublas status %d\n", (int)st); return 1; }
    cudaMemcpy(C.data(), dC, C.size(), cudaMemcpyDeviceToHost);
    for (size_t t = 0; t < bs.size(); ++t) {
      const float v = get(c.out, &C[((size_t)t + (size_t)t * ldc) * fb(c.out)]) / (c.alpha * kOne);
      const long r = std::lround(v);
      std::printf("%d %d %ld\n", a, bs[t], std::fabs(v - (float)r) < 1e-3f ? r : -1L);
    }
  }
  cublasDestroy(h);
  return 0;
}
