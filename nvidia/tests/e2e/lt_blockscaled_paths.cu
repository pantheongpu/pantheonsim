// cuBLASLt's block-scaled matmuls: MXFP8 (E4M3 with a UE8M0 scale per 32
// elements of K), NVFP4 (E2M1 with a UE4M3 scale per 16), the Hopper-style
// 128-element and 128x128 FP32 block scales, and D's block quantization with
// the output scales it computes -- all as NVIDIA's cuBLAS documentation
// defines them ("Narrow Precision Data Types Usage": the scaling modes, the
// 1D block scaling factors' tiled layout, 1D block quantization).
//
// Documentation-derived, not card-verified: these modes need an sm_100 (or,
// for the 128-element ones, sm_90) GPU, and the only card here is an RTX 3060
// (sm_86). So this runs on VirtualGPU (VGPU_GPU set, as run_lib_check.sh sets
// it) or on a card of compute capability 9.0 or later, and anywhere else
// prints SKIP. The reference below is computed from the documentation's
// formulas, independently of the library; the values are chosen so every
// product, scale and sum is exact.
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static int failures = 0;
static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}

// Named by value: CUDA 12.0's headers, which CI builds with, have none of these.
constexpr int kAScaleMode = 31, kBScaleMode = 32, kDScaleMode = 34, kDOutScalePointer = 36, kDOutScaleMode = 37;
constexpr int32_t kVec16UE4M3 = 1, kVec32UE8M0 = 2, kVec128 = 4, kBlk128 = 5;
constexpr cudaDataType kFP4 = (cudaDataType)33, kE4M3 = (cudaDataType)28;

// E4M3 and E2M1 by the formats' definitions.
static double e4m3(uint8_t b) {
  const int e = (b >> 3) & 15, m = b & 7;
  double v = e == 0 ? std::ldexp(m, -9) : (e == 15 && m == 7) ? NAN : std::ldexp(8 + m, e - 10);
  return (b & 0x80) ? -v : v;
}
static uint8_t to_e4m3(double v) {   // exact values only (the test's are)
  for (int b = 0; b < 0x7f; ++b)
    if (e4m3((uint8_t)b) == std::fabs(v)) return (uint8_t)(b | (v < 0 ? 0x80 : 0));
  return 0x7e;
}
static const double kE2M1[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
static double e2m1(uint8_t nib) { return (nib & 8) ? -kE2M1[nib & 7] : kE2M1[nib & 7]; }
static uint8_t to_e2m1(double v) {
  for (int i = 0; i < 8; ++i)
    if (kE2M1[i] == std::fabs(v)) return (uint8_t)(i | (v < 0 ? 8 : 0));
  return 7;
}
// The documentation's tile layout: (outer, inner block) to a byte offset, with
// `blocks` inner blocks per outer index.
static size_t tiled(size_t o, size_t b, size_t blocks) {
  const size_t dim = (blocks + 3) / 4 * 4;
  return ((b / 4) * 4 + (o / 128) * dim) * 128 + (o % 32) * 16 + (o % 128 / 32) * 4 + b % 4;
}
static size_t tiled_size(size_t outer, size_t blocks) { return (outer + 127) / 128 * 128 * ((blocks + 3) / 4 * 4); }

template <class T> static T* up(const std::vector<T>& h) {
  T* d = nullptr;
  cudaMalloc(&d, h.size() * sizeof(T) + 256);
  cudaMemset(d, 0, h.size() * sizeof(T) + 256);
  cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
  return d;
}
template <class T> static std::vector<T> down(const void* d, size_t n) {
  std::vector<T> h(n);
  cudaMemcpy(h.data(), d, n * sizeof(T), cudaMemcpyDeviceToHost);
  return h;
}

static cublasLtHandle_t lt;

struct Desc {
  cublasLtMatmulDesc_t d;
  Desc() {
    cublasLtMatmulDescCreate(&d, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    const cublasOperation_t t = CUBLAS_OP_T;   // A stored K x M, op(A) = A^T: K contiguous, as these modes want
    cublasLtMatmulDescSetAttribute(d, CUBLASLT_MATMUL_DESC_TRANSA, &t, sizeof t);
  }
  ~Desc() { cublasLtMatmulDescDestroy(d); }
  template <class V> void set(int attr, V v) { cublasLtMatmulDescSetAttribute(d, (cublasLtMatmulDescAttributes_t)attr, &v, sizeof v); }
};

// D (m x n) = op(A) B with A stored k x m and B k x n, both K-contiguous.
static int matmul(Desc& desc, cudaDataType ab, cudaDataType dt, int m, int n, int k, const void* A, const void* B,
                  void* D, int* heur = nullptr) {
  cublasLtMatrixLayout_t la, lb, ld;
  cublasLtMatrixLayoutCreate(&la, ab, k, m, k);
  cublasLtMatrixLayoutCreate(&lb, ab, k, n, k);
  cublasLtMatrixLayoutCreate(&ld, dt, m, n, m);
  const float one = 1, zero = 0;
  cublasLtMatmulPreference_t pref;
  cublasLtMatmulPreferenceCreate(&pref);
  cublasLtMatmulHeuristicResult_t hr;
  int got = 0;
  const int hs = (int)cublasLtMatmulAlgoGetHeuristic(lt, desc.d, la, lb, ld, ld, pref, 1, &hr, &got);
  if (heur) *heur = hs;
  const int st = (int)cublasLtMatmul(lt, desc.d, &one, A, la, B, lb, &zero, D, ld, D, ld, nullptr, nullptr, 0, 0);
  cudaDeviceSynchronize();
  cublasLtMatmulPreferenceDestroy(pref);
  for (auto l : {la, lb, ld}) cublasLtMatrixLayoutDestroy(l);
  return st;
}

// MXFP8: E4M3 operands, a power-of-two scale per 32 elements of K.
static void mxfp8() {
  const int m = 8, n = 6, k = 64, blocks = 2;
  std::vector<uint8_t> A((size_t)k * m), B((size_t)k * n);
  std::vector<double> a(A.size()), b(B.size());
  for (size_t i = 0; i < A.size(); ++i) a[i] = (double)((int)(i * 7 % 9) - 4) / 2, A[i] = to_e4m3(a[i]);
  for (size_t i = 0; i < B.size(); ++i) b[i] = (double)((int)(i * 5 % 7) - 3), B[i] = to_e4m3(b[i]);
  std::vector<uint8_t> sa(tiled_size(m, blocks), 127), sb(tiled_size(n, blocks), 127);
  auto pa = [&](int i, int blk) { return 127 + (i + blk) % 3 - 1; };   // 2^-1, 2^0 or 2^1
  auto pb = [&](int j, int blk) { return 127 + (j * 2 + blk) % 4 - 2; };
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < blocks; ++q) sa[tiled(i, q, blocks)] = (uint8_t)pa(i, q);
  for (int j = 0; j < n; ++j)
    for (int q = 0; q < blocks; ++q) sb[tiled(j, q, blocks)] = (uint8_t)pb(j, q);
  uint8_t *dA = up(A), *dB = up(B), *dsa = up(sa), *dsb = up(sb);
  float* dD = up(std::vector<float>((size_t)m * n));
  Desc desc;
  desc.set(kAScaleMode, kVec32UE8M0);
  desc.set(kBScaleMode, kVec32UE8M0);
  desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa);
  desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsb);
  const int st = matmul(desc, kE4M3, CUDA_R_32F, m, n, k, dA, dB, dD);
  const auto D = down<float>(dD, (size_t)m * n);
  bool ok = st == 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int p = 0; p < k; ++p)
        s += a[(size_t)i * k + p] * std::ldexp(1.0, pa(i, p / 32) - 127) * b[(size_t)j * k + p] *
             std::ldexp(1.0, pb(j, p / 32) - 127);
      ok = ok && D[(size_t)j * m + i] == (float)s;
    }
  check(ok, "MXFP8: E4M3 A and B, a UE8M0 scale per 32 of K in the tiled layout, fp32 D");

  // D in E4M3, quantized by blocks of 32 down each column, its UE8M0 scales
  // written to D_OUT_SCALE_POINTER: S = amax / 448, rounded up to a power of
  // two, each value multiplied by 1/S.
  const int md = 40;   // two blocks down a column, the second partial
  std::vector<uint8_t> A2((size_t)k * md);
  std::vector<double> a2(A2.size());
  for (size_t i = 0; i < A2.size(); ++i) a2[i] = (double)((int)(i * 11 % 13) - 6) * 4, A2[i] = to_e4m3(a2[i]);
  uint8_t* dA2 = up(A2);
  std::vector<uint8_t> sa2(tiled_size(md, blocks), 127);
  uint8_t* dsa2 = up(sa2);
  uint8_t* dD8 = up(std::vector<uint8_t>((size_t)md * n));
  const size_t dblocks = 2;
  std::vector<uint8_t> so(tiled_size(n, dblocks), 0xEE);
  uint8_t* dso = up(so);
  Desc q;
  q.set(kAScaleMode, kVec32UE8M0);
  q.set(kBScaleMode, kVec32UE8M0);
  q.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa2);
  q.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsb);
  q.set(kDOutScaleMode, kVec32UE8M0);
  q.set(kDOutScalePointer, (void*)dso);
  const int st2 = matmul(q, kE4M3, kE4M3, md, n, k, dA2, dB, dD8);
  const auto D8 = down<uint8_t>(dD8, (size_t)md * n);
  so = down<uint8_t>(dso, so.size());
  ok = st2 == 0;
  for (int j = 0; j < n; ++j)
    for (size_t blk = 0; blk < dblocks; ++blk) {
      std::vector<double> v;
      for (int i = (int)blk * 32; i < std::min(md, (int)blk * 32 + 32); ++i) {
        double s = 0;
        for (int p = 0; p < k; ++p) s += a2[(size_t)i * k + p] * b[(size_t)j * k + p] * std::ldexp(1.0, pb(j, p / 32) - 127);
        v.push_back(s);
      }
      double amax = 0;
      for (double x : v) amax = std::fmax(amax, std::fabs(x));
      const float s32 = (float)(amax / 448.0);
      int e;
      const float frac = std::frexp(s32, &e);   // s32 = frac 2^e, frac in [0.5, 1)
      const int expo = frac == 0.5f ? e - 1 : e;   // the power of two at or above s32
      const uint8_t code = (uint8_t)(expo + 127);
      ok = ok && so[tiled(j, blk, dblocks)] == code;
      for (size_t r = 0; r < v.size(); ++r) {
        const double want = e4m3(D8[(size_t)j * md + blk * 32 + r]), exact = v[r] * std::ldexp(1.0, -expo);
        ok = ok && std::fabs(want - exact) <= std::fabs(exact) / 16 + 1e-12;   // E4M3 rounding: 3 mantissa bits
      }
    }
  check(ok, "MXFP8 D: block amax over 448 rounded up to a UE8M0 scale, written to the D_OUT scale tensor");
  for (void* p : {(void*)dA, (void*)dB, (void*)dsa, (void*)dsb, (void*)dD, (void*)dA2, (void*)dsa2, (void*)dD8, (void*)dso})
    cudaFree(p);
}

// NVFP4: E2M1 operands two to a byte, a UE4M3 scale per 16 elements of K;
// and an FP4 D quantized with D's input scale and UE4M3 block scales.
static void nvfp4() {
  const int m = 4, n = 3, k = 32, blocks = 2;
  auto pack = [](const std::vector<double>& v) {
    std::vector<uint8_t> p((v.size() + 1) / 2, 0);
    for (size_t i = 0; i < v.size(); ++i) p[i / 2] |= (uint8_t)(to_e2m1(v[i]) << (4 * (i & 1)));
    return p;
  };
  std::vector<double> a((size_t)k * m), b((size_t)k * n);
  for (size_t i = 0; i < a.size(); ++i) a[i] = kE2M1[i * 3 % 8] * ((i % 5) == 2 ? -1 : 1);
  for (size_t i = 0; i < b.size(); ++i) b[i] = kE2M1[i * 5 % 7] * ((i % 3) == 1 ? -1 : 1);
  const double ue4m3[3] = {0.5, 1.5, 2};   // exact in E4M3
  std::vector<uint8_t> sa(tiled_size(m, blocks), 0), sb(tiled_size(n, blocks), 0);
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < blocks; ++q) sa[tiled(i, q, blocks)] = to_e4m3(ue4m3[(i + q) % 3]);
  for (int j = 0; j < n; ++j)
    for (int q = 0; q < blocks; ++q) sb[tiled(j, q, blocks)] = to_e4m3(ue4m3[(j + 2 * q) % 3]);
  uint8_t *dA = up(pack(a)), *dB = up(pack(b)), *dsa = up(sa), *dsb = up(sb);
  float* dD = up(std::vector<float>((size_t)m * n));
  Desc desc;
  desc.set(kAScaleMode, kVec16UE4M3);
  desc.set(kBScaleMode, kVec16UE4M3);
  desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa);
  desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsb);
  const int st = matmul(desc, kFP4, CUDA_R_32F, m, n, k, dA, dB, dD);
  const auto D = down<float>(dD, (size_t)m * n);
  std::vector<double> ref((size_t)m * n);
  bool ok = st == 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int p = 0; p < k; ++p)
        s += a[(size_t)i * k + p] * ue4m3[(i + p / 16) % 3] * b[(size_t)j * k + p] * ue4m3[(j + 2 * (p / 16)) % 3];
      ref[(size_t)j * m + i] = s;
      ok = ok && D[(size_t)j * m + i] == (float)s;
    }
  check(ok, "NVFP4: E2M1 A and B packed two to a byte, a UE4M3 scale per 16 of K, fp32 D");

  // FP4 D: values times D's input scale, a UE4M3 scale per 16 down a
  // column of e4m3(amax / 6), each value times its reciprocal in E2M1.
  uint8_t* dD4 = up(std::vector<uint8_t>((size_t)(m * n + 1) / 2));
  std::vector<uint8_t> so(tiled_size(n, 1), 0xEE);
  uint8_t* dso = up(so);
  float* din = up(std::vector<float>{0.25f});
  Desc q;
  q.set(kAScaleMode, kVec16UE4M3);
  q.set(kBScaleMode, kVec16UE4M3);
  q.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa);
  q.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsb);
  q.set(CUBLASLT_MATMUL_DESC_D_SCALE_POINTER, (const void*)din);
  q.set(kDOutScaleMode, kVec16UE4M3);
  q.set(kDOutScalePointer, (void*)dso);
  const int st2 = matmul(q, kFP4, kFP4, m, n, k, dA, dB, dD4);
  const auto D4 = down<uint8_t>(dD4, (size_t)(m * n + 1) / 2);
  so = down<uint8_t>(dso, so.size());
  ok = st2 == 0;
  for (int j = 0; j < n; ++j) {
    double amax = 0;
    for (int i = 0; i < m; ++i) amax = std::fmax(amax, std::fabs(0.25 * ref[(size_t)j * m + i]));
    const uint8_t code = so[tiled(j, 0, 1)];
    const double s = e4m3(code);
    ok = ok && (code & 0x80) == 0 && std::fabs(s - amax / 6) <= amax / 6 / 16;   // e4m3(amax / 6)
    for (int i = 0; i < m; ++i) {
      const size_t e = (size_t)j * m + i;
      const double got = e2m1((D4[e / 2] >> (4 * (e & 1))) & 15), exact = 0.25 * ref[e] / s;
      // E2M1 is coarse: the nearest of its values (ties to even).
      double best = 0;
      for (int c = 0; c < 16; ++c)
        if (std::fabs(e2m1((uint8_t)c) - exact) < std::fabs(best - exact)) best = e2m1((uint8_t)c);
      ok = ok && std::fabs(got - exact) <= std::fabs(best - exact) + 1e-12;
    }
  }
  check(ok, "NVFP4 D: D's input scale, then e4m3(amax / 6) per 16, the values rounded to E2M1");
  for (void* p : {(void*)dA, (void*)dB, (void*)dsa, (void*)dsb, (void*)dD, (void*)dD4, (void*)dso, (void*)din}) cudaFree(p);
}

// The Hopper forms: FP32 scales per 128 elements of K (M- or N-major), or per
// 128 x 128 block (K-major, the block columns padded to 4).
static void hopper() {
  const int m = 8, n = 4, k = 256, L = 2;
  std::vector<uint8_t> A((size_t)k * m), B((size_t)k * n);
  std::vector<double> a(A.size()), b(B.size());
  for (size_t i = 0; i < A.size(); ++i) a[i] = (double)((int)(i % 5) - 2), A[i] = to_e4m3(a[i]);
  for (size_t i = 0; i < B.size(); ++i) b[i] = (double)((int)(i % 3) - 1), B[i] = to_e4m3(b[i]);
  std::vector<float> sa((size_t)m * L), sb((size_t)4 * 1);   // A: VEC128, M x L M-major; B: BLK128x128, L4 x 1
  for (int i = 0; i < m; ++i)
    for (int q = 0; q < L; ++q) sa[(size_t)i + (size_t)q * m] = (float)(0.25 * (1 + (i + q) % 4));
  sb = {2.0f, -0.5f, 99.0f, 99.0f};
  uint8_t *dA = up(A), *dB = up(B);
  float *dsa = up(sa), *dsb = up(sb), *dD = up(std::vector<float>((size_t)m * n));
  Desc desc;
  desc.set(kAScaleMode, kVec128);
  desc.set(kBScaleMode, kBlk128);
  desc.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa);
  desc.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsb);
  const int st = matmul(desc, kE4M3, CUDA_R_32F, m, n, k, dA, dB, dD);
  const auto D = down<float>(dD, (size_t)m * n);
  bool ok = st == 0;
  for (int j = 0; j < n; ++j)
    for (int i = 0; i < m; ++i) {
      double s = 0;
      for (int p = 0; p < k; ++p) s += a[(size_t)i * k + p] * sa[(size_t)i + (size_t)(p / 128) * m] * b[(size_t)j * k + p] * sb[p / 128];
      ok = ok && D[(size_t)j * m + i] == (float)s;
    }
  check(ok, "FP8 with VEC128_32F A scales and BLK128x128_32F B scales");

  // What these modes refuse.
  int heur = 0;
  Desc both;
  both.set(kAScaleMode, kBlk128);
  both.set(kBScaleMode, kBlk128);
  both.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsb);
  both.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsb);
  check(matmul(both, kE4M3, CUDA_R_32F, m, n, k, dA, dB, dD, &heur) == CUBLAS_STATUS_NOT_SUPPORTED && heur == 15,
        "BLK128x128 for both A and B is not supported");
  check(matmul(desc, kE4M3, CUDA_R_32F, m, 3, k, dA, dB, dD) == CUBLAS_STATUS_INVALID_VALUE,
        "the 128-element modes need M and N multiples of 4");
  Desc mixed;
  mixed.set(kAScaleMode, kVec32UE8M0);
  mixed.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa);
  check(matmul(mixed, kE4M3, CUDA_R_32F, m, n, k, dA, dB, dD) == CUBLAS_STATUS_NOT_SUPPORTED,
        "a block scale for A without one for B is not supported");
  Desc wrong;
  wrong.set(kAScaleMode, kVec16UE4M3);
  wrong.set(kBScaleMode, kVec16UE4M3);
  wrong.set(CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, (const void*)dsa);
  wrong.set(CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, (const void*)dsa);
  check(matmul(wrong, kE4M3, CUDA_R_32F, m, n, k, dA, dB, dD) == CUBLAS_STATUS_NOT_SUPPORTED,
        "VEC16_UE4M3 scales go with FP4, not FP8");
  Desc none;
  check(matmul(none, kFP4, CUDA_R_32F, m, n, 32, dA, dB, dD) == CUBLAS_STATUS_NOT_SUPPORTED,
        "FP4 operands without block scales are not supported");
  Desc noptr;
  noptr.set(kAScaleMode, kVec32UE8M0);
  noptr.set(kBScaleMode, kVec32UE8M0);
  check(matmul(noptr, kE4M3, CUDA_R_32F, m, n, 64, dA, dB, dD) == CUBLAS_STATUS_INVALID_VALUE,
        "a block-scaled matmul without its scale tensors is refused");
  Desc dmode;
  dmode.set(kDScaleMode, kVec32UE8M0);
  check(matmul(dmode, kE4M3, kE4M3, m, n, 64, dA, dB, dD) == CUBLAS_STATUS_NOT_SUPPORTED,
        "D's block scales are set through D_OUT_SCALE_MODE, not D_SCALE_MODE");
  for (void* p : {(void*)dA, (void*)dB, (void*)dsa, (void*)dsb, (void*)dD}) cudaFree(p);
}

int main() {
  std::setvbuf(stdout, nullptr, _IOLBF, 0);
  cudaDeviceProp prop{};
  cudaGetDeviceProperties(&prop, 0);
  if (!std::getenv("VGPU_GPU") && prop.major < 9) {
    std::printf("SKIP: block-scaled matmuls need sm_90 or sm_100 (this is sm_%d%d) or VirtualGPU\nPASS (skipped)\n",
                prop.major, prop.minor);
    return 0;
  }
  if (cublasLtCreate(&lt)) {
    std::printf("FAIL: cublasLtCreate\n");
    return 1;
  }
  mxfp8();
  nvfp4();
  hopper();
  cublasLtDestroy(lt);
  std::printf(failures ? "FAIL: %d block-scaled cuBLASLt checks\n" : "PASS: every block-scaled cuBLASLt check\n",
              failures);
  return failures ? 1 : 0;
}
