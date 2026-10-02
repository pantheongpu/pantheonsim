// cuSPARSELt: descriptors and their checks, attributes, algorithm selection,
// STRIP and TILE pruning, the prune check, compression (sizes, the kept
// values and the metadata), and Matmul in fp16, bf16, tf32 and int8 (into
// int8, int32, fp16 and bf16) with transposes, both orders, a structured B,
// batches, bias, ReLU with its bounds, GELU, alpha-vector scaling, the
// search, argument checks and graph capture.
//
// The expectations are NVIDIA's: this program also runs against NVIDIA's
// libcusparseLt 0.10 on an RTX 3060, and every status, default, size, pruned
// pattern and rounding below is what that library answered. The pruning and
// metadata rules are written out here from those measurements and checked
// against whichever library is loaded.
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <vector>

#include "../../include/vgpu_cusparselt.h"

static int failures = 0;

static void check(bool ok, const char* what) {
  std::printf("%-4s %s\n", ok ? "ok" : "FAIL", what);
  if (!ok) ++failures;
}
#define IS(call, want) check((int)(call) == (int)(want), #call " -> " #want)

static const cusparseOperation_t N = CUSPARSE_OPERATION_NON_TRANSPOSE, T = CUSPARSE_OPERATION_TRANSPOSE;
static const cusparseOrder_t ROW = CUSPARSE_ORDER_ROW, COL = CUSPARSE_ORDER_COL;
static const cusparseLtSparsity_t S50 = CUSPARSELT_SPARSITY_50_PERCENT;
static cusparseLtHandle_t h;

// ---- host conversions ----

static float h2f(uint16_t x) {
  const uint32_t sign = (uint32_t)(x & 0x8000) << 16;
  int e = (x >> 10) & 0x1f;
  uint32_t m = x & 0x3ff, bits;
  if (e == 0x1f) bits = sign | 0x7f800000u | (m << 13);
  else if (e == 0) {
    if (!m) bits = sign;
    else {
      e = 1;
      while (!(m & 0x400)) { m <<= 1; --e; }
      bits = sign | (uint32_t)(e + 112) << 23 | ((m & 0x3ff) << 13);
    }
  } else bits = sign | (uint32_t)(e + 112) << 23 | (m << 13);
  float f;
  std::memcpy(&f, &bits, 4);
  return f;
}
static uint16_t f2h(float f) {  // round to nearest even, normal range only
  uint32_t x;
  std::memcpy(&x, &f, 4);
  const uint32_t sign = (x >> 16) & 0x8000;
  if ((x & 0x7fffffff) == 0) return (uint16_t)sign;
  const int e = (int)((x >> 23) & 0xff) - 112;
  if (e >= 31) return (uint16_t)(sign | 0x7c00);
  if (e <= 0) {
    uint32_t m = (x & 0x7fffff) | 0x800000;
    const int shift = 14 - e;
    if (shift > 24) return (uint16_t)sign;
    uint32_t hv = m >> shift;
    const uint32_t rem = m & ((1u << shift) - 1), half = 1u << (shift - 1);
    if (rem > half || (rem == half && (hv & 1))) ++hv;
    return (uint16_t)(sign | hv);
  }
  uint32_t hv = ((uint32_t)e << 10) | ((x & 0x7fffff) >> 13);
  const uint32_t rem = x & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (hv & 1))) ++hv;
  return (uint16_t)(sign | hv);
}
static float b2f(uint16_t b) {
  const uint32_t x = (uint32_t)b << 16;
  float f;
  std::memcpy(&f, &x, 4);
  return f;
}
static uint16_t f2b(float f) {
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x += 0x7fff + ((x >> 16) & 1);
  return (uint16_t)(x >> 16);
}
static float tf32(float f) {  // the card's tf32: half a unit added, low 13 bits dropped
  uint32_t x;
  std::memcpy(&x, &f, 4);
  x = (x + 0x1000u) & 0xffffe000u;
  std::memcpy(&f, &x, 4);
  return f;
}

// ---- device memory ----

template <class V>
static void* upload(const V& v) {
  void* p = nullptr;
  cudaMalloc(&p, v.size() * sizeof(v[0]) + 256);
  cudaMemcpy(p, v.data(), v.size() * sizeof(v[0]), cudaMemcpyHostToDevice);
  return p;
}
static void* zeros(size_t bytes) {
  void* p = nullptr;
  cudaMalloc(&p, bytes + 256);
  cudaMemset(p, 0, bytes + 256);
  return p;
}
template <class E>
static std::vector<E> download(const void* p, size_t n) {
  std::vector<E> v(n);
  cudaDeviceSynchronize();
  cudaMemcpy(v.data(), p, n * sizeof(E), cudaMemcpyDeviceToHost);
  return v;
}

// ---- the pruning rules, as measured ----

// STRIP: the larger magnitudes of each group, the lower position on a tie.
static unsigned strip_mask(const float* g, int n) {
  int idx[4] = {0, 1, 2, 3};
  std::stable_sort(idx, idx + n, [&](int a, int b) { return std::fabs(g[a]) > std::fabs(g[b]); });
  unsigned keep = 0;
  for (int j = 0; j < n / 2; ++j) keep |= 1u << idx[j];
  return keep;
}

// TILE: the 2-per-row-and-column pattern of largest L1 norm, the first of
// this order on a tie (rows of the tile are memory lines).
static const uint16_t kTileOrder[90] = {
    0xc3c3, 0xa5c3, 0xc3a5, 0xa5a5, 0x69c3, 0x96c3, 0xc369, 0xc396, 0x69a5, 0x96a5,
    0xa569, 0xa596, 0x5ac3, 0xc35a, 0x5aa5, 0xa55a, 0x6969, 0x9669, 0x6996, 0x9696,
    0x3cc3, 0x5a69, 0x5a96, 0x695a, 0x965a, 0xc33c, 0x3ca5, 0xa53c, 0x5a5a, 0x3c69,
    0x3c96, 0x693c, 0x963c, 0x3c5a, 0x5a3c, 0x3c3c, 0xcc33, 0xaa55, 0x6699, 0x9966,
    0xac35, 0xac53, 0xca35, 0x55aa, 0xca53, 0x6c39, 0x9c36, 0x6c93, 0x9c63, 0xc639,
    0xc936, 0x33cc, 0xc693, 0xc963, 0x5c3a, 0x5ca3, 0xc53a, 0x6a59, 0x9a56, 0xc5a3,
    0x6a95, 0x9a65, 0xa659, 0xa956, 0xa695, 0xa965, 0x3a5c, 0x3ac5, 0xa35c, 0x569a,
    0x596a, 0xa3c5, 0x56a9, 0x59a6, 0x659a, 0x956a, 0x65a9, 0x95a6, 0x369c, 0x396c,
    0x36c9, 0x39c6, 0x639c, 0x936c, 0x63c9, 0x93c6, 0x35ac, 0x35ca, 0x53ac, 0x53ca,
};
static unsigned tile_mask(const float t[4][4]) {
  int best = 0;
  float bs = -1.f;
  for (int p = 0; p < 90; ++p) {
    float s = 0.f;
    for (int i = 0; i < 4; ++i)
      for (int j = 0; j < 4; ++j)
        if (kTileOrder[p] >> (4 * i + j) & 1) s += std::fabs(t[i][j]);
    if (s > bs) { bs = s; best = p; }
  }
  return kTileOrder[best];
}

// The 16-bit metadata layout (64-row x 32-K blocks), as measured.
static size_t nibble16(int64_t r, int64_t g, int64_t nk) {
  const int64_t rows32 = (nk + 31) / 32 * 32;
  const int64_t wpk = (rows32 / 64) * 128 + (rows32 % 64) * 2;
  const int64_t kb = g / 8, kc = (g % 8) / 4, q = g % 4, rt = r / 64, rr = r % 64;
  const int64_t w = ((rr >> 3) & 1) | (kc << 1) | (((rr >> 4) & 1) << 2) | ((rr & 7) << 3) | (((rr >> 5) & 1) << 6);
  return (size_t)((kb * wpk + rt * 128 + w) * 4 + q);
}
static size_t nibble8(int64_t r, int64_t g, int64_t nk) {
  const int64_t rtiles = (nk + 63) / 64;
  const int64_t kt = g / 16, kc = (g % 16) / 4, q = g % 4, rt = r / 64, rr = r % 64;
  const int64_t w = (kc & 1) | ((rr & 1) << 1) | ((kc >> 1) << 2) | (((rr >> 3) & 7) << 3) | (((rr >> 1) & 3) << 6);
  return (size_t)(((kt * rtiles + rt) * 256 + w) * 4 + q);
}

// ---- a small harness around one product ----

struct Problem {
  cudaDataType ab, cd;
  cusparseComputeType ct;
  int64_t m, n, k;
  cusparseOperation_t opA = N, opB = N;
  cusparseOrder_t oA = ROW, oB = ROW, oC = ROW;
  bool sparseB = false;
  int batches = 1;
  bool broadcastSparse = false;  // the structured operand's batch stride is 0
};

struct Setup {
  Problem p;
  cusparseLtMatDescriptor_t A, B, C;
  cusparseLtMatmulDescriptor_t md;
  cusparseLtMatmulAlgSelection_t alg;
  cusparseLtMatmulPlan_t plan;
  int sA = -1, sB = -1, sC = -1, smd = -1;
  int64_t ar, ac, lda, br, bc, ldb, ldc;
  explicit Setup(const Problem& pr) : p(pr) {
    ar = p.opA == N ? p.m : p.k; ac = p.opA == N ? p.k : p.m;
    br = p.opB == N ? p.k : p.n; bc = p.opB == N ? p.n : p.k;
    lda = p.oA == ROW ? ac : ar; ldb = p.oB == ROW ? bc : br; ldc = p.oC == ROW ? p.n : p.m;
    sA = p.sparseB ? cusparseLtDenseDescriptorInit(&h, &A, ar, ac, lda, 16, p.ab, p.oA)
                   : cusparseLtStructuredDescriptorInit(&h, &A, ar, ac, lda, 16, p.ab, p.oA, S50);
    sB = p.sparseB ? cusparseLtStructuredDescriptorInit(&h, &B, br, bc, ldb, 16, p.ab, p.oB, S50)
                   : cusparseLtDenseDescriptorInit(&h, &B, br, bc, ldb, 16, p.ab, p.oB);
    sC = cusparseLtDenseDescriptorInit(&h, &C, p.m, p.n, ldc, 16, p.cd, p.oC);
    if (p.batches > 1)
      for (cusparseLtMatDescriptor_t* d : {&A, &B, &C}) cusparseLtMatDescSetAttribute(&h, d, CUSPARSELT_MAT_NUM_BATCHES, &p.batches, 4);
    if (p.broadcastSparse) {
      int64_t z = 0;
      cusparseLtMatDescSetAttribute(&h, p.sparseB ? &B : &A, CUSPARSELT_MAT_BATCH_STRIDE, &z, 8);
    }
    smd = cusparseLtMatmulDescriptorInit(&h, &md, p.opA, p.opB, &A, &B, &C, &C, p.ct);
  }
  int planit() {
    int s = cusparseLtMatmulAlgSelectionInit(&h, &alg, &md, CUSPARSELT_MATMUL_ALG_DEFAULT);
    return s ? s : cusparseLtMatmulPlanInit(&h, &plan, &md, &alg);
  }
  int64_t sizeA() const { return (p.oA == ROW ? ar : ac) * lda; }
  int64_t sizeB() const { return (p.oB == ROW ? br : bc) * ldb; }
  int64_t sizeC() const { return (p.oC == ROW ? p.m : p.n) * ldc; }
  int64_t at(int64_t r, int64_t c, int64_t ld, cusparseOrder_t o) const { return o == ROW ? r * ld + c : c * ld + r; }
  // op(A)(i,kk) and op(B)(kk,j) offsets within one batch
  int64_t offA(int64_t i, int64_t kk) const { return p.opA == N ? at(i, kk, lda, p.oA) : at(kk, i, lda, p.oA); }
  int64_t offB(int64_t kk, int64_t j) const { return p.opB == N ? at(kk, j, ldb, p.oB) : at(j, kk, ldb, p.oB); }
  int64_t offC(int64_t i, int64_t j) const { return at(i, j, ldc, p.oC); }
};

static size_t esize(cudaDataType t) {
  return t == CUDA_R_32F || t == CUDA_R_32I ? 4 : (t == CUDA_R_8I ? 1 : 2);
}
static float getv(const std::vector<uint8_t>& raw, int64_t off, cudaDataType t) {
  const uint8_t* q = raw.data() + off * esize(t);
  switch (t) {
    case CUDA_R_32F: { float f; std::memcpy(&f, q, 4); return f; }
    case CUDA_R_32I: { int32_t i; std::memcpy(&i, q, 4); return (float)i; }
    case CUDA_R_16F: { uint16_t v; std::memcpy(&v, q, 2); return h2f(v); }
    case CUDA_R_16BF: { uint16_t v; std::memcpy(&v, q, 2); return b2f(v); }
    default: return (float)(int8_t)*q;
  }
}
static void setv(std::vector<uint8_t>& raw, int64_t off, cudaDataType t, float v) {
  uint8_t* q = raw.data() + off * esize(t);
  switch (t) {
    case CUDA_R_32F: std::memcpy(q, &v, 4); break;
    case CUDA_R_32I: { int32_t i = (int32_t)v; std::memcpy(q, &i, 4); break; }
    case CUDA_R_16F: { uint16_t x = f2h(v); std::memcpy(q, &x, 2); break; }
    case CUDA_R_16BF: { uint16_t x = f2b(v); std::memcpy(q, &x, 2); break; }
    default: *q = (uint8_t)(int8_t)v; break;
  }
}
// The output rounding the card applies (round to nearest even; integers saturate).
static float round_out(float v, cudaDataType t) {
  switch (t) {
    case CUDA_R_16F: return h2f(f2h(v));
    case CUDA_R_16BF: return b2f(f2b(v));
    case CUDA_R_8I: return std::min(127.f, std::max(-128.f, std::nearbyint(v)));
    case CUDA_R_32I: return (float)std::nearbyint((double)v);
    default: return v;
  }
}

struct Epilogue {
  float alpha = 1.f, beta = 0.f;
  bool relu = false, gelu = false, alphaVec = false, betaVec = false;
  float ub = 3.40282347e38f, th = 0.f, geluScale = 1.f;
  std::vector<float> bias, av, bv;  // per row of D
  int64_t biasStride = 0;           // set only with batches
};

static float gelu_ref(float x) {
  return 0.5f * x * (1.f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

// Runs one product end to end (prune by STRIP, compress, multiply) and
// compares D with the host reference, within `tol` (in units of D's own
// rounding for floats, absolute for integers). Returns the mismatch count.
static int run_product(const char* name, const Problem& pr, const std::function<float(int64_t, int64_t, int64_t)>& fa,
                       const std::function<float(int64_t, int64_t, int64_t)>& fb, Epilogue ep, float tol = 0.f,
                       bool search = false) {
  Setup s(pr);
  char what[200];
  std::snprintf(what, sizeof what, "%s: descriptors", name);
  check(s.sA == 0 && s.sB == 0 && s.sC == 0 && s.smd == 0, what);
  if (s.smd) return 1;
  int one = 1;
  int st = 0;
  if (ep.relu) {
    st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_ACTIVATION_RELU, &one, 4);
    st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &ep.ub, 4);
    st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_ACTIVATION_RELU_THRESHOLD, &ep.th, 4);
  }
  if (ep.gelu) {
    st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_ACTIVATION_GELU, &one, 4);
    st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &ep.geluScale, 4);
  }
  if (ep.alphaVec) st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &one, 4);
  if (ep.betaVec) st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_BETA_VECTOR_SCALING, &one, 4);
  const int64_t m = pr.m, n = pr.n, k = pr.k, nb = pr.batches;
  // bias: float for int8 inputs, D's type otherwise (measured)
  const cudaDataType biasT = pr.ab == CUDA_R_8I ? CUDA_R_32F : pr.cd;
  void* dbias = nullptr;
  if (!ep.bias.empty()) {
    std::vector<uint8_t> raw((size_t)(m * nb) * esize(biasT));
    for (int64_t i = 0; i < m * nb; ++i) setv(raw, i, biasT, ep.bias[(size_t)i]);
    dbias = upload(raw);
    st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_BIAS_POINTER, &dbias, sizeof dbias);
    if (nb > 1) { int64_t bsd = m; st |= cusparseLtMatmulDescSetAttribute(&h, &s.md, CUSPARSELT_MATMUL_BIAS_STRIDE, &bsd, 8); }
  }
  std::snprintf(what, sizeof what, "%s: attributes and plan", name);
  st |= s.planit();
  check(st == 0, what);
  if (st) return 1;
  // Operands: every batch of A and B (a broadcast operand holds one).
  const int64_t nbA = (!pr.sparseB && pr.broadcastSparse) ? 1 : nb;
  const int64_t nbB = (pr.sparseB && pr.broadcastSparse) ? 1 : nb;
  std::vector<uint8_t> A((size_t)(s.sizeA() * nbA) * esize(pr.ab)), B((size_t)(s.sizeB() * nbB) * esize(pr.ab));
  std::vector<uint8_t> C((size_t)(s.sizeC() * nb) * esize(pr.cd));
  for (int64_t b = 0; b < nbA; ++b)
    for (int64_t i = 0; i < m; ++i)
      for (int64_t kk = 0; kk < k; ++kk) setv(A, b * s.sizeA() + s.offA(i, kk), pr.ab, fa(b, i, kk));
  for (int64_t b = 0; b < nbB; ++b)
    for (int64_t kk = 0; kk < k; ++kk)
      for (int64_t j = 0; j < n; ++j) setv(B, b * s.sizeB() + s.offB(kk, j), pr.ab, fb(b, kk, j));
  for (int64_t b = 0; b < nb; ++b)
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j) setv(C, b * s.sizeC() + s.offC(i, j), pr.cd, (float)((i * 3 + j + b) % 5) - 2.f);
  void* dA = upload(A);
  void* dB = upload(B);
  void* dC = upload(C);
  void* dD = zeros(C.size());
  void* dS = pr.sparseB ? dB : dA;
  // Prune (STRIP) in place, and check it.
  int* dvalid = (int*)zeros(4);
  st = cusparseLtSpMMAPrune(&h, &s.md, dS, dS, CUSPARSELT_PRUNE_SPMMA_STRIP, nullptr);
  st |= cusparseLtSpMMAPruneCheck(&h, &s.md, dS, dvalid, nullptr);
  std::snprintf(what, sizeof what, "%s: prune and check", name);
  check(st == 0 && download<int>(dvalid, 1)[0] == 0, what);
  std::vector<uint8_t>& Sh = pr.sparseB ? B : A;
  Sh = download<uint8_t>(dS, Sh.size());
  size_t cs = 0, cbs = 0;
  st = cusparseLtSpMMACompressedSize(&h, &s.plan, &cs, &cbs);
  void* dcomp = zeros(cs);
  void* dbuf = zeros(cbs);
  st |= cusparseLtSpMMACompress(&h, &s.plan, dS, dcomp, dbuf, nullptr);
  size_t ws = 0;
  st |= cusparseLtMatmulGetWorkspace(&h, &s.plan, &ws);
  void* dws = zeros(ws);
  std::vector<float> avs(ep.av), bvs(ep.bv);
  void* dav = ep.alphaVec ? upload(avs) : nullptr;
  void* dbv = ep.betaVec ? upload(bvs) : nullptr;
  const void* alpha = ep.alphaVec ? dav : (const void*)&ep.alpha;
  const void* beta = ep.betaVec ? dbv : (const void*)&ep.beta;
  const void* opA = pr.sparseB ? dA : dcomp;
  const void* opB = pr.sparseB ? dcomp : dB;
  if (search) st |= cusparseLtMatmulSearch(&h, &s.plan, alpha, opA, opB, beta, dC, dD, dws, nullptr, 0);
  else st |= cusparseLtMatmul(&h, &s.plan, alpha, opA, opB, beta, dC, dD, dws, nullptr, 0);
  std::snprintf(what, sizeof what, "%s: compress and multiply", name);
  check(st == 0 && cudaDeviceSynchronize() == cudaSuccess, what);
  const std::vector<uint8_t> D = download<uint8_t>(dD, C.size());
  int bad = 0;
  for (int64_t b = 0; b < nb; ++b)
    for (int64_t i = 0; i < m; ++i)
      for (int64_t j = 0; j < n; ++j) {
        const int64_t ba = nbA == 1 ? 0 : b, bb = nbB == 1 ? 0 : b;
        double acc = 0;
        for (int64_t kk = 0; kk < k; ++kk) {
          float x = getv(A, ba * s.sizeA() + s.offA(i, kk), pr.ab), y = getv(B, bb * s.sizeB() + s.offB(kk, j), pr.ab);
          if (pr.ab == CUDA_R_32F) { x = tf32(x); y = tf32(y); }
          acc += (double)x * y;
        }
        const float a = ep.alphaVec ? ep.av[(size_t)i] : ep.alpha;
        const float bt = ep.alphaVec ? (ep.betaVec ? ep.bv[(size_t)i] : 0.f) : ep.beta;
        float v = a * (float)acc + bt * getv(C, b * s.sizeC() + s.offC(i, j), pr.cd);
        if (!ep.bias.empty()) v += ep.bias[(size_t)(b * m + i)];
        if (ep.gelu) v = ep.geluScale * gelu_ref(v);
        else if (ep.relu) v = v <= ep.th ? std::copysign(0.f, v) : std::min(v, ep.ub);
        const float want = round_out(v, pr.cd);
        const float got = getv(D, b * s.sizeC() + s.offC(i, j), pr.cd);
        float slack = tol;
        if (pr.cd == CUDA_R_16F || pr.cd == CUDA_R_16BF || pr.cd == CUDA_R_32F) {
          const float ulp = pr.cd == CUDA_R_16F ? std::ldexp(1.f, std::max(-24, (int)std::floor(std::log2(std::fabs(want) + 1e-30f)) - 10))
                          : pr.cd == CUDA_R_16BF ? std::ldexp(1.f, (int)std::floor(std::log2(std::fabs(want) + 1e-30f)) - 7)
                                                 : std::ldexp(1.f, (int)std::floor(std::log2(std::fabs(want) + 1e-30f)) - 23);
          slack = tol * ulp;
        }
        if (!(std::fabs(got - want) <= slack)) {
          if (bad < 3) std::printf("     %s: D[%lld][%lld][%lld] = %.9g, expected %.9g\n", name, (long long)b, (long long)i, (long long)j, got, want);
          ++bad;
        }
      }
  std::snprintf(what, sizeof what, "%s: D matches the host product", name);
  check(bad == 0, what);
  for (void* p : {dA, dB, dC, dD, dcomp, dbuf, dws, (void*)dvalid, dbias, dav, dbv}) cudaFree(p);
  cusparseLtMatmulPlanDestroy(&s.plan);
  cusparseLtMatmulAlgSelectionDestroy(&s.alg);
  return bad;
}

// Small integers keep every product exact in every type.
static float smallint(int64_t a, int64_t b, int64_t c, int mod) { return (float)((a * 7 + b * 3 + c * 5) % mod) - (float)(mod / 2); }

int main() {
  IS(cusparseLtInit(&h), 0);

  // ---- library ----
  int v = -1;
  IS(cusparseLtGetVersion(&h, &v), 0);
  check(v == 1000, "version 0.10.0 reads 1000");
  int major = -1, minor = -1, patch = -1;
  IS(cusparseLtGetProperty(MAJOR_VERSION, &major), 0);
  IS(cusparseLtGetProperty(MINOR_VERSION, &minor), 0);
  IS(cusparseLtGetProperty(PATCH_LEVEL, &patch), 0);
  check(major == 0 && minor == 10 && patch == 0, "properties 0, 10, 0");
  IS(cusparseLtGetProperty((libraryPropertyType)3, &v), 3);
  IS(cusparseLtGetVersion(&h, nullptr), 3);
  check(!std::strcmp(cusparseLtGetErrorName(CUSPARSE_STATUS_NOT_SUPPORTED), "CUSPARSE_STATUS_NOT_SUPPORTED") &&
            !std::strcmp(cusparseLtGetErrorString(CUSPARSE_STATUS_INVALID_VALUE), "invalid value") &&
            !std::strcmp(cusparseLtGetErrorName((cusparseStatus_t)12), "unrecognized error code"),
        "error names and strings");
  check(sizeof(cusparseLtHandle_t) == 512 && alignof(cusparseLtMatmulPlan_t) == 16, "opaque objects: 512 bytes, aligned 16");
  IS(cusparseLtInit(nullptr), 3);
  {
    cusparseLtHandle_t h2;
    IS(cusparseLtInit(&h2), 0);
    IS(cusparseLtDestroy(&h2), 0);
    IS(cusparseLtDestroy(&h2), 3);
    cusparseLtMatDescriptor_t d;
    std::memset(&h2, 0, sizeof h2);
    IS(cusparseLtDenseDescriptorInit(&h2, &d, 64, 64, 64, 16, CUDA_R_16F, ROW), 3);
  }

  // ---- descriptors (measured on the card) ----
  {
    cusparseLtMatDescriptor_t d;
    // dense: rows, cols and ld multiples of 16 bytes' worth; structured 32 bytes'
    IS(cusparseLtDenseDescriptorInit(&h, &d, 8, 128, 128, 16, CUDA_R_16F, ROW), 0);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 4, 128, 128, 16, CUDA_R_16F, ROW), 10);
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 8, 128, 128, 16, CUDA_R_16F, ROW, S50), 10);
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 16, 128, 128, 16, CUDA_R_16F, ROW, S50), 0);
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 48, 128, 128, 16, CUDA_R_8I, ROW, S50), 10);
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 96, 128, 128, 16, CUDA_R_8I, ROW, S50), 0);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 4, 128, 128, 16, CUDA_R_32F, ROW), 0);
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 4, 128, 128, 16, CUDA_R_32F, ROW, S50), 10);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 136, 16, CUDA_R_16F, ROW), 0);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 132, 16, CUDA_R_16F, ROW), 10);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 127, 16, CUDA_R_16F, ROW), 3);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 64, 64, 16, CUDA_R_16F, COL), 3);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 128, 8, CUDA_R_16F, ROW), 10);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 128, 24, CUDA_R_16F, ROW), 10);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 128, 0, CUDA_R_16F, ROW), 10);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 128, 128, 128, 256, CUDA_R_16F, ROW), 0);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 0, 64, 64, 16, CUDA_R_16F, ROW), 3);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 64, 64, 64, 16, CUDA_R_64F, ROW), 3);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 64, 64, 64, 16, CUDA_R_8U, ROW), 3);
    IS(cusparseLtDenseDescriptorInit(&h, &d, 64, 64, 64, 16, CUDA_R_16F, (cusparseOrder_t)3), 3);
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 64, 64, 64, 16, CUDA_R_16F, ROW, (cusparseLtSparsity_t)1), 3);
    IS(cusparseLtDenseDescriptorInit(nullptr, &d, 64, 64, 64, 16, CUDA_R_16F, ROW), 3);
    IS(cusparseLtDenseDescriptorInit(&h, nullptr, 64, 64, 64, 16, CUDA_R_16F, ROW), 3);
    // batch attributes
    IS(cusparseLtStructuredDescriptorInit(&h, &d, 64, 64, 64, 16, CUDA_R_16F, ROW, S50), 0);
    int nb = -1;
    int64_t bs = -1;
    IS(cusparseLtMatDescGetAttribute(&h, &d, CUSPARSELT_MAT_NUM_BATCHES, &nb, 4), 0);
    IS(cusparseLtMatDescGetAttribute(&h, &d, CUSPARSELT_MAT_BATCH_STRIDE, &bs, 8), 0);
    check(nb == 1 && bs == 4096, "batch defaults: 1 batch, stride rows x ld");
    IS(cusparseLtMatDescGetAttribute(&h, &d, CUSPARSELT_MAT_BATCH_STRIDE, &nb, 4), 3);
    IS(cusparseLtMatDescGetAttribute(&h, &d, (cusparseLtMatDescAttribute_t)2, &bs, 8), 3);
    nb = 3;
    IS(cusparseLtMatDescSetAttribute(&h, &d, CUSPARSELT_MAT_NUM_BATCHES, &nb, 4), 0);
    nb = 0;
    IS(cusparseLtMatDescSetAttribute(&h, &d, CUSPARSELT_MAT_NUM_BATCHES, &nb, 4), 3);
    for (int64_t s : {(int64_t)0, (int64_t)4096, (int64_t)4104, (int64_t)8192}) {
      bs = s;
      IS(cusparseLtMatDescSetAttribute(&h, &d, CUSPARSELT_MAT_BATCH_STRIDE, &bs, 8), 0);
    }
    for (int64_t s : {(int64_t)100, (int64_t)4095, (int64_t)-8}) {
      bs = s;
      IS(cusparseLtMatDescSetAttribute(&h, &d, CUSPARSELT_MAT_BATCH_STRIDE, &bs, 8), 3);
    }
    IS(cusparseLtMatDescSetAttribute(&h, &d, CUSPARSELT_MAT_NUM_BATCHES, nullptr, 4), 3);
    IS(cusparseLtMatDescGetAttribute(&h, &d, CUSPARSELT_MAT_NUM_BATCHES, &nb, 4), 0);
    IS(cusparseLtMatDescGetAttribute(&h, &d, CUSPARSELT_MAT_BATCH_STRIDE, &bs, 8), 0);
    check(nb == 3 && bs == 8192, "batch attributes read back");
    cusparseLtMatDescriptor_t c;
    IS(cusparseLtDenseDescriptorInit(&h, &c, 64, 32, 64, 16, CUDA_R_16F, COL), 0);
    IS(cusparseLtMatDescGetAttribute(&h, &c, CUSPARSELT_MAT_BATCH_STRIDE, &bs, 8), 0);
    check(bs == 2048, "a column-major matrix's default stride is cols x ld");
    IS(cusparseLtMatDescriptorDestroy(&d), 0);
  }

  // ---- matmul descriptors: types and layouts sm_86 takes ----
  {
    struct C { cudaDataType ab, cd; cusparseComputeType ct; cusparseOperation_t opB; cusparseOrder_t oB; int md, alg; } cases[] = {
        {CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, N, ROW, 0, 0},
        {CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_16F, N, ROW, 0, 10},
        {CUDA_R_16F, CUDA_R_32F, CUSPARSE_COMPUTE_32F, N, ROW, 10, -1},
        {CUDA_R_16BF, CUDA_R_16BF, CUSPARSE_COMPUTE_32F, N, ROW, 0, 0},
        {CUDA_R_16BF, CUDA_R_16F, CUSPARSE_COMPUTE_32F, N, ROW, 10, -1},
        {CUDA_R_32F, CUDA_R_32F, CUSPARSE_COMPUTE_32F, N, ROW, 0, 0},
        {CUDA_R_8I, CUDA_R_8I, CUSPARSE_COMPUTE_32I, T, ROW, 0, 0},
        {CUDA_R_8I, CUDA_R_32I, CUSPARSE_COMPUTE_32I, T, ROW, 0, 0},
        {CUDA_R_8I, CUDA_R_16F, CUSPARSE_COMPUTE_32I, T, ROW, 0, 0},
        {CUDA_R_8I, CUDA_R_16BF, CUSPARSE_COMPUTE_32I, T, ROW, 0, 0},
        {CUDA_R_8I, CUDA_R_32F, CUSPARSE_COMPUTE_32I, T, ROW, 10, -1},
        {CUDA_R_8I, CUDA_R_8I, CUSPARSE_COMPUTE_32F, T, ROW, 10, -1},
        {CUDA_R_8I, CUDA_R_8I, CUSPARSE_COMPUTE_32I, N, ROW, 10, -1},  // B not K-contiguous
        {CUDA_R_8I, CUDA_R_8I, CUSPARSE_COMPUTE_32I, N, COL, 0, 0},
    };
    for (const C& c : cases) {
      Problem p{c.ab, c.cd, c.ct, 64, 64, 64};
      p.opB = c.opB;
      p.oB = c.oB;
      Setup s(p);
      char what[160];
      std::snprintf(what, sizeof what, "types %d -> %d compute %d opB %d orderB %d: matmul descriptor %d", (int)c.ab, (int)c.cd, (int)c.ct, (int)c.opB, (int)c.oB, c.md);
      check(s.smd == c.md, what);
      if (s.smd == 0) {
        const int a = cusparseLtMatmulAlgSelectionInit(&h, &s.alg, &s.md, CUSPARSELT_MATMUL_ALG_DEFAULT);
        std::snprintf(what, sizeof what, "  ... algorithm selection %d", c.alg);
        check(a == c.alg, what);
      }
    }
    // shapes, structure, batches, C against D
    cusparseLtMatDescriptor_t A, B, Cd, D, Dld, Ccol, Abig, Bs;
    cusparseLtMatmulDescriptor_t md;
    cusparseLtStructuredDescriptorInit(&h, &A, 64, 128, 128, 16, CUDA_R_16F, ROW, S50);
    cusparseLtDenseDescriptorInit(&h, &B, 128, 32, 32, 16, CUDA_R_16F, ROW);
    cusparseLtDenseDescriptorInit(&h, &Cd, 64, 32, 32, 16, CUDA_R_16F, ROW);
    cusparseLtDenseDescriptorInit(&h, &D, 64, 32, 32, 16, CUDA_R_16F, ROW);
    cusparseLtDenseDescriptorInit(&h, &Dld, 64, 32, 40, 16, CUDA_R_16F, ROW);
    cusparseLtDenseDescriptorInit(&h, &Ccol, 64, 32, 64, 16, CUDA_R_16F, COL);
    cusparseLtDenseDescriptorInit(&h, &Abig, 64, 128, 128, 16, CUDA_R_16F, ROW);
    cusparseLtStructuredDescriptorInit(&h, &Bs, 128, 32, 32, 16, CUDA_R_16F, ROW, S50);
    const cusparseComputeType F = CUSPARSE_COMPUTE_32F;
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Cd, &D, F), 0);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Cd, &Dld, F), 10);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Ccol, &D, F), 10);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, T, N, &A, &B, &Cd, &D, F), 3);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &Abig, &B, &Cd, &D, F), 3);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &Bs, &Cd, &D, F), 3);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, (cusparseOperation_t)2, N, &A, &B, &Cd, &D, F), 3);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Cd, &D, (cusparseComputeType)7), 3);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, nullptr, &B, &Cd, &D, F), 3);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Cd, nullptr, F), 3);
    int two = 2;
    cusparseLtMatDescriptor_t A2 = A;
    cusparseLtStructuredDescriptorInit(&h, &A2, 64, 128, 128, 16, CUDA_R_16F, ROW, S50);
    cusparseLtMatDescSetAttribute(&h, &A2, CUSPARSELT_MAT_NUM_BATCHES, &two, 4);
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A2, &B, &Cd, &Cd, F), 3);

    // matmul attributes: defaults, sizes, what fp16 refuses
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Cd, &D, F), 0);
    int iv = -1;
    float fv = -1;
    int64_t lv = -1;
    void* pv = (void*)1;
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_RELU, &iv, 4), 0);
    check(iv == 0, "ReLU off by default");
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_RELU_UPPERBOUND, &fv, 4), 0);
    check(fv == 3.40282347e38f, "ReLU upper bound defaults to FLT_MAX");
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &fv, 4), 0);
    check(fv == 1.f, "GELU scaling defaults to 1");
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_BIAS_STRIDE, &lv, 8), 0);
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_BIAS_POINTER, &pv, sizeof pv), 0);
    check(lv == 0 && pv == nullptr, "no bias by default");
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_RELU, &lv, 8), 3);
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_BIAS_POINTER, &iv, 4), 3);
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, (cusparseLtMatmulDescAttribute_t)20, &lv, 8), 3);
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_RELU, &two, 4), 0);
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_RELU, &iv, 4), 0);
    check(iv == 1, "ReLU reads back as 1");
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_GELU, &two, 4), 3);
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_ACTIVATION_GELU_SCALING, &fv, 4), 3);
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_ALPHA_VECTOR_SCALING, &two, 4), 0);
    lv = -5;
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_BIAS_STRIDE, &lv, 8), 3);
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_BIAS_STRIDE, &iv, 4), 3);
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_BIAS_POINTER, nullptr, 8), 3);
    pv = (void*)0x1234;
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &pv, sizeof pv), 0);
    IS(cusparseLtMatmulDescGetAttribute(&h, &md, CUSPARSELT_MATMUL_SPARSE_MAT_POINTER, &pv, sizeof pv), 7);

    // algorithm selection
    cusparseLtMatmulAlgSelection_t alg;
    IS(cusparseLtMatmulDescriptorInit(&h, &md, N, N, &A, &B, &Cd, &D, F), 0);
    IS(cusparseLtMatmulAlgSelectionInit(&h, &alg, &md, CUSPARSELT_MATMUL_ALG_DEFAULT), 0);
    int vals[6];
    bool got = true;
    for (int a = 0; a < 6; ++a) got &= cusparseLtMatmulAlgGetAttribute(&h, &alg, (cusparseLtMatmulAlgAttribute_t)a, &vals[a], 4) == 0;
    check(got && vals[0] == 0 && vals[1] == 4 && vals[2] == 5 && vals[3] == 1 && vals[4] == 1 && vals[5] == 0,
          "algorithm defaults: config 0 of 4, 5 iterations, split-K 1, one kernel, no buffers");
    IS(cusparseLtMatmulAlgGetAttribute(&h, &alg, CUSPARSELT_MATMUL_ALG_CONFIG_ID, &lv, 8), 3);
    IS(cusparseLtMatmulAlgGetAttribute(&h, &alg, (cusparseLtMatmulAlgAttribute_t)6, &iv, 4), 3);
    struct AV { cusparseLtMatmulAlgAttribute_t a; int v, want; } avs[] = {
        {CUSPARSELT_MATMUL_ALG_CONFIG_ID, 3, 0}, {CUSPARSELT_MATMUL_ALG_CONFIG_ID, 4, 3}, {CUSPARSELT_MATMUL_ALG_CONFIG_ID, -1, 3},
        {CUSPARSELT_MATMUL_ALG_CONFIG_MAX_ID, 1, 3}, {CUSPARSELT_MATMUL_SEARCH_ITERATIONS, 0, 0},
        {CUSPARSELT_MATMUL_SEARCH_ITERATIONS, 100, 0}, {CUSPARSELT_MATMUL_SPLIT_K, 0, 3}, {CUSPARSELT_MATMUL_SPLIT_K, 65, 0},
        {CUSPARSELT_MATMUL_SPLIT_K_MODE, 0, 3}, {CUSPARSELT_MATMUL_SPLIT_K_MODE, 2, 0}, {CUSPARSELT_MATMUL_SPLIT_K_MODE, 3, 3},
        {CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, -1, 3}, {CUSPARSELT_MATMUL_SPLIT_K_BUFFERS, 17, 0},
    };
    for (const AV& a : avs) {
      char what[120];
      std::snprintf(what, sizeof what, "algorithm attribute %d = %d -> %d", (int)a.a, a.v, a.want);
      check(cusparseLtMatmulAlgSetAttribute(&h, &alg, a.a, &a.v, 4) == a.want, what);
    }
    for (int a = 0; a < 6; ++a) cusparseLtMatmulAlgGetAttribute(&h, &alg, (cusparseLtMatmulAlgAttribute_t)a, &vals[a], 4);
    check(vals[0] == 3 && vals[2] == 100 && vals[3] == 65 && vals[4] == 2 && vals[5] == 17, "algorithm attributes read back");
    IS(cusparseLtMatmulAlgSelectionDestroy(&alg), 0);
    IS(cusparseLtMatmulAlgSelectionInit(&h, &alg, &md, (cusparseLtMatmulAlg_t)1), 3);
    IS(cusparseLtMatmulAlgSelectionDestroy(nullptr), 3);
    IS(cusparseLtMatmulPlanDestroy(nullptr), 3);
    // a scale mode on an fp16 product is accepted (it belongs to FP8)
    int mode = CUSPARSELT_MATMUL_MATRIX_SCALE_SCALAR_32F;
    IS(cusparseLtMatmulDescSetAttribute(&h, &md, CUSPARSELT_MATMUL_A_SCALE_MODE, &mode, 4), 0);
    IS(cusparseLtMatmulAlgSelectionInit(&h, &alg, &md, CUSPARSELT_MATMUL_ALG_DEFAULT), 0);
  }

  // ---- pruning: STRIP and TILE, as the card prunes ----
  {
    const int64_t m = 64, k = 64;
    std::vector<uint16_t> a((size_t)(m * k));
    // Rows 0-39: all 625 groups of {-2,-1,0,1,2}; the rest small integers
    // with many ties.
    for (int g = 0; g < 625; ++g) {
      int x = g;
      for (int j = 0; j < 4; ++j) { a[(size_t)(g * 4 + j)] = f2h((float)(x % 5 - 2)); x /= 5; }
    }
    for (int64_t i = 2500; i < m * k; ++i) a[(size_t)i] = f2h((float)((i * 37 + (i >> 3) * 11) % 7) - 3.f);
    // and a few tiles of 1s and 2s, where TILE's ties decide
    for (int64_t i = 48 * 64; i < m * k; ++i) a[(size_t)i] = f2h(1.f + (float)((i * 2654435761u >> 7) & 1));
    for (cusparseOrder_t order : {ROW, COL}) {
      for (cusparseOperation_t op : {N, T}) {
        Problem p{CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, 64, 64, 64};
        p.opA = op;
        p.oA = order;
        Setup s(p);
        void* din = upload(a);
        void* dout = zeros(a.size() * 2);
        for (int alg : {CUSPARSELT_PRUNE_SPMMA_STRIP, CUSPARSELT_PRUNE_SPMMA_TILE}) {
          const int st = cusparseLtSpMMAPrune(&h, &s.md, din, dout, (cusparseLtPruneAlg_t)alg, nullptr);
          const std::vector<uint16_t> o = download<uint16_t>(dout, a.size());
          // expected: memory holds lines of 64; K runs along lines when
          // (op N) == (row-major).
          std::vector<uint16_t> want(a);
          const bool k_along_lines = (op == N) == (order == ROW);
          if (alg == CUSPARSELT_PRUNE_SPMMA_STRIP) {
            for (int64_t l = 0; l < 64; ++l)
              for (int64_t q = 0; q < 64; q += 4) {
                float g[4];
                int64_t off[4];
                for (int j = 0; j < 4; ++j) {
                  off[j] = k_along_lines ? l * 64 + q + j : (q + j) * 64 + l;
                  g[j] = h2f(a[(size_t)off[j]]);
                }
                const unsigned keep = strip_mask(g, 4);
                for (int j = 0; j < 4; ++j)
                  if (!(keep >> j & 1)) want[(size_t)off[j]] = 0;
              }
          } else {
            for (int64_t l = 0; l < 64; l += 4)
              for (int64_t q = 0; q < 64; q += 4) {
                float t[4][4];
                for (int i = 0; i < 4; ++i)
                  for (int j = 0; j < 4; ++j) t[i][j] = h2f(a[(size_t)((l + i) * 64 + q + j)]);
                const unsigned keep = tile_mask(t);
                for (int i = 0; i < 4; ++i)
                  for (int j = 0; j < 4; ++j)
                    if (!(keep >> (4 * i + j) & 1)) want[(size_t)((l + i) * 64 + q + j)] = 0;
              }
          }
          int bad = 0;
          for (size_t i = 0; i < a.size(); ++i) bad += o[i] != want[i];
          char what[120];
          std::snprintf(what, sizeof what, "%s pruning, op%s, %s-major: %d of 4096 differ", alg ? "STRIP" : "TILE", op == N ? "N" : "T", order == ROW ? "row" : "column", bad);
          check(st == 0 && bad == 0, what);
          int* dv = (int*)zeros(4);
          IS(cusparseLtSpMMAPruneCheck(&h, &s.md, dout, dv, nullptr), 0);
          const int valid_pruned = download<int>(dv, 1)[0];
          IS(cusparseLtSpMMAPruneCheck(&h, &s.md, din, dv, nullptr), 0);
          check(valid_pruned == 0 && download<int>(dv, 1)[0] == 1, "PruneCheck: 0 for the pruned matrix, 1 for the original");
          cudaFree(dv);
        }
        // Prune2 on the descriptor alone agrees with Prune.
        cudaMemset(dout, 0, a.size() * 2);
        IS(cusparseLtSpMMAPrune2(&h, &s.A, 1, op, din, dout, CUSPARSELT_PRUNE_SPMMA_TILE, nullptr), 0);
        void* dout1 = zeros(a.size() * 2);
        IS(cusparseLtSpMMAPrune(&h, &s.md, din, dout1, CUSPARSELT_PRUNE_SPMMA_TILE, nullptr), 0);
        check(download<uint16_t>(dout, a.size()) == download<uint16_t>(dout1, a.size()), "Prune2 matches Prune");
        cudaFree(din);
        cudaFree(dout);
        cudaFree(dout1);
      }
    }
    // fp32 TILE is 2x2: the diagonal when strictly larger, else the anti-diagonal
    {
      Problem p{CUDA_R_32F, CUDA_R_32F, CUSPARSE_COMPUTE_32F, 64, 64, 64};
      Setup s(p);
      std::vector<float> f((size_t)(m * k));
      for (int64_t i = 0; i < m * k; ++i) f[(size_t)i] = (float)((i * 2654435761u >> 9) % 5);
      void* din = upload(f);
      void* dout = zeros(f.size() * 4);
      IS(cusparseLtSpMMAPrune(&h, &s.md, din, dout, CUSPARSELT_PRUNE_SPMMA_TILE, nullptr), 0);
      const std::vector<float> o = download<float>(dout, f.size());
      int bad = 0;
      for (int64_t l = 0; l < 64; l += 2)
        for (int64_t q = 0; q < 64; q += 2) {
          const float a00 = f[(size_t)(l * 64 + q)], a01 = f[(size_t)(l * 64 + q + 1)], a10 = f[(size_t)((l + 1) * 64 + q)], a11 = f[(size_t)((l + 1) * 64 + q + 1)];
          const bool diag = a00 + a11 > a01 + a10;
          bad += o[(size_t)(l * 64 + q)] != (diag ? a00 : 0.f) || o[(size_t)(l * 64 + q + 1)] != (diag ? 0.f : a01) ||
                 o[(size_t)((l + 1) * 64 + q)] != (diag ? 0.f : a10) || o[(size_t)((l + 1) * 64 + q + 1)] != (diag ? a11 : 0.f);
        }
      check(bad == 0, "fp32 TILE: 2x2 tiles, the anti-diagonal on a tie");
      cudaFree(din);
      cudaFree(dout);
    }
  }

  // ---- compression: sizes, kept values, metadata ----
  {
    struct Sz { cudaDataType t; int64_t r, c; size_t cs, cbs; cusparseOrder_t o; } sizes[] = {
        {CUDA_R_16F, 64, 64, 4608, 512},    {CUDA_R_16F, 16, 32, 768, 256},     {CUDA_R_16F, 64, 32, 2560, 512},
        {CUDA_R_16F, 96, 160, 17920, 2560}, {CUDA_R_16BF, 128, 128, 18432, 2048}, {CUDA_R_8I, 64, 64, 4096, 1024},
        {CUDA_R_8I, 96, 160, 15872, 4096},  {CUDA_R_8I, 256, 64, 16384, 4096},  {CUDA_R_32F, 64, 64, 10240, 1024},
        {CUDA_R_32F, 8, 16, 768, 256},      {CUDA_R_32F, 96, 160, 40960, 5120},
        // int8 sizes are not symmetric: K is taken as the contiguous dimension
        {CUDA_R_8I, 64, 128, 8192, 2048, COL}, {CUDA_R_8I, 128, 64, 6144, 1024, COL}, {CUDA_R_8I, 128, 64, 8192, 2048},
    };
    for (const Sz& z : sizes) {
      cusparseLtMatDescriptor_t d;
      size_t cs = 0, cbs = 0;
      const cusparseOrder_t o = z.o == COL ? COL : ROW;
      const int s0 = cusparseLtStructuredDescriptorInit(&h, &d, z.r, z.c, o == ROW ? z.c : z.r, 16, z.t, o, S50);
      const int s1 = cusparseLtSpMMACompressedSize2(&h, &d, &cs, &cbs);
      char what[120];
      std::snprintf(what, sizeof what, "type %d %lldx%lld %s-major compresses to %zu bytes with a %zu-byte buffer (got %zu, %zu)", (int)z.t, (long long)z.r, (long long)z.c, o == ROW ? "row" : "column", z.cs, z.cbs, cs, cbs);
      check(s0 == 0 && s1 == 0 && cs == z.cs && cbs == z.cbs, what);
    }
    // fp16 64x64: values row-major, then the metadata in the card's layout
    for (cudaDataType t : {CUDA_R_16F, CUDA_R_8I, CUDA_R_32F}) {
      const int64_t m = 64, k = 64;
      const size_t es = esize(t);
      std::vector<uint8_t> a((size_t)(m * k) * es, 0);
      // Groups with 0, 1 and 2 nonzeros in every position (fp32: pairs).
      const int g = t == CUDA_R_32F ? 2 : 4;
      for (int64_t r = 0; r < m; ++r)
        for (int64_t q = 0; q < k; q += g) {
          const int mask = (int)((r * 7 + q / g * 3) % (g == 4 ? 11 : 3));
          static const int masks4[11] = {0, 1, 2, 4, 8, 3, 5, 6, 9, 10, 12};
          const int bits = g == 4 ? masks4[mask] : mask;  // 0, 1, 2 for pairs
          for (int j = 0; j < g; ++j)
            if (bits >> j & 1) setv(a, r * k + q + j, t, (float)(1 + (r + q + j) % 50));
        }
      cusparseLtMatDescriptor_t d;
      cusparseLtStructuredDescriptorInit(&h, &d, m, k, k, 16, t, ROW, S50);
      size_t cs = 0, cbs = 0;
      cusparseLtSpMMACompressedSize2(&h, &d, &cs, &cbs);
      void* din = upload(a);
      void* dc = zeros(cs);
      void* db = zeros(cbs);
      const int st = cusparseLtSpMMACompress2(&h, &d, 1, N, din, dc, db, nullptr);
      const std::vector<uint8_t> c = download<uint8_t>(dc, cs);
      const size_t vb = (size_t)(m * k / 2) * es;
      int badv = 0, badm = 0;
      for (int64_t r = 0; r < m; ++r) {
        if (t == CUDA_R_32F) {
          for (int64_t p = 0; p < k / 2; ++p) {
            const float x0 = getv(a, r * k + 2 * p, t);
            const int pick = x0 != 0.f ? 0 : 1;
            uint32_t bits, want;
            std::memcpy(&want, a.data() + (size_t)(r * k + 2 * p + pick) * 4, 4);
            want += 0x1000u;  // the tf32 rounding half-unit, stored (measured)
            std::memcpy(&bits, c.data() + (size_t)(r * (k / 2) + p) * 4, 4);
            badv += bits != want;
            const uint8_t code = pick ? 0xE : 0x4;
            const size_t n0 = (size_t)(r * (k / 2) + p), n1 = cs - vb + nibble16(r, p, m);  // plain copy, then blocked
            badm += ((c[vb + n0 / 2] >> (4 * (n0 & 1))) & 15) != code;
            badm += ((c[vb + n1 / 2] >> (4 * (n1 & 1))) & 15) != code;
          }
          continue;
        }
        for (int64_t q = 0; q < k / 4; ++q) {
          int nz[4], n = 0;
          for (int j = 0; j < 4; ++j)
            if (getv(a, r * k + 4 * q + j, t) != 0.f) nz[n++] = j;
          int i0 = 2, i1 = 3;  // the card pads from the right (measured)
          if (n >= 2) { i0 = nz[0]; i1 = nz[1]; }
          else if (n == 1) { i0 = nz[0] == 3 ? 2 : nz[0]; i1 = 3; }
          badv += std::memcmp(c.data() + (size_t)(r * (k / 2) + 2 * q) * es, a.data() + (size_t)(r * k + 4 * q + i0) * es, es) != 0;
          badv += std::memcmp(c.data() + (size_t)(r * (k / 2) + 2 * q + 1) * es, a.data() + (size_t)(r * k + 4 * q + i1) * es, es) != 0;
          const size_t nib = t == CUDA_R_8I ? nibble8(r, q, m) : nibble16(r, q, m);
          badm += ((c[vb + nib / 2] >> (4 * (nib & 1))) & 15) != (i0 | (i1 << 2));
        }
      }
      char what[160];
      std::snprintf(what, sizeof what, "type %d 64x64 compressed: %d values and %d metadata codes differ from the card's layout", (int)t, badv, badm);
      check(st == 0 && badv == 0 && badm == 0, what);
      cudaFree(din);
      cudaFree(dc);
      cudaFree(db);
    }
  }

  // ---- the product ----
  {
    Epilogue plain;
    plain.alpha = 1.f;
    plain.beta = 0.f;
    Epilogue ab;
    ab.alpha = 0.5f;
    ab.beta = 0.25f;
    auto fa = [](int64_t b, int64_t i, int64_t kk) { return smallint(i + b, kk, 1, 7); };
    auto fb = [](int64_t b, int64_t kk, int64_t j) { return smallint(kk, j + 2 * b, 3, 5); };
    // fp16: every transpose and order, a structured B, rounding to fp16
    for (cusparseOperation_t opA : {N, T})
      for (cusparseOperation_t opB : {N, T})
        for (cusparseOrder_t o : {ROW, COL}) {
          Problem p{CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, 64, 32, 64};
          p.opA = opA;
          p.opB = opB;
          p.oA = p.oB = p.oC = o;
          char name[80];
          std::snprintf(name, sizeof name, "fp16 op%s%s %s", opA == N ? "N" : "T", opB == N ? "N" : "T", o == ROW ? "row" : "col");
          run_product(name, p, fa, fb, ab);
        }
    {
      Problem p{CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, 32, 64, 64};
      p.sparseB = true;
      run_product("fp16 structured B", p, fa, fb, ab);
      p.opB = T;
      p.oC = COL;
      run_product("fp16 structured B transposed, column-major D", p, fa, fb, ab);
    }
    {
      // Rounding into fp16: 2048 + j lands on every rounding case.
      Problem p{CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, 64, 64, 64};
      run_product("fp16 rounding to nearest even", p,
                  [](int64_t, int64_t i, int64_t kk) { return kk < 2 ? 1.f : (kk % 4 < 2 ? (float)((i + kk) % 3) : 0.f); },
                  [](int64_t, int64_t kk, int64_t j) { return kk == 0 ? 2048.f : (kk == 1 ? (float)j : 0.f); }, plain);
      Epilogue e = ab;
      e.relu = true;
      e.th = 1.f;
      e.ub = 6.f;
      run_product("fp16 ReLU with threshold and upper bound", p, fa, fb, e);
      Epilogue r = ab;
      r.relu = true;
      run_product("fp16 ReLU (negative results are -0)", p, fa, fb, r);
      Epilogue bz = ab;
      for (int i = 0; i < 64; ++i) bz.bias.push_back(0.5f * (float)i - 8.f);
      run_product("fp16 bias (fp16 values)", p, fa, fb, bz);
      Epilogue av;
      av.alphaVec = true;
      av.beta = 1.f;  // measured: with an alpha vector and no beta vector, beta is not applied
      for (int i = 0; i < 64; ++i) av.av.push_back(1.f + (float)(i % 3));
      run_product("fp16 alpha vector", p, fa, fb, av);
      av.betaVec = true;
      for (int i = 0; i < 64; ++i) av.bv.push_back(0.5f * (float)(i % 4));
      run_product("fp16 alpha and beta vectors", p, fa, fb, av);
      run_product("fp16 through MatmulSearch", p, fa, fb, ab, 0.f, true);
    }
    {
      Problem p{CUDA_R_16BF, CUDA_R_16BF, CUSPARSE_COMPUTE_32F, 64, 64, 128};
      Epilogue e = ab;
      for (int i = 0; i < 64; ++i) e.bias.push_back(100.f + (float)i);
      run_product("bf16 with a bf16 bias", p, fa, fb, e);
    }
    {
      // tf32: low mantissa bits round to nearest, ties away (measured)
      Problem p{CUDA_R_32F, CUDA_R_32F, CUSPARSE_COMPUTE_32F, 64, 64, 64};
      run_product("tf32 operand rounding", p,
                  [](int64_t, int64_t i, int64_t kk) { return (kk % 2 == (i + kk / 2) % 2) ? 1.f + (float)((i * 3 + kk) % 64) * std::ldexp(1.f, -16) : 0.f; },
                  [](int64_t, int64_t kk, int64_t j) { return kk < 2 ? 1.f + (float)j * std::ldexp(1.f, -12) : 0.f; }, plain, 1.f);
      p.opA = T;
      p.oC = COL;
      run_product("tf32 transposed A, column-major D", p, fa, fb, ab);
    }
    {
      // int8, both operands K-contiguous: A row-major op N, B row-major op T
      for (cudaDataType out : {CUDA_R_8I, CUDA_R_32I, CUDA_R_16F, CUDA_R_16BF}) {
        Problem p{CUDA_R_8I, out, CUSPARSE_COMPUTE_32I, 64, 64, 64};
        p.opB = T;
        auto ia = [](int64_t, int64_t i, int64_t kk) { return (float)((i * 7 + kk * 3) % 23) - 11.f; };
        auto ib = [](int64_t, int64_t kk, int64_t j) { return (float)((j * 5 + kk) % 19) - 9.f; };
        char name[80];
        std::snprintf(name, sizeof name, "int8 into type %d: halves round to even, saturation", (int)out);
        run_product(name, p, ia, ib, ab);
        Epilogue e = ab;
        for (int i = 0; i < 64; ++i) e.bias.push_back(0.25f * (float)i - 3.f);
        std::snprintf(name, sizeof name, "int8 into type %d with a float bias", (int)out);
        run_product(name, p, ia, ib, e);
        Epilogue av;
        av.alphaVec = true;
        av.beta = 1.f;
        for (int i = 0; i < 64; ++i) av.av.push_back(0.5f + 0.25f * (float)(i % 4));
        std::snprintf(name, sizeof name, "int8 into type %d with an alpha vector", (int)out);
        run_product(name, p, ia, ib, av);
      }
      Problem p{CUDA_R_8I, CUDA_R_8I, CUSPARSE_COMPUTE_32I, 64, 64, 64};
      p.opB = T;
      Epilogue g;
      g.alpha = 1.f / 64;
      g.gelu = true;
      g.geluScale = 10.f;
      run_product("int8 GELU with scaling", p, [](int64_t, int64_t i, int64_t kk) { return kk == 0 ? (float)(i - 32) : 0.f; },
                  [](int64_t, int64_t kk, int64_t j) { return kk == 0 ? (float)(j - 32) : 0.f; }, g, 1.f);
      g.relu = true;  // with both set, GELU applies and ReLU does not (measured)
      g.th = 0.5f;
      run_product("int8 GELU and ReLU together", p, [](int64_t, int64_t i, int64_t kk) { return kk == 0 ? (float)(i - 32) : 0.f; },
                  [](int64_t, int64_t kk, int64_t j) { return kk == 0 ? (float)(j - 32) : 0.f; }, g, 1.f);
      p.oA = COL;
      p.opA = T;
      p.opB = N;
      p.oB = COL;
      run_product("int8 transposed A column-major, B column-major", p, fa, fb, ab);
      Problem q{CUDA_R_8I, CUDA_R_32I, CUSPARSE_COMPUTE_32I, 32, 64, 64};
      q.sparseB = true;
      q.opB = T;
      run_product("int8 structured B", q, fa, fb, ab);
    }
    {
      // batches: three of everything, then one structured A for all
      Problem p{CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, 64, 32, 64};
      p.batches = 3;
      Epilogue e = ab;
      for (int i = 0; i < 3 * 64; ++i) e.bias.push_back((float)(i % 64) * 0.25f + (float)(i / 64));
      run_product("fp16 3 batches with a strided bias", p, fa, fb, e);
      p.broadcastSparse = true;
      run_product("fp16 3 batches sharing one structured A", p, fa, fb, ab);
      p.sparseB = true;
      p.m = 32;
      p.n = 64;
      run_product("fp16 3 batches sharing one structured B", p, fa, fb, ab);
    }
  }

  // ---- Matmul's argument checks, and a captured graph ----
  {
    Problem p{CUDA_R_16F, CUDA_R_16F, CUSPARSE_COMPUTE_32F, 64, 64, 64};
    Setup s(p);
    IS(s.planit(), 0);
    std::vector<uint16_t> A(64 * 64, 0), B(64 * 64, f2h(1.f)), C(64 * 64, f2h(2.f));
    for (int i = 0; i < 64; ++i) A[(size_t)(i * 64)] = f2h(1.f + (float)(i % 4));
    void* dA = upload(A);
    void* dB = upload(B);
    void* dC = upload(C);
    void* dD = zeros(C.size() * 2);
    size_t cs = 0, cbs = 0, ws = 0;
    IS(cusparseLtSpMMACompressedSize(&h, &s.plan, &cs, &cbs), 0);
    IS(cusparseLtSpMMACompressedSize(&h, &s.plan, nullptr, &cbs), 3);
    IS(cusparseLtSpMMACompressedSize(&h, &s.plan, &cs, nullptr), 3);
    void* dc = zeros(cs);
    IS(cusparseLtSpMMACompress(&h, &s.plan, dA, dc, nullptr, nullptr), 0);  // the buffer may be NULL
    IS(cusparseLtSpMMACompress(&h, &s.plan, nullptr, dc, nullptr, nullptr), 3);
    IS(cusparseLtSpMMACompress(&h, &s.plan, dA, nullptr, nullptr, nullptr), 3);
    IS(cusparseLtMatmulGetWorkspace(&h, &s.plan, &ws), 0);
    IS(cusparseLtMatmulGetWorkspace(&h, &s.plan, nullptr), 3);
    void* dws = zeros(ws);
    float alpha = 1.f, beta = 1.f, zero = 0.f;
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, &beta, dC, dD, nullptr, nullptr, 0), 0);  // no workspace
    IS(cusparseLtMatmul(&h, &s.plan, nullptr, dc, dB, &beta, dC, dD, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, nullptr, dB, &beta, dC, dD, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, nullptr, &beta, dC, dD, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, nullptr, dC, dD, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, &zero, nullptr, dD, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, &beta, dC, nullptr, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, &beta, dC, dD, dws, nullptr, -1), 3);
    IS(cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, &beta, dC, dD, dws, nullptr, 1), 3);
    IS(cusparseLtMatmul(nullptr, &s.plan, &alpha, dc, dB, &beta, dC, dD, dws, nullptr, 0), 3);
    IS(cusparseLtMatmul(&h, nullptr, &alpha, dc, dB, &beta, dC, dD, dws, nullptr, 0), 3);
    IS(cusparseLtSpMMAPrune(&h, &s.md, dA, dA, (cusparseLtPruneAlg_t)2, nullptr), 3);
    IS(cusparseLtSpMMAPrune(&h, &s.md, nullptr, dA, CUSPARSELT_PRUNE_SPMMA_STRIP, nullptr), 3);
    IS(cusparseLtSpMMAPruneCheck(&h, &s.md, dA, nullptr, nullptr), 3);
    IS(cusparseLtSpMMAPrune2(&h, &s.A, 1, (cusparseOperation_t)2, dA, dA, CUSPARSELT_PRUNE_SPMMA_STRIP, nullptr), 3);

    // A product captured into a graph reads B when the graph runs.
    cudaStream_t st;
    cudaStreamCreate(&st);
    cudaGraph_t g = nullptr;
    cudaGraphExec_t ge = nullptr;
    bool captured = cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal) == cudaSuccess;
    const int mm = cusparseLtMatmul(&h, &s.plan, &alpha, dc, dB, &zero, dC, dD, dws, &st, 1);
    captured = cudaStreamEndCapture(st, &g) == cudaSuccess && captured && mm == 0;
    captured = captured && cudaGraphInstantiate(&ge, g, 0) == cudaSuccess;
    check(captured, "Matmul captured into a CUDA graph");
    if (captured) {
      std::vector<uint16_t> B2(B.size(), f2h(3.f));
      cudaMemcpy(dB, B2.data(), B2.size() * 2, cudaMemcpyHostToDevice);
      cudaGraphLaunch(ge, st);
      cudaStreamSynchronize(st);
      const std::vector<uint16_t> D = download<uint16_t>(dD, C.size());
      bool ok = true;
      for (int i = 0; i < 64; ++i)
        for (int j = 0; j < 64; ++j) ok &= h2f(D[(size_t)(i * 64 + j)]) == 3.f * (1.f + (float)(i % 4));
      check(ok, "the graph's product used the B written after capture");
      cudaGraphExecDestroy(ge);
      cudaGraphDestroy(g);
    }
    cudaStreamDestroy(st);
    for (void* q : {dA, dB, dC, dD, dc, dws}) cudaFree(q);
  }

  IS(cusparseLtDestroy(&h), 0);
  std::printf("%s (%d failures)\n", failures ? "FAIL" : "PASS", failures);
  return failures ? 1 : 0;
}
