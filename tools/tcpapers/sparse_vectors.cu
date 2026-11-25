// Runs dot-product vectors through the sparse tensor-core forms (mma.sp.sync.aligned, 2:4 for f16 and bf16, 1:2 for tf32; HMMA.SP in SASS) on whatever
// device the CUDA runtime gives it (a card, or the simulator), and either prints the results (--dump: a file a card made, which a test replays) or
// compares them with the d column of the file. Our own program (issue #664).
//
//   sparse_vectors <mode> <file> [--show N] [--dump]
//   mode: f16k16 f16k32 bf16k16 bf16k32 tf32k8 tf32k16 (the shape m16n8kK, K the dense length; f32 accumulator) and f16k32h, f16k16h (f16 accumulator)
//
// A line of <file>:  s_0..s_{S-1}  b_0..b_{K-1}  pattern  c  d     (8 hex digits a token: the bits of the float32 holding the value; S = K/2 stored
//                    values of A's row 0; the pattern is a number, below). The vector is row 0 of A and column 0 of B of a 16x8xK product, every other
//                    element zero. In every group of 4 dense columns (of 2 for tf32) the stored values sit at the positions the pattern names, and every
//                    lane gets the same metadata word, so no fragment-to-lane mapping of the metadata enters: f16 and bf16 patterns 0..5 are the pairs of
//                    positions (0,1) (0,2) (0,3) (1,2) (1,3) (2,3), tf32 patterns 0 and 1 are position 0 and 1.
// Prints  sparse <mode> vectors=<n> mismatches=<m>.
// Build: nvcc -std=c++17 -arch=sm_86 -cudart shared sparse_vectors.cu -o sparse_vectors      (sm_80 or newer)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

__device__ uint32_t pack_h(float lo, float hi) { const __half2 h = __floats2half2_rn(lo, hi); return *reinterpret_cast<const uint32_t*>(&h); }
__device__ uint32_t pack_b(float lo, float hi) { const __nv_bfloat162 h = __floats2bfloat162_rn(lo, hi); return *reinterpret_cast<const uint32_t*>(&h); }
__device__ uint32_t meta_f16(int pat) { static const uint32_t n[6] = {0x4, 0x8, 0xC, 0x9, 0xD, 0xE}; uint32_t v = n[pat]; return v * 0x11111111u; }
__device__ uint32_t meta_tf32(int pat) { return (pat ? 0xEu : 0x4u) * 0x11111111u; }

enum Mode { F16K16, F16K32, BF16K16, BF16K32, TF32K8, TF32K16, F16K32H, F16K16H };

// One warp a vector. Lane t (0..3) of row 0 holds stored values and B's column 0; every lane the metadata. D(0,0) is lane 0's first register.
template <int MODE>
__global__ void kern(const float* s, const float* b, const int* pat, const float* c, float* d, int N, int S, int K) {
  const int p = blockIdx.x; if (p >= N) return;
  const int t = threadIdx.x;
  const float* sv = s + (size_t)p * S; const float* bv = b + (size_t)p * K;
  float cc = t == 0 ? c[p] : 0.f;
  uint32_t a[4] = {0, 0, 0, 0}, bb[4] = {0, 0, 0, 0};
  const bool lead = t < 4;
  constexpr bool H = MODE == F16K16 || MODE == F16K32 || MODE == F16K32H || MODE == F16K16H;
  constexpr bool BF = MODE == BF16K16 || MODE == BF16K32;
  constexpr bool T = MODE == TF32K8 || MODE == TF32K16;
  constexpr int NA = (MODE == F16K32 || MODE == BF16K32 || MODE == F16K32H || MODE == TF32K16) ? 4 : 2;   // registers of A and of B
  if (lead && (H || BF)) {
    auto pk = [&](float x, float y) { return BF ? pack_b(x, y) : pack_h(x, y); };
    a[0] = pk(sv[2 * t], sv[2 * t + 1]);
    if (NA == 4) a[2] = pk(sv[8 + 2 * t], sv[9 + 2 * t]);
    for (int r = 0; r < NA; ++r) bb[r] = pk(bv[8 * r + 2 * t], bv[8 * r + 2 * t + 1]);
  }
  if (lead && T) {
    a[0] = __float_as_uint(sv[t]);
    if (NA == 4) a[2] = __float_as_uint(sv[t + 4]);
    for (int r = 0; r < NA; ++r) bb[r] = __float_as_uint(bv[t + 4 * r]);
  }
  const uint32_t meta = T ? meta_tf32(pat[p]) : meta_f16(pat[p]);
  float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
  uint32_t h0 = 0, h1 = 0;
  if constexpr (MODE == F16K32)
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]), "r"(bb[2]), "r"(bb[3]), "f"(cc), "f"(0.f), "f"(0.f), "f"(0.f), "r"(meta));
  if constexpr (MODE == BF16K32)
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]), "r"(bb[2]), "r"(bb[3]), "f"(cc), "f"(0.f), "f"(0.f), "f"(0.f), "r"(meta));
  if constexpr (MODE == F16K16)
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a[0]), "r"(a[1]), "r"(bb[0]), "r"(bb[1]), "f"(cc), "f"(0.f), "f"(0.f), "f"(0.f), "r"(meta));
  if constexpr (MODE == BF16K16)
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a[0]), "r"(a[1]), "r"(bb[0]), "r"(bb[1]), "f"(cc), "f"(0.f), "f"(0.f), "f"(0.f), "r"(meta));
  if constexpr (MODE == TF32K8)
    asm volatile("mma.sp.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11}, %12, 0x0;"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a[0]), "r"(a[1]), "r"(bb[0]), "r"(bb[1]), "f"(cc), "f"(0.f), "f"(0.f), "f"(0.f), "r"(meta));
  if constexpr (MODE == TF32K16)
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f32.tf32.tf32.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9,%10,%11}, {%12,%13,%14,%15}, %16, 0x0;"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]), "r"(bb[2]), "r"(bb[3]), "f"(cc), "f"(0.f), "f"(0.f), "f"(0.f), "r"(meta));
  if constexpr (MODE == F16K32H) {
    const uint32_t c0 = t == 0 ? pack_h(c[p], 0.f) : 0u;
    asm volatile("mma.sp.sync.aligned.m16n8k32.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7,%8,%9}, {%10,%11}, %12, 0x0;"
                 : "=r"(h0), "=r"(h1) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bb[0]), "r"(bb[1]), "r"(bb[2]), "r"(bb[3]), "r"(c0), "r"(0u), "r"(meta));
  }
  if constexpr (MODE == F16K16H) {
    const uint32_t c0 = t == 0 ? pack_h(c[p], 0.f) : 0u;
    asm volatile("mma.sp.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3}, {%4,%5}, {%6,%7}, %8, 0x0;"
                 : "=r"(h0), "=r"(h1) : "r"(a[0]), "r"(a[1]), "r"(bb[0]), "r"(bb[1]), "r"(c0), "r"(0u), "r"(meta));
  }
  if (t == 0) d[p] = (MODE == F16K32H || MODE == F16K16H) ? __half2float(__ushort_as_half((unsigned short)(h0 & 0xffff))) : d0;
}

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s mode file [--show N] [--dump]\n", argv[0]); return 2; }
  const std::string m = argv[1];
  int show = 3; bool dump = false;
  for (int i = 3; i < argc; ++i) { if (!strcmp(argv[i], "--show") && i + 1 < argc) show = atoi(argv[++i]); else if (!strcmp(argv[i], "--dump")) dump = true; }
  struct M { const char* n; int mode, K; } modes[] = {{"f16k16", F16K16, 16}, {"f16k32", F16K32, 32}, {"bf16k16", BF16K16, 16}, {"bf16k32", BF16K32, 32},
                                                     {"tf32k8", TF32K8, 8}, {"tf32k16", TF32K16, 16}, {"f16k32h", F16K32H, 32}, {"f16k16h", F16K16H, 16}};
  int mode = -1, K = 0;
  for (auto& x : modes) if (m == x.n) { mode = x.mode; K = x.K; }
  if (mode < 0) { fprintf(stderr, "bad mode\n"); return 2; }
  const int S = K / 2, W = S + K + 3;
  FILE* f = fopen(argv[2], "r"); if (!f) { perror(argv[2]); return 2; }
  std::vector<float> s, b, c, d; std::vector<int> pat; char tok[64]; std::vector<uint32_t> row;
  while (fscanf(f, "%63s", tok) == 1) {
    row.push_back((uint32_t)strtoul(tok, nullptr, 16));
    if ((int)row.size() == W) {
      for (int k = 0; k < S; ++k) s.push_back(bits2f(row[k]));
      for (int k = 0; k < K; ++k) b.push_back(bits2f(row[S + k]));
      pat.push_back((int)row[S + K]); c.push_back(bits2f(row[S + K + 1])); d.push_back(bits2f(row[S + K + 2])); row.clear();
    }
  }
  fclose(f);
  const int N = (int)d.size();
  if (!row.empty() || N == 0) { fprintf(stderr, "file does not hold whole vectors of %d tokens\n", W); return 2; }
  float *ds, *db, *dc, *dd; int* dp;
  CK(cudaMalloc(&ds, s.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, N * 4)); CK(cudaMalloc(&dd, N * 4)); CK(cudaMalloc(&dp, N * 4));
  CK(cudaMemcpy(ds, s.data(), s.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dp, pat.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dd, 0, N * 4));
  switch (mode) {
    case F16K16: kern<F16K16><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    case F16K32: kern<F16K32><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    case BF16K16: kern<BF16K16><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    case BF16K32: kern<BF16K32><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    case TF32K8: kern<TF32K8><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    case TF32K16: kern<TF32K16><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    case F16K32H: kern<F16K32H><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
    default: kern<F16K16H><<<N, 32>>>(ds, db, dp, dc, dd, N, S, K); break;
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> r(N); CK(cudaMemcpy(r.data(), dd, N * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (dump) {
      for (int k = 0; k < S; ++k) printf("%08x ", f2bits(s[(size_t)i * S + k]));
      for (int k = 0; k < K; ++k) printf("%08x ", f2bits(b[(size_t)i * K + k]));
      printf("%08x %08x %08x\n", (unsigned)pat[i], f2bits(c[i]), f2bits(r[i]));
    }
    if (f2bits(r[i]) == f2bits(d[i])) continue;
    if (bad < show) fprintf(stderr, "  vector %d: recorded %08x  here %08x\n", i, f2bits(d[i]), f2bits(r[i]));
    ++bad;
  }
  fprintf(dump ? stderr : stdout, "sparse %s vectors=%d mismatches=%d\n", m.c_str(), N, bad);
  cudaFree(ds); cudaFree(db); cudaFree(dc); cudaFree(dd); cudaFree(dp);   // free what was allocated: the leak sweep (tests/lint/leak_known.txt) lists no program of this test
  return 0;
}
