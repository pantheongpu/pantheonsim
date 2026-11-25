// Runs the f16, bf16 and tf32 test vectors that Khattak and Mikaitis (arXiv 2512.07004, github.com/north-numerical-computing/MATLAB-tensor-core,
// BSD-2-Clause, model_validation/) recorded on an H100, an H200 and a B200 through wgmma.mma_async on whatever device the CUDA runtime gives it (a card, or
// the simulator), and reports how many results are bit-equal to the recorded ones. Their f16, bf16 and tf32 files were recorded with wmma (mma.sync), so
// on a Hopper card a mismatch says that wgmma does NOT sum as the same card's mma.sync does; MMA-Sim (arXiv 2511.10909) says it does. The accumulator
// is the vector's c (in thread 0's first register), so the form with a non-zero accumulator is run too. Our own program (issue #664); not run on a card yet.
//
//   khattak_wgmma_vectors <f16|bf16|tf32|f16h> <file> [--show N] [--dump]
//   f16, bf16: m64n8k16, f32 accumulator, K = 16. tf32: m64n8k8, K = 8 (a file of 4 products is padded with zeros). f16h: m64n8k16, f16 accumulator.
//
// A line of <file>: a_0..a_{K-1} b_0..b_{K-1} c d (8 hex digits a token, the bits of the float32 holding the value), K = 16 (f16, bf16, f16h) or 4 (tf32).
// The vector is row 0 of A and column 0 of B, zero elsewhere. Prints  <mode> vectors=<n> mismatches=<m>  (--dump: one line a vector with the result computed
// here, and the summary on stderr).
// Build: nvcc -std=c++17 -gencode arch=compute_90a,code=[sm_90a,compute_90a] -cudart shared khattak_wgmma_vectors.cu -o khattak_wgmma_vectors   (CUDA 12.0 or later)
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

// K-major, no swizzle: 8-row core matrices of 16-byte rows; the next 16 bytes of K are 128 bytes on, the next 8 rows 256.
constexpr int kLbo = 128, kSbo = 256, kBBase = 4096;
__device__ int smem_at(int mn, int kb) { return (mn % 8) * 16 + (mn / 8) * kSbo + kb % 16 + (kb / 16) * kLbo; }
__device__ uint64_t desc(uint32_t addr) { return uint64_t((addr & 0x3FFFF) >> 4) | (uint64_t(kLbo >> 4) << 16) | (uint64_t(kSbo >> 4) << 32); }

enum Mode { F16, BF16, TF32, F16H };

template <int MODE>
__global__ void kern(const float* a, const float* b, const float* c, float* d, int N, int K) {
  __shared__ __align__(1024) uint8_t smem[8192];
  const int p = blockIdx.x; if (p >= N) return;
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem[i] = 0;
  __syncthreads();
  if (threadIdx.x < K) {
    const float x = a[(size_t)p * K + threadIdx.x], y = b[(size_t)p * K + threadIdx.x];
    if (MODE == TF32) {
      *reinterpret_cast<float*>(&smem[smem_at(0, 4 * threadIdx.x)]) = x;
      *reinterpret_cast<float*>(&smem[kBBase + smem_at(0, 4 * threadIdx.x)]) = y;
    } else if (MODE == BF16) {
      *reinterpret_cast<__nv_bfloat16*>(&smem[smem_at(0, 2 * threadIdx.x)]) = __float2bfloat16(x);
      *reinterpret_cast<__nv_bfloat16*>(&smem[kBBase + smem_at(0, 2 * threadIdx.x)]) = __float2bfloat16(y);
    } else {
      *reinterpret_cast<__half*>(&smem[smem_at(0, 2 * threadIdx.x)]) = __float2half(x);
      *reinterpret_cast<__half*>(&smem[kBBase + smem_at(0, 2 * threadIdx.x)]) = __float2half(y);
    }
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint64_t da = desc(base), db = desc(base + kBBase);
  float d0 = threadIdx.x == 0 ? c[p] : 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  uint32_t h0 = 0, h1 = 0;
  if (MODE == F16H && threadIdx.x == 0) { const __half2 hc = __floats2half2_rn(c[p], 0.f); h0 = *reinterpret_cast<const uint32_t*>(&hc); }
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  if constexpr (MODE == F16)
    asm volatile("{\n.reg .pred pz;\nsetp.ne.b32 pz, 1, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%0,%1,%2,%3}, %4, %5, pz, 1, 1, 0, 0;\n}\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db));
  if constexpr (MODE == BF16)
    asm volatile("{\n.reg .pred pz;\nsetp.ne.b32 pz, 1, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 {%0,%1,%2,%3}, %4, %5, pz, 1, 1, 0, 0;\n}\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db));
  if constexpr (MODE == TF32)
    asm volatile("{\n.reg .pred pz;\nsetp.ne.b32 pz, 1, 0;\nwgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 {%0,%1,%2,%3}, %4, %5, pz, 1, 1;\n}\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db));
  if constexpr (MODE == F16H)
    asm volatile("{\n.reg .pred pz;\nsetp.ne.b32 pz, 1, 0;\nwgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 {%0,%1}, %2, %3, pz, 1, 1, 0, 0;\n}\n"
                 : "+r"(h0), "+r"(h1) : "l"(da), "l"(db));
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  if (threadIdx.x == 0) d[p] = MODE == F16H ? __half2float(__ushort_as_half((unsigned short)(h0 & 0xffff))) : d0;   // thread 0, register 0: row 0, column 0
}

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s f16|bf16|tf32|f16h file [--show N] [--dump]\n", argv[0]); return 2; }
  const std::string m = argv[1];
  int show = 3; bool dump = false;
  for (int i = 3; i < argc; ++i) { if (!strcmp(argv[i], "--show") && i + 1 < argc) show = atoi(argv[++i]); else if (!strcmp(argv[i], "--dump")) dump = true; }
  const int mode = m == "f16" ? F16 : m == "bf16" ? BF16 : m == "tf32" ? TF32 : m == "f16h" ? F16H : -1;
  if (mode < 0) { fprintf(stderr, "bad mode\n"); return 2; }
  const int K = mode == TF32 ? 8 : 16, Kf = mode == TF32 ? 4 : 16, W = 2 * Kf + 2;   // K of the instruction, K of the file
  FILE* f = fopen(argv[2], "r"); if (!f) { perror(argv[2]); return 2; }
  std::vector<float> a, b, c, d; char tok[64]; std::vector<uint32_t> row;
  while (fscanf(f, "%63s", tok) == 1) {
    row.push_back((uint32_t)strtoul(tok, nullptr, 16));
    if ((int)row.size() == W) {
      for (int k = 0; k < K; ++k) a.push_back(k < Kf ? bits2f(row[k]) : 0.f);
      for (int k = 0; k < K; ++k) b.push_back(k < Kf ? bits2f(row[Kf + k]) : 0.f);
      c.push_back(bits2f(row[2 * Kf])); d.push_back(bits2f(row[2 * Kf + 1])); row.clear();
    }
  }
  fclose(f);
  const int N = (int)d.size();
  if (!row.empty() || N == 0) { fprintf(stderr, "file does not hold whole vectors of %d tokens\n", W); return 2; }
  float *da, *db, *dc, *dd;
  CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, N * 4)); CK(cudaMalloc(&dd, N * 4));
  CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dd, 0, N * 4));
  switch (mode) {
    case F16: kern<F16><<<N, 128>>>(da, db, dc, dd, N, K); break;
    case BF16: kern<BF16><<<N, 128>>>(da, db, dc, dd, N, K); break;
    case TF32: kern<TF32><<<N, 128>>>(da, db, dc, dd, N, K); break;
    default: kern<F16H><<<N, 128>>>(da, db, dc, dd, N, K); break;
  }
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> r(N); CK(cudaMemcpy(r.data(), dd, N * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (dump) {
      for (int k = 0; k < Kf; ++k) printf("%08x ", f2bits(a[(size_t)i * K + k]));
      for (int k = 0; k < Kf; ++k) printf("%08x ", f2bits(b[(size_t)i * K + k]));
      printf("%08x %08x\n", f2bits(c[i]), f2bits(r[i]));
    }
    if (f2bits(r[i]) == f2bits(d[i])) continue;
    if (bad < show) fprintf(stderr, "  vector %d: recorded %08x  here %08x\n", i, f2bits(d[i]), f2bits(r[i]));
    ++bad;
  }
  fprintf(dump ? stderr : stdout, "%s vectors=%d mismatches=%d\n", m.c_str(), N, bad);
  cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd);   // free what was allocated: the leak sweep (tests/lint/leak_known.txt) lists no program of this test
  return 0;
}
