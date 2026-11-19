// Runs the e4m3 / e5m2 f32-accumulator test vectors that Khattak and Mikaitis (arXiv 2512.07004, github.com/north-numerical-computing/
// MATLAB-tensor-core, BSD-2-Clause, model_validation/<H100|H200>/E4M3 and E5M2) recorded through wgmma, through wgmma.mma_async
// m64n8k32 on whatever device the CUDA runtime gives it (a card, or the simulator), and reports how many results are bit-equal to the
// recorded ones. Our own program, as khattak_fp8_vectors.cu is for mma.sync; khattak_convert.py writes the files.
//
//   khattak_fp8_wgmma_vectors <e4m3|e5m2> <file> [--show N]
//
// A line of <file>:   a_0 .. a_31  b_0 .. b_31  c  d     (8 hex digits a token: the bits of the float32 holding the value). The vector is
//                     row 0 of A and column 0 of B of a m64n8k32, zero elsewhere. C is zero: the paper's H100 and H200 sets were recorded
//                     through wgmma with a zero accumulator, so the c column is not an input.
// Prints  <fmt> vectors=<n> mismatches=<m>  and the first mismatches.
// Build: nvcc -std=c++17 -arch=sm_90a -cudart shared khattak_fp8_wgmma_vectors.cu -o khattak_fp8_wgmma_vectors      (CUDA 12.0 or later)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp8.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

// K-major, no swizzle: 8-row core matrices of 16-byte rows; the next 16 bytes of K are 128 bytes on, the next 8 rows 256.
constexpr int kLbo = 128, kSbo = 256, kBBase = 4096;
__device__ int smem_at(int mn, int k) { return (mn % 8) * 16 + (mn / 8) * kSbo + k % 16 + (k / 16) * kLbo; }
__device__ uint64_t desc(uint32_t addr) { return uint64_t((addr & 0x3FFFF) >> 4) | (uint64_t(kLbo >> 4) << 16) | (uint64_t(kSbo >> 4) << 32); }

template <bool E5> __device__ uint8_t cvt(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, E5 ? __NV_E5M2 : __NV_E4M3);
}

template <bool E5>
__global__ void kfp8(const float* a, const float* b, float* d, int N) {
  __shared__ __align__(1024) uint8_t smem[8192];
  const int p = blockIdx.x; if (p >= N) return;
  for (int i = threadIdx.x; i < 8192; i += blockDim.x) smem[i] = 0;
  __syncthreads();
  if (threadIdx.x < 32) {
    smem[smem_at(0, threadIdx.x)] = cvt<E5>(a[(size_t)p * 32 + threadIdx.x]);
    smem[kBBase + smem_at(0, threadIdx.x)] = cvt<E5>(b[(size_t)p * 32 + threadIdx.x]);
  }
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  const uint32_t base = (uint32_t)__cvta_generic_to_shared(smem);
  const uint64_t da = desc(base), db = desc(base + kBBase);
  float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f;
  asm volatile("wgmma.fence.sync.aligned;" ::: "memory");
  if (E5)
    asm volatile("{\n.reg .pred pz;\nsetp.ne.b32 pz, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e5m2 {%0,%1,%2,%3}, %4, %5, pz, 1, 1;\n}\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db));
  else
    asm volatile("{\n.reg .pred pz;\nsetp.ne.b32 pz, 1, 0;\n"
                 "wgmma.mma_async.sync.aligned.m64n8k32.f32.e4m3.e4m3 {%0,%1,%2,%3}, %4, %5, pz, 1, 1;\n}\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "l"(da), "l"(db));
  asm volatile("wgmma.commit_group.sync.aligned;" ::: "memory");
  asm volatile("wgmma.wait_group.sync.aligned 0;" ::: "memory");
  if (threadIdx.x == 0) d[p] = d0;   // thread 0, register 0: row 0, column 0 of the first warp's rows
}

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s e4m3|e5m2 file [--show N]\n", argv[0]); return 2; }
  std::string fmt = argv[1];
  int show = 3;
  for (int i = 3; i < argc; ++i) if (!strcmp(argv[i], "--show") && i + 1 < argc) show = atoi(argv[++i]);
  if (fmt != "e4m3" && fmt != "e5m2") { fprintf(stderr, "bad format\n"); return 2; }
  FILE* f = fopen(argv[2], "r"); if (!f) { perror(argv[2]); return 2; }
  std::vector<float> a, b, d; char tok[64]; std::vector<uint32_t> row;
  while (fscanf(f, "%63s", tok) == 1) {
    row.push_back((uint32_t)strtoul(tok, nullptr, 16));
    if (row.size() == 66) {
      for (int k = 0; k < 32; ++k) a.push_back(bits2f(row[k]));
      for (int k = 0; k < 32; ++k) b.push_back(bits2f(row[32 + k]));
      d.push_back(bits2f(row[65])); row.clear();
    }
  }
  fclose(f);
  int N = (int)d.size();
  if (!row.empty() || N == 0) { fprintf(stderr, "file does not hold whole vectors of 66 tokens\n"); return 2; }
  float *da, *db, *dd; CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dd, N * 4));
  CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dd, 0, N * 4));
  if (fmt == "e5m2") kfp8<true><<<N, 128>>>(da, db, dd, N); else kfp8<false><<<N, 128>>>(da, db, dd, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> r(N); CK(cudaMemcpy(r.data(), dd, N * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (f2bits(r[i]) == f2bits(d[i])) continue;
    if (bad < show) printf("  vector %d: recorded %08x  here %08x\n", i, f2bits(d[i]), f2bits(r[i]));
    ++bad;
  }
  printf("%s vectors=%d mismatches=%d\n", fmt.c_str(), N, bad);
  return 0;
}
