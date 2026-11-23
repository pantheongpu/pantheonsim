// Runs dot-product test vectors through mma.sync.aligned.m8n8k4 (Volta's f16 form, HMMA.884) on whatever device the CUDA runtime gives it (a card,
// or the simulator), and reports how many results are bit-equal to the recorded ones. The vectors are those Khattak and Mikaitis (arXiv 2512.07004,
// github.com/north-numerical-computing/MATLAB-tensor-core, BSD-2-Clause, model_validation/V100/fp16) recorded on a V100; khattak_convert.py writes
// the one-line-a-vector files. Our own program (issue #664).
//
//   m8n8k4_vectors <fp32|fp16> <file> [--show N] [--dump]
//
// A line of <file>:  a_0..a_3 b_0..b_3 c d    (8 hex digits a token: the bits of the float32 holding the value; for fp16 the accumulator is rounded to
//                    binary16 on the way in and d is a float32 that holds a binary16 value, as khattak_vectors does). The vector is row 0 of A and column 0
//                    of B of an 8x8x4 product, every other element zero; D(0,0) is what is compared.
// Prints  m8n8k4 out=<fp32|fp16> vectors=<n> mismatches=<m>  and the first mismatches; with --dump, one line a vector: the inputs and the result
// computed here (a file of results of a card, which a test replays).
// Build: nvcc -std=c++17 -arch=sm_75 -cudart shared m8n8k4_vectors.cu -o m8n8k4_vectors      (a Volta or Turing card; CUDA 13 does not build sm_70)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

__device__ uint32_t pack(float lo, float hi) {
  const __half2 h = __floats2half2_rn(lo, hi);
  return *reinterpret_cast<const uint32_t*>(&h);
}

// One warp a vector; lane 0 holds row 0 of A, column 0 of B and C(0,0), every other lane zeros. The result is lane 0's first D register.
__global__ void k32(const float* a, const float* b, const float* c, float* d, int N) {
  const int p = blockIdx.x; if (p >= N) return;
  uint32_t a0 = 0, a1 = 0, b0 = 0, b1 = 0;
  float c0 = 0.f;
  if (threadIdx.x == 0) {
    const float* x = a + (size_t)p * 4; const float* y = b + (size_t)p * 4;
    a0 = pack(x[0], x[1]); a1 = pack(x[2], x[3]); b0 = pack(y[0], y[1]); b1 = pack(y[2], y[3]); c0 = c[p];
  }
  float d0, d1, d2, d3, d4, d5, d6, d7;
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, {%12,%13,%14,%15,%16,%17,%18,%19};"
               : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3), "=f"(d4), "=f"(d5), "=f"(d6), "=f"(d7)
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "f"(c0), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f), "f"(0.f));
  if (threadIdx.x == 0) d[p] = d0;
}
__global__ void k16(const float* a, const float* b, const float* c, float* d, int N) {
  const int p = blockIdx.x; if (p >= N) return;
  uint32_t a0 = 0, a1 = 0, b0 = 0, b1 = 0, c0 = 0;
  if (threadIdx.x == 0) {
    const float* x = a + (size_t)p * 4; const float* y = b + (size_t)p * 4;
    a0 = pack(x[0], x[1]); a1 = pack(x[2], x[3]); b0 = pack(y[0], y[1]); b1 = pack(y[2], y[3]); c0 = pack(c[p], 0.f);
  }
  uint32_t d0, d1, d2, d3;
  asm volatile("mma.sync.aligned.m8n8k4.row.col.f16.f16.f16.f16 {%0,%1,%2,%3}, {%4,%5}, {%6,%7}, {%8,%9,%10,%11};"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3)
               : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(c0), "r"(0u), "r"(0u), "r"(0u));
  if (threadIdx.x == 0) { const __half h = __ushort_as_half((unsigned short)(d0 & 0xffff)); d[p] = __half2float(h); }
}

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  if (argc < 3) { fprintf(stderr, "usage: %s fp32|fp16 file [--show N] [--dump]\n", argv[0]); return 2; }
  const std::string out = argv[1];
  int show = 3; bool dump = false;
  for (int i = 3; i < argc; ++i) { if (!strcmp(argv[i], "--show") && i + 1 < argc) show = atoi(argv[++i]); else if (!strcmp(argv[i], "--dump")) dump = true; }
  if (out != "fp32" && out != "fp16") { fprintf(stderr, "bad output format\n"); return 2; }
  FILE* f = fopen(argv[2], "r"); if (!f) { perror(argv[2]); return 2; }
  std::vector<float> a, b, c, d; char tok[64]; std::vector<uint32_t> row;
  while (fscanf(f, "%63s", tok) == 1) {
    row.push_back((uint32_t)strtoul(tok, nullptr, 16));
    if (row.size() == 10) {
      for (int k = 0; k < 4; ++k) a.push_back(bits2f(row[k]));
      for (int k = 0; k < 4; ++k) b.push_back(bits2f(row[4 + k]));
      c.push_back(bits2f(row[8])); d.push_back(bits2f(row[9])); row.clear();
    }
  }
  fclose(f);
  const int N = (int)d.size();
  if (!row.empty() || N == 0) { fprintf(stderr, "file does not hold whole vectors of 10 tokens\n"); return 2; }
  float *da, *db, *dc, *dd;
  CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, N * 4)); CK(cudaMalloc(&dd, N * 4));
  CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dd, 0, N * 4));
  if (out == "fp16") k16<<<N, 32>>>(da, db, dc, dd, N); else k32<<<N, 32>>>(da, db, dc, dd, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> r(N); CK(cudaMemcpy(r.data(), dd, N * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (dump) { for (int k = 0; k < 4; ++k) printf("%08x ", f2bits(a[i * 4 + k])); for (int k = 0; k < 4; ++k) printf("%08x ", f2bits(b[i * 4 + k])); printf("%08x %08x\n", f2bits(c[i]), f2bits(r[i])); }
    if (f2bits(r[i]) == f2bits(d[i])) continue;
    if (bad < show) fprintf(stderr, "  vector %d: recorded %08x  here %08x\n", i, f2bits(d[i]), f2bits(r[i]));
    ++bad;
  }
  printf("m8n8k4 out=%s vectors=%d mismatches=%d\n", out.c_str(), N, bad);
  return 0;
}
