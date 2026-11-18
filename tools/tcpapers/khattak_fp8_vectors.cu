// Runs the FP8 test vectors of Khattak and Mikaitis (arXiv 2512.07004, github.com/north-numerical-computing/MATLAB-tensor-core,
// BSD-2-Clause, model_validation/<card>/E4M3 and E5M2) through mma.sync m16n8k32 on whatever device the CUDA runtime gives it (a card,
// or the simulator) and reports how many results are bit-equal to the recorded ones. Our own program, as khattak_vectors.cu is for
// the wmma forms; khattak_convert.py writes the files.
//
//   khattak_fp8_vectors <e4m3|e5m2> <fp32|fp16> <file> [--show N] [--zero-c]
//
// A line of <file>:   a_0 .. a_31  b_0 .. b_31  c  d     (8 hex digits a token: the bits of the float32 holding the value; the a and b
//                     values are exactly representable in the format). The vector is row 0 of A and column 0 of B of a m16n8k32, zero
//                     elsewhere, c in element (0, 0) of C. --zero-c puts 0 there instead (the paper's H100 and H200 sets were recorded
//                     through wgmma with a zero accumulator, so their c column is not an input).
// Prints  <fmt> out=<out> vectors=<n> mismatches=<m>  and the first mismatches.
// Build: nvcc -std=c++17 -arch=sm_<NN> -cudart shared khattak_fp8_vectors.cu -o khattak_fp8_vectors      (sm_89 or newer)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

template <bool E5> __device__ uint32_t cvt(float x) {
  return __nv_cvt_float_to_fp8(x, __NV_SATFINITE, E5 ? __NV_E5M2 : __NV_E4M3);
}
// four consecutive elements of a vector, packed little-endian
template <bool E5> __device__ uint32_t pack4(const float* v) {
  return cvt<E5>(v[0]) | (cvt<E5>(v[1]) << 8) | (cvt<E5>(v[2]) << 16) | (cvt<E5>(v[3]) << 24);
}

template <bool E5, bool F16OUT>
__global__ void kfp8(const float* a, const float* b, const float* c, float* d, int N) {
  int p = blockIdx.x; if (p >= N) return;
  int lane = threadIdx.x, t = lane & 3, g = lane >> 2;
  const float* va = a + (size_t)p * 32; const float* vb = b + (size_t)p * 32;
  uint32_t a0 = 0, a1 = 0, a2 = 0, a3 = 0, b0 = 0, b1 = 0;
  if (g == 0) {                        // row 0 of A and column 0 of B live in lanes 0..3
    a0 = pack4<E5>(va + 4 * t); a2 = pack4<E5>(va + 16 + 4 * t);
    b0 = pack4<E5>(vb + 4 * t); b1 = pack4<E5>(vb + 16 + 4 * t);
  }
  if (!F16OUT) {
    float c0 = (lane == 0) ? c[p] : 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, d0, d1, d2, d3;
    if (E5)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                   : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    else
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                   : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                   : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
    if (lane == 0) d[p] = d0;
  } else {
    __half h = __float2half(lane == 0 ? c[p] : 0.f);
    uint32_t c0 = lane == 0 ? (uint32_t)__half_as_ushort(h) : 0u, c1 = 0u, d0, d1;
    if (E5)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e5m2.e5m2.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                   : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    else
      asm volatile("mma.sync.aligned.m16n8k32.row.col.f16.e4m3.e4m3.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%8,%9};\n"
                   : "=r"(d0), "=r"(d1) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "r"(c0), "r"(c1));
    if (lane == 0) d[p] = __half2float(__ushort_as_half((unsigned short)(d0 & 0xffff)));
  }
}

static float bits2f(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static uint32_t f2bits(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }

int main(int argc, char** argv) {
  if (argc < 4) { fprintf(stderr, "usage: %s e4m3|e5m2 fp32|fp16 file [--show N] [--zero-c]\n", argv[0]); return 2; }
  std::string fmt = argv[1], out = argv[2];
  int show = 3; bool zero_c = false;
  for (int i = 4; i < argc; ++i) { if (!strcmp(argv[i], "--show") && i + 1 < argc) show = atoi(argv[++i]); else if (!strcmp(argv[i], "--zero-c")) zero_c = true; }
  if ((fmt != "e4m3" && fmt != "e5m2") || (out != "fp32" && out != "fp16")) { fprintf(stderr, "bad format\n"); return 2; }
  FILE* f = fopen(argv[3], "r"); if (!f) { perror(argv[3]); return 2; }
  std::vector<float> a, b, c, d; char tok[64]; std::vector<uint32_t> row;
  while (fscanf(f, "%63s", tok) == 1) {
    row.push_back((uint32_t)strtoul(tok, nullptr, 16));
    if (row.size() == 66) {
      for (int k = 0; k < 32; ++k) a.push_back(bits2f(row[k]));
      for (int k = 0; k < 32; ++k) b.push_back(bits2f(row[32 + k]));
      c.push_back(zero_c ? 0.f : bits2f(row[64])); d.push_back(bits2f(row[65])); row.clear();
    }
  }
  fclose(f);
  int N = (int)d.size();
  if (!row.empty() || N == 0) { fprintf(stderr, "file does not hold whole vectors of 66 tokens\n"); return 2; }
  float *da, *db, *dc, *dd; CK(cudaMalloc(&da, a.size() * 4)); CK(cudaMalloc(&db, b.size() * 4)); CK(cudaMalloc(&dc, N * 4)); CK(cudaMalloc(&dd, N * 4));
  CK(cudaMemcpy(da, a.data(), a.size() * 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(db, b.data(), b.size() * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dc, c.data(), N * 4, cudaMemcpyHostToDevice)); CK(cudaMemset(dd, 0, N * 4));
  const bool e5 = fmt == "e5m2", h = out == "fp16";
  if (!e5 && !h) kfp8<false, false><<<N, 32>>>(da, db, dc, dd, N);
  else if (!e5 && h) kfp8<false, true><<<N, 32>>>(da, db, dc, dd, N);
  else if (e5 && !h) kfp8<true, false><<<N, 32>>>(da, db, dc, dd, N);
  else kfp8<true, true><<<N, 32>>>(da, db, dc, dd, N);
  CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  std::vector<float> r(N); CK(cudaMemcpy(r.data(), dd, N * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < N; ++i) {
    if (f2bits(r[i]) == f2bits(d[i])) continue;
    if (bad < show) printf("  vector %d: recorded %08x  here %08x\n", i, f2bits(d[i]), f2bits(r[i]));
    ++bad;
  }
  printf("%s out=%s vectors=%d mismatches=%d\n", fmt.c_str(), out.c_str(), N, bad);
  return 0;
}
