// Inputs that tell truncation from round-to-zero in a tensor core's accumulation, and show that the order of the products does not matter
// (Valpey, Li, Pai, Gopalakrishnan, arXiv 2502.15999, section 4.2 and table 3; properties 3 and 4). Our own program, not the paper's.
// wmma 16x16x16, f16 inputs, f32 accumulator, one product (or eight) in row 0 of A and column 0 of B. One line a case, result bits in hex:
//
//     <name> <bits>
//
// Build: nvcc -std=c++17 -arch=sm_<NN> -cudart shared valpey_probe.cu -o valpey_probe
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); exit(2); } } while (0)

__global__ void k(const half* a, const half* b, float* c) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> fb;   // column 0 is b[0..15]
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> fc;
  wmma::load_matrix_sync(fa, a, 16); wmma::load_matrix_sync(fb, b, 16); wmma::load_matrix_sync(fc, c, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(c, fc, 16, wmma::mem_row_major);
}

static half *dA, *dB; static float* dC;
// d = a . b + c for the first n products (the rest zero)
static uint32_t run(const float* a, const float* b, int n, float c) {
  half hA[256], hB[256]; float hC[256];
  for (int i = 0; i < 256; ++i) { hA[i] = __float2half(0.f); hB[i] = __float2half(0.f); hC[i] = 0.f; }
  for (int i = 0; i < n; ++i) { hA[i] = __float2half(a[i]); hB[i] = __float2half(b[i]); }
  hC[0] = c;
  CK(cudaMemcpy(dA, hA, sizeof hA, cudaMemcpyHostToDevice)); CK(cudaMemcpy(dB, hB, sizeof hB, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dC, hC, sizeof hC, cudaMemcpyHostToDevice));
  k<<<1, 32>>>(dA, dB, dC); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hC, dC, sizeof hC, cudaMemcpyDeviceToHost));
  uint32_t r; memcpy(&r, &hC[0], 4); return r;
}

int main() {
  CK(cudaMalloc(&dA, 512)); CK(cudaMalloc(&dB, 512)); CK(cudaMalloc(&dC, 1024));
  // one product p = a * 1 and an accumulator c: the effective subtraction of table 3 (a = 2, c = -2^-40), and its neighbours
  struct { const char* name; float a, c; } one[] = {
    {"p2_c-2^-40", 2.f, -ldexpf(1, -40)}, {"p2_c-2^-26", 2.f, -ldexpf(1, -26)}, {"p2_c-2^-25", 2.f, -ldexpf(1, -25)},
    {"p2_c-2^-24", 2.f, -ldexpf(1, -24)}, {"p2_c-1.5*2^-24", 2.f, -1.5f * ldexpf(1, -24)}, {"p2_c-2^-23", 2.f, -ldexpf(1, -23)},
    {"p2_c+2^-25", 2.f, ldexpf(1, -25)}, {"p2_c+2^-24", 2.f, ldexpf(1, -24)}, {"p-2_c+2^-25", -2.f, ldexpf(1, -25)},
    {"p-2_c+2^-24", -2.f, ldexpf(1, -24)}, {"p1_c-2^-26", 1.f, -ldexpf(1, -26)}, {"p1_c-2^-25", 1.f, -ldexpf(1, -25)},
    {"p1_c-2^-24", 1.f, -ldexpf(1, -24)}, {"p1_c-3*2^-25", 1.f, -3.f * ldexpf(1, -25)},
    {"p1.5_c-2^-25", 1.5f, -ldexpf(1, -25)}, {"p1.5_c-2^-24", 1.5f, -ldexpf(1, -24)},
  };
  const float one_b = 1.f;
  for (auto& t : one) printf("%s %08x\n", t.name, run(&t.a, &one_b, 1, t.c));
  // eight products of very different size, forward and in reverse: the order must not matter (property 3), and the sum is not normalised
  // between terms (property 4: 1 + 2^-24 + 2^-24 is 1 + 2^-23 here, not 1)
  float f[8] = {1.f, ldexpf(1, -11), ldexpf(1, -12), ldexpf(1, -13), ldexpf(1, -24), -ldexpf(1, -24), ldexpf(1, -25), 1.5f * ldexpf(1, -26)};
  float r[8]; for (int i = 0; i < 8; ++i) r[i] = f[7 - i];
  float ones[8] = {1, 1, 1, 1, 1, 1, 1, 1};
  printf("eight_forward %08x\n", run(f, ones, 8, 0.f));
  printf("eight_reverse %08x\n", run(r, ones, 8, 0.f));
  float g[3] = {1.f, ldexpf(1, -24), ldexpf(1, -24)}, g1[3] = {1, 1, 1};
  printf("one_plus_two_halves_ulp %08x\n", run(g, g1, 3, 0.f));
  float h[3] = {ldexpf(1, -24), ldexpf(1, -24), 1.f};
  printf("one_plus_two_halves_ulp_reverse %08x\n", run(h, g1, 3, 0.f));
  return 0;
}
