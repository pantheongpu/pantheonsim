// The corners of the tensor-core sum in the subnormal range, one wmma 16x16x16 each (element (0,0); row 0 of A and column 0 of B hold
// the vector, everything else is zero), printed as the result's bits beside what each rule gives. The inputs are those of
// tests/unit/test_tc_ampere.cpp, the_subnormal_corners:
//   S1, S2   separate the floor under a zero accumulator (and, S2, the cut of a subnormal f32 result) -- on an RTX 3060: 0x0000 and
//            0x00000041 (2026-11-15); an H100's floors are one lower and its published model keeps 0x0001 and 0x00000042
//   R1..R4   the published models' answers (MMA-Sim, arXiv 2511.10909; Khattak and Mikaitis, arXiv 2512.07004): not yet on a card
// Build: nvcc -std=c++17 -O1 -cudart shared -arch=sm_<N> tc_corners.cu -o tc_corners   (-DNO80 below sm_80: the f16 cases only)
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>
using namespace nvcuda;
template <typename T, typename C>
__global__ void k(const T* a, const T* b, const C* c, C* d) {
  wmma::fragment<wmma::matrix_a, 16, 16, 16, T, wmma::row_major> fa;
  wmma::fragment<wmma::matrix_b, 16, 16, 16, T, wmma::col_major> fb;
  wmma::fragment<wmma::accumulator, 16, 16, 16, C> fc;
  wmma::load_matrix_sync(fa, a, 16);
  wmma::load_matrix_sync(fb, b, 16);
  wmma::load_matrix_sync(fc, c, 16, wmma::mem_row_major);
  wmma::mma_sync(fc, fa, fb, fc);
  wmma::store_matrix_sync(d, fc, 16, wmma::mem_row_major);
}
// a, b: up to 8 operands as the 16 bits of their format; c: the accumulator's bits (16 or 32).
template <typename T, typename C>
uint32_t run(const uint16_t* a, const uint16_t* b, int n, uint32_t c) {
  uint16_t ha[256] = {0}, hb[256] = {0};
  C hc[256], hd[256];
  std::memset(hc, 0, sizeof hc);
  std::memcpy(&hc[0], &c, sizeof(C));
  for (int i = 0; i < n; ++i) ha[i] = a[i], hb[i] = b[i];   // row 0 of A (row-major), column 0 of B (column-major)
  T *da, *db;
  C *dc, *dd;
  cudaMalloc(&da, 512); cudaMalloc(&db, 512); cudaMalloc(&dc, sizeof hc); cudaMalloc(&dd, sizeof hc);
  cudaMemcpy(da, ha, 512, cudaMemcpyHostToDevice); cudaMemcpy(db, hb, 512, cudaMemcpyHostToDevice); cudaMemcpy(dc, hc, sizeof hc, cudaMemcpyHostToDevice);
  k<T, C><<<1, 32>>>(da, db, dc, dd);
  cudaDeviceSynchronize();
  cudaMemcpy(hd, dd, sizeof hc, cudaMemcpyDeviceToHost);
  cudaFree(da); cudaFree(db); cudaFree(dc); cudaFree(dd);
  uint32_t r = 0;
  std::memcpy(&r, &hd[0], sizeof(C));
  return r;
}
static uint16_t h16(uint32_t float_bits) {   // the binary16 of a float that holds one exactly
  float f;
  std::memcpy(&f, &float_bits, 4);
  const __half h = __float2half(f);
  uint16_t u;
  std::memcpy(&u, &h, 2);
  return u;
}
int main() {
  cudaDeviceProp p;
  cudaGetDeviceProperties(&p, 0);
  std::printf("%s (sm_%d%d)\n", p.name, p.major, p.minor);
  {
    const uint16_t a[] = {0x1000, 0x0001}, b[] = {0x0400, 0x0004};
    std::printf("S1 f16 -> f16   0x%04x   (Turing/Ampere/Ada 0x0000; Hopper model 0x0001; no floor 0x0001)\n", run<__half, __half>(a, b, 2, 0));
  }
  {
    const uint32_t af[] = {0x3c87c000u, 0x3b3f0000u, 0x38650000u, 0x39afa000u, 0xb991c000u, 0x38c2a000u, 0xbbeb6000u, 0x37b08000u};
    const uint32_t bfl[] = {0x3aa6a000u, 0xbb728000u, 0xbce7e000u, 0xbac20000u, 0x3b624000u, 0xb5a80000u, 0x3a860000u, 0x3b194000u};
    uint16_t a[8], b[8];
    for (int i = 0; i < 8; ++i) a[i] = h16(af[i]), b[i] = h16(bfl[i]);
    std::printf("R4 f16 -> f16   0x%04x   (published 0x0000: +0; a sign kept would be 0x8000)\n", run<__half, __half>(a, b, 8, 0));
  }
#ifndef NO80
  {
    const uint16_t a[] = {0x1C81, 0x1A7F}, b[] = {0x1B01, 0x1A00};
    std::printf("S2 bf16 -> f32  0x%08x   (Ampere/Ada 0x00000041; Hopper model 0x00000042; no floor 0x00000042)\n", run<__nv_bfloat16, float>(a, b, 2, 0));
  }
  {
    const uint16_t a[] = {0x9d3c, 0x8d1d, 0x014a, 0x1993, 0x9c64, 0x036f, 0x94a6, 0x16c8}, b[] = {0x8e3e, 0x8162, 0x973e, 0x8433, 0xa27f, 0x095d, 0x10d8, 0xa1d8};
    std::printf("R1 bf16 -> f32  0x%08x   (published Ampere 0x006b1f35; rounded instead of cut 0x006b1f36)\n", run<__nv_bfloat16, float>(a, b, 8, 0x004ebd07u));
  }
  {
    const uint16_t a[] = {0x123e, 0x0fe8, 0x8d0b, 0x8261, 0x0ac1, 0x0b33, 0x93fd, 0x20ee}, b[] = {0x8e9a, 0x0265, 0x86eb, 0x911a, 0x126a, 0x9d2c, 0x8002, 0x0185};
    std::printf("R2 bf16 -> f32  0x%08x   (published Ampere 0x00000000; no floor 0x80000000)\n", run<__nv_bfloat16, float>(a, b, 8, 0));
  }
  {
    const uint16_t a[] = {0x836c, 0x0768, 0x182e, 0x83a3, 0x19f0, 0x8132, 0x814c, 0x0cc3}, b[] = {0x1eeb, 0x099b, 0x9eff, 0x2345, 0x850d, 0x893a, 0x21de, 0x0c3e};
    std::printf("R3 bf16 -> f32  0x%08x   (published Ampere 0x00411fcd; accumulator at its own leading bit 0x00411fcc)\n", run<__nv_bfloat16, float>(a, b, 8, 0x00411ff8u));
  }
#endif
  return 0;
}
