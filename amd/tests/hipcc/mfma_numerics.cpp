// The binary16 and bfloat16 matrix instructions with a binary32 accumulator, for the numerics tests
// (amd/tests/unit/test_cdna2_matrix.cpp): D = A x B + C through rocWMMA, 16x16x16, with C read from memory (so a test can make it a
// subnormal or a number that cancels), and the GEMM of the FTTN paper's motivating example, D = C - sum over KT tiles of A x B.
// Built for gfx90a (one v_mfma_f32_16x16x16f16 / v_mfma_f32_16x16x16bf16_1k a tile) and gfx942 (v_mfma_f32_16x16x16_f16 / _bf16).
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
template <typename In>
__device__ void mma_of(const In* a, const In* b, const float* c, float* d) {
  fragment<matrix_a, 16, 16, 16, In, row_major> fa;
  fragment<matrix_b, 16, 16, 16, In, col_major> fb;
  fragment<accumulator, 16, 16, 16, float> fc;
  load_matrix_sync(fa, a, 16);
  load_matrix_sync(fb, b, 16);
  load_matrix_sync(fc, c, 16, mem_row_major);
  mma_sync(fc, fa, fb, fc);
  store_matrix_sync(d, fc, 16, mem_row_major);
}
__global__ void mma_f16(const float16_t* a, const float16_t* b, const float* c, float* d) { mma_of<float16_t>(a, b, c, d); }
__global__ void mma_bf16(const bfloat16_t* a, const bfloat16_t* b, const float* c, float* d) { mma_of<bfloat16_t>(a, b, c, d); }
// A is 16 rows of 16 * kt (row-major), B 16 columns of 16 * kt (column-major), C and D 16 x 16 (row-major): the accumulator starts
// at zero, takes kt tiles, and D = C - accumulator.
__global__ void gemm_minus_f16(const float16_t* a, const float16_t* b, const float* c, float* d, int kt) {
  fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
  fragment<matrix_b, 16, 16, 16, float16_t, col_major> fb;
  fragment<accumulator, 16, 16, 16, float> acc, fc;
  fill_fragment(acc, 0.0f);
  for (int t = 0; t < kt; ++t) {
    load_matrix_sync(fa, a + t * 16, 16 * kt);
    load_matrix_sync(fb, b + t * 16, 16 * kt);
    mma_sync(acc, fa, fb, acc);
  }
  load_matrix_sync(fc, c, 16, mem_row_major);
  for (uint32_t i = 0; i < fc.num_elements; ++i) fc.x[i] = fc.x[i] - acc.x[i];
  store_matrix_sync(d, fc, 16, mem_row_major);
}
