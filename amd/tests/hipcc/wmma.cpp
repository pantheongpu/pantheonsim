// A 16x16x16 GEMM through rocWMMA, AMD's own fragment library: its loads and
// stores place each element where the hardware expects it, which is what
// checks the simulator's matrix instruction against the real layout.
#include <hip/hip_runtime.h>
#include <rocwmma/rocwmma.hpp>
using namespace rocwmma;
__global__ void gemm(const float16_t* a, const float16_t* b, const float* c, float* d) {
  fragment<matrix_a, 16, 16, 16, float16_t, row_major> fa;
  fragment<matrix_b, 16, 16, 16, float16_t, col_major> fb;
  fragment<accumulator, 16, 16, 16, float> fc;
  load_matrix_sync(fa, a, 16);
  load_matrix_sync(fb, b, 16);
  load_matrix_sync(fc, c, 16, mem_row_major);
  mma_sync(fc, fa, fb, fc);
  store_matrix_sync(d, fc, 16, mem_row_major);
}

// The same through the float and double matrix instructions Tensile's GEMMs
// use: a block of M rows and columns, K deep -- several of the instruction's
// own steps of K, which rocWMMA chains -- A row-major, B column-major and C
// and D row-major.
template <typename T, int M, int K>
__device__ void gemm_of(const T* a, const T* b, const T* c, T* d) {
  fragment<matrix_a, M, M, K, T, row_major> fa;
  fragment<matrix_b, M, M, K, T, col_major> fb;
  fragment<accumulator, M, M, K, T> fc;
  load_matrix_sync(fa, a, K);
  load_matrix_sync(fb, b, K);
  load_matrix_sync(fc, c, M, mem_row_major);
  mma_sync(fc, fa, fb, fc);
  store_matrix_sync(d, fc, M, mem_row_major);
}
__global__ void gemm_f32_16x16x16(const float* a, const float* b, const float* c, float* d) {
  gemm_of<float, 16, 16>(a, b, c, d);
}
__global__ void gemm_f32_32x32x8(const float* a, const float* b, const float* c, float* d) {
  gemm_of<float, 32, 8>(a, b, c, d);
}
__global__ void gemm_f64_16x16x16(const double* a, const double* b, const double* c, double* d) {
  gemm_of<double, 16, 16>(a, b, c, d);
}

// Halves, bfloat16s and bytes into float and int32 accumulators: the matrix
// instructions the half, bfloat16 and int8 GEMMs use.
template <typename In, typename Acc, int M, int K>
__device__ void gemm_mixed(const In* a, const In* b, const Acc* c, Acc* d) {
  fragment<matrix_a, M, M, K, In, row_major> fa;
  fragment<matrix_b, M, M, K, In, col_major> fb;
  fragment<accumulator, M, M, K, Acc> fc;
  load_matrix_sync(fa, a, K);
  load_matrix_sync(fb, b, K);
  load_matrix_sync(fc, c, M, mem_row_major);
  mma_sync(fc, fa, fb, fc);
  store_matrix_sync(d, fc, M, mem_row_major);
}
__global__ void gemm_bf16_16x16x16(const bfloat16_t* a, const bfloat16_t* b, const float* c, float* d) {
  gemm_mixed<bfloat16_t, float, 16, 16>(a, b, c, d);
}
__global__ void gemm_bf16_32x32x16(const bfloat16_t* a, const bfloat16_t* b, const float* c, float* d) {
  gemm_mixed<bfloat16_t, float, 32, 16>(a, b, c, d);
}
__global__ void gemm_f16_32x32x16(const float16_t* a, const float16_t* b, const float* c, float* d) {
  gemm_mixed<float16_t, float, 32, 16>(a, b, c, d);
}
__global__ void gemm_i8_32x32x16(const int8_t* a, const int8_t* b, const int32_t* c, int32_t* d) {
  gemm_mixed<int8_t, int32_t, 32, 16>(a, b, c, d);
}
