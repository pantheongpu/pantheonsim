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
