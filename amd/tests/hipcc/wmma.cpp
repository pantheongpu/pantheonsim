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
