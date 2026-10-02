// A TMA store whose coordinates differ from lane to lane. UTMASTG takes its
// coordinates from uniform registers, so ptxas loops: R2UR (and R2UR.OR, for
// each further coordinate) take one lane's values and mark, per lane, whether
// its own differ; the store runs for them; the lanes that differed go round
// again (a "waterfall"). Every lane's tile must land where its coordinates say.
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>

constexpr int kW = 64, kH = 16, kBox = 8;   // a 64 x 16 int matrix, 8 x 8 boxes

__global__ void scatter_tiles(const __grid_constant__ CUtensorMap map) {
  __shared__ alignas(128) int tile[kBox * kBox];
  const int lane = threadIdx.x;
  for (int i = lane; i < kBox * kBox; i += 32) tile[i] = 1000 + i;
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncwarp();
  // Lanes 0-15 each own one box (x from bits 0-2, y from bit 3); the rest
  // repeat lane & 15's.
  const int x = (lane & 7) * kBox, y = (lane >> 3 & 1) * kBox;
  const unsigned s = static_cast<unsigned>(__cvta_generic_to_shared(tile));
  asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group [%0, {%1, %2}], [%3];"
               :: "l"(&map), "r"(x), "r"(y), "r"(s) : "memory");
  asm volatile("cp.async.bulk.commit_group;");
  asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
}

int main() {
  int* d;
  cudaMalloc(&d, kW * kH * sizeof(int));
  cudaMemset(d, 0, kW * kH * sizeof(int));
  CUtensorMap map;
  const cuuint64_t dims[2] = {kW, kH}, strides[1] = {kW * sizeof(int)};
  const cuuint32_t box[2] = {kBox, kBox}, es[2] = {1, 1};
  if (cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_INT32, 2, d, dims, strides, box, es,
                             CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                             CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) != CUDA_SUCCESS) {
    std::printf("FAIL: cuTensorMapEncodeTiled\n");
    return 1;
  }
  scatter_tiles<<<1, 32>>>(map);
  int h[kW * kH];
  const cudaError_t e = cudaMemcpy(h, d, sizeof h, cudaMemcpyDeviceToHost);
  int bad = e != cudaSuccess;
  for (int r = 0; r < kH; ++r)
    for (int c = 0; c < kW; ++c) {
      const int want = 1000 + r % kBox * kBox + c % kBox;
      if (h[r * kW + c] != want && bad++ < 5) std::printf("FAIL: [%d][%d] = %d, want %d\n", r, c, h[r * kW + c], want);
    }
  std::printf(bad ? "FAIL (%d)\n" : "PASS\n", bad);
  return bad != 0;
}
