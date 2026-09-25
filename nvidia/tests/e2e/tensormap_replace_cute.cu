// Retargeting a TMA descriptor on the device, through CuTe's own helpers --
// the path CUTLASS's grouped and pointer-array GEMMs take to point one
// descriptor at each group's tensors in turn:
//
// 1. in shared memory: copy the descriptor in, replace its address, extents
//    and row pitch (tma_descriptor_replace_addr_in_shared_mem and
//    tma_descriptor_replace_dims_strides_in_shared_mem), publish it to global
//    memory with tensormap.cp_fenceproxy, acquire it, and load a tile that
//    hangs past the new tensor's edges;
// 2. in global memory: replace the address in place
//    (tma_descriptor_replace_addr_in_global_mem), fence, and load again.
//
// The strides CuTe passes follow the CUDA version it is compiled with, so
// this checks the simulator's reading of global_stride against CuTe's.
// Every element is compared exactly. Prints PASS on the last line.
#include <cstdio>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

using namespace cute;

constexpr int T = 32;                  // the box: T x T floats
constexpr int M0 = 32, N0 = 64;        // the tensor the descriptor is built for
constexpr int M1 = 48, N1 = 80, P1 = 96;   // the one it is retargeted to: 48 x 80, row pitch 96
constexpr int R0 = 32, C0 = 64;        // the tile loaded from it: rows 32.., columns 64..

template <class Tma>
__global__ void retarget(CUTE_GRID_CONSTANT Tma const tma, TmaDescriptor* workspace, float const* a1,
                         float const* a2, float* out1, float* out2) {
  __shared__ alignas(128) TmaDescriptor smem_desc;
  __shared__ alignas(128) float tile[T * T];
  __shared__ alignas(8) uint64_t mbar;
  const int lane = threadIdx.x % 32;

  if (threadIdx.x < 32) {
    if (lane == 0) {
      smem_desc = *tma.get_tma_descriptor();
      tma_descriptor_replace_addr_in_shared_mem(smem_desc, a1);
      cute::array<uint32_t, 5> shape = {uint32_t(N1), uint32_t(M1), 1, 1, 1};
      // Byte strides, as CUTLASS computes them; index 0 is the element's.
      cute::array<uint64_t, 5> stride = {4, uint64_t(P1) * 4, 0, 0, 0};
      tma_descriptor_replace_dims_strides_in_shared_mem(smem_desc, shape, stride);
    }
    __syncwarp();
    tma_descriptor_cp_fence_release(workspace, smem_desc);   // the whole warp, .sync.aligned
    tma_descriptor_fence_acquire(workspace);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    initialize_barrier(mbar, 1);
    set_barrier_transaction_bytes(mbar, T * T * sizeof(float));
    const int32_t col = C0, row = R0;   // copy takes them by reference
    SM90_TMA_LOAD_2D::copy(workspace, &mbar, 0, tile, col, row);
  }
  __syncthreads();
  wait_barrier(mbar, 0);
  for (int i = threadIdx.x; i < T * T; i += blockDim.x) out1[i] = tile[i];
  __syncthreads();

  if (threadIdx.x == 0) {
    tma_descriptor_replace_addr_in_global_mem(workspace, a2);
    tma_descriptor_fence_release();
    tma_descriptor_fence_acquire(workspace);
    set_barrier_transaction_bytes(mbar, T * T * sizeof(float));
    SM90_TMA_LOAD_2D::copy(workspace, &mbar, 0, tile, 0, 0);
  }
  __syncthreads();
  wait_barrier(mbar, 1);
  for (int i = threadIdx.x; i < T * T; i += blockDim.x) out2[i] = tile[i];
}

int main() {
  std::vector<float> h0(M0 * N0, -1.0f), h1(M1 * P1), h2(M1 * P1);
  for (int i = 0; i < M1 * P1; ++i) {
    h1[i] = float(i);
    h2[i] = float(100000 + i);
  }
  float *a0, *a1, *a2, *out1, *out2;
  TmaDescriptor* workspace;
  cudaMalloc(&a0, h0.size() * 4);
  cudaMalloc(&a1, h1.size() * 4);
  cudaMalloc(&a2, h2.size() * 4);
  cudaMalloc(&out1, T * T * 4);
  cudaMalloc(&out2, T * T * 4);
  cudaMalloc(&workspace, sizeof(TmaDescriptor));
  cudaMemcpy(a0, h0.data(), h0.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(a1, h1.data(), h1.size() * 4, cudaMemcpyHostToDevice);
  cudaMemcpy(a2, h2.data(), h2.size() * 4, cudaMemcpyHostToDevice);

  Tensor g0 = make_tensor(make_gmem_ptr(a0), make_layout(make_shape(M0, N0), LayoutRight{}));
  auto smem_layout = make_layout(Shape<Int<T>, Int<T>>{}, LayoutRight{});
  auto tma = make_tma_copy(SM90_TMA_LOAD{}, g0, smem_layout);
  retarget<<<1, 128>>>(tma, workspace, a1, a2, out1, out2);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("FAIL: %s\n", cudaGetErrorString(err));
    return 1;
  }
  std::vector<float> o1(T * T), o2(T * T);
  cudaMemcpy(o1.data(), out1, T * T * 4, cudaMemcpyDeviceToHost);
  cudaMemcpy(o2.data(), out2, T * T * 4, cudaMemcpyDeviceToHost);
  int bad1 = 0, bad2 = 0;
  for (int r = 0; r < T; ++r)
    for (int c = 0; c < T; ++c) {
      const int gr = R0 + r, gc = C0 + c;
      const float w1 = (gr < M1 && gc < N1) ? h1[gr * P1 + gc] : 0.0f;
      const float w2 = h2[r * P1 + c];
      if (o1[r * T + c] != w1 && bad1++ < 4)
        std::printf("  shared: (%d,%d) = %g, want %g\n", r, c, o1[r * T + c], w1);
      if (o2[r * T + c] != w2 && bad2++ < 4)
        std::printf("  global: (%d,%d) = %g, want %g\n", r, c, o2[r * T + c], w2);
    }
  std::printf("replaced in shared memory: %d of %d wrong\n", bad1, T * T);
  std::printf("replaced in global memory: %d of %d wrong\n", bad2, T * T);
  const bool ok = bad1 == 0 && bad2 == 0;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
