// TMA reductions the way CuTe issues them: SM90_TMA_REDUCE_ADD, which
// compiles to cp.reduce.async.bulk.tensor. CuTe builds the tensor map and
// partitions the tiles, so what is checked is agreement with NVIDIA's own
// layout code, not with this simulator's reading of the ISA.
//
// Three blocks add their tiles into each tile of C at once -- grid z is the
// contribution -- so every element is reduced into by several blocks running
// on different host threads, and a lost update shows. Values are small
// integers, exact in every type used, so the expected C is exact whatever
// order the blocks run in. Runs f32 through a 128-byte-swizzled tile and f16
// through an unswizzled one. Prints PASS on the last line.
#include <cstdio>
#include <vector>

#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm90_gmma.hpp>

using namespace cute;

constexpr int M = 64, N = 96, kContribs = 3, T = 32;

template <class Elem, class Tma, class SmemLayout>
__global__ void reduce_tiles(Elem const* a, CUTE_GRID_CONSTANT Tma const tma, SmemLayout smem_layout) {
  __shared__ alignas(128) Elem smem[cosize_v<SmemLayout>];
  Tensor sC = make_tensor(make_smem_ptr(smem), smem_layout);                  // (T,T)
  // A holds kContribs contributions, each M x N row-major.
  Tensor mA = make_tensor(make_gmem_ptr(a), make_shape(M, N, kContribs), make_stride(N, Int<1>{}, M * N));
  Tensor gA = local_tile(mA(_, _, blockIdx.z), Shape<Int<T>, Int<T>>{}, make_coord(blockIdx.x, blockIdx.y));
  for (int i = threadIdx.x; i < size(sC); i += blockDim.x) sC(i) = gA(i);
  __syncthreads();
  tma_store_fence();   // make the shared writes visible to the async proxy

  Tensor mC = tma.get_tma_tensor(make_shape(M, N));
  Tensor gC = local_tile(mC, Shape<Int<T>, Int<T>>{}, make_coord(blockIdx.x, blockIdx.y));
  auto cta_tma = tma.get_slice(Int<0>{});
  Tensor tCsC = cta_tma.partition_S(sC);
  Tensor tCgC = cta_tma.partition_D(gC);
  if (threadIdx.x == 0) {
    copy(tma, tCsC, tCgC);
    tma_store_arrive();
    tma_store_wait<0>();
  }
  __syncthreads();
}

template <class Elem, class SmemLayout>
bool run(const char* name, SmemLayout smem_layout) {
  std::vector<Elem> h_a(size_t(M) * N * kContribs), h_c(size_t(M) * N);
  std::vector<float> want(size_t(M) * N);
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      const float c0 = float((m + n) % 4);
      h_c[m * N + n] = Elem(c0);
      float sum = c0;
      for (int r = 0; r < kContribs; ++r) {
        const float v = float((m * 7 + n * 3 + r * 5) % 11 - 5);
        h_a[size_t(r) * M * N + m * N + n] = Elem(v);
        sum += v;
      }
      want[m * N + n] = sum;
    }
  Elem *d_a = nullptr, *d_c = nullptr;
  cudaMalloc(&d_a, h_a.size() * sizeof(Elem));
  cudaMalloc(&d_c, h_c.size() * sizeof(Elem));
  cudaMemcpy(d_a, h_a.data(), h_a.size() * sizeof(Elem), cudaMemcpyHostToDevice);
  cudaMemcpy(d_c, h_c.data(), h_c.size() * sizeof(Elem), cudaMemcpyHostToDevice);

  Tensor gC = make_tensor(make_gmem_ptr(d_c), make_layout(make_shape(M, N), LayoutRight{}));
  auto tma = make_tma_copy(SM90_TMA_REDUCE_ADD{}, gC, smem_layout);
  reduce_tiles<<<dim3(M / T, N / T, kContribs), 128>>>(d_a, tma, smem_layout);
  const cudaError_t err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::printf("FAIL: %s: %s\n", name, cudaGetErrorString(err));
    return false;
  }
  cudaMemcpy(h_c.data(), d_c, h_c.size() * sizeof(Elem), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (size_t i = 0; i < h_c.size(); ++i)
    if (float(h_c[i]) != want[i]) {
      if (bad < 5) std::printf("  %s: C[%zu] = %g, want %g\n", name, i, float(h_c[i]), want[i]);
      ++bad;
    }
  std::printf("%s: %d of %d wrong\n", name, bad, M * N);
  cudaFree(d_a);
  cudaFree(d_c);
  return bad == 0;
}

int main() {
  bool ok = true;
  // f32, 128-byte swizzle: the GMMA K-major atom, tiled to 32 x 32.
  ok &= run<float>("f32, 128B swizzle",
                   tile_to_shape(GMMA::Layout_K_SW128_Atom<float>{}, Shape<Int<T>, Int<T>>{}));
  // f16, no swizzle, row-major.
  ok &= run<half_t>("f16, no swizzle", make_layout(Shape<Int<T>, Int<T>>{}, LayoutRight{}));
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
