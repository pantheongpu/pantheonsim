/***************************************************************************************************
 * Copyright (c) 2024 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
#include <cstdlib>
// A Hopper GEMM the way CUTLASS writes one, checked element by element.
//
// Adapted from CUTLASS's examples/cute/tutorial/hopper/wgmma_tma_sm90.cu (the
// notice above is that file's), which runs but does not check its answer.
// What it exercises together, which the smaller tests take one at a time:
//   - 2-CTA thread-block clusters and barrier.cluster (cute::cluster_sync),
//   - a 3-stage pipeline: TMA tile loads completing on mbarriers through
//     expect-tx byte counts, consumer barriers with 128 arrivals, parity waits
//     that wrap around the stages,
//   - tensor maps from cuTensorMapEncodeTiled, reached through
//     cudaGetDriverEntryPoint and passed as __grid_constant__ parameters,
//   - wgmma reading the TMA-swizzled tiles through descriptors.
// K is four times the tile's K, so every stage is refilled and every barrier
// changes phase more than once. Inputs are multiples of 1/2 and sums stay
// exact in f32, so the comparison is exact.
#include <cutlass/cutlass.h>
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>
#include <cutlass/cluster_launch.hpp>
#include <cutlass/device_kernel.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

template <class TA, class TB, class SmemLayoutA, class SmemLayoutB>
struct SharedStorage {
  alignas(1024) ArrayEngine<TA, cosize_v<SmemLayoutA>> A;
  alignas(1024) ArrayEngine<TB, cosize_v<SmemLayoutB>> B;
  uint64_t tma_barrier[size<2>(SmemLayoutA{})];
  uint64_t mma_barrier[size<2>(SmemLayoutA{})];
};

template <class Shape, class Tiler, class TA, class SmemLayoutA, class TmaA, class TB,
          class SmemLayoutB, class TmaB, class TC, class TiledMma>
__global__ static __launch_bounds__(decltype(size(TiledMma{}))::value) void gemm_device(
    Shape shape_MNK, Tiler cta_tiler, CUTLASS_GRID_CONSTANT TmaA const tma_a,
    CUTLASS_GRID_CONSTANT TmaB const tma_b, TC* out, TiledMma mma) {
  auto [M, N, K] = shape_MNK;
  Tensor mA = tma_a.get_tma_tensor(make_shape(M, K));
  Tensor mB = tma_b.get_tma_tensor(make_shape(N, K));
  Tensor mC = make_tensor(make_gmem_ptr(out), make_shape(M, N), make_stride(Int<1>{}, M));
  auto coord = make_coord(blockIdx.x, blockIdx.y, _);
  Tensor gA = local_tile(mA, cta_tiler, coord, Step<_1, X, _1>{});
  Tensor gB = local_tile(mB, cta_tiler, coord, Step<X, _1, _1>{});
  Tensor gC = local_tile(mC, cta_tiler, coord, Step<_1, _1, X>{});

  extern __shared__ char shared_memory[];
  using Storage = SharedStorage<TA, TB, SmemLayoutA, SmemLayoutB>;
  Storage& smem = *reinterpret_cast<Storage*>(shared_memory);
  Tensor sA = make_tensor(make_smem_ptr(smem.A.begin()), SmemLayoutA{});
  Tensor sB = make_tensor(make_smem_ptr(smem.B.begin()), SmemLayoutB{});

  auto [tAgA, tAsA] = tma_partition(tma_a, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sA),
                                    group_modes<0, 2>(gA));
  auto [tBgB, tBsB] = tma_partition(tma_b, Int<0>{}, Layout<_1>{}, group_modes<0, 2>(sB),
                                    group_modes<0, 2>(gB));
  constexpr int kTxBytes =
      sizeof(make_tensor_like(tensor<0>(tAsA))) + sizeof(make_tensor_like(tensor<0>(tBsB)));

  const int pipes = size<1>(tAsA);
  int k_tiles = size<1>(tAgA);
  int k_tile = 0;
  const bool producer = cutlass::canonical_warp_idx_sync() == 0 && cute::elect_one_sync();
  using Full = cutlass::arch::ClusterTransactionBarrier;
  using Empty = cutlass::arch::ClusterBarrier;
  if (producer)
    for (int p = 0; p < pipes; ++p) {
      Full::init(&smem.tma_barrier[p], 1);
      Empty::init(&smem.mma_barrier[p], 128);
    }
  cluster_sync();

  for (int p = 0; p < pipes; ++p) {
    if (producer) {
      Full::arrive_and_expect_tx(&smem.tma_barrier[p], kTxBytes);
      copy(tma_a.with(smem.tma_barrier[p]), tAgA(_, k_tile), tAsA(_, p));
      copy(tma_b.with(smem.tma_barrier[p]), tBgB(_, k_tile), tBsB(_, p));
    }
    --k_tiles;
    ++k_tile;
  }

  ThrMMA thr = mma.get_thread_slice(threadIdx.x);
  Tensor tCsA = thr.partition_A(sA);
  Tensor tCsB = thr.partition_B(sB);
  Tensor tCgC = thr.partition_C(gC);
  Tensor tCrC = thr.make_fragment_C(tCgC);
  clear(tCrC);
  Tensor tCrA = thr.make_fragment_A(tCsA);
  Tensor tCrB = thr.make_fragment_B(tCsB);

  auto write_state = cutlass::PipelineState<size<2>(SmemLayoutA{})>();
  auto read_state = cutlass::PipelineState<size<2>(SmemLayoutA{})>();
  while (k_tiles > -pipes) {
    const int rp = read_state.index();
    Full::wait(&smem.tma_barrier[rp], read_state.phase());
    warpgroup_arrive();
    gemm(mma, tCrA(_, _, _, rp), tCrB(_, _, _, rp), tCrC);
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    Empty::arrive(&smem.mma_barrier[rp]);
    ++read_state;
    if (producer && k_tiles > 0) {
      const int wp = write_state.index();
      Empty::wait(&smem.mma_barrier[wp], write_state.phase());
      Full::arrive_and_expect_tx(&smem.tma_barrier[wp], kTxBytes);
      copy(tma_a.with(smem.tma_barrier[wp]), tAgA(_, k_tile), tAsA(_, wp));
      copy(tma_b.with(smem.tma_barrier[wp]), tBgB(_, k_tile), tBsB(_, wp));
      ++write_state;
    }
    --k_tiles;
    ++k_tile;
  }
  copy(tCrC, tCgC);
}

template <class T>
static double v(int i) { return ((static_cast<int>((static_cast<unsigned>(i) * 2654435761u) >> 20) % 9) - 4) * 0.5; }

// A (M x K) and B (N x K) in global memory, K-major ("TN") or M/N-major
// ("NT"); C is M x N column-major.
template <class T, bool KMajor, class AtomA, class AtomB, class MmaAtom, int BK>
static bool run(const char* name) {
  constexpr int M = 256, N = 128, K = 4 * BK;
  std::vector<T> a(M * K), b(N * K);
  auto ia = [](int m, int k) { return KMajor ? m * K + k : k * M + m; };
  auto ib = [](int n, int k) { return KMajor ? n * K + k : k * N + n; };
  for (int m = 0; m < M; ++m)
    for (int k = 0; k < K; ++k) a[ia(m, k)] = T(v<T>(m * 977 + k));
  for (int n = 0; n < N; ++n)
    for (int k = 0; k < K; ++k) b[ib(n, k)] = T(v<T>(n * 613 + k + 5));
  T *da, *db;
  float* dc;
  CK(cudaMalloc(&da, sizeof(T) * a.size()));
  CK(cudaMalloc(&db, sizeof(T) * b.size()));
  CK(cudaMalloc(&dc, sizeof(float) * M * N));
  CK(cudaMemcpy(da, a.data(), sizeof(T) * a.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b.data(), sizeof(T) * b.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dc, 0xFF, sizeof(float) * M * N));

  auto shape = make_shape(M, N, K);
  auto dA = [&] { if constexpr (KMajor) return make_stride(K, Int<1>{}); else return make_stride(Int<1>{}, M); }();
  auto dB = [&] { if constexpr (KMajor) return make_stride(K, Int<1>{}); else return make_stride(Int<1>{}, N); }();
  auto tiler = make_shape(Int<128>{}, Int<128>{}, Int<BK>{});
  auto sA = tile_to_shape(AtomA{}, make_shape(Int<128>{}, Int<BK>{}, Int<3>{}));
  auto sB = tile_to_shape(AtomB{}, make_shape(Int<128>{}, Int<BK>{}, Int<3>{}));
  TiledMMA mma = make_tiled_mma(MmaAtom{});
  Tensor mA = make_tensor(da, make_shape(M, K), dA);
  Tensor mB = make_tensor(db, make_shape(N, K), dB);
  auto tmaA = make_tma_atom(SM90_TMA_LOAD{}, mA, sA(_, _, 0), make_shape(Int<128>{}, Int<BK>{}));
  auto tmaB = make_tma_atom(SM90_TMA_LOAD{}, mB, sB(_, _, 0), make_shape(Int<128>{}, Int<BK>{}));
  const int smem = int(sizeof(SharedStorage<T, T, decltype(sA), decltype(sB)>));
  void const* k = reinterpret_cast<void const*>(
      &gemm_device<decltype(shape), decltype(tiler), T, decltype(sA), decltype(tmaA), T,
                   decltype(sB), decltype(tmaB), float, decltype(mma)>);
  CK(cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  cutlass::ClusterLaunchParams params = {dim3(M / 128, N / 128), dim3(size(mma)), dim3(2, 1, 1), smem};
  if (cutlass::launch_kernel_on_cluster(params, k, shape, tiler, tmaA, tmaB, dc, mma) !=
      cutlass::Status::kSuccess) {
    printf("FAIL %s: launch\n", name);
    return false;
  }
  CK(cudaDeviceSynchronize());
  std::vector<float> c(M * N);
  CK(cudaMemcpy(c.data(), dc, sizeof(float) * c.size(), cudaMemcpyDeviceToHost));
  CK(cudaFree(da));
  CK(cudaFree(db));
  CK(cudaFree(dc));
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double want = 0;
      for (int kk = 0; kk < K; ++kk)
        want += double(float(a[ia(m, kk)])) * double(float(b[ib(n, kk)]));
      if (double(c[m + n * M]) != want && bad++ < 4)
        printf("  %s: C[%d][%d] = %g, expected %g\n", name, m, n, c[m + n * M], want);
    }
  printf("%s %s\n", bad ? "FAIL" : "ok  ", name);
  return bad == 0;
}

int main() {
  using half = cute::half_t;
  using bf16 = cute::bfloat16_t;
  namespace G = SM90::GMMA;
  bool ok = true;
  ok &= run<half, true, G::Layout_K_SW128_Atom<half>, G::Layout_K_SW128_Atom<half>,
            SM90_64x64x16_F32F16F16_SS<G::Major::K, G::Major::K>, 64>("TN f16 SW128");
  ok &= run<half, false, G::Layout_MN_SW128_Atom<half>, G::Layout_MN_SW128_Atom<half>,
            SM90_64x64x16_F32F16F16_SS<G::Major::MN, G::Major::MN>, 64>("NT f16 SW128");
  ok &= run<bf16, true, G::Layout_K_SW64_Atom<bf16>, G::Layout_K_SW64_Atom<bf16>,
            SM90_64x128x16_F32BF16BF16_SS<G::Major::K, G::Major::K>, 32>("TN bf16 SW64 n128");
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
