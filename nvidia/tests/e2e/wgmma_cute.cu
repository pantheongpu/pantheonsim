// Hopper warpgroup MMA (wgmma), checked against CuTe.
//
// The layouts wgmma reads are easy to get plausibly wrong: a misread
// descriptor field or swizzle returns a matrix full of real input values in
// the wrong places, and a test that builds its shared-memory tiles with the
// same misunderstanding agrees with it. So none of the layouts here are
// written by hand. CuTe (CUTLASS's layout library, BSD-3-Clause, fetched by the
// runner script) builds the shared-memory tiles, the matrix descriptors and
// the register fragments from NVIDIA's own definitions, the same code that
// runs on H100s in production; the only thing this file supplies is the
// arithmetic to check the answer against. The kernel is the shape of CUTLASS's
// examples/cute/tutorial/hopper/wgmma_sm90.cu, reduced to one tile and made to
// verify its result.
//
// Every input is a small multiple of 1/2, so every product and every partial
// sum is exact in every accumulator type used and the comparison is exact:
// any difference is a wrong element, not rounding.
#include <cute/tensor.hpp>
#include <cutlass/arch/barrier.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace cute;
namespace G = SM90::GMMA;

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  printf("FAIL %s -> %s\n", #x, cudaGetErrorString(e_)); exit(1); } } while (0)

// One tile, the whole problem: M = 64 x warpgroups, N = the atom's N, and K
// in `KT` tiles of BK so the loop, the pipeline refill and scale-d all run.
template <class TA_, class TB_, class TC_, class Atom, class SmemA_, class SmemB_, int WG_,
          int BK_, int KT_, bool RS_ = false, bool ZeroFirst_ = false>
struct Cfg {
  using TA = TA_;
  using TB = TB_;
  using TC = TC_;
  using MmaAtom = Atom;
  static constexpr int WG = WG_, BK = BK_, KT = KT_;
  static constexpr bool RS = RS_, ZeroFirst = ZeroFirst_;
  static constexpr int BM = 64 * WG;
  static constexpr int BN = decltype(size<1>(typename MMA_Traits<Atom>::Shape_MNK{}))::value;
  using SmemA = decltype(tile_to_shape(SmemA_{}, make_shape(Int<BM>{}, Int<BK>{})));
  using SmemB = decltype(tile_to_shape(SmemB_{}, make_shape(Int<BN>{}, Int<BK>{})));
};

template <class C>
__global__ void gemm_tile(typename C::TA const* A, typename C::TB const* B, typename C::TC* D) {
  using TA = typename C::TA;
  using TB = typename C::TB;
  using TC = typename C::TC;
  __shared__ alignas(1024) TA smem_a[cosize_v<typename C::SmemA>];
  __shared__ alignas(1024) TB smem_b[cosize_v<typename C::SmemB>];

  // Global: A is M x K and B is N x K, both column-major.
  Tensor gA = make_tensor(make_gmem_ptr(A), make_shape(Int<C::BM>{}, Int<C::BK * C::KT>{}));
  Tensor gB = make_tensor(make_gmem_ptr(B), make_shape(Int<C::BN>{}, Int<C::BK * C::KT>{}));
  Tensor gD = make_tensor(make_gmem_ptr(D), make_shape(Int<C::BM>{}, Int<C::BN>{}));
  Tensor sA = make_tensor(make_smem_ptr(smem_a), typename C::SmemA{});
  Tensor sB = make_tensor(make_smem_ptr(smem_b), typename C::SmemB{});

  auto mma = make_tiled_mma(typename C::MmaAtom{}, Layout<Shape<Int<C::WG>, _1, _1>>{});
  ThrMMA thr = mma.get_slice(threadIdx.x);
  Tensor tCsA = thr.partition_A(sA);
  Tensor tCsB = thr.partition_B(sB);
  Tensor tCgD = thr.partition_C(gD);
  Tensor tCrB = thr.make_fragment_B(tCsB);
  Tensor tCrD = thr.make_fragment_C(tCgD);
  // With ZeroFirst the accumulator starts as garbage and the first product
  // is issued with scale-d false, which must ignore it.
  if constexpr (C::ZeroFirst) fill(tCrD, TC(7));
  else clear(tCrD);
  mma.accumulate_ = C::ZeroFirst ? G::ScaleOut::Zero : G::ScaleOut::One;

  for (int kt = 0; kt < C::KT; ++kt) {
    Tensor gA_k = local_tile(gA, make_shape(Int<C::BM>{}, Int<C::BK>{}), make_coord(0, kt));
    Tensor gB_k = local_tile(gB, make_shape(Int<C::BN>{}, Int<C::BK>{}), make_coord(0, kt));
    // CuTe's tensor indexing applies the swizzle; the position-independent
    // view takes it from the address bits, as the hardware does.
    Tensor wA = as_position_independent_swizzle_tensor(sA);
    Tensor wB = as_position_independent_swizzle_tensor(sB);
    for (int i = threadIdx.x; i < size(wA); i += blockDim.x) wA(i) = gA_k(i);
    for (int i = threadIdx.x; i < size(wB); i += blockDim.x) wB(i) = gB_k(i);
    __syncthreads();
    cutlass::arch::fence_view_async_shared();   // generic writes -> async-proxy reads

    // A is descriptors into shared memory, or for the RS atoms registers
    // loaded from it first.
    Tensor tCrA = thr.make_fragment_A(tCsA);
    if constexpr (C::RS) copy(tCsA, tCrA);
    warpgroup_fence_operand(tCrD);
    warpgroup_arrive();
    for (int kb = 0; kb < size<2>(tCrA); ++kb) {
      cute::gemm(mma, tCrA(_, _, kb), tCrB(_, _, kb), tCrD);
      mma.accumulate_ = G::ScaleOut::One;
    }
    warpgroup_commit_batch();
    warpgroup_wait<0>();
    warpgroup_fence_operand(tCrD);
    __syncthreads();   // before the next tile overwrites shared memory
  }
  copy(tCrD, tCgD);
}

static int value_code(int i) { return static_cast<int>((static_cast<unsigned>(i) * 1103515245u + 12345u) >> 16); }

// A small multiple of 1/2 in [-2, 2] (or [0, 4] unsigned, integers for the
// integer types), different for every index.
template <class T>
static double sample(int i) {
  const int c = value_code(i);
  if constexpr (std::is_same_v<T, int8_t>) return (c % 9) - 4;
  else if constexpr (std::is_same_v<T, uint8_t>) return c % 9;
  else return ((c % 9) - 4) * 0.5;
}

template <class T>
static double to_double(T v) {
  if constexpr (std::is_integral_v<T>) return static_cast<double>(v);
  else return static_cast<double>(static_cast<float>(v));
}

template <class C>
static bool run(const char* name, double sign = 1.0) {
  using TA = typename C::TA;
  using TB = typename C::TB;
  using TC = typename C::TC;
  constexpr int M = C::BM, N = C::BN, K = C::BK * C::KT;
  std::vector<TA> a(M * K);
  std::vector<TB> b(N * K);
  for (int i = 0; i < M * K; ++i) a[i] = TA(sample<TA>(i));
  for (int i = 0; i < N * K; ++i) b[i] = TB(sample<TB>(i + 7919));
  TA* da;
  TB* db;
  TC* dd;
  CK(cudaMalloc(&da, sizeof(TA) * M * K));
  CK(cudaMalloc(&db, sizeof(TB) * N * K));
  CK(cudaMalloc(&dd, sizeof(TC) * M * N));
  CK(cudaMemcpy(da, a.data(), sizeof(TA) * M * K, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, b.data(), sizeof(TB) * N * K, cudaMemcpyHostToDevice));
  gemm_tile<C><<<1, 128 * C::WG>>>(da, db, dd);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<TC> d(M * N);
  CK(cudaMemcpy(d.data(), dd, sizeof(TC) * M * N, cudaMemcpyDeviceToHost));
  CK(cudaFree(da));
  CK(cudaFree(db));
  CK(cudaFree(dd));
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double want = 0;
      for (int k = 0; k < K; ++k) want += to_double(a[m + k * M]) * to_double(b[n + k * N]);
      want *= sign;
      const double got = to_double(d[m + n * M]);
      if (got != want && bad++ < 4)
        printf("  %s: D[%d][%d] = %g, expected %g\n", name, m, n, got, want);
    }
  printf("%s %s\n", bad ? "FAIL" : "ok  ", name);
  return bad == 0;
}

int main() {
  using half = cute::half_t;
  using bf16 = cute::bfloat16_t;
  using tf32 = cute::tfloat32_t;
  using e4m3 = cute::float_e4m3_t;
  using e5m2 = cute::float_e5m2_t;
  using K = G::Major;
  bool ok = true;
  // f16 with an f16 accumulator, both layouts in each operand, 128B swizzle.
  ok &= run<Cfg<half, half, half, G::MMA_64x64x16_F16F16F16_SS<K::K, K::K>,
                G::Layout_K_SW128_Atom<half>, G::Layout_K_SW128_Atom<half>, 1, 64, 2>>(
      "f16->f16  K/K    SW128");
  ok &= run<Cfg<half, half, half, G::MMA_64x64x16_F16F16F16_SS<K::MN, K::MN>,
                G::Layout_MN_SW128_Atom<half>, G::Layout_MN_SW128_Atom<half>, 1, 64, 2>>(
      "f16->f16  MN/MN  SW128");
  // f32 accumulator; mixed majorness and the two narrower swizzles.
  ok &= run<Cfg<half, half, float, G::MMA_64x64x16_F32F16F16_SS<K::K, K::MN>,
                G::Layout_K_SW64_Atom<half>, G::Layout_MN_SW32_Atom<half>, 1, 32, 2>>(
      "f16->f32  K/MN   SW64/SW32");
  ok &= run<Cfg<half, half, float, G::MMA_64x64x16_F32F16F16_SS<K::MN, K::K>,
                G::Layout_MN_SW64_Atom<half>, G::Layout_K_SW32_Atom<half>, 1, 16, 2>>(
      "f16->f32  MN/K   SW64/SW32");
  // No swizzle at all (the "interleaved" canonical layouts), where LBO and SBO
  // both matter and trade roles with majorness.
  ok &= run<Cfg<bf16, bf16, float, G::MMA_64x64x16_F32BF16BF16_SS<K::MN, K::K>,
                G::Layout_MN_INTER_Atom<bf16>, G::Layout_K_INTER_Atom<bf16>, 1, 32, 2>>(
      "bf16->f32 MN/K   none");
  ok &= run<Cfg<bf16, bf16, float, G::MMA_64x64x16_F32BF16BF16_SS<K::K, K::MN>,
                G::Layout_K_INTER_Atom<bf16>, G::Layout_MN_INTER_Atom<bf16>, 1, 32, 2>>(
      "bf16->f32 K/MN   none");
  // The widest N, and two warpgroups sharing one B.
  ok &= run<Cfg<bf16, bf16, float, G::MMA_64x256x16_F32BF16BF16_SS<K::K, K::K>,
                G::Layout_K_SW128_Atom<bf16>, G::Layout_K_SW128_Atom<bf16>, 2, 64, 1>>(
      "bf16->f32 K/K    SW128  n256  2 warpgroups");
  // tf32, 8-bit float and integer: K-major only, as the ISA requires.
  ok &= run<Cfg<tf32, tf32, float, G::MMA_64x64x8_F32TF32TF32_SS_TN<>,
                G::Layout_K_SW128_Atom<tf32>, G::Layout_K_SW128_Atom<tf32>, 1, 32, 2>>(
      "tf32->f32 K/K    SW128");
  ok &= run<Cfg<tf32, tf32, float, G::MMA_64x64x8_F32TF32TF32_SS_TN<>,
                G::Layout_K_INTER_Atom<tf32>, G::Layout_K_SW32_Atom<tf32>, 1, 16, 2>>(
      "tf32->f32 K/K    none/SW32");
  ok &= run<Cfg<e4m3, e5m2, float, G::MMA_64x64x32_F32E4M3E5M2_SS_TN<>,
                G::Layout_K_SW64_Atom<e4m3>, G::Layout_K_SW128_Atom<e5m2>, 1, 128, 2>>(
      "e4m3*e5m2->f32 K/K SW64/SW128");
  ok &= run<Cfg<int8_t, int8_t, int32_t, G::MMA_64x64x32_S32S8S8_SS_TN,
                G::Layout_K_SW128_Atom<int8_t>, G::Layout_K_INTER_Atom<int8_t>, 1, 128, 2>>(
      "s8->s32   K/K    SW128/none");
  // Mixed signedness is PTX ISA 8.4 (CUDA 12.4); older compilers cannot emit it.
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
  ok &= run<Cfg<uint8_t, int8_t, int32_t, G::MMA_64x32x32_S32U8S8_SS_TN,
                G::Layout_K_SW32_Atom<uint8_t>, G::Layout_K_SW64_Atom<int8_t>, 1, 64, 2>>(
      "u8*s8->s32 K/K   SW32/SW64");
#endif
  // A from registers.
  ok &= run<Cfg<half, half, float, G::MMA_64x64x16_F32F16F16_RS<K::K, K::MN>,
                G::Layout_K_SW128_Atom<half>, G::Layout_MN_SW128_Atom<half>, 2, 64, 2, true>>(
      "f16->f32  regs/MN SW128  2 warpgroups");
  ok &= run<Cfg<tf32, tf32, float, G::MMA_64x64x8_F32TF32TF32_RS_TN<>,
                G::Layout_K_SW128_Atom<tf32>, G::Layout_K_SW64_Atom<tf32>, 1, 32, 1, true>>(
      "tf32->f32 regs/K SW64");
#if __CUDACC_VER_MAJOR__ > 12 || (__CUDACC_VER_MAJOR__ == 12 && __CUDACC_VER_MINOR__ >= 4)
  ok &= run<Cfg<int8_t, uint8_t, int32_t, G::MMA_64x16x32_S32S8U8_RS_TN,
                G::Layout_K_SW128_Atom<int8_t>, G::Layout_K_SW128_Atom<uint8_t>, 1, 128, 1, true>>(
      "s8*u8->s32 regs/K SW128");
#else
  ok &= run<Cfg<int8_t, int8_t, int32_t, G::MMA_64x16x32_S32S8S8_RS_TN,
                G::Layout_K_SW128_Atom<int8_t>, G::Layout_K_SW128_Atom<int8_t>, 1, 128, 1, true>>(
      "s8->s32   regs/K SW128");
#endif
  // The negate flag on A, and scale-d false on the very first product.
  ok &= run<Cfg<half, half, float,
                G::MMA_64x64x16_F32F16F16_SS<K::K, K::K, G::ScaleIn::Neg, G::ScaleIn::One>,
                G::Layout_K_SW128_Atom<half>, G::Layout_K_SW128_Atom<half>, 1, 64, 2>>(
      "f16->f32  -A", -1.0);
  ok &= run<Cfg<half, half, float, G::MMA_64x64x16_F32F16F16_SS<K::K, K::K>,
                G::Layout_K_SW128_Atom<half>, G::Layout_K_SW128_Atom<half>, 1, 64, 2, false, true>>(
      "f16->f32  scale-d=0 first");
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
