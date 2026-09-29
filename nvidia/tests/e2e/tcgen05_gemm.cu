// Blackwell's tensor core through tcgen05, written out in PTX (sm_100a):
// Tensor Memory allocated, A and B put in shared memory in the canonical
// K-major layout, their descriptors built, M=128 x N=64 products of several
// kinds accumulated over K and committed to an mbarrier, the accumulator read
// back with tcgen05.ld and checked exactly against the host; then A taken
// from Tensor Memory (tcgen05.cp), a store/load round trip in other shapes,
// and tcgen05.shift; and Blackwell's TMA forms (.tile::gather4/scatter4, a
// CTA pair loading onto one barrier). run_sass_archs.sh builds it for sm_100a
// and checks the SASS run against the PTX one. Prints PASS on the last line.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>

constexpr int M = 128, N = 64, KSTEPS = 4;

// A kind's shape: bytes per element, K per instruction, and the
// instruction descriptor's type fields (Table 51).
struct Kind {
  const char* name;
  int eb;         // bytes per A/B element
  int k;          // K per tcgen05.mma
  uint32_t idesc;
};

__device__ __forceinline__ unsigned smem_addr(const void* p) {
  return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

// The canonical K-major layout without swizzle: 8-row core matrices of 16
// bytes, K-adjacent ones LBO bytes apart, 8-row groups SBO bytes apart.
__device__ __host__ inline uint32_t kmajor_offset(int mn, int kbyte, int kbytes) {
  const int lbo = 128, sbo = (kbytes / 16) * 128;
  return (mn % 8) * 16 + kbyte % 16 + (kbyte / 16) * lbo + (mn / 8) * sbo;
}

__device__ __forceinline__ uint64_t desc(unsigned addr, int kbytes) {
  const uint64_t lbo = 128, sbo = (kbytes / 16) * 128;
  return (uint64_t(addr >> 4) & 0x3FFF) | ((lbo >> 4) << 16) | ((sbo >> 4) << 32) | (uint64_t(1) << 46);
}

// D (+)= A * B over KSTEPS instructions. A is M x K and B is N x K (both
// K-major), raw bytes; D comes back as 32-bit words, row-major M x N.
template <int KIND>
__global__ void gemm(const uint8_t* a, const uint8_t* b, uint32_t idesc, int eb, int kinst, uint32_t* d) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  extern __shared__ __align__(1024) uint8_t smem[];
  __shared__ uint32_t taddr_slot;
  __shared__ __align__(8) uint64_t bar;
  const int t = threadIdx.x, warp = t / 32;
  const int kbytes = kinst * eb * KSTEPS;               // one row of A or B
  uint8_t* sa = smem;
  uint8_t* sb = smem + M * kbytes;
  for (int i = t; i < M * kbytes; i += blockDim.x) sa[kmajor_offset(i / kbytes, i % kbytes, kbytes)] = a[i];
  for (int i = t; i < N * kbytes; i += blockDim.x) sb[kmajor_offset(i / kbytes, i % kbytes, kbytes)] = b[i];
  if (warp == 0)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 64;" :: "r"(smem_addr(&taddr_slot)));
  if (t == 0) asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(smem_addr(&bar)));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tm = taddr_slot;
  if (t == 0) {
    for (int s = 0; s < KSTEPS; ++s) {
      // Step s starts kinst * eb bytes further along K: whole 16-byte core
      // matrix columns, s * (bytes per step / 16) * LBO.
      const int step = s * (kinst * eb / 16) * 128;
      const uint64_t da = desc(smem_addr(sa) + step, kbytes), db = desc(smem_addr(sb) + step, kbytes);
      const uint32_t acc = s > 0;
      if (KIND == 0)
        asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, q; }"
                     :: "r"(tm), "l"(da), "l"(db), "r"(idesc), "r"(acc));
      else if (KIND == 1)
        asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, q; }"
                     :: "r"(tm), "l"(da), "l"(db), "r"(idesc), "r"(acc));
      else if (KIND == 2)
        asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, q; }"
                     :: "r"(tm), "l"(da), "l"(db), "r"(idesc), "r"(acc));
      else
        asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, q; }"
                     :: "r"(tm), "l"(da), "l"(db), "r"(idesc), "r"(acc));
    }
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(smem_addr(&bar)));
  }
  unsigned done = 0;
  while (!done)
    asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                 : "=r"(done) : "r"(smem_addr(&bar)) : "memory");
  asm volatile("tcgen05.fence::after_thread_sync;");
  // Each warp reads its 32 lanes (rows), 64 columns, 16 at a time.
  for (int c = 0; c < N; c += 16) {
    uint32_t r[16];
    asm volatile("tcgen05.ld.sync.aligned.32x32b.x16.b32 {%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15}, [%16];"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]), "=r"(r[4]), "=r"(r[5]), "=r"(r[6]), "=r"(r[7]),
                   "=r"(r[8]), "=r"(r[9]), "=r"(r[10]), "=r"(r[11]), "=r"(r[12]), "=r"(r[13]), "=r"(r[14]), "=r"(r[15])
                 : "r"(tm + ((32 * warp) << 16) + c));
    asm volatile("tcgen05.wait::ld.sync.aligned;");
    for (int i = 0; i < 16; ++i) d[(32 * warp + t % 32) * N + c + i] = r[i];
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  if (warp == 0) {
    asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 64;" :: "r"(tm));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
#endif
}

// Tensor Memory moves: a pattern stored in one shape, read back in others,
// copied in from shared memory (tcgen05.cp .128x256b) and shifted down a lane
// (tcgen05.shift) -- rows 0-30 of the copy one lane down in each warp's 32
// lanes. out: per thread, 16 words.
__global__ void moves(uint32_t* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __shared__ __align__(1024) uint8_t smem[128 * 32];
  __shared__ uint32_t taddr_slot;
  const int t = threadIdx.x, warp = t / 32, l = t % 32;
  for (int i = t; i < 128 * 32; i += blockDim.x) smem[kmajor_offset(i / 32, i % 32, 32)] = uint8_t(i * 7 + 3);
  if (warp == 0)
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 32;" :: "r"(smem_addr(&taddr_slot)));
  asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  const uint32_t tm = taddr_slot, mine = tm + ((32 * warp) << 16);
  uint32_t* o = out + 16 * t;
  // 16x64b: two values a thread, in the warp's lanes.
  asm volatile("tcgen05.st.sync.aligned.16x64b.x2.b32 [%0], {%1, %2};" :: "r"(mine), "r"(t * 3 + 1), "r"(t * 5 + 2));
  asm volatile("tcgen05.wait::st.sync.aligned;");
  uint32_t r[4];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(mine));
  asm volatile("tcgen05.ld.sync.aligned.16x128b.x1.b32 {%0, %1}, [%2];" : "=r"(r[2]), "=r"(r[3]) : "r"(mine));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  for (int i = 0; i < 4; ++i) o[i] = r[i];
  // 16-bit packing: stored two halves to a register, read unpacked.
  asm volatile("tcgen05.st.sync.aligned.32x32b.x1.unpack::16b.b32 [%0], {%1};" :: "r"(mine + 8), "r"(0x12340000u + t));
  asm volatile("tcgen05.wait::st.sync.aligned;");
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x2.b32 {%0, %1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(mine + 8));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  o[4] = r[0];
  o[5] = r[1];
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  if (t == 0) {
    const uint64_t d = desc(smem_addr(smem), 32);
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(tm + 16), "l"(d));
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" :: "r"(tm + 16));
  }
  asm volatile("tcgen05.fence::before_thread_sync;");
  __syncthreads();
  asm volatile("tcgen05.fence::after_thread_sync;");
  uint32_t c[8];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x8.b32 {%0, %1, %2, %3, %4, %5, %6, %7}, [%8];"
               : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3]), "=r"(c[4]), "=r"(c[5]), "=r"(c[6]), "=r"(c[7])
               : "r"(mine + 16));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  for (int i = 0; i < 8; ++i) o[6 + i] = c[i];
  o[14] = l;
  o[15] = warp;
  __syncthreads();
  if (warp == 0) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 32;" :: "r"(tm));
#endif
}

// Blackwell's TMA forms: four rows gathered from a tensor (.tile::gather4)
// and scattered back in reverse order (.tile::scatter4); then a CTA pair
// (.cta_group::2) each loading a tile that completes on the leader's barrier.
// The tensor is 16 rows of 64 floats; the box is 32 floats by 1 row.
__global__ void gather_scatter(const __grid_constant__ CUtensorMap src, const __grid_constant__ CUtensorMap dst,
                               int col, int r0, int r1, int r2, int r3) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __shared__ __align__(128) float buf[4 * 32];
  __shared__ __align__(8) uint64_t bar;
  const unsigned sb = smem_addr(buf), b = smem_addr(&bar);
  if (threadIdx.x == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(b));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"(4 * 32 * 4));
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.tile::gather4.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5, %6}], [%7];"
                 :: "r"(sb), "l"(&src), "r"(col), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(b) : "memory");
    unsigned done = 0;
    while (!done)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(b) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.tile::scatter4.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
                 :: "l"(&dst), "r"(col), "r"(r3), "r"(r2), "r"(r1), "r"(r0), "r"(sb) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group 0;");
  }
#endif
}

__global__ void __cluster_dims__(2, 1, 1) pair_load(const __grid_constant__ CUtensorMap src, float* out) {
#if defined(__CUDA_ARCH_FEAT_SM100_ALL) || defined(__CUDA_ARCH_FEAT_SM103_ALL)
  __shared__ __align__(128) float buf[32];
  __shared__ __align__(8) uint64_t bar;
  unsigned rank;
  asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
  const unsigned b = smem_addr(&bar);
  if (threadIdx.x == 0 && rank == 0) {
    asm volatile("mbarrier.init.shared.b64 [%0], 1;" :: "r"(b));
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
    asm volatile("mbarrier.arrive.expect_tx.shared.b64 _, [%0], %1;" :: "r"(b), "r"(2 * 32 * 4));
  }
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  if (threadIdx.x == 0) {
    unsigned leader_bar;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(leader_bar) : "r"(b));
    // Each CTA loads row `rank` + 3 into its own buffer; both complete on rank 0's barrier.
    asm volatile("cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(smem_addr(buf)), "l"(&src), "r"(0), "r"(static_cast<int>(rank) + 3), "r"(leader_bar) : "memory");
  }
  if (rank == 0) {
    unsigned done = 0;
    while (!done)
      asm volatile("{ .reg .pred p; mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0; selp.u32 %0, 1, 0, p; }"
                   : "=r"(done) : "r"(b) : "memory");
  }
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
  out[rank * 32 + threadIdx.x] = buf[threadIdx.x];
  asm volatile("barrier.cluster.arrive.aligned; barrier.cluster.wait.aligned;" ::: "memory");
#endif
}

static int g_fail = 0;

// A 2D float tensor map: 16 rows of 64, box 32 x 1.
static bool tensor_map(CUtensorMap* m, float* base) {
  const cuuint64_t dims[2] = {64, 16}, strides[1] = {64 * sizeof(float)};
  const cuuint32_t box[2] = {32, 1}, elem[2] = {1, 1};
  return cuTensorMapEncodeTiled(m, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, base, dims, strides, box, elem,
                                CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
                                CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE) == CUDA_SUCCESS;
}

static void run_tma() {
  std::vector<float> h(16 * 64);
  for (int i = 0; i < 16 * 64; ++i) h[i] = float(i);
  float *dsrc, *ddst, *dout;
  cudaMalloc(&dsrc, h.size() * 4);
  cudaMalloc(&ddst, h.size() * 4);
  cudaMalloc(&dout, 64 * 4);
  cudaMemcpy(dsrc, h.data(), h.size() * 4, cudaMemcpyHostToDevice);
  cudaMemset(ddst, 0, h.size() * 4);
  CUtensorMap ms, md;
  if (!tensor_map(&ms, dsrc) || !tensor_map(&md, ddst)) {
    std::printf("FAIL: cuTensorMapEncodeTiled\n");
    ++g_fail;
    return;
  }
  gather_scatter<<<1, 32>>>(ms, md, 32, 1, 5, 9, 14);
  cudaError_t e = cudaDeviceSynchronize();
  std::vector<float> d(h.size());
  cudaMemcpy(d.data(), ddst, d.size() * 4, cudaMemcpyDeviceToHost);
  int bad = 0;
  const int rows[4] = {1, 5, 9, 14};
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 64; ++c) {
      float want = 0;
      for (int i = 0; i < 4; ++i)
        if (r == rows[3 - i] && c >= 32) want = h[rows[i] * 64 + c];
      bad += d[r * 64 + c] != want;
    }
  std::printf("%-28s %s, %d of %d wrong\n", "gather4 then scatter4", cudaGetErrorString(e), bad, 16 * 64);
  g_fail += bad || e != cudaSuccess;
  pair_load<<<2, 32>>>(ms, dout);
  e = cudaDeviceSynchronize();
  std::vector<float> o(64);
  cudaMemcpy(o.data(), dout, o.size() * 4, cudaMemcpyDeviceToHost);
  bad = 0;
  for (int r = 0; r < 2; ++r)
    for (int c = 0; c < 32; ++c) bad += o[r * 32 + c] != h[(r + 3) * 64 + c];
  std::printf("%-28s %s, %d of 64 wrong\n", "CTA-pair load", cudaGetErrorString(e), bad);
  g_fail += bad || e != cudaSuccess;
  cudaFree(dsrc);
  cudaFree(ddst);
  cudaFree(dout);
}

static float half_to_float(uint16_t h) {
  const uint32_t s = (h >> 15) & 1, e = (h >> 10) & 0x1f, m = h & 0x3ff;
  uint32_t f;
  if (e == 0) f = s << 31;   // the test's values are normal or zero
  else f = (s << 31) | ((e + 112) << 23) | (m << 13);
  float x;
  std::memcpy(&x, &f, 4);
  return x;
}

template <int KIND>
static void run_gemm(const char* name, int eb, int kinst, uint32_t type_bits, int dtype) {
  const int kbytes = kinst * eb * KSTEPS, K = kinst * KSTEPS;
  std::vector<uint8_t> a(M * kbytes), b(N * kbytes);
  std::vector<double> av(M * K), bv(N * K);
  // Small integers, exact in every type used: f16 (0x3c00 = 1), tf32 (as
  // f32), s8, e4m3 (0x38 = 1).
  auto put = [&](std::vector<uint8_t>& raw, std::vector<double>& val, int row, int k, int v) {
    val[row * K + k] = v;
    uint8_t* p = &raw[row * kbytes + k * eb];
    if (KIND == 0) {
      uint16_t h = 0;
      if (v) {
        const int m = v < 0 ? -v : v;
        int e = 0;
        while ((m >> e) > 1) ++e;
        h = uint16_t(((v < 0) << 15) | ((e + 15) << 10) | ((m << (10 - e)) & 0x3ff));
      }
      std::memcpy(p, &h, 2);
    } else if (KIND == 1) {
      const float f = float(v);
      std::memcpy(p, &f, 4);
    } else if (KIND == 2) {
      p[0] = uint8_t(int8_t(v));
    } else {
      // e4m3: 1 sign, 4 exponent (bias 7), 3 mantissa.
      uint8_t q = 0;
      if (v) {
        const int m = v < 0 ? -v : v;
        int e = 0;
        while ((m >> e) > 1) ++e;
        q = uint8_t(((v < 0) << 7) | ((e + 7) << 3) | ((m << (3 - e)) & 7));
      }
      p[0] = q;
    }
  };
  for (int r = 0; r < M; ++r)
    for (int k = 0; k < K; ++k) put(a, av, r, k, (r * 3 + k * 5) % 7 - 3);
  for (int r = 0; r < N; ++r)
    for (int k = 0; k < K; ++k) put(b, bv, r, k, (r * 5 + k * 3) % 5 - 2);
  const uint32_t idesc = (uint32_t(dtype) << 4) | type_bits | (uint32_t(N >> 3) << 17) | (uint32_t(M >> 4) << 24);
  uint8_t *da, *db;
  uint32_t* dd;
  cudaMalloc(&da, a.size());
  cudaMalloc(&db, b.size());
  cudaMalloc(&dd, M * N * 4);
  cudaMemcpy(da, a.data(), a.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(db, b.data(), b.size(), cudaMemcpyHostToDevice);
  cudaMemset(dd, 0, M * N * 4);
  const int smem = (M + N) * kbytes;
  cudaFuncSetAttribute(gemm<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
  gemm<KIND><<<1, 128, smem>>>(da, db, idesc, eb, kinst, dd);
  const cudaError_t e = cudaDeviceSynchronize();
  std::vector<uint32_t> d(M * N);
  cudaMemcpy(d.data(), dd, d.size() * 4, cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double want = 0;
      for (int k = 0; k < K; ++k) want += av[m * K + k] * bv[n * K + k];
      double got;
      if (KIND == 2) got = int32_t(d[m * N + n]);
      else if (dtype == 0) got = half_to_float(uint16_t(d[m * N + n]));
      else {
        float f;
        std::memcpy(&f, &d[m * N + n], 4);
        got = f;
      }
      bad += got != want;
    }
  std::printf("%-28s %s, %d of %d wrong\n", name, cudaGetErrorString(e), bad, M * N);
  g_fail += bad || e != cudaSuccess;
  cudaFree(da);
  cudaFree(db);
  cudaFree(dd);
}

int main() {
  // Table 51: A and B types at 7-9 and 10-12; D's at 4-5 (0 f16, 1 f32, 2 s32).
  run_gemm<0>("f16 x f16 -> f32", 2, 16, 0, 1);
  run_gemm<0>("f16 x f16 -> f16", 2, 16, 0, 0);
  run_gemm<1>("tf32 x tf32 -> f32", 4, 8, (2u << 7) | (2u << 10), 1);
  run_gemm<2>("s8 x s8 -> s32", 1, 32, (1u << 7) | (1u << 10), 2);
  run_gemm<3>("e4m3 x e4m3 -> f32", 1, 32, 0, 1);
  run_tma();

  uint32_t* dm;
  cudaMalloc(&dm, 128 * 16 * 4);
  moves<<<1, 128>>>(dm);
  const cudaError_t e = cudaDeviceSynchronize();
  std::vector<uint32_t> o(128 * 16);
  cudaMemcpy(o.data(), dm, o.size() * 4, cudaMemcpyDeviceToHost);
  // A checksum of the moves: the SASS and PTX runs must agree on every word,
  // and a few are checked here.
  uint64_t sum = 0;
  for (uint32_t v : o) sum = sum * 31 + v;
  std::printf("moves: %s, checksum %016llx\n", cudaGetErrorString(e), static_cast<unsigned long long>(sum));
  g_fail += e != cudaSuccess;
  for (int t = 0; t < 128; ++t) {
    const uint32_t* w = &o[16 * t];
    if (w[14] != uint32_t(t % 32) || w[15] != uint32_t(t / 32)) {
      std::printf("FAIL: moves thread %d wrote the wrong ids\n", t);
      ++g_fail;
      break;
    }
    // Unpacked 16-bit store: low half in column 8, high half in column 9.
    if (w[4] != (uint32_t(t) & 0xFFFF) || w[5] != 0x1234) {
      std::printf("FAIL: moves thread %d unpacked store read back %08x %08x\n", t, w[4], w[5]);
      ++g_fail;
      break;
    }
  }
  std::printf(g_fail ? "FAIL (%d)\n" : "PASS\n", g_fail);
  return g_fail != 0;
}
