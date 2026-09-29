// Decoder probe: tcgen05 -- alloc/dealloc, every mma kind (A in shared memory
// or Tensor Memory, block-scaled, sparse, .ws), commit, cp in each shape,
// shift, ld/st, the waits and fences. sm_100a.
#include <cstdint>
__global__ void k(uint64_t da, uint64_t db, uint32_t idesc, uint32_t* out, int p) {
  __shared__ uint32_t taddr;
  __shared__ uint64_t bar;
  unsigned sa = static_cast<unsigned>(__cvta_generic_to_shared(&taddr));
  unsigned mb = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x < 32) {
    asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0], 128;" :: "r"(sa));
    asm volatile("tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;");
  }
  __syncthreads();
  uint32_t t = taddr;
  if (threadIdx.x == 0) {
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], %1, %2, %3, q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::tf32 [%0], %1, %2, %3, q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::f8f6f4 [%0], %1, %2, %3, q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::i8 [%0], %1, %2, %3, q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.cta_group::1.kind::f16 [%0], [%1], %2, %3, q; }" :: "r"(t), "r"(t + 64), "l"(db), "r"(idesc), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %6, 0; tcgen05.mma.cta_group::1.kind::mxf8f6f4.block_scale [%0], %1, %2, %3, [%4], [%5], q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(t + 96), "r"(t + 100), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %6, 0; tcgen05.mma.cta_group::1.kind::mxf4nvf4.block_scale.block16 [%0], %1, %2, %3, [%4], [%5], q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(t + 96), "r"(t + 100), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %5, 0; tcgen05.mma.sp.cta_group::1.kind::f16 [%0], %1, %2, [%3], %4, q; }" :: "r"(t), "l"(da), "l"(db), "r"(t + 80), "r"(idesc), "r"(p));
    asm volatile("{ .reg .pred q; setp.ne.b32 q, %4, 0; tcgen05.mma.ws.cta_group::1.kind::f16 [%0], %1, %2, %3, q; }" :: "r"(t), "l"(da), "l"(db), "r"(idesc), "r"(p));
    asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [%0];" :: "r"(mb));
    asm volatile("tcgen05.cp.cta_group::1.128x256b [%0], %1;" :: "r"(t), "l"(da));
    asm volatile("tcgen05.cp.cta_group::1.4x256b [%0], %1;" :: "r"(t), "l"(da));
    asm volatile("tcgen05.cp.cta_group::1.128x128b [%0], %1;" :: "r"(t), "l"(da));
    asm volatile("tcgen05.cp.cta_group::1.64x128b.warpx2::02_13 [%0], %1;" :: "r"(t), "l"(da));
    asm volatile("tcgen05.cp.cta_group::1.32x128b.warpx4 [%0], %1;" :: "r"(t), "l"(da));
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" :: "r"(t));
  }
  uint32_t r[4];
  asm volatile("tcgen05.ld.sync.aligned.32x32b.x4.b32 {%0, %1, %2, %3}, [%4];" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(t));
  asm volatile("tcgen05.wait::ld.sync.aligned;");
  asm volatile("tcgen05.st.sync.aligned.16x64b.x4.b32 [%0], {%1, %2, %3, %4};" :: "r"(t), "r"(r[0]), "r"(r[1]), "r"(r[2]), "r"(r[3]));
  asm volatile("tcgen05.wait::st.sync.aligned;");
  asm volatile("tcgen05.fence::before_thread_sync;");
  asm volatile("tcgen05.fence::after_thread_sync;");
  uint32_t q[2];
  asm volatile("tcgen05.ld.sync.aligned.16x32bx2.x2.b32 {%0, %1}, [%2], 2;" : "=r"(q[0]), "=r"(q[1]) : "r"(t));
  asm volatile("tcgen05.ld.sync.aligned.16x128b.x1.pack::16b.b32 {%0, %1}, [%2];" : "=r"(q[0]), "=r"(q[1]) : "r"(t));
  out[threadIdx.x] = r[0] + r[1] + r[2] + r[3] + q[0] + q[1];
  __syncthreads();
  if (threadIdx.x < 32) asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0, 128;" :: "r"(t));
}

// Guarded forms: ptxas puts a uniform predicate on the single-thread
// tcgen05 operations here (from nvidia/tests/e2e/tcgen05_gemm.cu).
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

// Tensor Memory moves: a pattern stored in one shape, read back in others,
// copied in from shared memory (tcgen05.cp .128x256b) and shifted down a lane
// (tcgen05.shift). out: per thread, 16 words.
__global__ void guarded(uint32_t* out) {
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
    asm volatile("tcgen05.shift.cta_group::1.down [%0];" :: "r"(tm + 24));
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

