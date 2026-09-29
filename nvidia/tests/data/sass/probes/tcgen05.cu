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
