// Decoder probe: wgmma in every type family (HGMMA/IGMMA/BGMMA/QGMMA), A in
// registers, sparse, negation and transposes, the scale-d predicate. sm_90a.
#include <cstdint>
#define F4 "{%0,%1,%2,%3}"
__global__ void k(float* o, uint64_t da, uint64_t db, int p) {
  float d[4] = {0,0,0,0}; unsigned h[2] = {0,0}; int s[4] = {0,0,0,0};
  asm volatile("wgmma.fence.sync.aligned;");
  // m64n8k16 f32 += f16*f16
  asm volatile("{.reg .pred q; setp.ne.b32 q, %6, 0; wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 " F4 ", %4, %5, q, 1, 1, 0, 0;}"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db), "r"(p));
  // f16 accumulate
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f16.f16.f16 {%0,%1}, %2, %3, 1, 1, 1, 0, 0;" : "+r"(h[0]), "+r"(h[1]) : "l"(da), "l"(db));
  // negated A and B, transposes
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.bf16.bf16 " F4 ", %4, %5, 1, -1, -1, 1, 1;"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db));
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k8.f32.tf32.tf32 " F4 ", %4, %5, 1, 1, 1;"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db));
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f32.e5m2.e4m3 " F4 ", %4, %5, 1, 1, 1;"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db));
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.f16.e4m3.e4m3 {%0,%1}, %2, %3, 1, 1, 1;" : "+r"(h[0]), "+r"(h[1]) : "l"(da), "l"(db));
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.u8 " F4 ", %4, %5, 1;"
    : "+r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]) : "l"(da), "l"(db));
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k32.s32.s8.s8.satfinite " F4 ", %4, %5, 1;"
    : "+r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]) : "l"(da), "l"(db));
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k256.s32.b1.b1.and.popc " F4 ", %4, %5, 1;"
    : "+r"(s[0]), "+r"(s[1]), "+r"(s[2]), "+r"(s[3]) : "l"(da), "l"(db));
  // A from registers
  unsigned a0 = p, a1 = p + 1, a2 = p + 2, a3 = p + 3;
  asm volatile("wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 " F4 ", {%4,%5,%6,%7}, %8, 1, 1, 1, 1;"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "l"(db));
  // sparse
  asm volatile("wgmma.mma_async.sp.sync.aligned.m64n8k32.f32.f16.f16 " F4 ", %4, %5, %6, 0, 1, 1, 1, 0, 0;"
    : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "l"(da), "l"(db), "r"(p));
  asm volatile("wgmma.commit_group.sync.aligned;");
  asm volatile("wgmma.wait_group.sync.aligned 1;");
  asm volatile("wgmma.wait_group.sync.aligned 0;");
  o[threadIdx.x] = d[0] + d[1] + d[2] + d[3] + h[0] + h[1] + s[0] + s[1] + s[2] + s[3];
}
