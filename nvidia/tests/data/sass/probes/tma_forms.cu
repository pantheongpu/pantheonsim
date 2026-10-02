// Decoder probe: the TMA forms tma_bulk.cu leaves out -- 1D to 5D tiles,
// multicast, im2col in 3D to 5D, every tensor reduction, bulk reductions,
// shared-to-shared bulk copies, L2 cache hints, prefetches. sm_90a.
#include <cstdint>
#define C5 "{%2, %3, %4, %5, %6}"
__global__ void k(uint64_t tmap, int a, int b, int c, int d, int e, unsigned short o0, unsigned short o1,
                  unsigned short o2, unsigned short mask, uint64_t pol, uint32_t* out) {
  __shared__ alignas(128) uint32_t buf[4096];
  __shared__ uint64_t bar;
  unsigned sb = static_cast<unsigned>(__cvta_generic_to_shared(buf));
  unsigned mb = static_cast<unsigned>(__cvta_generic_to_shared(&bar));
  if (threadIdx.x == 0) {
#define LD(n, crd, ...) asm volatile("cp.async.bulk.tensor." #n "d.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1, " crd "], [%7];" \
                                     :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(mb) : "memory")
    LD(1, "{%2}");
    LD(4, "{%2, %3, %4, %5}");
    LD(5, C5);
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(mb), "h"(mask) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint [%0], [%1, {%2, %3}], [%4], %5;"
                 :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(mb), "l"(pol) : "memory");
    asm volatile("cp.async.bulk.tensor.3d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4}], [%5], {%6};"
                 :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(mb), "h"(o0) : "memory");
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3, %4, %5}], [%6], {%7, %8};"
                 :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(d), "r"(mb), "h"(o0), "h"(o1) : "memory");
    asm volatile("cp.async.bulk.tensor.5d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes [%0], [%1, " C5 "], [%7], {%8, %9, %10};"
                 :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(mb), "h"(o0), "h"(o1), "h"(o2) : "memory");
    asm volatile("cp.async.bulk.tensor.4d.shared::cluster.global.im2col.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1, {%2, %3, %4, %5}], [%6], {%7, %8}, %9;"
                 :: "r"(sb), "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(d), "r"(mb), "h"(o0), "h"(o1), "h"(mask) : "memory");
    asm volatile("cp.async.bulk.tensor.1d.global.shared::cta.bulk_group [%0, {%1}], [%2];" :: "l"(tmap), "r"(a), "r"(sb) : "memory");
    asm volatile("cp.async.bulk.tensor.5d.global.shared::cta.bulk_group [%0, {%1, %2, %3, %4, %5}], [%6];"
                 :: "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(d), "r"(e), "r"(sb) : "memory");
    asm volatile("cp.async.bulk.tensor.2d.global.shared::cta.bulk_group.L2::cache_hint [%0, {%1, %2}], [%3], %4;"
                 :: "l"(tmap), "r"(a), "r"(b), "r"(sb), "l"(pol) : "memory");
#define RD(op) asm volatile("cp.reduce.async.bulk.tensor.2d.global.shared::cta." #op ".tile.bulk_group [%0, {%1, %2}], [%3];" \
                            :: "l"(tmap), "r"(a), "r"(b), "r"(sb) : "memory")
    RD(min); RD(max); RD(inc); RD(dec); RD(and); RD(or); RD(xor);
    asm volatile("cp.reduce.async.bulk.tensor.3d.global.shared::cta.add.tile.bulk_group [%0, {%1, %2, %3}], [%4];"
                 :: "l"(tmap), "r"(a), "r"(b), "r"(c), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.s32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.u64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.s64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.f16 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.min.bf16 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.u64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.s64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.f16 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.max.bf16 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.inc.u32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.dec.u32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.and.b64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.or.b64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b32 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.xor.b64 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [%0], [%1], 256;" :: "l"(out), "r"(sb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.s32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.add.u64 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.u32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.min.s32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.u32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.max.s32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.inc.u32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.dec.u32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.and.b32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.or.b32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.reduce.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes.xor.b32 [%0], [%1], 256, [%2];" :: "r"(sb + 4096), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.shared::cta.mbarrier::complete_tx::bytes [%0], [%1], 256, [%2];"
                 :: "r"(sb + 8192), "r"(sb), "r"(mb) : "memory");
    asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster [%0], [%1], 256, [%2], %3;"
                 :: "r"(sb), "l"(out), "r"(mb), "h"(mask) : "memory");
    asm volatile("cp.async.bulk.prefetch.L2.global [%0], 256;" :: "l"(out) : "memory");
    asm volatile("cp.async.bulk.prefetch.tensor.2d.L2.global [%0, {%1, %2}];" :: "l"(tmap), "r"(a), "r"(b) : "memory");
    asm volatile("cp.async.bulk.commit_group;");
    asm volatile("cp.async.bulk.wait_group 1;");
    asm volatile("cp.async.bulk.wait_group.read 0;");
  }
  out[threadIdx.x] = buf[threadIdx.x];
}
